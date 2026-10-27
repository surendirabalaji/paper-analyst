#!/usr/bin/env bash
# Build the image with Cloud Build and deploy it to Cloud Run, then smoke-test the public URL.
#
# Usage: GCP_PROJECT_ID=my-project scripts/deploy.sh [--dry-run] [--no-smoke] [--yes]
#
# Settings (environment variables, defaults in brackets):
#   GCP_PROJECT_ID (required)   GCP_REGION [us-west1]   CLOUD_RUN_SERVICE [paper-analyst]
#   AR_REPO [paper-analyst]     ENV_FILE [.env]
#   MEMORY [2Gi] CPU [1] CONCURRENCY [10] TIMEOUT [300] MIN_INSTANCES [0] MAX_INSTANCES [3]
#
# Credentials: ENV_FILE is converted by scripts/env_to_yaml.py into a 0600 YAML file inside a fresh
# mktemp directory outside the repo, passed with --env-vars-file, and deleted on exit. Nothing secret is
# uploaded to Cloud Build or baked into the image (.gcloudignore and .dockerignore are allowlists).
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_FILE="${ENV_FILE:-.env}"
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-west1}"
SERVICE="${CLOUD_RUN_SERVICE:-paper-analyst}"
REPO="${AR_REPO:-paper-analyst}"
IMAGE="paper-analyst"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
CONCURRENCY="${CONCURRENCY:-10}"
TIMEOUT="${TIMEOUT:-300}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
SMOKE_QUERY="${SMOKE_QUERY:-How does Self-RAG decide when to retrieve?}"
DRY_RUN=0
SMOKE=1
ASSUME_YES=0

WORK_DIR=""
cleanup() { if [ -n "$WORK_DIR" ]; then rm -rf -- "$WORK_DIR"; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log() { printf '[deploy] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '[dry-run] '; printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-smoke) SMOKE=0 ;;
    --yes) ASSUME_YES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

# ---- preflight (changes nothing, also in dry-run) ----
for tool in gcloud python3 curl; do
  command -v "$tool" >/dev/null 2>&1 || die "install $tool first"
done
[ -n "$PROJECT_ID" ] || die "set GCP_PROJECT_ID"
[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE (copy .env.example and fill in the keys)"
for f in Dockerfile .dockerignore .gcloudignore requirements.txt app/main.py; do
  [ -e "$f" ] || die "missing $f"
done
[ -f data/chroma_index/chroma.sqlite3 ] || die "no Chroma index in data/chroma_index; run scripts/ingest.py first"
[ -f data/papers.db ] || die "no data/papers.db; run scripts/ingest.py first"
# The router needs both generation providers (OpenAI direct + Gemini via OpenRouter).
grep -Eq '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY=.+' "$ENV_FILE" && ! grep -Eq '^[[:space:]]*(export[[:space:]]+)?OPENROUTER_API_KEY=.*REPLACE_ME' "$ENV_FILE" \
  || die "OPENROUTER_API_KEY missing or placeholder in $ENV_FILE (needed for Gemini routing)"
# Live retrieval and the shared MCP tool (Phase 5) need the Tavily key.
grep -Eq '^[[:space:]]*(export[[:space:]]+)?TAVILY_API_KEY=.+' "$ENV_FILE" && ! grep -Eq '^[[:space:]]*(export[[:space:]]+)?TAVILY_API_KEY=.*REPLACE_ME' "$ENV_FILE" \
  || die "TAVILY_API_KEY missing or placeholder in $ENV_FILE (needed for Tavily Remote MCP)"
for n in "$CONCURRENCY" "$TIMEOUT" "$MIN_INSTANCES" "$MAX_INSTANCES"; do
  case "$n" in ''|*[!0-9]*) die "CONCURRENCY, TIMEOUT, MIN_INSTANCES and MAX_INSTANCES must be integers" ;; esac
done

ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
[ -n "$ACCOUNT" ] || die "no active gcloud account; run: gcloud auth login"
log "Account: $ACCOUNT | Project: $PROJECT_ID | Region: $REGION | Service: $SERVICE"
log "Resources: memory=$MEMORY cpu=$CPU concurrency=$CONCURRENCY timeout=${TIMEOUT}s instances=$MIN_INSTANCES..$MAX_INSTANCES"
if [ "$ASSUME_YES" -eq 0 ]; then
  printf 'Continue? [y/N] ' >&2
  read -r answer || die "no confirmation; use --yes for non-interactive runs"
  case "$answer" in y|Y|yes|YES) ;; *) die "cancelled" ;; esac
fi

ENABLED="$(gcloud services list --enabled --project "$PROJECT_ID" --format='value(config.name)')"
for api in run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com; do
  printf '%s\n' "$ENABLED" | grep -qx "$api" || die "enable the required APIs first:
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT_ID"
done

# ---- credentials: temp YAML outside the repo ----
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paper-analyst-deploy.XXXXXX")"
chmod 700 "$WORK_DIR"
ENV_YAML="$WORK_DIR/env.yaml"
python3 scripts/env_to_yaml.py --input "$ENV_FILE" --output "$ENV_YAML" || die "could not build the env-vars file"

# ---- image tag ----
if GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null)"; then
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then GIT_SHA="${GIT_SHA}-dirty"; fi
else
  GIT_SHA="nogit"  # an extracted ZIP has no git history
fi
TAG="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

# ---- Artifact Registry repo (created once) ----
if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
  log "creating Artifact Registry repo $REPO in $REGION"
  run gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location "$REGION" --project "$PROJECT_ID" --quiet
fi

# ---- build (Cloud Build, linux/amd64) and deploy ----
log "uploading the .gcloudignore allowlist (Dockerfile, requirements, app/, ingest/, data index + DB) to Cloud Build"
run gcloud builds submit . --tag "$IMAGE_URL" --project "$PROJECT_ID" \
  --ignore-file .gcloudignore
run gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --image "$IMAGE_URL" \
  --allow-unauthenticated --port 8080 \
  --memory "$MEMORY" --cpu "$CPU" --cpu-boost \
  --concurrency "$CONCURRENCY" --timeout "$TIMEOUT" \
  --min-instances "$MIN_INSTANCES" --max-instances "$MAX_INSTANCES" \
  --env-vars-file "$ENV_YAML" --quiet

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry run complete: nothing was created, built or deployed"
  exit 0
fi

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
REV="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format='value(status.latestReadyRevisionName)')"
[ -n "$URL" ] && [ -n "$REV" ] || die "could not read the service URL and revision"
log "URL: $URL"
log "Revision: $REV"
log "Image: $IMAGE_URL"
[ "$SMOKE" -eq 1 ] || exit 0

# ---- smoke tests: /health, then one streamed corpus question ----
H="$(curl -sS --max-time 90 -w ' %{http_code} %{time_total}' "$URL/health")" || die "/health request failed"
log "GET /health -> HTTP $(printf '%s' "$H" | awk '{print $(NF-1)" in "$NF"s"}')"
case "$H" in *'"status":"ok"'*' 200 '*) ;; *) die "/health did not return status ok" ;; esac

SSE="$WORK_DIR/chat.sse"
BODY="$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$SMOKE_QUERY")"
curl -sS -N --max-time 240 -X POST "$URL/api/chat" -H 'content-type: application/json' -d "$BODY" -o "$SSE" \
  || die "/api/chat request failed"
python3 - "$SSE" <<'PY' || die "chat smoke test failed"
import json, sys
counts, errors, event, done, failed_calls = {}, [], None, {}, []
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith("event:"):
        event = line[6:].strip()
    elif line.startswith("data:"):
        counts[event] = counts.get(event, 0) + 1
        data = json.loads(line[5:])
        if event == "error":
            errors.append(data.get("message_safe", "")[:160])
        elif event == "llm_result" and data.get("status") != "ok":
            failed_calls.append(f"{data.get('provider')}:{data.get('error_kind')}")
        elif event == "done":
            done = data
print(f"[deploy] /api/chat events: {counts}", file=sys.stderr)
print(f"[deploy] providers used: {done.get('providers_used')}, fallbacks: {done.get('fallbacks')}, "
      f"failed LLM attempts: {failed_calls or 'none'}", file=sys.stderr)
ok = counts.get("token", 0) > 0 and counts.get("citation", 0) > 0 and counts.get("done") == 1 and not errors
if not ok:
    print(f"[deploy] expected tokens, citations and one done event without errors; errors={errors}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
log "smoke tests passed"
