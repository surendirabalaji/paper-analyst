#!/usr/bin/env python3
"""Writes eval/testset/testset_v1.jsonl from the reviewed drafts below (step 2 of the construction method).

Every RAGAS item's reference answer restates only what its reference chunk(s) say. The chunk ids come from
eval/testset/sampled_chunks.jsonl (single-paper items) or from a paper-scoped vector search recorded in
eval/testset/README.md (comparison items). Data items were checked against data/papers.db with the SQL shown.
Drafted by Claude (claude-opus-5-5) in Claude Code on 2026-10-27; to be reviewed and edited by the student
before freezing. Usage: .venv/bin/python eval/testset/draft_v1.py  (no network)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (id, question, reference answer, reference chunk ids)
SINGLE = [
 ("s01", "In MMOA-RAG, how is the Generator agent rewarded?",
  "The Generator's reward is the shared reward R_shared (the F1 score of the predicted answer, shared by all agents) plus a penalty P_G that constrains overly long output: P_G is -0.5 when the generated answer exceeds a certain length and 0 otherwise. The Generator observes its prompt, the question and the documents chosen by the Selector.",
  ["2501.15228v2::S3.SS3.SSS3::000"]),
 ("s02", "Which iterative RAG baselines does TeaRAG compare against, and what retriever do the baselines use?",
  "Besides IRCoT and SelfRAG, TeaRAG compares against R1-Searcher (two-stage training with outcome and format rewards on Reinforce++, trained on HotpotQA and 2WikiMultiHopQA; Llama-8B and Qwen-7B versions) and Search-R1 (PPO on NQ and HotpotQA with outcome rewards; four 7B/14B instruct and base versions). All baselines use E5-base-V2 as the retriever unless stated otherwise; the \"+R\" variants add BGE-reranker-v2 reranking.",
  ["2511.05385v2::S5.SS1.SSS2::001"]),
 ("s03", "How do language models perform on the synthetic key-value retrieval task in Lost in the Middle?",
  "With 75, 140 and 300 key-value pairs, Claude-1.3 and Claude-1.3 (100K) are nearly perfect at every length, but other models struggle, especially with 140 or 300 pairs, even though the task only needs an exact match. GPT-3.5-Turbo, GPT-3.5-Turbo (16K) and MPT-30B-Instruct do worst when the pair is in the middle of the context. LongChat-13B (16K) behaves differently at 140 pairs: when the key is at the start it tends to generate code to retrieve the key instead of outputting the value.",
  ["2307.03172v3::S3.SS2::000"]),
 ("s04", "How does GraphRAG generate the global sensemaking questions it uses for evaluation?",
  "An LLM is given a high-level description of the corpus and prompted to describe K personas of potential users, then N tasks per user, then M high-level questions per user-task pair that require understanding the entire corpus and do not require retrieving specific low-level facts. With K = N = M = 5 this gives 125 test questions per dataset.",
  ["2404.16130v2::S3.SS2::000"]),
 ("s05", "On which reasoning tasks is multiagent debate evaluated, and against which baselines?",
  "Three reasoning tasks of increasing difficulty: arithmetic expressions over six two-digit numbers, GSM8K grade-school math, and chess move prediction (scored by Stockfish pawn advantage). Baselines are a single agent answering directly, single-agent self-reflection, and multiple agents with majority voting, all with the same starting prompt and model, zero-shot, using a ChatGPT-based model.",
  ["2305.14325v1::S3.SS1::000"]),
 ("s06", "What limitations do the ChatDev authors acknowledge?",
  "Agents' capabilities may be overestimated: they often implement simple logic with low information density and struggle without clear, detailed requirements, so the technology suits prototypes more than complex real-world applications. Automatically evaluating general-purpose software is hard and manual verification does not scale. Multiple agents need more tokens and time than single-agent approaches, raising computational and environmental cost.",
  ["2307.07924v5::S6::000"]),
 ("s07", "How is a Mixture-of-Agents layer structured, and how are the layer outputs combined?",
  "MoA has l layers, each with n LLMs; models can be reused within or across layers. Each LLM in a layer processes the input and generates a continuation; the outputs are combined with an Aggregate-and-Synthesize prompt and concatenated with the original prompt to form the next layer's input. No fine-tuning is needed, only prompting, and only one LLM is used in the last layer.",
  ["2406.04692v1::S2.SS2::000"]),
 ("s08", "Does the sampling-and-voting method from More Agents Is All You Need combine well with other methods?",
  "Yes, mostly. Integrated with other methods it gives gains of 10-21% on GSM8K, 1-15% on MATH, 1-13% on Chess, 1-11% on MMLU and 2-7% on code generation. The exceptions are failures when combined with debate on Llama2-13B and Llama2-70B for code generation, attributed to noise from referencing other agents' answers, which disrupts code logic.",
  ["2402.05120v2::S5.SS2::000"]),
 ("s09", "What problem-solving patterns does SWE-agent show when fixing issues?",
  "Trajectories start with reproduction (create a file with reproduction code, edit, run with python) and/or localization (find_file or search_dir), moving from a broad directory-level search to specific files and lines (search_file, goto). Later turns are mostly edit-then-execute loops with occasional extra localization. Editing remains hard: 51.7% of SWE-agent with GPT-4 Turbo trajectories have at least one failed edit that raises a linting error.",
  ["2405.15793v3::S5.SS2::000"]),
 ("s10", "How does SWE-bench choose which files to give the model as context?",
  "Codebases are far too large for a context window, so baselines use retrieval. Sparse retrieval uses BM25, because dense retrieval suits long code documents with natural-language queries poorly, and retrieves as many files as fit in a context limit; models do best with the shortest limit. An oracle setting uses the files edited by the reference patch. BM25 retrieves a superset of the oracle files in about 40% of instances at the 27,000-token limit, but none of them in almost half.",
  ["2310.06770v3::S4.SS1::000"]),
 ("s11", "How do GPT-3.5 and GPT-4 perform on Mind2Web with in-context learning?",
  "With three demonstrations in the MindAct multiple-choice format, both are comparable to the baselines. GPT-3.5 has only about 20% element selection accuracy, partly because it tends to choose the None option, claiming the task cannot be done on the current page. GPT-4 is on par with fine-tuned Flan-T5 models on Cross-Website and Cross-Domain element selection, though its high operational cost is a concern.",
  ["2306.06070v3::S4.SS3::001"]),
 ("s12", "In the filesystem-based memory study, how do curated stores and the episode log differ as memory grows?",
  "Curated stores mature: the management agent's store mostly stops growing a quarter of the way into the chain (14 of its final 16 files exist by task 35) and is edited and merged afterwards, and curated skills+memory (GS) has the cheapest per-task retrieval. The episode log grows without bound (140 files, 270 KB by the end at mini), and its per-task retrieval price climbs with the store and stays a multiple of GS's. Accumulation helps most for weak execution agents: the episode log gains +7.1 at the weak agent versus +1.4 at the strong one.",
  ["2607.26637v1::S4.SS4::002"]),
 ("s13", "How does A-MEM retrieve relevant memories for a new interaction?",
  "It embeds the query text with the same encoder used for memory notes, computes cosine similarity between the query embedding and every memory note, and retrieves the top-k most similar memories to build the prompt context.",
  ["2502.12110v11::S3.SS4::000"]),
 ("s14", "Does running agents on AIOS change their benchmark performance?",
  "On HumanEval, MINT (code subset), GAIA and SWE-Bench-Lite with GPT-4o-mini at temperature 1.0, AIOS consistently maintains success rate and sometimes improves it. On code benchmarks the gain comes from prompt enhancement with more structured input and output; on GAIA tool calling it comes from pre-execution parameter validation with structural regex and from conflict-resolution hashmaps for concurrent access.",
  ["2403.16971v5::S4.SS2::000"]),
 ("s15", "Does Self-Refine work with weaker models such as Vicuna-13B?",
  "Not well. Vicuna-13B can produce initial outputs but struggles with refinement: it cannot consistently produce feedback in the required format and, even with oracle or hard-coded feedback, often repeats its output or hallucinates a conversation. The authors suggest this is because it was trained on conversations and generalizes worse to test-time few-shot tasks than instruction-tuned models. Separately, Self-Refine outputs are still preferred by humans over all k=4 initial ChatGPT samples.",
  ["2303.17651v2::S4::003"]),
 ("s16", "How does ReWOO use specialization to make a small model plan well?",
  "Because ReWOO decouples reasoning from tool calls, the Planner can be trained without tool responses. text-davinci-003 generates 4,000 (Plan, #E) blueprints on HotpotQA and TriviaQA; about 2,000 that lead to correct answers are kept as Planner instruction data. LLaMA 7B is instruction-tuned on 52k self-instruct data (Alpaca 7B) and then on the Planner data, giving Planner 7B, which is compared with GPT-3.5 and Alpaca 7B as the Planner.",
  ["2305.18323v1::S2.SS3::000"]),
 ("s17", "How is Toolformer decoded and evaluated on downstream tasks?",
  "In a prompted zero-shot setup with no in-context examples, so the model must decide on its own which tools to use. Decoding is greedy, except that an API call may start whenever <API> is among the k = 10 most likely tokens (k = 1 would be ordinary greedy decoding), and at most one API call is allowed per input so the model does not loop calling APIs.",
  ["2302.04761v1::S4.SS2::000"]),
 ("s18", "How does tool-result faithfulness vary by domain in ToolFailBench?",
  "Across 17 disciplined models, finance has the highest median rule Result-Ignore Rate (12.24%) and cybersecurity the lowest (0.68%), possibly because plausible prior values like stock prices compete with tool returns. The authors caution that domain RIR also depends on how salient the trap values are, so they treat it as descriptive, and it supports their five-domain design.",
  ["2607.04686v1::S5.SS5::000"]),
 ("s19", "How does the InfoDeepSeek agentic RAG framework carry out the retrieval stage?",
  "The agent first plans how to seek information, then runs an information-seeking trajectory of up to T steps: at each step it reflects on the current observation and its memory of the trajectory, updates the plan, and chooses a tool (search engines such as Google, Bing, Yahoo, DuckDuckGo, Selenium browsing, time utilities, or termination). It stops when it has enough information or reaches T, producing a sequence of observations.",
  ["2505.15872v2::S3::000"]),
 ("s20", "How does the Rise and Potential of LLM-based Agents survey suggest evaluating agents' ability to keep learning?",
  "It names three aspects: continual learning, evaluated by overall performance on tasks learned so far, memory stability on old tasks and plasticity on new ones; autotelic learning, where agents set and achieve their own goals, evaluated e.g. in a simulated survival environment by how fast and how far they acquire skills; and adaptability and generalization to new environments, evaluated with diverse simulated environments and unseen tasks.",
  ["2309.07864v3::S6.SS2::003"]),
]

COMPARE = [
 ("c01", "How do MemGPT and A-MEM differ in how they organize an agent's memory?",
  "MemGPT treats the context window like main memory in an operating system: the prompt holds read-only system instructions, a fixed-size working context writable only through function calls, and a FIFO message queue with a recursive summary of evicted messages, and the LLM pages information between context and external storage with function calls. A-MEM builds Zettelkasten-style atomic memory notes with content, timestamp, LLM-generated keywords, tags and a contextual description, an embedding and links to related notes, and evolves existing neighbouring notes when a new memory is added.",
  ["2310.08560v2::S1::001", "2310.08560v2::S2.SS1::000", "2502.12110v11::S3.SS1::000", "2502.12110v11::S3.SS3::000"]),
 ("c02", "Compare how multiagent debate and Mixture-of-Agents combine the outputs of several LLMs.",
  "In multiagent debate, agents answer and then read each other's answers over several rounds until they converge on a shared answer; prompts that make agents more stubborn lead to longer debates and better answers. Mixture-of-Agents arranges LLMs in layers: proposers generate reference responses and aggregators synthesize them into one higher-quality answer, with the outputs passed to the next layer through an Aggregate-and-Synthesize prompt.",
  ["2305.14325v1::S2.SS2::000", "2406.04692v1::S2.SS1::000", "2406.04692v1::S2.SS2::000"]),
 ("c03", "How do GraphRAG and RAPTOR each build summaries for retrieval?",
  "GraphRAG extracts entities and relationships into a knowledge graph, detects communities (Leiden) and summarizes them; for a query it maps community summaries to partial answers with helpfulness scores in parallel and reduces the best ones to a global answer. RAPTOR clusters text chunks with Gaussian Mixture Models (cluster count chosen by BIC), summarizes each cluster with gpt-3.5-turbo, and repeats recursively to build a tree; about 4% of its summaries had minor hallucinations.",
  ["2404.16130v2::S3.SS1.SSS6::000", "2404.16130v2::S4.SS1.SSS3::000", "2401.18059v1::S3::002"]),
 ("c04", "How do RAGAS and ARES differ in how they evaluate RAG systems?",
  "RAGAS uses reference-free metrics computed by prompting an LLM: faithfulness is the share of answer statements supported by the context, answer relevance is the mean similarity between the question and questions generated from the answer, and context relevance measures how focused the context is. ARES trains LLM judges on synthetic data to label context relevance, answer faithfulness and answer relevance, then uses prediction-powered inference with a small human-annotated validation set to give confidence intervals and rank RAG systems.",
  ["2309.15217v2::S3::000", "2309.15217v2::S3::001", "2311.09476v2::S3.SS3::000", "2311.09476v2::S3.SS3::001"]),
 ("c05", "How do Toolformer and Gorilla teach language models to use APIs?",
  "Toolformer is self-supervised: it samples API calls in text and keeps only those whose call and result reduce the language-modeling loss on the following tokens by at least a threshold tau_f, then fine-tunes on the kept calls. Gorilla fine-tunes LLaMA-7B on self-instruct {instruction, API} pairs from TorchHub, TensorHub and HuggingFace (APIBench), optionally with retrieved API documentation appended to the prompt (retriever-aware training), which reduces hallucination and lets it adapt to documentation changes.",
  ["2302.04761v1::S2::002", "2305.15334v1::S3.SS2::000", "2305.15334v1::S1::001"]),
 ("c06", "How do Search-o1 and Search-R1 integrate search into a model's reasoning?",
  "Search-o1 works at inference time: the reasoning model emits search queries between special tokens, retrieves documents, and a Reason-in-Documents step uses the same model to refine them into concise knowledge inserted into the reasoning chain. Search-R1 trains the model with reinforcement learning (PPO or GRPO) on rollouts that interleave reasoning with search-engine calls, masking retrieved tokens in the loss.",
  ["2501.05366v1::S3.SS4::000", "2501.05366v1::S3.SS4::002", "2503.09516v5::S3.SS1::000", "2503.09516v5::S3.SS1::002"]),
]

DATA = [  # system tests: the evidence is SQL rows, not chunk text, so not in the RAGAS set
 ("d01", "How many papers in the corpus were published in 2026?", "14.",
  "SELECT COUNT(*) FROM papers WHERE year = 2026"),
 ("d02", "Who are the authors of the Voyager paper?",
  "Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar.",
  "SELECT a.name FROM authors a JOIN paper_authors pa USING(author_id) WHERE pa.arxiv_id='2305.16291' ORDER BY pa.position"),
 ("d03", "Which primary arXiv category has the most papers in the corpus, and how many?", "cs.CL, with 47 papers.",
  "SELECT primary_category, COUNT(*) FROM papers GROUP BY 1 ORDER BY 2 DESC LIMIT 1"),
 ("d04", "How many papers in the corpus link to a code repository?", "58.", "SELECT COUNT(*) FROM repos"),
]
MULTI = [  # multi-agent (Data -> Summary) questions: latency category "multi"; checked for the hand-off, not RAGAS
 ("m01", "What is the most recent paper in the corpus about the Model Context Protocol, and what does it find?",
  "The newest MCP paper is 2606.05339, A Taxonomy of Runtime Faults in Model Context Protocol Servers (2026-06-03); the answer should summarize it with C# citations.",
  "SELECT arxiv_id, published, title FROM papers WHERE title LIKE '%Model Context Protocol%' OR title LIKE '%MCP%' ORDER BY published DESC"),
 ("m02", "Which 2026 papers in the corpus are about agent memory? Summarize the earliest one.",
  "2026 memory papers include SimpleMem (2601.02553, 2026-01-05), Memory is Reconstructed, Not Retrieved (2606.06036), Governed Shared Memory (2606.24535) and Filesystem-Based Memory (2607.26637); the earliest is SimpleMem, which the answer should summarize with C# citations.",
  "SELECT arxiv_id, published, title FROM papers WHERE year = 2026 AND title LIKE '%memory%' ORDER BY published"),
]
SYSTEM = [  # behaviour checks outside RAGAS
 ("x01", "How does photosynthesis convert light into chemical energy?", "out_of_corpus",
  "No C# corpus citations; the answer says the corpus does not cover this, and anything it adds comes only from W# web results with the retrieval date, or it abstains."),
 ("x02", "What does the Voyager-2 paper by OpenAI say about agent swarms?", "missing_paper",
  "States that no such paper is in the corpus; no invented citations; may point to related corpus papers."),
 ("x03", "What are the latest developments in the Model Context Protocol specification?", "live",
  "Uses tavily_search; W# citations with a retrieval date; distinguishes current web information from the corpus."),
 ("x04", "What has happened recently in tool learning for LLMs, and was the Toolformer paper published at a peer-reviewed venue?", "mcp_sharing",
  "Research and Summary both call the same tavily_search tool with role-specific queries; answer has W# and C# citations."),
]


def main() -> None:
    sampled = {json.loads(l)["chunk_id"]: json.loads(l) for l in (ROOT / "eval/testset/sampled_chunks.jsonl").read_text().splitlines()}
    rows = []
    for i, q, ref, ids in SINGLE:
        src = sampled[ids[0]]
        rows.append({"id": i, "category": "rag_single", "latency_category": "simple", "ragas": True, "question": q,
                     "reference": ref, "reference_chunk_ids": ids, "arxiv_ids": [src["arxiv_id"]],
                     "topic": src["topic"], "section_type": src["section_type"]})
    for i, q, ref, ids in COMPARE:
        rows.append({"id": i, "category": "rag_compare", "latency_category": "simple", "ragas": True, "question": q,
                     "reference": ref, "reference_chunk_ids": ids,
                     "arxiv_ids": sorted({c.split("v")[0] for c in ids})})
    for i, q, ref, sql in DATA:
        rows.append({"id": i, "category": "data", "latency_category": "simple", "ragas": False, "question": q,
                     "reference": ref, "reference_sql": sql})
    for i, q, ref, sql in MULTI:
        rows.append({"id": i, "category": "multi", "latency_category": "multi", "ragas": False, "question": q,
                     "reference": ref, "reference_sql": sql})
    for i, q, kind, expect in SYSTEM:
        rows.append({"id": i, "category": "system", "kind": kind, "latency_category": None, "ragas": False,
                     "question": q, "expected_behaviour": expect})
    out = ROOT / "eval/testset/testset_v1.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} items ({sum(r['ragas'] for r in rows)} RAGAS) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
