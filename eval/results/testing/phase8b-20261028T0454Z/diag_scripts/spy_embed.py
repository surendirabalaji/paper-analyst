import pytest
calls = []
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    import app.rag as rag
    orig = rag.Retriever.search
    async def spy(self, query, k, doc_id=None):
        calls.append(query)
        return await orig(self, query, k, doc_id)
    rag.Retriever.search = spy
def pytest_unconfigure(config):
    print("\nREAL Retriever.search calls:", calls)
