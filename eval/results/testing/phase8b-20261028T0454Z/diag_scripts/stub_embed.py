import pytest
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    import app.rag as rag
    async def stub(self, query, k, doc_id=None):
        raise RuntimeError("network embedding disabled in unit tests (tester stub)")
    rag.Retriever.search = stub
