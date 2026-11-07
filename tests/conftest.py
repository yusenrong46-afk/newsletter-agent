import pytest


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    """Every test gets an empty data folder, so no test can touch real credentials or caches."""
    home = tmp_path / "home"
    monkeypatch.setenv("NEWSLETTER_AGENT_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home
