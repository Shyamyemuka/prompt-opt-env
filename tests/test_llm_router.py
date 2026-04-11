import os

from prompt_opt_env.llm_router import build_provider_specs


def test_openai_provider_keeps_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1/")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    providers = build_provider_specs(
        default_model="Qwen/Qwen2.5-72B-Instruct",
        default_base_url=os.environ["API_BASE_URL"],
    )

    assert len(providers) == 1
    assert providers[0].name == "openai"
    assert providers[0].base_url == "https://api.openai.com/v1/"


def test_hf_provider_uses_api_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1/")
    monkeypatch.delenv("HF_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    providers = build_provider_specs(
        default_model="Qwen/Qwen2.5-72B-Instruct",
        default_base_url=os.environ["API_BASE_URL"],
    )

    assert len(providers) == 1
    assert providers[0].name == "hf"
    assert providers[0].base_url == "https://router.huggingface.co/v1/"
