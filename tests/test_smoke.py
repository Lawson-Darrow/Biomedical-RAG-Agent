"""Smoke tests — confirm the package imports and config loads. Real tests land per-milestone."""

import biomed_rag


def test_version():
    assert biomed_rag.__version__ == "0.1.0"


def test_config_imports():
    from biomed_rag.config import settings

    assert settings.retrieval_top_k > 0
