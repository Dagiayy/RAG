from app.config.settings import Settings, get_settings


def test_settings_have_sane_defaults():
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.retrieval_final_evidence_count > 0
    assert settings.retrieval_rerank_candidates >= settings.retrieval_final_evidence_count


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
