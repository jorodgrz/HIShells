from src.utils.config import resolve_config
import os

def test_resolve_and_override(tmp_path, monkeypatch):
    # use the repo pv_config.yaml
    cfg = resolve_config("pv_config.yaml", ["optim.lr=0.0005"], write_resolved=False)
    assert "cube_path" in cfg and "galaxy" in cfg
    # env override
    monkeypatch.setenv("PV_model.base_filters", "48")
    cfg2 = resolve_config("pv_config.yaml", write_resolved=False)
    assert cfg2["model"]["base_filters"] == 48
    assert "_meta" in cfg2 and cfg2["_meta"]["_hash"]