from pathlib import Path

from push_ui_to_hf import REMOTE_DELETE_PATTERNS, stage_hf_bundle


def test_stage_hf_bundle_excludes_local_logs(tmp_path):
    staged = stage_hf_bundle(tmp_path)

    assert "README.md" in staged
    assert "Dockerfile" in staged
    assert "web_ui.py" in staged
    assert not (tmp_path / "docker_build.log").exists()
    assert not (tmp_path / "uv_error.txt").exists()
    assert "*.log" in REMOTE_DELETE_PATTERNS
    assert "*.txt" in REMOTE_DELETE_PATTERNS


def test_stage_hf_bundle_copies_server_tree(tmp_path):
    stage_hf_bundle(tmp_path)

    assert (tmp_path / "server" / "app.py").exists()
    assert (tmp_path / "server" / "Dockerfile").exists()
    assert (tmp_path / "README.md").exists()
