from stego.colab import in_colab, latest_checkpoint, run_dir


def test_not_in_colab(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    assert not in_colab()


def test_run_dir_and_latest_checkpoint(tmp_path):
    path = run_dir("e1-sft", root=tmp_path)
    assert path == tmp_path / "e1-sft" and path.is_dir()
    assert latest_checkpoint(path) is None
    for step in (50, 100, 150):
        (path / f"checkpoint-{step}").mkdir()
    (path / "checkpoint-notes").mkdir()
    (path / "checkpoint-999.txt").write_text("not a checkpoint")
    assert latest_checkpoint(path) == path / "checkpoint-150"
