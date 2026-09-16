import json, pytest, torch
from cbis_ddsm.data.dataset import make_fake_patch_dir
from cbis_ddsm.training.config import TrainConfig, load_config
from cbis_ddsm.training.trainer import load_checkpoint, overfit_batch, train

def _cfg(tmp_path, **kw):
    root = make_fake_patch_dir(tmp_path / "fake", n=12, size=64)
    base = dict(patch_root=str(root), output_root=str(tmp_path / "exp"), encoder="resnet18", encoder_weights=None,
                device="cpu", amp=False, num_workers=0, batch_size=4, epochs=1, augment=False)
    base.update(kw)
    return TrainConfig(**base)

def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"; path.write_text("patch_root: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)

def test_train_writes_a_complete_run_directory(tmp_path):
    result = train(_cfg(tmp_path))
    for name in ("config.yaml", "metrics.jsonl", "checkpoint_best.pt", "run.json"):
        assert (result.run_dir / name).is_file()
    lines = (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["epoch"] == 0 and 0.0 <= result.best_val_dice <= 1.0

def test_same_seed_gives_identical_first_epoch_loss(tmp_path):
    a = train(_cfg(tmp_path / "a", seed=3)).history[0]["train_loss"]
    b = train(_cfg(tmp_path / "b", seed=3)).history[0]["train_loss"]
    assert a == pytest.approx(b, rel=1e-6)

def test_checkpoint_reload_reproduces_predictions(tmp_path):
    result = train(_cfg(tmp_path))
    model, _ = load_checkpoint(result.run_dir / "checkpoint_best.pt")
    again, _ = load_checkpoint(result.run_dir / "checkpoint_best.pt")
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), again.eval()(x))

@pytest.mark.slow
def test_overfit_gate_on_synthetic_patches(tmp_path):
    assert overfit_batch(_cfg(tmp_path, learning_rate=1e-3), n=8, steps=150) > 0.9
