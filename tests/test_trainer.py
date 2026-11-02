import json, pytest, torch
from pathlib import Path
from cbis_ddsm.data.dataset import make_fake_patch_dir
from cbis_ddsm.training.config import TrainConfig, load_config, parse_overrides
from cbis_ddsm.training.trainer import load_checkpoint, overfit_batch, stop_reason, train

def _cfg(tmp_path, **kw):
    root = make_fake_patch_dir(tmp_path / "fake", n=12, size=64)
    base = dict(patch_root=str(root), output_root=str(tmp_path / "exp"), encoder="resnet18", encoder_weights=None,
                device="cpu", amp=False, num_workers=0, batch_size=4, epochs=1, augment=False)
    base.update(kw)
    return TrainConfig(**base)

def test_parse_overrides_parses_yaml_values(tmp_path):
    result = parse_overrides(["epochs=3", "encoder_weights=null", "amp=false"])
    assert result == {"epochs": 3, "encoder_weights": None, "amp": False}

def test_parse_overrides_raises_on_malformed_pair(tmp_path):
    with pytest.raises(ValueError, match="got: bad_pair"):
        parse_overrides(["bad_pair"])

def test_parse_overrides_raises_on_empty_key(tmp_path):
    with pytest.raises(ValueError, match="got: =value"):
        parse_overrides(["=value"])

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

def test_config_rejects_an_unknown_scheduler(tmp_path):
    with pytest.raises(ValueError, match="scheduler"):
        _cfg(tmp_path, scheduler="bogus")

def test_config_rejects_an_unknown_selection_metric(tmp_path):
    with pytest.raises(ValueError, match="selection_metric"):
        _cfg(tmp_path, selection_metric="bogus")

def test_baseline_config_is_loadable_and_longer_than_the_smoke_config(tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    baseline = load_config(configs / "unet_baseline.yaml")
    smoke = load_config(configs / "unet_smoke.yaml")
    assert baseline.epochs > smoke.epochs and baseline.resume_from is None
    assert baseline.max_hours is not None and baseline.early_stopping_patience is not None

@pytest.mark.parametrize("since_best, patience, expected", [
    (0, 3, None), (2, 3, None), (3, 3, "early_stopping"), (4, 3, "early_stopping"), (9, None, None),
])
def test_stop_reason_reports_early_stopping_at_the_patience(since_best, patience, expected):
    assert stop_reason(epochs_since_best=since_best, patience=patience, elapsed_seconds=0.0, max_hours=None) == expected

def test_stop_reason_reports_an_exhausted_time_budget():
    assert stop_reason(epochs_since_best=0, patience=None, elapsed_seconds=10.0, max_hours=0.001) == "time_budget"

def test_stop_reason_prefers_the_time_budget_over_early_stopping():
    assert stop_reason(epochs_since_best=9, patience=1, elapsed_seconds=10.0, max_hours=0.001) == "time_budget"

def test_train_writes_a_resumable_last_checkpoint(tmp_path):
    result = train(_cfg(tmp_path))
    state = torch.load(result.run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert {"model", "config", "optimizer", "scaler", "epoch", "best_epoch", "best_score"} <= state.keys()
    assert state["epoch"] == 0 and result.completed_epochs == 1 and not result.stopped_early

def test_best_checkpoint_stays_weights_only_for_inference(tmp_path):
    result = train(_cfg(tmp_path))
    state = torch.load(result.run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert {"model", "config", "epoch", "val_dice"} <= state.keys()
    assert "optimizer" not in state

def test_resume_continues_in_the_same_run_directory(tmp_path):
    first = train(_cfg(tmp_path, epochs=1))
    second = train(_cfg(tmp_path, epochs=3, resume_from=str(first.run_dir)))
    assert second.run_dir == first.run_dir and second.completed_epochs == 3
    lines = (second.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [0, 1, 2]
    assert len(second.history) == 3

def test_resume_writes_its_own_config_snapshot(tmp_path):
    first = train(_cfg(tmp_path, epochs=1))
    train(_cfg(tmp_path, epochs=2, resume_from=str(first.run_dir)))
    assert (first.run_dir / "config.yaml").is_file() and (first.run_dir / "config_resume_1.yaml").is_file()

def test_resume_rejects_a_different_patch_root(tmp_path):
    first = train(_cfg(tmp_path, epochs=1))
    other = make_fake_patch_dir(tmp_path / "other", n=12, size=64)
    with pytest.raises(ValueError, match="patch_root"):
        train(_cfg(tmp_path, epochs=2, resume_from=str(first.run_dir), patch_root=str(other)))

def test_resume_rejects_a_different_architecture(tmp_path):
    first = train(_cfg(tmp_path, epochs=1))
    with pytest.raises(ValueError, match="encoder"):
        train(_cfg(tmp_path, epochs=2, resume_from=str(first.run_dir), encoder="resnet34"))

def test_resume_rejects_a_missing_run_directory(tmp_path):
    with pytest.raises(ValueError, match="resume_from"):
        train(_cfg(tmp_path, resume_from=str(tmp_path / "nope")))

def test_resume_of_a_finished_run_runs_no_further_epoch(tmp_path):
    first = train(_cfg(tmp_path, epochs=1))
    second = train(_cfg(tmp_path, epochs=1, resume_from=str(first.run_dir)))
    lines = (second.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [0]
    assert second.completed_epochs == 1 and len(second.history) == 1

def test_cosine_scheduler_lowers_the_learning_rate_each_epoch(tmp_path):
    history = train(_cfg(tmp_path, epochs=3, scheduler="cosine")).history
    assert history[0]["lr"] > history[1]["lr"] > history[2]["lr"]

def test_no_scheduler_keeps_the_learning_rate_constant(tmp_path):
    history = train(_cfg(tmp_path, epochs=2)).history
    assert history[0]["lr"] == history[1]["lr"]

def test_time_budget_stops_the_run_and_records_the_reason(tmp_path):
    result = train(_cfg(tmp_path, epochs=5, max_hours=0.0))
    assert result.completed_epochs == 1 and result.stopped_early and result.stop_reason == "time_budget"
    run_info = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_info["stop_reason"] == "time_budget" and run_info["completed_epochs"] == 1

def test_validation_dice_separates_lesion_patches_from_background(tmp_path):
    root = make_fake_patch_dir(tmp_path / "mixed", n=12, size=64, kinds=("lesion", "lesion", "background"))
    record = train(_cfg(tmp_path, patch_root=str(root), epochs=1)).history[0]
    assert 0.0 <= record["val_dice_lesion"] <= 1.0
    assert record["val_dice"] != record["val_dice_lesion"]

def test_lesion_dice_is_none_when_validation_holds_no_lesion_patch(tmp_path):
    root = make_fake_patch_dir(tmp_path / "bg", n=6, size=64, kinds=("background",))
    assert train(_cfg(tmp_path, patch_root=str(root), epochs=1)).history[0]["val_dice_lesion"] is None

def test_selection_metric_lesion_only_drives_the_best_score(tmp_path):
    root = make_fake_patch_dir(tmp_path / "mixed", n=12, size=64, kinds=("lesion", "lesion", "background"))
    result = train(_cfg(tmp_path, patch_root=str(root), epochs=1, selection_metric="val_dice_lesion"))
    assert result.best_val_dice == pytest.approx(result.history[0]["val_dice_lesion"])

def test_config_rejects_a_yaml_exponent_that_parsed_as_a_string(tmp_path):
    path = tmp_path / "c.yaml"
    # YAML 1.1 reads `3e-4` as a string: the config must say so instead of failing in the optimizer.
    path.write_text("patch_root: x\nlearning_rate: 3e-4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate must be a number"):
        load_config(path)

def test_config_accepts_a_dotted_yaml_exponent(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("patch_root: x\nlearning_rate: 3.0e-4\n", encoding="utf-8")
    assert load_config(path).learning_rate == pytest.approx(3e-4)

def test_every_shipped_config_is_loadable(tmp_path):
    for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("unet_*.yaml")):
        assert load_config(path).epochs >= 1, path

def test_a_resumed_cosine_schedule_matches_an_uninterrupted_one(tmp_path):
    """The Kaggle case: the time budget cuts the session, `epochs` stays the horizon.

    Resuming must not replay the scheduler step, which would stretch the schedule by one epoch
    on every resume.
    """
    straight = train(_cfg(tmp_path / "straight", epochs=3, scheduler="cosine"))
    first = train(_cfg(tmp_path / "cut", epochs=3, scheduler="cosine", max_hours=0.0))
    assert first.completed_epochs == 1 and first.stop_reason == "time_budget"
    resumed = train(_cfg(tmp_path / "cut", epochs=3, scheduler="cosine", resume_from=str(first.run_dir)))
    assert [record["lr"] for record in resumed.history] == pytest.approx([record["lr"] for record in straight.history])

def test_resuming_with_more_epochs_stretches_the_cosine_horizon(tmp_path):
    """A longer run must not inherit the interrupted run's T_max and jump back to the base rate."""
    first = train(_cfg(tmp_path, epochs=2, scheduler="cosine", max_hours=0.0))
    resumed = train(_cfg(tmp_path, epochs=4, scheduler="cosine", resume_from=str(first.run_dir)))
    rates = [record["lr"] for record in resumed.history]
    assert rates == sorted(rates, reverse=True) and rates[-1] < rates[0]

def test_a_resumed_run_reproduces_an_uninterrupted_one(tmp_path):
    """A session cut must cost nothing: the same seed and horizon give the same epoch records."""
    straight = train(_cfg(tmp_path / "straight", epochs=3, scheduler="cosine", augment=True))
    first = train(_cfg(tmp_path / "cut", epochs=3, scheduler="cosine", augment=True, max_hours=0.0))
    resumed = train(_cfg(tmp_path / "cut", epochs=3, scheduler="cosine", augment=True, resume_from=str(first.run_dir)))
    assert [record["train_loss"] for record in resumed.history] == pytest.approx(
        [record["train_loss"] for record in straight.history], rel=1e-4
    )
    assert resumed.best_val_dice == pytest.approx(straight.best_val_dice, rel=1e-4)
    assert resumed.best_epoch == straight.best_epoch
