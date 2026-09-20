# SDD ledger — plan: docs/superpowers/plans/2026-09-16-kaggle-smoke-pipeline.md

Spec: PLAN.md, docs/PROTOCOLE.md, docs/ARCHITECTURE.md, schemas/coding_plan.svg, schemas/labels_and_splits.svg (reachable).
Branch: s2-s6-kaggle-smoke (no worktree: data/raw and .venv are git-ignored and would be absent from a worktree).
Ruling: work on branch s2-s6-kaggle-smoke in the main checkout instead of a worktree — the pipeline needs data/raw (5.9 GB) and .venv, both untracked — cost if wrong: none beyond a less isolated tree; branch is not main.
Credit rule (user): monitor 5-hour usage window; commit everything when >= 90% used.

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1→T6,T7 | smp/torchvision/sklearn installed | OK |
| T1→T9 | capture --record-only | OK |
| T2→T3,T4 | ImageRecord 11 fields in order (image_key..subtlety); group_images, select_masses, read_excluded_identities, fuse_masks | OK, T3 test constructs positionally in same order |
| T2 manifest fields ↔ existing tests/test_audit.py, test_data_clis.py | ManifestRow constructed by keyword without new fields | CONFLICT → Ruling R1 |
| T2 manifest header ↔ data/interim/manifest.csv (old header) | read_manifest_csv strict header | stale local file → Ruling R2 |
| T3→T4,T9 | build_split(n_folds=5 default) | CONFLICT: CLI/integration tests with 2–4 patients cannot make 5 stratified folds → Ruling R3 |
| T4→T5 | PatchRecord 15 fields, write/read_patch_index | OK (T4 test positional order matches definition) |
| T5→T7,T8 | PatchDataset(root, partition, transform, seed, limit), set_epoch, make_fake_patch_dir | OK |
| T6→T7,T8 | build_model, BCEDiceLoss(bce_weight), dice_score, iou_score, threshold_probabilities | OK |
| T7→T8,T9 | TrainConfig, train→RunResult.run_dir, load_checkpoint, overfit_batch(cfg,n,steps) | run dir name to the second may collide → Ruling R4 |
| T8→T9 | scripts/predict.py | OK |
| T1 self | versions left to uv resolution | Ruling R5 |
| T2 self | tests vs code | OK |
| T3 self | CLI test "2 patients" vs 5 folds | see R3 |
| T4 self | CLI test "2 train patients, 4 rows" vs split with 5 folds | see R3 |
| T5 self | hash((seed, epoch, i)) deterministic for ints | OK |
| T6 self | OK | OK |
| T7 self | OK | see R4 |
| T8 self | overlay test fixed pre-flight | OK |
| T9 self | integration test 4 train patients vs 5 folds | see R3 |

Ruling R1: new ManifestRow fields `subtlety`, `shape_or_type`, `margins_or_distribution` get default "" (they are appended last) — keeps every existing keyword construction valid — cost if wrong: a row silently missing morphology if a caller forgets them; the CSV path always fills them.
Ruling R2: data/interim/manifest.csv built before T2 must be regenerated with scripts/build_manifest.py; read_manifest_csv stays strict — cost if wrong: one ~75 s rebuild.
Ruling R3: small-data tests use `n_folds=2` with at least 2 malignant + 2 benign train patients; `scripts/kaggle_smoke.py` exposes `--n-folds` (default 5); T4 CLI test uses 4 train patients split with n_folds=2 and expects 2 rows (lesion + background) per image in the requested partitions — cost if wrong: small-data tests stop exercising the default 5-fold path (covered by T3 unit tests).
Ruling R4: `train` appends `_2`, `_3`… to the run directory when the timestamped name already exists — cost if wrong: none.
Ruling R5: implementer of T1 picks the newest segmentation-models-pytorch and scikit-learn that resolve with torch 2.5.1+cu121, numpy 1.26.4 and Python 3.12, and records them — cost if wrong: a later pin bump.

## Tasks
Task 1: verified ⚠️ items: imports OK (torch 2.5.1+cu121, torchvision 0.20.1+cu121, smp 0.5.0, sklearn 1.9.1, CUDA True); trailer reads "Claude Sonnet 5"
Ruling: accept subagent commit trailers naming the model that actually wrote the commit (Sonnet) instead of rewriting history — honest attribution, no history rewrite — cost if wrong: cosmetic trailer mismatch on a few commits
Task 1: minor (deferred): test_capture_environment.py uses len(LOCK_NAMES) instead of literal; mixed pytest/unittest style (brief-mandated)
Task 1: complete (commits fab997e..d0b2218, review clean)
Task 2: minor (deferred): targets.py:142 `_identity(row)` duplicates ManifestRow.lesion_identity — reuse the property
Task 2: complete (commits d0b2218..1fdc4d5, review clean)
Task 3: minor (deferred): make_splits.py:68 recomputes patient_labels; read_split_json has no version/schema validation
Task 3: complete (commits 1fdc4d5..6bce3da, review clean)
Task 4: minor (deferred): patches.py cv2.imwrite return unchecked; read_patch_index no header validation; no test for background shortfall path; masks read twice per image
Task 4: complete (commits 6bce3da..98a1235, review clean)
Task 5: minor (deferred): dataset.py:85-86 global random/np.random seeding is inert (set_random_seed does the work) and mutates global RNG; no test for partition=None
Ruling: Task 7 trainer must not use persistent_workers=True with PatchDataset.set_epoch (workers keep a stale copy) — carry to Task 7 dispatch — cost if wrong: identical augmentations every epoch
Task 5: complete (commits 98a1235..bbdc0a4, review clean)
Ruling: Task 6 plan-mandated unseeded torch.rand in tests/test_models.py gets an explicit generator seed — global constraint "every random operation takes an explicit seed" outranks the plan's test snippet — cost if wrong: none
Task 6: minor (deferred): no ValueError shape-mismatch test (added in fix round 1 request)
Task 6: fix round 1/5 (3 addressed, 0 open — dispatch table, metrics helper, seeded test; commits 28c8dcc..cb9596a)
Task 6: complete (commits bbdc0a4..cb9596a, review clean)
Ruling: from Task 7 on, implementers commit after each green step (several commits per task) instead of one commit — the user's credit limit may stop the session mid-task and uncommitted work would be lost — cost if wrong: noisier history
Task 7: review approved with 1 Important open: scripts/train.py:109-117 and scripts/overfit_batch.py:48-56 duplicate _parse_overrides verbatim — move to cbis_ddsm.training.config.parse_overrides (fix round 1 NOT yet dispatched: session stopped at the user's 90% credit threshold)
Task 7: minor (deferred): train/overfit CLIs don't wrap train()/overfit_batch() errors in parser.error; untested CUDA-OOM fallback in overfit_batch; no test for empty validation set
Session stop 2026-09-16 (5-hour window 88%): Tasks 1-6 complete, Task 7 implemented+reviewed (fix round 1 pending), Tasks 8-9 not started, final whole-branch review not run.

## Session 2026-09-20 (resumed)
Ruling: nnU-Net runs on ROI patches, not whole images (user decision 2026-09-20) — same images as the U-Net baseline, so the comparison is like-for-like; whole-image detection stays a perspective — cost if wrong: nnU-Net numbers are patch-level and must never be presented as detection.
Plan extended with Tasks 10-12 (nnU-Net export / folds+runner / prediction import), committed.
Pre-flight scan for Tasks 10-12:
| Pair | Produces vs consumes | Finding |
| T4->T10 | read_patch_index, PatchRecord, patch PNGs 0/255 | OK (labels rewritten to 0/1 by export) |
| T5->T10,T12 | make_fake_patch_dir(root, n, size, seed, partitions) | OK: tests need a "test" partition — helper accepts any partitions tuple |
| T10->T11,T12 | case_index.csv + CASE_INDEX_FIELDS, DATASET_NAME | OK |
| T3->T11 | split JSON keys folds/n_folds/partitions | OK |
| T8->T12 | evaluate_patches schema (metrics CSV columns, summary keys), make_overlay | RISK: Task 8 not written yet — T12 must be dispatched after T8 and reuse its exact schema |
| T11 self | nnunetv2 never imported by tests; commands checked as argv lists | OK |
Ruling: Task 12 is dispatched only after Task 8 is complete, so it can copy Task 8's real output schema instead of a guessed one — cost if wrong: two divergent metric files.
Task 7: fix round 1/5 (1 addressed, 0 open — parse_overrides shared; commits d06a592..8abd1d8)
Task 7: complete (commits cb9596a..8abd1d8, review clean)
Task 8: minor (deferred): tests/test_predict.py:360 tautological shape assertion (panel.shape[0] compared to itself) — fix in the final wave; no test for a partition with zero lesion patches
Task 8: complete (commits 8abd1d8..64a95b1, review clean)
Task 9: implemented, real local dry run OK (exit 0): manifest 3567 rows / 3.0 min, audit masses 78 exclusions + 1618 kept / 17.75 min, splits train 525 / val 132 / test 196 patients, 52 patches, overfit dice 0.9934 (gate passed), 1 epoch val dice 0.1136, predict mean_dice_lesion 0.227 on 28 val patches. Commits 64a95b1..61c3e3b. REVIEW NOT YET RUN (session stopped at the user's 90% credit threshold; review package already written: review-64a95b1..61c3e3b.diff).
Session stop 2026-09-20 (5-hour window 93%): Tasks 1-8 complete and reviewed; Task 9 implemented + dry-run proven, review pending; Tasks 10-12 (nnU-Net) specified in the plan, not started; final whole-branch review not run.

## Reprise (handoff) — mis à jour 2026-09-20

1. Revue de la tâche 9 (paquet déjà généré : `review-64a95b1..61c3e3b.diff` dans l'espace de travail SDD).
2. Tâches 10, 11, 12 (nnU-Net sur imagettes) : export au format nnU-Net, plis par patiente + lanceur, import des prédictions dans le schéma de la tâche 8.
3. Revue finale de toute la branche `s2-s6-kaggle-smoke`, puis décision de fusion / push (action partagée, à valider par l'utilisateur).
Mineurs différés à traiter dans la vague finale : assertion tautologique `tests/test_predict.py:360` ; `targets.py` `_identity` qui duplique `ManifestRow.lesion_identity` ; graines globales inertes dans `dataset.py` ; absence de tests pour la validation vide et le repli mémoire de `overfit_batch`.
