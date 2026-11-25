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

## Session 2026-11-02 — survivabilité d'un entraînement long sur Kaggle

Branche : `kaggle-long-run` (depuis `main` @ 25cd979). Rien commité par l'agent ; commandes rendues à l'utilisateur.

Scan complet du code demandé par l'utilisateur, puis correction des trois manques bloquants qu'il a relevés : aucune sauvegarde intermédiaire, aucune configuration d'entraînement réel, aucun ordonnancement du learning rate.

Implémenté :
- `checkpoint_last.pt` (poids + optimiseur + scaler + scheduler) réécrit à chaque époque par écriture atomique (`_save_atomic` : fichier temporaire + `os.replace`), et `run.json` réécrit à chaque époque ; `checkpoint_best.pt` reste poids seuls, donc rechargeable par `load_checkpoint` / `predict.py` et sous la limite de 200 Mo du zip de `kaggle_smoke`.
- `resume_from` : reprise dans le même répertoire de run, à l'époque suivante, en réécrivant `config_resume_<n>.yaml` et en réouvrant `metrics.jsonl`. Refus si l'un de `RESUME_INVARIANT_FIELDS` (`patch_root`, `architecture`, `encoder`, `batch_size`, `accumulation_steps`) change.
- Scheduler (`none` | `cosine` | `plateau`), `early_stopping_patience`, `max_hours` ; `stop_reason()` fait primer le budget de temps sur l'arrêt anticipé.
- `val_dice_lesion` journalisé à côté de `val_dice`, et `selection_metric` (`val_dice` par défaut, `val_dice_lesion` dans la config de référence) : le `val_dice` toutes imagettes vaut 1,0 sur une imagette de fond non prédite et n'est pas comparable au `mean_dice_lesion` de `predict.py`.
- `configs/unet_baseline.yaml` : 60 époques, cosinus, patience 12, `max_hours: 8.0`, sélection lésion seule.
- `kaggle_smoke.py` : `--train-config`, `--all-patients`, `--resume-run` ; `--epochs` et `--encoder-weights` ne sont plus injectés sauf demande explicite, pour ne pas écraser la config.
- `make_fake_patch_dir(kinds=...)` produit des imagettes de fond à masque vide (traite aussi le mineur différé « absence de test pour la validation vide ») ; `scripts/train.py` enveloppe les `ValueError` dans `parser.error`.

Trois défauts trouvés et corrigés pendant la vérification, qui auraient tous frappé sur Kaggle après l'heure d'extraction :
1. YAML 1.1 lit `3e-4` comme une **chaîne** (un flottant exige un point) : `learning_rate` arrivait en `str` jusqu'à l'optimiseur. Config corrigée en `3.0e-4` et validation de type ajoutée dans `TrainConfig.__post_init__`, avec message explicite.
2. L'état du scheduler était sauvegardé avant son `step()` : chaque reprise rejouait un pas et étirait le calendrier d'une époque. Ordre corrigé.
3. `CosineAnnealingLR.state_dict()` porte `T_max` : une reprise avec plus d'époques gardait l'ancien horizon et renvoyait le LR à sa valeur de départ. `T_max` est réaligné sur la config après `load_state_dict`.
4. Le générateur du DataLoader était semé une seule fois : l'ordre des lots d'une reprise différait d'un run continu. Il est désormais resemé depuis `(graine, époque)` à chaque époque, donc l'ordre de l'époque N ne dépend que de N. (Change l'ordre des lots par rapport aux runs antérieurs ; aucune référence n'était figée.)

Preuves :
- `uv run pytest` : **143 passed** en 6 min 45 (106 avant, +37 tests). Exécuté avec `.venv/Scripts/python.exe -m pytest`.
- Preuve CLI sur imagettes synthétiques (CPU, resnet18, 4 époques, `scratchpad`) : run coupé par `max_hours=0.0` après l'époque 0 puis repris, comparé à un run continu de même graine → `lr`, `train_loss` et `val_dice_lesion` **identiques sur les 4 époques**, `best_epoch` et `best_val_dice` identiques.

Limites — ce qui n'est pas prouvé :
- **Toujours jamais exécuté sur Kaggle** : l'item correspondant de `PLAN.md` étape 3 reste décoché. Tout ci-dessus est vérifié en local, sur CPU et sur données synthétiques.
- Aucun entraînement de référence mesuré : `configs/unet_baseline.yaml` n'a jamais tourné sur données réelles. Les valeurs d'`epochs`, `batch_size`, `max_hours` et de patience sont des points de départ, pas des réglages mesurés.
- Coût de `extract_patches` sur les 657 patientes train+val non mesuré (mesuré sur 30 seulement).
- Équivalence d'une reprise entre GPU différents, ou CPU↔GPU, non vérifiée.
- Non traité, et hors du périmètre demandé : la partition reste gravée dans `index.csv` à l'extraction (changer `val_fold` impose de réextraire) ; pas de pilote de validation croisée ; HD95, distance moyenne symétrique, sensibilité par lésion, faux positifs par image et bootstrap groupé par patiente toujours absents (exigés par `docs/PROTOCOLE.md` pour l'étape 5) ; pas d'inférence image entière ; pas de code nnU-Net (tâches 10-12).

Prochaine action : lancer la première session réelle sur Kaggle avec `--train-config configs/unet_baseline.yaml --all-patients`, puis consigner ici la durée par époque, la VRAM observée et la raison d'arrêt, avant de régler `max_hours` et `batch_size` sur des mesures.

## Session 2026-11-24 — premier run réel sur Kaggle (fumée, 30 patientes)

Branche `kaggle-long-run` @ c625b2e, poussée sur `origin` puis clonée dans un notebook Kaggle (GPU T4, Internet On). Seul `segmentation-models-pytorch==0.5.0` installé en plus de l'image. Runtime : Python 3.13.16, torch 2.11.0+cu130, CUDA disponible — écart au verrou (Python 3.12, cu121) enregistré par `capture_environment --record-only`, non bloquant.
Données montées sous `/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset` (nouveau chemin de montage Kaggle, différent de celui de `docs/KAGGLE.md`).

Commande : `kaggle_smoke.py --raw-dir <ci-dessus> --work-dir /kaggle/working/smoke --limit-patients 30` (config `unet_smoke.yaml`). Les 8 étapes à `ok` :
| Étape | Durée | Résultat |
|---|---|---|
| build_manifest | 348.7 s | 3567 lignes (identique au local), 1 exclusion d'acquisition |
| audit | 912.1 s | 78 exclusions, 1618 retenues (identique au local) |
| make_splits | 1.6 s | train 525 / val 132 / test 196 patientes (identique au local) ; part maligne train 0.512, val 0.515 |
| extract_patches | 25.7 s | 251 imagettes (30 patientes par partition) : train 55 fond + 69 lésion, val 61 fond + 66 lésion |
| overfit | 185.1 s | Dice 0.9975 ≥ 0.95, gate passée |
| train | 27.9 s | 2 époques, best_epoch 1, best_val_dice 0.3495 |
| predict | 11.4 s | `val_summary.json` écrit |

Preuves : manifeste, audit et partitions reproduisent exactement les chiffres du run local sur un autre runtime ; la gate de surapprentissage passe sur GPU.

Limites :
- `best_val_dice` 0.3495 est le Dice **toutes imagettes** (la config de fumée garde `selection_metric: val_dice`), pas le Dice lésion.
- `val_summary.json` (127 imagettes de validation, 30 patientes) : `mean_dice_lesion` 0.673, `mean_iou_lesion` 0.527, `background_false_positive_rate` 1.0. Ce Dice **n'est pas une mesure de segmentation** : les imagettes de lésion sont centrées sur la boîte englobante du masque réel (marge 15 %), la lésion couvre en médiane 15 % de l'imagette, et les superpositions montrent une ellipse centrale plus un cadre parasite le long des bords de l'imagette. Le modèle a appris « une tache au centre », pas la lésion. Prédire toute l'imagette donnerait déjà 0.30 ; une référence « disque centré » reste à calculer pour savoir ce que le modèle ajoute à ce biais de position. Sur les imagettes de fond, il prédit toujours quelque chose (médiane 4.6 % des pixels).
- Superpositions (8 figures, récupérées dans `smoke_run/`) : le contour de référence tombe bien sur la lésion visible, donc pas de décalage image/masque.
- L'exclusion d'acquisition est un cas de calcification (`P_01563 RIGHT MLO`, référence DICOM non résolue) : sans effet sur la cible masses.
- Toujours aucun entraînement de référence ; durée par époque et VRAM sur toutes les patientes non mesurées.
- `docs/KAGGLE.md` et le README citent encore l'ancien chemin `/kaggle/input/cbis-ddsm-breast-cancer-image-dataset`.

Prochaine action : lancer `--all-patients --train-config configs/unet_baseline.yaml` en *Save & Run All* et consigner durée par époque, VRAM et raison d'arrêt.

## Session 2026-11-25 — documentation et feuille de route de la comparaison

Branche `kaggle-long-run` @ c625b2e, aucun code modifié. Rien commité par l'agent.

Fait :
- `docs/CONTEXTE.md` : contexte complet du projet en un fichier, pour une nouvelle personne ou session. Il renvoie à ce journal, à `PLAN.md` et à `docs/PROTOCOLE.md`, qui font autorité.
- Supports de l'oral, hors code : `reports/hyperparametres_modele.{pdf,html}` (21 diapositives sur les réglages) et `reports/fichiers_cles_pipeline.pptx` (28 diapositives sur `audit.py`, `patches.py`, `transforms.py`, `losses.py`, `unet_baseline.yaml`, `trainer.py`).
- Chemin de montage Kaggle corrigé dans le README et `docs/KAGGLE.md` (`/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset`) : la limite relevée le 2026-11-24 est levée.
- `PLAN.md` étape 3 : l'exécution sur Kaggle est cochée sur la preuve du run de fumée. L'entraînement de référence reste décoché. Ajout de la vérification du run du 2026-11-25 et de la référence « disque centré ».
- Feuille de route décidée par l'utilisateur, inscrite à l'étape 4 de `PLAN.md` : vérifier la sortie du run lancé, lancer 3 runs avec des hyperparamètres différents, faire les statistiques de comparaison, conclure sur le modèle.
- `docs/PROTOCOLE.md`, nouvelle section « Comparaison de réglages », écrite avant tout résultat de comparaison :
  - un facteur à la fois, mêmes partitions, imagettes, graine et seuil ;
  - comparaison sur la validation seulement ;
  - métrique principale `mean_dice_lesion` ;
  - bootstrap des patientes avec différences appariées ;
  - disque centré dans le même tableau ;
  - portée limitée par l'unique graine par réglage.
  Ces règles sont à faire relire par l'équipe, car elles touchent aux métriques.
- README : marche à suivre pour comparer plusieurs réglages, à savoir un YAML par réglage (`kaggle_smoke.py` n'a pas de `--set`), le même `--work-dir` avec `--skip-audit-if-present`, et la récupération de `smoke_run.zip` après chaque run (le run suivant le réécrit).

État des preuves :
- Run lancé sur Kaggle par l'utilisateur le 2026-11-25 : **sortie non encore vérifiée**, rien n'en est consigné ici.
- Toujours aucun entraînement de référence mesuré ni aucune comparaison.
- Non implémentés, et nécessaires à l'étape « statistiques » : le bootstrap des patientes et la référence disque centré. `predict.py` écrit déjà `val_metrics.csv` par imagette, avec `patient_id` : c'est assez pour le bootstrap. Le disque centré demande en plus les masques des imagettes de validation, donc le `--work-dir` du run.
- Budget : trois runs de 8 h (`max_hours`) consomment environ 24 h des 30 h hebdomadaires de GPU Kaggle. Chaque nouvelle session refait aussi manifeste, audit et extraction, sauf si le `--work-dir` précédent est remonté en input.

Prochaine action : vérifier `smoke_report.json`, `val_summary.json` et les superpositions du run du 2026-11-25, puis les consigner ici. Ensuite, choisir les trois réglages et écrire leurs YAML avant de lancer.
