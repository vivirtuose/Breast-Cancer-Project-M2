# Contexte complet du projet — à lire en premier par une nouvelle session

> Mis à jour le 2026-11-25, branche `kaggle-long-run` @ `c625b2e`.
> Ce fichier rassemble le contexte. Il **ne remplace pas** les sources qui font autorité :
> l'avancement daté et ses preuves sont dans
> [`docs/superpowers/plans/2026-09-16-kaggle-smoke-pipeline.progress.md`](superpowers/plans/2026-09-16-kaggle-smoke-pipeline.progress.md),
> les règles scientifiques dans [`docs/PROTOCOLE.md`](PROTOCOLE.md), les jalons dans [`PLAN.md`](../PLAN.md).
> En cas de désaccord, ce sont eux qui ont raison : corriger ce fichier.

---

## 1. Le projet en une page

| | |
|---|---|
| **Cadre** | Projet M2, module MLB, Rennes 2026–2027. Équipe de 3 personnes. |
| **Livrable noté** | Un **oral de 15 minutes**, noté avec la grille [`Evaluation.pdf`](../Evaluation.pdf) : plan, introduction, théorie et diversité des méthodes, pratique et validation, résultats, conclusion, qualité des transparents, des graphiques, gestion du temps, intérêt, note sur 10. Le dépôt n'est pas noté en lui-même. |
| **Exigence signalée, non confirmée** | L'utilisatrice ou l'utilisateur a mentionné qu'un **notebook Jupyter** serait exigé à l'évaluation. La grille n'en parle pas : à vérifier auprès de l'enseignant (le notebook doit-il réentraîner, ou seulement montrer méthode et résultats ?). |
| **Tâche** | **Segmentation binaire des masses** mammaires sur CBIS-DDSM : pour une imagette de mammographie, produire le masque pixel par pixel de la lésion. Pas de diagnostic bénin/malin. |
| **Périmètre** | Masses seulement (pas les calcifications). Les trois pathologies (`MALIGNANT`, `BENIGN`, `BENIGN_WITHOUT_CALLBACK`) sont des lésions positives ; la pathologie est une métadonnée d'analyse, **jamais une entrée du modèle**. |
| **Modèle actuel** | U-Net (`segmentation-models-pytorch`), encodeur ResNet-34 pré-entraîné ImageNet, entrée en niveaux de gris, sur imagettes de 512 px. |
| **Comparaison prévue** | nnU-Net sur **les mêmes imagettes** (décision du 2026-09-20). Spécifié (tâches 10–12 du plan), **non codé**. U-Net++ et DeepLabV3+ disponibles sans code. |
| **Matériel** | Poste local : GeForce MX350 2 Go, ~15 Go libres → développement et tests seulement. Entraînement sur **Kaggle** (GPU T4 16 Go, sessions de 12 h, environ 30 h de GPU par semaine). |
| **État au 2026-11-25** | Toute la chaîne fonctionne, en local et sur Kaggle (run de fumée réussi le 2026-11-24). Un nouveau run a été lancé le 2026-11-25, sortie pas encore vérifiée. **Aucun entraînement de référence n'a encore tourné** : il n'existe aucun résultat de segmentation interprétable. |

Décision de cadrage (2026-11-24) : **ne pas basculer vers une classification seule**. La difficulté matérielle venait de nnU-Net, pas de la segmentation. Le U-Net déjà construit reste le cœur ; nnU-Net devient optionnel ; une classification sur les mêmes imagettes reste une extension possible (`PLAN.md`, « Extensions »).

---

## 2. Les données

### 2.1 Source

- Miroir Kaggle `awsaf49/cbis-ddsm-breast-cancer-image-dataset` : sous-dossiers `csv/` (descriptions des cas) et `jpeg/` (images converties en JPEG).
- En local : `data/raw/` (jamais modifié, jamais copié dans un worktree).
- Sur Kaggle, le jeu est monté sous `/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset` (chemin constaté le 2026-11-24, repris dans `docs/KAGGLE.md` et le README depuis le 2026-11-25).
- Article de référence du jeu : Lee et al. 2017, *Scientific Data* ([`sdata2017177.pdf`](../sdata2017177.pdf)).

### 2.2 Effectifs relevés sur notre copie

| Quantité | Valeur |
|---|---|
| Participantes | 1 566 |
| Dossiers d'images | 6 774 |
| Lésions décrites (lignes du manifeste) | **3 567** |
| Lésions de type masse | 1 696 |
| Masses exclues par l'audit | **78** |
| Masses retenues | **1 618** |
| Images de masse portant au moins deux lésions | 71 |
| Patientes avec masses : entraînement / validation / test | **525 / 132 / 196** |

Les chiffres du manifeste, de l'audit et des partitions sont **identiques** entre le run local (2026-09-20, Windows, Python 3.12, CUDA 12.1) et le run Kaggle (2026-11-24, Linux, Python 3.13, CUDA 13).

---

## 3. Curation des données (étapes 1–2)

Chaque étape est un module pur dans `src/cbis_ddsm/` plus une CLI mince dans `scripts/`. Les étapes ne communiquent que par fichiers versionnés.

### 3.1 Manifeste — `src/cbis_ddsm/data/manifest.py`, `scripts/build_manifest.py`

- Lit les 4 CSV (`calc_/mass_case_description_{train,test}_set.csv`).
- Résout les références DICOM des CSV vers les fichiers JPEG réels du miroir, via un index des dossiers.
- Distingue l'imagette recadrée (*crop*) du masque ROI quand le miroir les inverse (cas connu d'inversion ROI/crop) ; une ambiguïté non résoluble lève une erreur explicite.
- Une ligne par lésion : patiente, côté, vue (CC/MLO), anomalie, type, pathologie, densité, BI-RADS, subtilité, forme, marges, partition officielle, chemins image / crop / masque (relatifs à `--raw-dir`).
- Sortie : `data/interim/manifest.csv` (**3 567 lignes**) et `acquisition_exclusions.csv` (**1 ligne** : une calcification, `P_01563 RIGHT MLO`, référence DICOM introuvable ; sans effet sur les masses).

### 3.2 Audit — `src/cbis_ddsm/data/audit.py`, `scripts/audit_data.py --lesion-type mass`

Une seule raison par ligne, avec une précédence stable :

1. `missing_path` : image, crop ou masque absent ;
2. `duplicate_lesion` : identité de lésion déjà vue ;
3. `dimension_mismatch` : le masque n'a pas la taille de l'image ;
4. `non_binary_mask` : présence de gris intermédiaires (valeurs ni ≤ 10 ni ≥ 245) ;
5. `empty_mask` : aucun pixel ≥ 245.

Résultat sur les masses : **78 exclusions, toutes `dimension_mismatch`**. Décision du 2026-09-16 : les exclure plutôt que redimensionner le masque. Un étirement donnait une erreur médiane de 40 px, contre 7 px pour les lignes normales. Sortie : `data/interim/exclusions.csv`.

### 3.3 Cibles par image — `src/cbis_ddsm/data/targets.py`

- Garde les masses, regroupe les lésions d'une même mammographie (une `ImageRecord` par image).
- **Une image dont au moins une lésion est exclue est exclue entière** : son masque fusionné serait incomplet.
- Masques binarisés à `pixel >= 128` (la compression JPEG déplace 255 vers ~249 et 0 vers de petites valeurs), puis **union pixel par pixel** de toutes les lésions de l'image (masque fusionné).

---

## 4. Création des partitions et du jeu de validation — `src/cbis_ddsm/data/splits.py`, `scripts/make_splits.py`

- **Unité de découpage : la patiente**, jamais l'image ni la lésion. Une patiente peut avoir jusqu'à 4 images (2 seins × 2 vues) : tirer par image mettrait la vue CC d'un sein à l'entraînement et sa vue MLO en validation.
- **Test = partition officielle « test » de CBIS-DDSM** (196 patientes avec masses). Il est **scellé** : jamais utilisé pour entraîner, choisir une époque ou régler un hyperparamètre. Le script vérifie qu'aucune patiente n'est à la fois dans le train et le test officiels, et refuse de continuer sinon.
- **Validation = un pli du train officiel** :
  - étiquette par patiente : 1 si l'une de ses masses du train est maligne, sinon 0 ;
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` sur les patientes (une ligne par patiente, donc le groupement est exact) ;
  - `val_fold = 0` → **132 patientes de validation**, les 4 autres plis → **525 patientes d'entraînement** ;
  - part de patientes malignes : 0,512 en entraînement, 0,515 en validation.
- Sortie : `data/splits/split_v1.json` (version, graine, `n_folds`, `val_fold`, SHA-256 du manifeste, pli de chaque patiente, listes train/val/test).
- La partition est ensuite **inscrite dans `index.csv` des imagettes** : changer `val_fold` impose de réextraire les imagettes.

---

## 5. Imagettes — `src/cbis_ddsm/data/patches.py`, `scripts/extract_patches.py`

- **Jamais de redimensionnement de la mammographie entière** pour l'entraînement : une réduction à 512 px effacerait les spicules, qui sont le signe de malignité.
- **Imagette de lésion** : une par lésion. Fenêtre carrée centrée sur la boîte englobante du masque de la lésion, de côté `max(512, ⌈c·(1 + 2·0,15)⌉)` où `c` est le côté de la boîte (marge `m = 0,15`, codée en dur). Elle est découpée dans l'image avec remplissage par des zéros si elle dépasse, puis ramenée à 512 px. Une petite lésion garde sa résolution native ; une grande est réduite. Le masque découpé est le **masque fusionné**, donc une autre lésion voisine apparaît aussi.
- **Imagette de fond** : `--background-per-image 1` par image, fenêtre de 512 px tirée au hasard (graine fixée) dont le centre est dans le tissu et qui ne contient **aucun** pixel de lésion. Jusqu'à 200 essais ; un manque est compté, pas une erreur.
- Sortie : `data/processed/patches_v1/images/*.png`, `masks/*.png` (0/255) et `index.csv` (identifiant, image d'origine, patiente, **partition**, type, géométrie de la fenêtre pour reprojeter la prédiction dans l'image d'origine, nombre de pixels de lésion).
- Seules les partitions `train` et `val` sont extraites par défaut.
- Run de fumée (30 patientes par partition) : 251 imagettes (train 69 lésion + 55 fond, val 66 lésion + 61 fond). Sur la validation, la lésion couvre en médiane 15 % de l'imagette.

**Biais connu, central pour l'oral** : comme la fenêtre est centrée sur la vraie lésion, la lésion est presque toujours au milieu de l'imagette. Un modèle peut obtenir un bon Dice en prédisant « une tache au centre ». Voir §9.

---

## 6. Chargement et augmentations — `src/cbis_ddsm/data/dataset.py`, `transforms.py`

- `PatchDataset` lit l'index, filtre une partition, charge image et masque en niveaux de gris.
- Image : `float32 / 255`, un canal, **sans normalisation ImageNet**. Masque : binaire.
- Augmentations d'entraînement (albumentations 1.4.24), appliquées **identiquement à l'image et au masque**, chacune avec p = 0,5 : miroir horizontal, miroir vertical, rotation de 90°, affine (translation ±5 %, zoom 0,9–1,1, rotation ±15°, remplissage 0), luminosité/contraste ±15 %. Aucune augmentation en validation ou en prédiction.
- Déterminisme : le générateur du DataLoader est resemé avec `graine + époque` à chaque époque (`trainer.py:325`), donc l'ordre des lots de l'époque N ne dépend que de N. Pas de `persistent_workers` (sinon les augmentations seraient figées).

---

## 7. Modèle — `src/cbis_ddsm/models/factory.py`

| Architecture (`architecture=`) | Poids (ResNet-34, 1 canal) |
|---|---|
| `unet` (**choisie**) | **24 430 097** |
| `unetplusplus` | 26 072 337 |
| `deeplabv3plus` | 22 431 185 |

- Encodeur `resnet34`, poids **ImageNet** (`encoder_weights: imagenet`). La bibliothèque adapte la première convolution à une entrée à 1 canal. 46 couches BatchNorm.
- Sortie : 1 carte de logits ; probabilité = sigmoïde ; masque = probabilité ≥ seuil τ = 0,5.
- **Poids entraînés disponibles** : uniquement ceux du run de fumée (2 époques sur 30 patientes, dans `MLB/smoke_run/…/checkpoint_best.pt`). Ils servent de preuve de fonctionnement, pas de modèle.

---

## 8. Entraînement — `src/cbis_ddsm/training/`

### 8.1 Mécanique

- Perte (`losses.py`) : `L = w·BCE + (1 − w)·(1 − Dice_soft)`, avec `Dice_soft = (2Σpy + s)/(Σp + Σy + s)` calculé **par imagette** puis moyenné, et `s = 1` codé en dur. Sur une imagette de fond, `Dice_soft = 1/(Σp + 1)` : la perte pousse à ne rien prédire.
- Optimiseur : **AdamW**, codé en dur (`trainer.py:271`).
- Demi-précision (AMP) avec `GradScaler`, accumulation de gradients sur `accumulation_steps` lots.
- Validation à chaque époque : `val_loss`, `val_dice` (toutes imagettes ; une imagette de fond sans prédiction vaut 1) et `val_dice_lesion` (imagettes de lésion seules, la même quantité que `mean_dice_lesion` de `predict.py`).
- Sélection de la meilleure époque sur `selection_metric` ; arrêt anticipé et ordonnanceur `plateau` sur cette même métrique.
- Fichiers du run (`experiments/<run_name>_<horodatage>/`) :
  - `config.yaml` ;
  - `metrics.jsonl` (une ligne par époque) ;
  - `run.json` (`best_epoch`, `completed_epochs`, `stop_reason`, métrique de sélection) ;
  - `checkpoint_best.pt` (poids seuls, rechargeables par `load_checkpoint` / `predict.py`) ;
  - `checkpoint_last.pt` (poids, optimiseur, scaler, ordonnanceur ; réécrit à chaque époque par écriture atomique).
- **Reprise** (`resume_from` / `--resume-run`) : reprend à l'époque suivante dans le même dossier. Elle est refusée si `patch_root`, `architecture`, `encoder`, `batch_size` ou `accumulation_steps` changent. Testé : un run coupé puis repris reproduit à l'identique un run continu (même LR, mêmes pertes, mêmes Dice). L'équivalence entre GPU différents n'est pas garantie.
- **Gate de surapprentissage** (`overfit_batch`) : apprendre 8 imagettes non augmentées en 300 pas doit donner un Dice ≥ 0,95. Si ce n'est pas le cas, le problème est dans les données (alignement), pas dans le modèle.

### 8.2 Hyperparamètres choisis

Défauts dans `TrainConfig` ([`config.py:32`](../src/cbis_ddsm/training/config.py)). Valeurs effectives dans les YAML.

| Clé | Fumée `configs/unet_smoke.yaml` | **Référence `configs/unet_baseline.yaml`** |
|---|---|---|
| `architecture` / `encoder` / `encoder_weights` | unet / resnet34 / imagenet | unet / resnet34 / imagenet |
| `epochs` | 2 | 60 |
| `batch_size` × `accumulation_steps` | 4 × 2 | 8 × 2 (lot effectif 16) |
| `learning_rate` | 3·10⁻⁴ (défaut) | 3.0e-4 |
| `weight_decay` | 10⁻⁴ (défaut) | 1.0e-4 |
| `bce_weight` | 0,5 | 0,5 |
| `scheduler` / `min_learning_rate` | none | cosine / 1.0e-6 |
| `early_stopping_patience` | aucune | 12 |
| `max_hours` | aucun | 8.0 |
| `selection_metric` | val_dice (défaut) | **val_dice_lesion** |
| `augment` / `amp` / `threshold` / `seed` | true / true / 0,5 / 0 | true / true / 0,5 / 0 |

Paramètres des données : imagettes de 512 px, marge de 0,15, 1 imagette de fond par image, 5 plis, `val_fold` 0, graine 0.

**Aucune de ces valeurs n'a été réglée sur des mesures** : ce sont des points de départ courants. Les effets attendus de chaque réglage sont détaillés dans [`reports/hyperparametres_modele.pdf`](../reports/hyperparametres_modele.pdf).

### 8.3 Modifier les hyperparamètres

- YAML (recommandé, tracé) : écrire une nouvelle config. Écrire `1.0e-4`, **jamais `1e-4`** : YAML 1.1 le lit comme une chaîne, et la config le refuse.
- `scripts/train.py --config <yaml> --set clé=valeur [--set …]` ; `null` pour « aucune valeur ».
- `scripts/overfit_batch.py --config … --n 8 --steps 300 --target-dice 0.95 --set …`.
- `scripts/kaggle_smoke.py` **n'a pas de `--set`**. Il accepte seulement `--train-config`, `--epochs`, `--encoder-weights`, `--seed`, `--size`, `--limit-patients` / `--all-patients`, `--n-folds`, `--overfit-steps`, `--overfit-n`, `--device`, `--skip-audit-if-present` et `--resume-run`.
- Données : `extract_patches.py --size --background-per-image --seed --limit-patients --partitions` ; `make_splits.py --n-folds --val-fold --seed` ; prédiction : `predict.py --threshold`.
- Codés en dur : augmentations (`transforms.py:16`), AdamW (`trainer.py:271`), lissage du Dice (`losses.py:16`), marge (`patches.py:54`).

---

## 9. Évaluation — `src/cbis_ddsm/evaluation/`, `src/cbis_ddsm/inference/predict.py`

- Dice et IoU binaires avec des règles explicites : les deux masques vides → 1 ; un seul vide → 0.
- `predict.py` écrit :
  - `predictions/<partition>_metrics.csv` : par imagette `patch_id, patient_id, kind, lesion_pixels, dice, iou, predicted_pixels` ;
  - `<partition>_summary.json` : `n_patches`, `mean_dice_lesion`, `mean_iou_lesion`, `background_false_positive_rate` (part des imagettes de fond où au moins un pixel est prédit) ;
  - jusqu'à 8 superpositions `figures/<patch_id>.png` : image / vérité terrain en vert / prédiction en rouge.
- **Exigés par le protocole mais pas encore implémentés** : HD95, distance moyenne symétrique, sensibilité par lésion, faux positifs par image, FROC, intervalles de confiance par rééchantillonnage des patientes, analyses par densité / taille / pathologie / vue, inférence sur image entière.
- **Référence « disque centré » (proposée, non calculée)** : prédire pour chaque imagette de lésion le même disque au centre, de rayon choisi sur l'entraînement, sans regarder l'image. Elle mesure la part du Dice due au centrage des imagettes. Le chiffre à défendre est l'écart entre le U-Net et ce disque, pas le Dice brut. Le calcul demande les masques des imagettes, absents de `smoke_run`.
- Limite à toujours énoncer : un score sur imagettes centrées sur la vraie lésion **n'est pas une détection** sur mammographie entière.

---

## 10. Exécution

### 10.1 En local

```powershell
uv run pytest
```
Ou bien `.venv/Scripts/python.exe -m pytest -q` : **143 tests verts** au 2026-11-02, sur données synthétiques, CPU, sans réseau.

```powershell
uv run python scripts/kaggle_smoke.py --raw-dir data/raw --work-dir <dossier> --limit-patients 30
```

Environnement verrouillé : Python 3.12, torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 1.26.4, opencv-python-headless 4.10.0.84, albumentations 1.4.24, scikit-image 0.24.0, segmentation-models-pytorch 0.5.0, scikit-learn 1.9.1, pytest 8.3.4 (`pyproject.toml` + `uv.lock`). `configs/env.yaml` = capture de l'environnement local.

### 10.2 Sur Kaggle (procédure validée le 2026-11-24)

Prérequis du notebook :
- **Internet On** : il faut un compte vérifié par téléphone ;
- **Accélérateur GPU T4** : sans lui, torch est la version `+cpu` ;
- jeu `awsaf49/cbis-ddsm-breast-cancer-image-dataset` ajouté en input ;
- la branche à cloner doit être **poussée** sur GitHub.

Cellules :

```python
!git clone --branch kaggle-long-run https://github.com/vivirtuose/Breast-Cancer-Project-M2.git /kaggle/working/code
%cd /kaggle/working/code
!pip install -q segmentation-models-pytorch==0.5.0
!python -c "import sys, torch; print(sys.version, torch.__version__, torch.cuda.is_available())"
RAW = "/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"
!python scripts/kaggle_smoke.py --raw-dir {RAW} --work-dir /kaggle/working/smoke --limit-patients 30
```

Référence complète, à lancer en **Save Version → Save & Run All** pour survivre à l'inactivité :

```python
!python scripts/kaggle_smoke.py --raw-dir {RAW} --work-dir /kaggle/working/baseline --all-patients --train-config configs/unet_baseline.yaml
```

- **Ne jamais faire `pip install -e .`** sur Kaggle : cela remplacerait le torch GPU de l'image.
- Écart d'environnement assumé (Python 3.13, torch 2.11+cu130 sur Kaggle) : enregistré par `capture_environment --record-only`, non bloquant.
- `/kaggle/working` n'est pas persistant : récupérer `smoke_run.zip`, ou réutiliser la sortie du notebook comme input d'une session suivante pour une reprise (`--skip-audit-if-present --resume-run <dossier du run>`).
- Durées mesurées : manifeste 5 min 49, audit 15 min 12 (ces deux étapes lisent toutes les masses), extraction de 30 patientes 26 s, environ 10 s par époque sur 124 imagettes. La durée d'une époque sur toutes les patientes n'est pas mesurée.

---

## 11. Résultats obtenus à ce jour

| Run | Date | Ce qu'il prouve | Chiffres |
|---|---|---|---|
| Fumée locale | 2026-09-20 | Chaîne complète en local | gate 0,9934 ; 1 époque ; `mean_dice_lesion` 0,227 sur 28 imagettes de validation |
| Fumée Kaggle (30 patientes / partition) | 2026-11-24 | Chaîne complète sur GPU T4, données identiques au local | 8/8 étapes ok en 25 min ; gate **0,9975** ; 2 époques ; `val_dice` 0,3495 ; `mean_dice_lesion` **0,673**, `mean_iou_lesion` 0,527 sur 66 imagettes de lésion ; `background_false_positive_rate` **1,0** sur 61 imagettes de fond |

Interprétation du run Kaggle, consignée dans le journal :
- l'alignement image/masque est vérifié par les superpositions ;
- le **0,673 n'est pas un résultat de segmentation** : le modèle prédit une ellipse centrale et un cadre parasite le long des bords ;
- prédire toute l'imagette donnerait déjà un Dice de 0,30 ;
- le modèle prédit des pixels sur toutes les imagettes de fond.

Le run est conservé hors dépôt dans `MLB/smoke_run/` et `MLB/smoke_run.zip`.

---

## 12. Livrables et supports existants

| Livrable | Emplacement | État |
|---|---|---|
| Code du pipeline (données → prédiction) | `src/cbis_ddsm/`, `scripts/` | Livré, testé |
| Protocole scientifique | [`docs/PROTOCOLE.md`](PROTOCOLE.md) | Fixé |
| Architecture logicielle | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | Fixée |
| Procédure Kaggle | [`docs/KAGGLE.md`](KAGGLE.md), README « Lancer un entraînement sur Kaggle » et « Comparer plusieurs réglages » | À jour (chemin de montage corrigé le 2026-11-25) |
| Références bibliographiques | [`docs/REFERENCES.md`](REFERENCES.md) | |
| Cours (4 modules : maladie et mammographie, donnée, métriques, modèles) | [`cours/`](../cours/) | Rédigés, avec corrections de séance |
| Schémas | [`schemas/*.svg`](../schemas/) : carte du projet (EN/FR), plan de cours, partitions, plan des modèles, plan d'exécution, plan de code | |
| Diaporama pédagogique « Un ours dans la neige » | `MLB/Un ours dans la neige.pdf` (hors dépôt), 31 diapositives | Charte visuelle de référence : marine, crème, serif, terracotta |
| Diaporama « Le run de fumée sur Kaggle » (15 diapositives) | Artifact privé `https://claude.ai/artifact/HvoFPJwk16qmcHfdReN9zr` (exportable en PPTX/PDF) | Publié |
| Diaporama « Les réglages du modèle » (21 diapositives) | [`reports/hyperparametres_modele.pdf`](../reports/hyperparametres_modele.pdf) + source `.html` | Non commité |
| PowerPoint « Six fichiers clés du pipeline » (28 diapositives, notes d'orateur) | [`reports/fichiers_cles_pipeline.pptx`](../reports/fichiers_cles_pipeline.pptx) : `audit.py`, `patches.py`, `transforms.py`, `losses.py`, `unet_baseline.yaml`, `trainer.py` | Non commité |
| Notebook pour l'évaluation | — | **À faire** : notebook 01 d'entraînement Kaggle et notebook 02 de résultats, qui lit les artefacts et appelle le package |

---

## 13. Ce qui reste à faire, par priorité pour l'oral

Feuille de route décidée le 2026-11-25 (`PLAN.md`, étapes 3 et 4) :

1. **Vérifier la sortie du run lancé le 2026-11-25** : `smoke_report.json`, `val_summary.json`, superpositions. La consigner dans le journal.
2. **Lancer 3 runs avec des hyperparamètres différents.** Chaque run change un seul facteur par rapport à `unet_baseline.yaml`, dans son propre YAML avec son `run_name`. Tous gardent les mêmes partitions, imagettes et graine (`--skip-audit-if-present` sur le même `--work-dir`). Candidats, à choisir avant de lancer :
   - taux d'apprentissage 10⁻⁴ / 3·10⁻⁴ / 10⁻³ ;
   - `bce_weight` ;
   - architecture (U-Net++ ou DeepLabV3+).
   Budget : environ 24 h de GPU sur 30 h par semaine.
3. **Statistiques de comparaison**, selon `docs/PROTOCOLE.md`, « Comparaison de réglages » :
   - `mean_dice_lesion` avec un intervalle par bootstrap des patientes ;
   - différences appariées entre runs ;
   - taux de faux positifs sur le fond ;
   - référence disque centré dans le même tableau.
   Le bootstrap et le disque centré sont **à coder**.
4. **Conclure sur le modèle**, sur la validation seulement. Limites à énoncer : une graine par réglage, imagettes centrées, pas de détection sur image entière.
5. Ensuite : métriques manquantes du protocole (HD95, sensibilité par lésion, faux positifs par image), graines supplémentaires pour le réglage retenu, puis évaluation **unique** sur le test, réglages figés.
6. Notebooks de l'évaluation, puis un seul diaporama de 15 minutes (12 à 15 diapositives). Il montre les courbes d'apprentissage, la comparaison avec ses intervalles et des exemples d'échec. Le préparer avec une répétition chronométrée à trois.

Petites dettes connues :
- assertion tautologique `tests/test_predict.py:360` ;
- `targets.py` duplique `ManifestRow.lesion_identity` ;
- graines globales inertes dans `dataset.py` ;
- `kaggle_smoke.py` ne transmet ni `--set` ni `--background-per-image`.

---

## 14. État Git et conventions

- Dépôt : `https://github.com/vivirtuose/Breast-Cancer-Project-M2.git`.
- Branches :
  - `main` @ `25cd979` ;
  - `kaggle-long-run` @ `c625b2e`, poussée, contient checkpoints, reprise et config de référence, **non fusionnée dans `main`** ;
  - anciennes : `s0-s1-environment-manifest`, `s2-s6-kaggle-smoke`.
- Non commités au 2026-11-25 : README, `PLAN.md`, `docs/PROTOCOLE.md`, `docs/KAGGLE.md`, le journal d'avancement, ce fichier, `reports/hyperparametres_modele.{pdf,html}`, `reports/fichiers_cles_pipeline.pptx`, `schemas/organigramme_fichiers_pipeline_fr.svg`, `AGENTS.md` (non suivi).
- Sur ce poste, `.git` appartient à un autre utilisateur Windows : utiliser `git -c safe.directory='*' …`.
- Code, commentaires et messages de commit en anglais ; documents du projet en français.
- **Aucune marque de Claude dans les commits ni les PR** : pas de `Co-Authored-By: Claude…`, même si un ancien plan le demande.
- Ne jamais écrire dans `data/raw`, ne jamais copier les données brutes ni `.venv` dans un worktree, ne pas nettoyer branches ou worktrees sans accord explicite.
- Séparer toujours : code implémenté / tests synthétiques / exécution réelle / résultat scientifique validé. Le test réservé ne sert jamais au réglage. Une comparaison sur imagettes n'est pas une validation de détection sur image entière.
- À la fin d'un jalon : mettre à jour le **journal d'avancement** (date, branche/commit, preuves, limites, prochaine action), pas un résumé concurrent.
