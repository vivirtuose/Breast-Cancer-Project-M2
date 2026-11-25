# Breast Cancer Project M2

Projet de recherche et d’apprentissage visant à segmenter les lésions visibles sur les mammographies de CBIS-DDSM avec des modèles de deep learning. Première étape : les masses ; extension éventuelle aux calcifications après audit des annotations.

## Objectif et périmètre

Produire un masque de lésion à partir d’une mammographie et évaluer sa qualité géométrique. La segmentation localise une anomalie ; elle ne détermine pas, à elle seule, sa malignité. Le projet pourra ensuite étudier la classification bénin/malin ou une segmentation spécifique des lésions malignes, avec un protocole explicite.

**État documenté :** le pipeline complet (manifeste, audit, partitions, imagettes, U-Net, entraînement avec reprise, évaluation) est livré et a tourné de bout en bout sur Kaggle lors d'un run de fumée (2026-11-24). Aucun entraînement de référence n'est encore mesuré. Prochaines étapes, dans l'ordre : vérifier la sortie du run lancé, entraîner trois réglages différents sur les mêmes partitions, les comparer statistiquement, conclure sur le modèle (détail dans [PLAN.md](PLAN.md), étape 4). Voir [le journal de livraison](docs/superpowers/plans/2026-09-16-kaggle-smoke-pipeline.progress.md) pour les preuves, commits et limites, et [docs/CONTEXTE.md](docs/CONTEXTE.md) pour le contexte complet en un fichier. Un composant livré ou un smoke test ne prouve pas un entraînement de référence ni une validation finale ; vérifier les artefacts du run concerné.

## Organisation

```text
cours/                    Quatre modules pédagogiques et questions
schemas/                  Cartes SVG du projet et du cours, FR et EN
docs/                     Architecture, protocole et références
configs/                  Configurations des expériences
data/
  raw/                    Téléchargements originaux, immuables
  interim/                Manifestes et contrôles intermédiaires
  processed/              Images et masques prêts pour l’entraînement
  splits/                 Répartitions par patiente
notebooks/                Exploration et visualisation
src/cbis_ddsm/
  data/                   Lecture, appariement, transformations
  models/                 Architectures et chargement des poids
  training/               Entraînement et pertes
  evaluation/             Métriques et analyses
  inference/              Prédiction et reconstruction des masques
scripts/                  Points d’entrée du pipeline
tests/                    Contrôles des données et du pipeline
experiments/              Journaux, configurations et poids locaux
reports/figures/           Rapports et figures sélectionnées
```

Les dossiers du pipeline contiennent désormais des composants implémentés ; leur état détaillé et leurs limites sont suivis dans le plan et son journal. L’environnement S0 cible Python 3.12 et ses versions sont fixées dans `pyproject.toml` et `uv.lock`.

## Démarrage

1. Lire le [plan](PLAN.md) et le [protocole](docs/PROTOCOLE.md).
2. Consulter le [cours](cours/00_index.md).
3. Recréer l’environnement avec `uv sync --frozen`, puis exécuter `uv run pytest`.
4. Produire la photographie de la machine avec `uv run python scripts/capture_environment.py` ; un code de sortie non nul signifie que le gate S0 reste bloqué.
5. Télécharger les données depuis [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset), puis les placer dans `data/raw/`.
6. Auditer les CSV, les chemins, les masques et les identités patient avant tout entraînement :
   ```bash
   uv run python scripts/build_manifest.py --raw-dir data/raw
   uv run python scripts/audit_data.py --raw-dir data/raw --lesion-type mass
   uv run python scripts/make_splits.py --raw-dir data/raw
   uv run python scripts/extract_patches.py --raw-dir data/raw
   ```
7. Lancer un run de fumée de bout en bout (petit sous-ensemble, entraînement court) avec
   `scripts/kaggle_smoke.py`, localement ou sur Kaggle — voir [docs/KAGGLE.md](docs/KAGGLE.md) pour
   la procédure Kaggle complète :
   ```bash
   uv run python scripts/kaggle_smoke.py --raw-dir data/raw --work-dir /tmp/cbis_smoke --limit-patients 30
   ```
8. Pour un entraînement complet sur GPU Kaggle, suivre [Lancer un entraînement sur Kaggle](#lancer-un-entraînement-sur-kaggle-pas-à-pas)
   ci-dessous.

Les données, secrets, poids et sorties volumineuses restent hors de Git. Les petits résultats retenus pour le rapport peuvent être ajoutés dans `reports/`.

## Lancer un entraînement sur Kaggle, pas à pas

Le GPU local (MX350, 2 Go) ne suffit pas pour un entraînement de référence : on développe en
local et on entraîne sur Kaggle. Cette section est le mode opératoire ; [docs/KAGGLE.md](docs/KAGGLE.md)
détaille la configuration du notebook et la définition du succès du run de fumée.

**Aucun entraînement de référence n'a encore été mesuré.** Les valeurs par défaut ci-dessous sont
des points de départ raisonnables, pas des réglages validés par une mesure.

### Étape 1 — Créer le notebook

Sur [kaggle.com](https://www.kaggle.com), nouveau notebook, puis dans *Notebook options* :
accélérateur **GPU T4 x1** et **Internet: On** (nécessaire pour `pip install` et pour télécharger
les poids `imagenet`).

### Étape 2 — Monter les données

Dans *Add Input*, ajouter `awsaf49/cbis-ddsm-breast-cancer-image-dataset`. Il apparaît en lecture
seule sous `/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/`, avec les `csv/` et `jpeg/`
attendus (chemin constaté le 2026-11-24 ; Kaggle l'a déjà changé une fois). En cas de doute,
`find /kaggle/input -maxdepth 4 -name csv` donne le bon `--raw-dir`.

### Étape 3 — Récupérer le code

```bash
git clone --branch <branche> https://github.com/vivirtuose/Breast-Cancer-Project-M2.git
```

Pour un dépôt privé sans exposer de jeton, publier une archive du dépôt (sans `data/`, `.venv/`,
`experiments/`) comme *Kaggle Dataset* privé et l'ajouter en second input.

### Étape 4 — Installer le strict nécessaire

**Ne pas exécuter `pip install -e .`** : le projet épingle des roues `cu121` qui ne correspondent
pas à l'image Kaggle et casseraient l'accélération GPU déjà en place.

```bash
pip install segmentation-models-pytorch==0.5.0
```

### Étape 5 — Lancer la première session

```bash
python scripts/kaggle_smoke.py --raw-dir /kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset --work-dir /kaggle/working/baseline --all-patients --train-config configs/unet_baseline.yaml
```

L'orchestrateur enchaîne les huit étapes (environnement → manifeste → audit → partitions →
imagettes → gate de surapprentissage → entraînement → inférence), journalise chacune dans
`<work-dir>/logs/<étape>.log` et affiche un résumé d'une ligne par étape.

### Étape 6 — Reprendre après la coupure de session

`configs/unet_baseline.yaml` fixe `max_hours: 8.0` : le run s'arrête de lui-même avant que Kaggle
ne le coupe, et l'étape `train` affiche alors la commande de reprise. Chaque époque réécrit
`checkpoint_last.pt` par écriture atomique, donc même un arrêt brutal reste reprenable.

```bash
python scripts/kaggle_smoke.py --raw-dir /kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset --work-dir /kaggle/working/baseline --all-patients --train-config configs/unet_baseline.yaml --skip-audit-if-present --resume-run /kaggle/working/baseline/experiments/unet_baseline_<horodatage>
```

`/kaggle/working/` n'est pas persistant : publier `--work-dir` comme *Kaggle Dataset* en fin de
session et le remonter en input à la suivante, sinon manifeste, audit et imagettes sont à refaire.

### Étape 7 — Récupérer les résultats

Depuis le panneau *Output* : `smoke_run.zip` (répertoire de run avec `checkpoint_best.pt`,
`metrics.jsonl`, `config.yaml`, figures de superposition, plus `env.yaml`, la partition et l'index
d'imagettes) et `smoke_report.json`. Les PNG d'imagettes ne sont pas dans l'archive, et
`checkpoint_last.pt` non plus car il dépasse la limite de 200 Mo — il est signalé dans le rapport.

### Comparer plusieurs réglages

`kaggle_smoke.py` n'accepte pas `--set` : chaque réglage comparé a donc **son propre YAML**, copie de
`configs/unet_baseline.yaml` où ne change que le facteur étudié et le `run_name`. Pour garder les
mêmes partitions et les mêmes imagettes, réutiliser le même `--work-dir` avec
`--skip-audit-if-present` et la même `--seed` : seules la gate et l'entraînement sont relancés, dans
un nouveau répertoire de run. Récupérer `smoke_run.zip` et `smoke_report.json` après chaque run,
car le suivant les réécrit. Les règles de comparaison sont dans [docs/PROTOCOLE.md](docs/PROTOCOLE.md),
section « Comparaison de réglages ».

### Paramètres modifiables à chaque run

**Options de l'orchestrateur** (`scripts/kaggle_smoke.py`), à passer en ligne de commande :

| Option | Défaut | Effet |
|---|---|---|
| `--raw-dir` | requis | Racine des données brutes, en lecture seule |
| `--work-dir` | requis | Toutes les sorties ; doit être hors de `--raw-dir` |
| `--train-config` | `configs/unet_smoke.yaml` | Configuration pilotant la gate et l'entraînement |
| `--limit-patients` | `30` | Nombre de patientes par partition (run de fumée) |
| `--all-patients` | — | Toutes les patientes ; exclusif avec `--limit-patients` |
| `--size` | `512` | Côté des imagettes en pixels **(réextraction)** |
| `--n-folds` | `5` | Nombre de plis de la validation groupée **(réextraction)** |
| `--epochs` | config | Écrase le nombre d'époques de la configuration |
| `--encoder-weights` | config | `imagenet` ou `none` ; écrase la configuration |
| `--device` | `auto` | `auto`, `cuda` ou `cpu` |
| `--seed` | `0` | Graine propagée aux partitions, aux imagettes et à l'entraînement |
| `--overfit-n`, `--overfit-steps` | `8`, `300` | Taille et durée de la gate de surapprentissage |
| `--skip-audit-if-present` | — | Réutilise manifeste, audit, partitions et imagettes déjà présents |
| `--resume-run` | — | Reprend l'entraînement dans ce répertoire de run |

**(réextraction)** signale un paramètre qui change le jeu d'imagettes : la partition est gravée
dans `index.csv` à l'extraction, donc toute modification de `--size` ou `--n-folds` impose de
réextraire, et donc de ne pas utiliser `--skip-audit-if-present`.

**Champs de la configuration d'entraînement** ([configs/unet_baseline.yaml](configs/unet_baseline.yaml)),
modifiables dans le YAML ou à la volée avec `--set clé=valeur` sur `scripts/train.py` :

| Champ | Défaut | Rôle |
|---|---|---|
| `patch_root` | requis | Racine des imagettes (`index.csv`, `images/`, `masks/`) |
| `output_root` | `experiments` | Où créer le répertoire de run |
| `run_name` | `unet_smoke` | Préfixe du répertoire, horodaté automatiquement |
| `architecture` | `unet` | `unet`, `unetplusplus` ou `deeplabv3plus` |
| `encoder` | `resnet34` | Encodeur `segmentation-models-pytorch` (`resnet18`, `resnet50`, …) |
| `encoder_weights` | `imagenet` | `imagenet` ou `null` pour un encodeur depuis zéro |
| `epochs` | `2` | Horizon du run ; fixe aussi l'horizon du cosinus |
| `batch_size` | `8` | Taille de lot ; premier levier en cas de saturation VRAM |
| `accumulation_steps` | `1` | Lots accumulés avant un pas d'optimisation |
| `learning_rate` | `3.0e-4` | Taux initial d'AdamW |
| `weight_decay` | `1.0e-4` | Régularisation d'AdamW |
| `bce_weight` | `0.5` | Poids du terme BCE dans `BCE + Dice` |
| `scheduler` | `none` | `none`, `cosine` ou `plateau` |
| `min_learning_rate` | `1.0e-6` | Plancher du cosinus et du plateau |
| `plateau_factor` | `0.5` | Facteur de réduction du plateau |
| `plateau_patience` | `5` | Époques sans progrès avant réduction du plateau |
| `early_stopping_patience` | `null` | Époques sans progrès avant arrêt ; `null` désactive |
| `max_hours` | `null` | Budget de temps ; prime sur l'arrêt anticipé |
| `selection_metric` | `val_dice` | `val_dice` ou `val_dice_lesion` (voir ci-dessous) |
| `threshold` | `0.5` | Seuil appliqué aux probabilités pour le Dice |
| `amp` | `true` | Précision mixte ; sans effet hors CUDA |
| `augment` | `true` | Augmentations synchronisées image/masque à l'entraînement |
| `num_workers` | `2` | Processus de chargement ; `0` pour déboguer |
| `device` | `auto` | `auto`, `cuda` ou `cpu` |
| `seed` | `0` | Graine ; fixe l'ordre des lots par `(graine, époque)` |
| `limit_train`, `limit_val` | `null` | Tronque les partitions, pour un essai rapide |
| `resume_from` | `null` | Répertoire de run à reprendre |

Trois pièges à connaître :

- **En YAML, `3e-4` est une chaîne, pas un nombre** : un flottant exponentiel exige un point,
  `3.0e-4`. Le chargement de configuration refuse désormais ces valeurs avec un message explicite,
  mais la règle vaut aussi pour `--set learning_rate=3e-4`.
- **L'orchestrateur écrase `patch_root`, `output_root`, `device` et `seed`** de la configuration,
  puisqu'il en est la source ; les y modifier reste sans effet. Pour les piloter, passer par
  `scripts/train.py` directement.
- **`selection_metric`** choisit l'époque conservée. Le `val_dice` toutes imagettes confondues vaut
  1,0 sur une imagette de fond dès que le modèle ne prédit rien : il récompense l'abstention et
  n'est pas comparable au `mean_dice_lesion` de `predict.py`. `configs/unet_baseline.yaml`
  sélectionne donc sur `val_dice_lesion`. Les deux restent journalisés par époque dans
  `metrics.jsonl`, et `run.json` indique lequel a piloté la sélection, avec `completed_epochs` et
  `stop_reason`.

Une reprise refuse de changer `patch_root`, `architecture`, `encoder`, `batch_size` ou
`accumulation_steps` : ce ne serait plus la même expérience. Changer `epochs` est permis et étire
l'horizon du cosinus.

## Évaluation prévue

Dice et IoU pour le recouvrement ; HD95 et distance moyenne symétrique pour les contours ; sensibilité par lésion et faux positifs par image pour la détection. Les règles d’appariement, seuils et cas de masques vides seront fixés avant le test final. Une éventuelle classification sera évaluée séparément.

## Documents et sources

- [Contexte complet du projet](docs/CONTEXTE.md), à lire en premier par une nouvelle personne ou session.
- [Architecture](docs/ARCHITECTURE.md), [procédure Kaggle](docs/KAGGLE.md) et [références](docs/REFERENCES.md).
- [Carte française du projet](schemas/cbis_ddsm_project_map_fr.svg).
- [Carte française du cours](schemas/course_map_4_modules_fr.svg).
- `Evaluation.pdf` : consignes locales de l’enseignement.
- `sdata2017177.pdf` : article de description du dataset.

Les cours et schémas constituent des supports de travail à relire ; les nuances scientifiques précisées dans le protocole prévalent pour les futures expériences. Les documents tiers conservent leurs droits et conditions d’utilisation propres. Ce dépôt privé ne constitue pas un dispositif médical validé.
