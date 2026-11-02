# Exécuter le run de fumée sur Kaggle

Ce document décrit comment lancer `scripts/kaggle_smoke.py` dans un notebook Kaggle : une chaîne
complète (environnement → manifeste → audit → partitions → imagettes → gate de surapprentissage →
entraînement → inférence) sur un petit sous-ensemble, pour vérifier que le pipeline tourne de bout
en bout avant un entraînement complet. Le code reste en anglais ; ce document est en français.

## 1. Créer le notebook

1. Sur [kaggle.com](https://www.kaggle.com), créer un nouveau notebook.
2. Dans les paramètres du notebook (icône ⋮ → *Notebook options* / *Accelerator*), choisir
   **GPU T4 x1** (ou T4 x2, une seule suffit pour le run de fumée).
3. Activer **Internet** (*Internet: On*) — nécessaire pour installer `segmentation-models-pytorch`
   et, selon la méthode choisie, pour cloner le dépôt.

## 2. Ajouter le jeu de données

Dans le panneau *Add Input* du notebook, ajouter le jeu de données Kaggle
`awsaf49/cbis-ddsm-breast-cancer-image-dataset`. Il apparaît alors monté en lecture seule sous
`/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/`, avec les sous-dossiers `csv/` et `jpeg/`
attendus par `build_manifest.py`.

## 3. Récupérer le code du dépôt

Deux méthodes, au choix :

**a) `git clone` (dépôt public ou avec jeton) :**

```bash
git clone --branch <nom-de-branche> https://github.com/vivirtuose/Breast-Cancer-Project-M2.git
cd Breast-Cancer-Project-M2
```

Remplacer `<nom-de-branche>` par la branche à tester (ex. `s2-s6-kaggle-smoke`). Pour un dépôt
privé, utiliser un token d'accès personnel dans l'URL (`https://<token>@github.com/...`) plutôt que
de saisir un mot de passe.

**b) Dépôt Kaggle privé (sans accès réseau au code) :**

Compresser le dépôt localement (`git archive` ou zip du répertoire de travail, en excluant
`data/`, `.venv/`, `experiments/`) et le publier comme *Kaggle Dataset* privé. L'ajouter ensuite au
notebook comme second input ; le code apparaît sous `/kaggle/input/<nom-du-dataset>/`. Cette
méthode évite d'exposer un jeton GitHub dans le notebook.

## 4. Installer uniquement ce qui manque

L'image Kaggle fournit déjà PyTorch avec CUDA. **Ne pas exécuter `pip install -e .`** : le projet
épingle Python 3.12 et des roues `torch`/`torchvision` en `cu121` (`uv.lock`) qui ne correspondent
pas forcément à l'image Kaggle, et une réinstallation risquerait de casser l'accélération GPU déjà
en place. Installer seulement le paquet que Kaggle n'a pas :

```bash
pip install segmentation-models-pytorch==0.5.0
```

Si `scikit-learn` installé par défaut est plus ancien que `1.9.1` (utilisé par `make_splits.py`
pour la stratification), l'installer aussi : `pip install scikit-learn==1.9.1`. Vérifier au besoin
avec `python -c "import sklearn; print(sklearn.__version__)"` avant de réinstaller.

## 5. Lancer le run de fumée

Depuis la racine du dépôt cloné (ou copié) :

```bash
python scripts/kaggle_smoke.py \
  --raw-dir /kaggle/input/cbis-ddsm-breast-cancer-image-dataset \
  --work-dir /kaggle/working/smoke \
  --limit-patients 30
```

Options utiles :

- `--size 512` (défaut) : taille des imagettes ; réduire si la mémoire GPU est limitée.
- `--epochs`, `--overfit-steps 300` : `--epochs` non fourni laisse la configuration
  d'entraînement décider de son nombre d'époques ; le fournir l'écrase.
- `--encoder-weights imagenet|none` : non fourni, la configuration décide ; `imagenet` télécharge
  les poids préentraînés (Internet On requis) ; `none` entraîne l'encodeur from scratch, utile
  pour un run plus rapide ou hors ligne.
- `--train-config` (défaut `configs/unet_smoke.yaml`) : configuration pilotant la gate de
  surapprentissage et l'entraînement — voir la section 8 pour l'entraînement complet.
- `--all-patients` : extrait toutes les patientes des partitions demandées, au lieu de
  `--limit-patients` (les deux options sont mutuellement exclusives).
- `--resume-run <répertoire>` : reprend l'entraînement dans un répertoire de run existant.
- `--device auto|cuda|cpu` : `auto` (défaut) utilise le GPU T4 si disponible.
- `--n-folds` (défaut 5) : nombre de plis pour la validation croisée groupée par patiente.
- `--skip-audit-if-present` : relancer sans refaire le manifeste, l'audit, les partitions ou les
  imagettes déjà présents dans `--work-dir` (utile pour reprendre un run interrompu).

Le script journalise chaque étape dans `<work-dir>/logs/<étape>.log` et affiche un résumé d'une
ligne par étape sur la sortie standard.

## 6. Récupérer les résultats

À la fin du run, télécharger depuis le panneau *Output* du notebook (ou via l'API Kaggle) :

- `/kaggle/working/smoke/smoke_run.zip` : l'archive contenant le répertoire de run
  (`experiments/<run>/`, avec `checkpoint_best.pt`, `metrics.jsonl`, `config.yaml`, les figures de
  superposition), `smoke_report.json`, `env.yaml`, la partition `splits/split_v1.json` et l'index
  des imagettes `processed/patches_v1/index.csv` (les PNG d'imagettes eux-mêmes ne sont pas
  inclus, pour limiter la taille de l'archive).
- `/kaggle/working/smoke/smoke_report.json` : le même rapport, à part, pour une lecture rapide
  sans décompresser l'archive.

## 7. Définition du succès

Le run de fumée est un succès si :

- `scripts/kaggle_smoke.py` se termine avec un code de sortie **0** (les étapes 1 à 5, 7 et 8 ont
  réussi ; seule la gate de surapprentissage — étape 6 — peut échouer sans faire échouer le run).
- `smoke_report.json` liste les huit étapes (`capture_environment`, `build_manifest`, `audit`,
  `make_splits`, `extract_patches`, `overfit`, `train`, `predict`) avec leur statut, leur durée et
  un résumé de leurs chiffres clés (lignes de manifeste, exclusions, patients par partition,
  imagettes extraites, Dice de surapprentissage, Dice de validation).
- `smoke_run.zip` existe et contient bien le répertoire de run, `smoke_report.json`, `env.yaml`,
  la partition et l'index d'imagettes.
- Le Dice de validation rapporté par l'étape `predict` (dans
  `experiments/<run>/predictions/val_summary.json`, aussi lisible dans le `detail` de l'étape
  `predict`) est calculé sur au moins une imagette de validation — un `null` signale une partition
  de validation vide, à corriger avant d'augmenter `--limit-patients`.
- Une figure de superposition (`experiments/<run>/figures/*.png`) montre visuellement le contour
  de la lésion annotée superposé à une lésion réellement visible sur l'imagette, sans décalage
  flagrant entre l'image et le masque.

Ce run de fumée ne constitue pas une première référence : les nombres qu'il produit sont mesurés
sur un sous-ensemble restreint et un entraînement volontairement court, uniquement pour vérifier
que la chaîne fonctionne de bout en bout avant de lancer un entraînement complet sur Kaggle.

## 8. Entraînement complet et reprise après coupure de session

Une session Kaggle s'arrête au bout de quelques heures, et l'entraînement de référence est plus
long que cela. `configs/unet_baseline.yaml` est donc dimensionné pour être **interrompu puis
repris** : `max_hours` arrête le run avant que Kaggle ne le coupe, et chaque époque réécrit
`checkpoint_last.pt` (poids + optimiseur + scaler + scheduler) par écriture atomique.

Première session, sur toutes les patientes :

```bash
python scripts/kaggle_smoke.py --raw-dir /kaggle/input/cbis-ddsm-breast-cancer-image-dataset --work-dir /kaggle/working/baseline --all-patients --train-config configs/unet_baseline.yaml
```

L'étape `train` affiche à la fin le répertoire du run et, si elle s'est arrêtée tôt, la commande
de reprise. Sessions suivantes, en réutilisant le manifeste, l'audit, les partitions et les
imagettes déjà produits dans `--work-dir` :

```bash
python scripts/kaggle_smoke.py --raw-dir /kaggle/input/cbis-ddsm-breast-cancer-image-dataset --work-dir /kaggle/working/baseline --all-patients --train-config configs/unet_baseline.yaml --skip-audit-if-present --resume-run /kaggle/working/baseline/experiments/unet_baseline_<horodatage>
```

Pour conserver `--work-dir` d'une session à l'autre, le publier comme *Kaggle Dataset* en fin de
session et le remonter en input à la suivante ; `/kaggle/working/` n'est pas persistant.

Ce qu'une reprise garantit, et ce qu'elle ne garantit pas :

- **Garanti** : à graine et `epochs` identiques, un run coupé puis repris produit les mêmes
  enregistrements d'époque qu'un run continu — même calendrier de learning rate, mêmes pertes,
  mêmes Dice (vérifié par `tests/test_trainer.py::test_a_resumed_run_reproduces_an_uninterrupted_one`).
  L'ordre des lots dépend de `(graine, époque)` seulement, et le scheduler reprend au pas suivant.
- **Refusé** : une reprise qui change `patch_root`, `architecture`, `encoder`, `batch_size` ou
  `accumulation_steps` échoue avec un message nommant le champ ; ce ne serait plus la même
  expérience. Changer `epochs` est permis et étire l'horizon du cosinus.
- **Non garanti** : l'équivalence entre GPU différents, ou entre CPU et GPU.

### Choix de la meilleure époque

`configs/unet_baseline.yaml` sélectionne sur `selection_metric: val_dice_lesion`, le Dice calculé
**sur les seules imagettes de lésion**. Le `val_dice` toutes imagettes confondues vaut 1,0 sur une
imagette de fond dès que le modèle ne prédit rien : comme la moitié des imagettes sont du fond, il
récompense l'abstention et n'est pas comparable au `mean_dice_lesion` rapporté par `predict.py`.
Les deux restent journalisés par époque dans `metrics.jsonl`, et `run.json` indique lequel a
piloté la sélection ainsi que `completed_epochs` et `stop_reason`.

L'arrêt anticipé (`early_stopping_patience`) se déclenche sur cette même métrique. Un budget de
temps épuisé l'emporte sur l'arrêt anticipé : sur un runtime à session limitée, c'est lui qui
impose la fin du run.
