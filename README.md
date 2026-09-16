# Breast Cancer Project M2

Projet de recherche et d’apprentissage visant à segmenter les lésions visibles sur les mammographies de CBIS-DDSM avec des modèles de deep learning. Première étape : les masses ; extension éventuelle aux calcifications après audit des annotations.

## Objectif et périmètre

Produire un masque de lésion à partir d’une mammographie et évaluer sa qualité géométrique. La segmentation localise une anomalie ; elle ne détermine pas, à elle seule, sa malignité. Le projet pourra ensuite étudier la classification bénin/malin ou une segmentation spécifique des lésions malignes, avec un protocole explicite.

**État actuel :** cours, schémas, structure et plan disponibles. Le jalon S0 fournit le verrou des dépendances, la capture reproductible de l’environnement et ses tests ; aucun pipeline de données ni modèle entraîné n’est encore livré.

## Organisation

```text
cours/                    Quatre modules pédagogiques et questions
schemas/                  Cartes SVG du projet et du cours, FR et EN
docs/                     Architecture, protocole et références
configs/                  Futures configurations des expériences
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
scripts/                  Futurs points d’entrée du pipeline
tests/                    Contrôles des données et du pipeline
experiments/              Journaux, configurations et poids locaux
reports/figures/           Rapports et figures sélectionnées
```

Les dossiers du pipeline restent des emplacements réservés. L’environnement S0 cible Python 3.12 et ses versions sont fixées dans `pyproject.toml` et `uv.lock`.

## Démarrage

1. Lire le [plan](PLAN.md) et le [protocole](docs/PROTOCOLE.md).
2. Consulter le [cours](cours/00_index.md).
3. Recréer l’environnement avec `uv sync --frozen`, puis exécuter `uv run pytest`.
4. Produire la photographie de la machine avec `uv run python scripts/capture_environment.py` ; un code de sortie non nul signifie que le gate S0 reste bloqué.
5. Télécharger les données depuis [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset), puis les placer dans `data/raw/`.
6. Auditer les CSV, les chemins, les masques et les identités patient avant tout entraînement.

Les données, secrets, poids et sorties volumineuses restent hors de Git. Les petits résultats retenus pour le rapport peuvent être ajoutés dans `reports/`.

## Évaluation prévue

Dice et IoU pour le recouvrement ; HD95 et distance moyenne symétrique pour les contours ; sensibilité par lésion et faux positifs par image pour la détection. Les règles d’appariement, seuils et cas de masques vides seront fixés avant le test final. Une éventuelle classification sera évaluée séparément.

## Documents et sources

- [Architecture](docs/ARCHITECTURE.md) et [références](docs/REFERENCES.md).
- [Carte française du projet](schemas/cbis_ddsm_project_map_fr.svg).
- [Carte française du cours](schemas/course_map_4_modules_fr.svg).
- `Evaluation.pdf` : consignes locales de l’enseignement.
- `sdata2017177.pdf` : article de description du dataset.

Les cours et schémas constituent des supports de travail à relire ; les nuances scientifiques précisées dans le protocole prévalent pour les futures expériences. Les documents tiers conservent leurs droits et conditions d’utilisation propres. Ce dépôt privé ne constitue pas un dispositif médical validé.
