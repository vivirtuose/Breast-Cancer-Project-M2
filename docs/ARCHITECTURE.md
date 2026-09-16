# Architecture du projet

Flux prévu : `data/raw` → audit et manifeste → partitions par patiente → préparation → entraînement → inférence → évaluation → rapport.

## Environnements d’exécution

Le développement, les tests unitaires et l’audit initial des données peuvent être réalisés localement. Les entraînements et les inférences de référence sont exécutés sur Kaggle. Les scripts utilisent donc des chemins configurables et portables, sans chemin absolu propre à une machine ni hypothèse sur le GPU local.

Chaque exécution Kaggle conserve sa configuration, ses versions, les caractéristiques de l’accélérateur et les chemins des artefacts. La capture d’environnement est relancée dans le runtime Kaggle afin que les résultats puissent être reliés à l’environnement réellement utilisé.

## Responsabilités

| Emplacement | Responsabilité |
|---|---|
| `src/cbis_ddsm/data` | Lire les métadonnées, apparier les masques, préparer les tenseurs sans mélange entre partitions |
| `src/cbis_ddsm/models` | Construire les réseaux, sans logique de partitionnement |
| `src/cbis_ddsm/training` | Optimisation, sauvegarde et sélection sur validation |
| `src/cbis_ddsm/inference` | Prédire et replacer les sorties dans le repère de l’image |
| `src/cbis_ddsm/evaluation` | Calculer les métriques et agréger les résultats |
| `configs` | Décrire les paramètres des expériences |
| `scripts` | Exposer les opérations reproductibles une fois implémentées |
| `notebooks` | Explorer ; le traitement reproductible sera dans `src` |
| `tests` | Vérifier appariements, partitions, géométrie et métriques sur cas synthétiques |

## Traçabilité

Une expérience devra conserver le commit, la configuration, les graines, les versions logicielles, l’identifiant du manifeste et des partitions, les métriques et le chemin des poids. Les artefacts volumineux restent locaux ; les résultats sélectionnés et configurations sont versionnés.

Les données originales ne sont jamais modifiées. Toute transformation conserve assez d’information pour projeter le masque prédit dans le repère d’origine. Les chemins utilisés par le futur code seront configurables et indépendants de cette machine.
