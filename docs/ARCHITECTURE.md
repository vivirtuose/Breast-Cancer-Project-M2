# Architecture du projet

Flux prévu : `data/raw` → audit et manifeste → partitions par patiente → préparation → entraînement → inférence → évaluation → rapport.

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
