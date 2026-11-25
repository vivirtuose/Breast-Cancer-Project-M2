# Plan de travail

## Règle transversale — tests et reproductibilité

Chaque étape majeure ajoute des tests `pytest` ciblés avant de franchir son critère de passage. Les tests vérifient le comportement attendu et la reproductibilité à partir de petites données synthétiques versionnées ; ils ne dépendent pas des données brutes locales. Les tests qui exigent le GPU ou le jeu de données réel sont marqués séparément et complètent, sans remplacer, la suite portable.

Un jalon n'est validé que si un partenaire peut recréer l'environnement depuis le verrou, exécuter la suite complète avec `uv run pytest`, puis obtenir les mêmes résultats déterministes pour les cas synthétiques avec les graines et configurations fixées. Toute modification d'une dépendance met à jour le verrou et relance la suite complète.

| Étape | Preuve `pytest` minimale au passage du jalon |
|---|---|
| 0 — Environnement | Cohérence de `pyproject.toml` et `uv.lock`, versions installées conformes au verrou, schéma de `env.yaml`, détection explicite des blocages Python/GPU/CUDA. |
| 1 — Acquisition | Résolution des chemins, appariement image/masque, détection du cas connu d'inversion ROI/crop, schéma et déterminisme du manifeste. |
| 2 — Audit et préparation | Absence de fuite entre patientes, exclusions traçables, fusion des lésions, transformations image/masque synchronisées et déterministes sous graine fixée. |
| 3 — Référence | Formes et types des tenseurs, pertes et métriques sur cas synthétiques, surapprentissage d'un petit lot, sauvegarde/rechargement et inférence reproductibles. |
| 4 — Comparaisons | Réutilisation stricte des partitions, isolation du facteur comparé, répétition déterministe d'un essai court et compatibilité des sorties entre modèles. |
| 5 — Évaluation finale | Gel du modèle et des seuils, cas limites des masques vides et lésions multiples, agrégations groupées par patiente et bootstrap reproductible. |
| 6 — Restitution | Commandes de reproduction exécutables, cohérence entre configurations, métriques et figures retenues, présence des métadonnées de traçabilité. |

## 0 — Cadrage et préparation

- [x] Rassembler les cours, schémas et ressources.
- [x] Définir l’arborescence et un protocole initial.
- [x] Relever le matériel disponible : GPU, mémoire et espace disque.
- [x] Fixer les versions de Python et des dépendances, puis verrouiller l’environnement.

## 1 — Acquisition et audit

- [ ] Consigner la version, la provenance et les conditions d’utilisation des données.
- [x] Inventorier les CSV et les images réellement présents.
- [x] Construire un manifeste : patiente, côté, vue, anomalie, pathologie, image, masque.
- [x] Vérifier chemins, doublons, dimensions, orientation, valeurs et superposition des masques.
- [ ] Examiner visuellement un échantillon stratifié et consigner les exclusions.

**Critère de passage :** chaque paire retenue est traçable ; erreurs et exclusions sont documentées. Ne pas confondre nombre de fichiers, de lésions, de vues et de patientes.

## 2 — Protocole et préparation

- [ ] Commencer par la segmentation des masses, bénignes et malignes, et analyser les malignes séparément.
- [x] Auditer les partitions officielles pour les chevauchements de patientes.
- [x] Créer une validation groupée par patiente à partir de l’entraînement.
- [ ] En cas de chevauchement officiel, décider et documenter une correction avant les expériences.
- [x] Fixer résolution, normalisation, stratégie d’imagettes et transformations synchronisées.
- [x] Définir les règles d’évaluation, notamment masques vides et lésions multiples.

**Critère de passage :** aucun chevauchement de patiente ; transformation image/masque vérifiée ; test réservé.

## 3 — Première référence reproductible

- [x] Implémenter le chargement et une architecture U-Net 2D.
- [x] Exécuter les entraînements et les inférences sur Kaggle avec des chemins configurables et une capture de l’environnement du runtime. *(Run de fumée du 2026-11-24, 30 patientes par partition : 8/8 étapes, gate 0,9975 ; voir le journal.)*
- [x] Vérifier qu’un petit lot peut être surappris pour détecter les erreurs du pipeline.
- [ ] Vérifier la sortie du run lancé le 2026-11-25 (`smoke_report.json`, `val_summary.json`, superpositions) et la consigner dans le journal.
- [ ] Entraîner une première référence avec BCE + Dice, configuration et graine enregistrées (`configs/unet_baseline.yaml`, toutes les patientes).
- [ ] Calculer la référence « disque centré » sur les mêmes imagettes de validation, pour mesurer la part du Dice due au centrage des imagettes.
- [x] Produire des superpositions image / référence / prédiction et un rapport de validation.

**Critère de passage :** entraînement et inférence reproductibles, absence d’erreur d’alignement, résultat de référence mesuré.

## 4 — Comparaisons contrôlées

Feuille de route retenue le 2026-11-25, dans l’ordre :

- [ ] Choisir et consigner les trois réglages comparés : un facteur changé à la fois par rapport à la référence, un YAML par réglage avec son propre `run_name`.
- [ ] Lancer les trois runs sur les mêmes partitions, les mêmes imagettes et la même graine (`--skip-audit-if-present` sur le même `--work-dir`).
- [ ] Produire les statistiques de comparaison sur la validation, selon `docs/PROTOCOLE.md`, « Comparaison de réglages » : Dice et IoU lésion, taux de faux positifs sur le fond, intervalles et différences appariées par rééchantillonnage des patientes, écart à la référence « disque centré ».
- [ ] Conclure sur le modèle retenu, sur la validation uniquement, avec les limites (une graine par réglage, imagettes centrées).
- [ ] Comparer un encodeur préentraîné et, si les ressources le permettent, nnU-Net.
- [ ] Étudier résolution, imagettes et pertes sensibles aux frontières, un facteur à la fois.
- [ ] Répéter les expériences importantes avec plusieurs graines.
- [ ] Comparer la qualité, le coût mémoire et le temps d’inférence.

**Critère de passage :** mêmes partitions et règles pour tous les modèles ; choix final réalisé sur validation uniquement.

## 5 — Évaluation finale

- [ ] Figer le modèle, les seuils et le post-traitement.
- [ ] Évaluer une fois sur le test réservé.
- [ ] Rapporter recouvrement, frontières et erreurs de détection selon le protocole.
- [ ] Analyser par densité, taille, pathologie et vue ; indiquer les effectifs.
- [ ] Estimer l’incertitude par rééchantillonnage groupé par patiente.

**Critère de passage :** résultats accompagnés d’effectifs, d’incertitudes et d’exemples d’échec ; aucune revendication de performance clinique générale.

## 6 — Rapport et restitution

- [ ] Rédiger méthodes, résultats, limites et comparaison bibliographique selon `Evaluation.pdf`.
- [ ] Documenter les commandes de reproduction et les versions des expériences.
- [ ] Préparer figures et présentation.

## Extensions après la référence

Calcifications, fusion CC/MLO, classification associée, segmentation des seules lésions malignes et validation externe. Chacune nécessite son propre protocole ; aucune n’est requise pour la première référence.
