# Plan de travail

## 0 — Cadrage et préparation

- [x] Rassembler les cours, schémas et ressources.
- [x] Définir l’arborescence et un protocole initial.
- [ ] Relever le matériel disponible : GPU, mémoire et espace disque.
- [ ] Fixer les versions de Python et des dépendances, puis verrouiller l’environnement.

## 1 — Acquisition et audit

- [ ] Consigner la version, la provenance et les conditions d’utilisation des données.
- [ ] Inventorier les CSV et les images réellement présents.
- [ ] Construire un manifeste : patiente, côté, vue, anomalie, pathologie, image, masque.
- [ ] Vérifier chemins, doublons, dimensions, orientation, valeurs et superposition des masques.
- [ ] Examiner visuellement un échantillon stratifié et consigner les exclusions.

**Critère de passage :** chaque paire retenue est traçable ; erreurs et exclusions sont documentées. Ne pas confondre nombre de fichiers, de lésions, de vues et de patientes.

## 2 — Protocole et préparation

- [ ] Commencer par la segmentation des masses, bénignes et malignes, et analyser les malignes séparément.
- [ ] Auditer les partitions officielles pour les chevauchements de patientes.
- [ ] Créer une validation groupée par patiente à partir de l’entraînement.
- [ ] En cas de chevauchement officiel, décider et documenter une correction avant les expériences.
- [ ] Fixer résolution, normalisation, stratégie d’imagettes et transformations synchronisées.
- [ ] Définir les règles d’évaluation, notamment masques vides et lésions multiples.

**Critère de passage :** aucun chevauchement de patiente ; transformation image/masque vérifiée ; test réservé.

## 3 — Première référence reproductible

- [ ] Implémenter le chargement et une architecture U-Net 2D.
- [ ] Vérifier qu’un petit lot peut être surappris pour détecter les erreurs du pipeline.
- [ ] Entraîner une première référence avec BCE + Dice, configuration et graine enregistrées.
- [ ] Produire des superpositions image / référence / prédiction et un rapport de validation.

**Critère de passage :** entraînement et inférence reproductibles, absence d’erreur d’alignement, résultat de référence mesuré.

## 4 — Comparaisons contrôlées

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
