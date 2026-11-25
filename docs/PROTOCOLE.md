# Protocole initial et précautions d’interprétation

## Cible

Première référence : segmenter les masses annotées, puis présenter une analyse spécifique des lésions malignes. Un masque de lésion visible n’est pas une carte histologique exacte du tissu cancéreux. Écarter les lésions bénignes de la cible serait une autre tâche, à définir séparément.

## Annotation et données

Auditer le miroir Kaggle avant de conclure sur les formats, la profondeur de bits ou la précision des masques. Les contours fournis sont une référence imparfaite. Le Dice publié entre une méthode de curation et un expert n’est pas un plafond mathématique pour un modèle entraîné sur ces annotations. Un score supérieur ne prouve pas à lui seul un surapprentissage.

Les quatre exemples de densité montrés dans la figure de l’article ne sont pas quatre moyennes de population. La morphologie des marges est informative, mais ne constitue pas à elle seule un diagnostic. Les supports pédagogiques existants contiennent des formulations simplifiées sur ces points.

## Séparation des données

Regrouper toutes les vues, côtés et anomalies d’une même patiente. Vérifier les partitions officielles plutôt que supposer leur indépendance. Les adaptations éventuelles sont documentées ; ne pas revendiquer une comparaison directe avec des résultats utilisant une autre partition. Aucun réglage sur le test.

## Entrées

La première référence reçoit uniquement l’image. Les descripteurs et la pathologie servent à définir la cible et à analyser les résultats. Un modèle utilisant des observations radiologiques en entrée serait un système assisté distinct. Ne pas fournir de recadrage issu du masque réel à une évaluation présentée comme détection autonome sur image complète.

## Métriques

- Dice et IoU : recouvrement des surfaces, avec agrégation explicitée.
- HD95 et distance moyenne symétrique : écart des frontières. Unités physiques seulement si l’espacement est fiable ; sinon pixels et résolution précisés.
- Sensibilité par lésion et faux positifs par image : fixer l’appariement des composantes, son seuil, et le traitement des correspondances multiples ; courbe FROC si des scores sont disponibles.
- Masques vides : distinguer vrais négatifs, lésions manquées et fausses détections ; ne pas attribuer arbitrairement une distance finie aux frontières absentes.
- Incertitude : rééchantillonner les patientes, pas les pixels indépendamment.

Les métriques de frontière ne garantissent pas de mesurer chaque spicule ; HD95 peut ignorer des erreurs très localisées. Dice/BCE ne récompensent pas systématiquement le lissage : cela dépend des références et des prédictions. La qualité des détails se vérifie expérimentalement et visuellement, pas à partir du seul nom de l’architecture.

Une classification future utilisera ses propres métriques et unités d’analyse. La valeur prédictive positive dépend de la prévalence : les performances d’un ensemble enrichi en lésions ne se transposent pas directement au dépistage.

## Comparaison de réglages

Règles fixées le 2026-11-25, avant les premiers runs de comparaison, pour que le choix du modèle ne dépende pas des résultats observés.

- **Même expérience sauf un facteur.** Tous les runs comparés utilisent la même partition (`split_v1.json`, même empreinte SHA-256 du manifeste), le même index d’imagettes, la même graine et le même seuil τ = 0,5. Chaque run diffère de la référence par un seul facteur, consigné dans son YAML. Les trois réglages et la métrique principale sont choisis et consignés avant de voir les résultats.
- **Validation seulement.** Comparaison, choix de l’époque et choix du réglage se font sur la validation. Le test reste scellé jusqu’au gel du modèle (étape 5).
- **Métrique principale :** Dice lésion moyen sur les imagettes de lésion de validation (`mean_dice_lesion` de `predict.py`, égal au `val_dice_lesion` de la meilleure époque). Secondaires : IoU lésion, taux de faux positifs sur les imagettes de fond, temps par époque et mémoire GPU.
- **Incertitude.** Intervalle à 95 % par bootstrap des patientes : tirer les patientes de validation avec remise, recalculer la moyenne sur toutes leurs imagettes, au moins 1 000 tirages, graine fixée. Pour comparer deux runs, calculer la **différence appariée** sur le même tirage de patientes, puisque les deux runs voient les mêmes imagettes.
- **Référence obligatoire.** La référence « disque centré » figure dans le même tableau. Le chiffre défendu est l’écart d’un run à cette référence, pas son Dice brut.
- **Portée de la conclusion.** Avec une seule graine par réglage, une différence appariée dont l’intervalle contient 0 n’est pas une différence. Même un intervalle qui exclut 0 ne sépare pas l’effet du réglage de la variabilité due à la graine : la répéter avec deux autres graines si le budget Kaggle le permet, sinon l’énoncer comme limite. Les résultats restent des mesures sur imagettes centrées sur la lésion, pas une détection sur mammographie entière.

## Reproductibilité logicielle

À chaque étape majeure, les décisions reproductibles — environnement, données, partitions, géométrie, entraînement, inférence et métriques — sont accompagnées de tests `pytest` ciblés. Avant de considérer un jalon terminé, un autre collaborateur doit pouvoir reconstruire un environnement propre à partir du verrou, exécuter les tests documentés et retrouver les mêmes invariants sans dépendre d’un état local implicite.

Les versions réellement installées sont comparées au verrou. Tout écart, dépendance manquante ou contrôle matériel non vérifié maintient le jalon bloqué. Les tests portables sur données synthétiques sont toujours exécutés ; les contrôles nécessitant le GPU ou les données réelles sont marqués séparément et apportent une preuve complémentaire.
