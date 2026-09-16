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

## Reproductibilité logicielle

À chaque étape majeure, les décisions reproductibles — environnement, données, partitions, géométrie, entraînement, inférence et métriques — sont accompagnées de tests `pytest` ciblés. Avant de considérer un jalon terminé, un autre collaborateur doit pouvoir reconstruire un environnement propre à partir du verrou, exécuter les tests documentés et retrouver les mêmes invariants sans dépendre d’un état local implicite.

Les versions réellement installées sont comparées au verrou. Tout écart, dépendance manquante ou contrôle matériel non vérifié maintient le jalon bloqué. Les tests portables sur données synthétiques sont toujours exécutés ; les contrôles nécessitant le GPU ou les données réelles sont marqués séparément et apportent une preuve complémentaire.
