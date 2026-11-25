# Breast Cancer — consignes de travail

Le projet combine recherche sur la segmentation CBIS-DDSM et apprentissage.
`PLAN.md` décrit les jalons ; `docs/PROTOCOLE.md` fixe l'évaluation et les frontières
scientifiques ; le journal `docs/superpowers/plans/2026-09-16-kaggle-smoke-pipeline.progress.md`
trace les livraisons, leurs commits et limites. Vérifier ces preuves avant d'affirmer
qu'une étape est terminée. Le README est un point d'entrée, pas un registre de scores.

En mode cours, expliquer le pipeline. En mode réalisation, implémenter le jalon
demandé. Revue indépendante pour splits, métriques ou conclusion scientifique.
Préserver les partitions par patiente, les données brutes, la géométrie image/masque,
les graines et l'environnement verrouillé. Le test réservé ne sert pas au tuning.
`uv run pytest` vérifie les cas synthétiques ; un run réel Kaggle est une preuve
distincte, avec sa configuration et sa capture d'environnement. Les comparaisons
sur patches ne constituent pas une validation de détection sur image entière.
Ne pas copier implicitement données brutes et `.venv` dans un worktree.

## Organisation locale

Répondre dans la langue de la demande ; code et messages de commit en anglais,
sauf convention propre au livrable. Lire seulement les documents utiles à la tâche.
Une modification bornée déjà demandée ne nécessite pas un nouveau cycle de
brainstorming. Conserver les validations scientifiques et autorisations coûteuses
spécifiques au projet.

Un agent principal par défaut. Ajouter une revue indépendante lorsque la décision
scientifique ou le risque sur les fichiers le justifie. Deux implémenteurs seulement
pour des tâches indépendantes, aux fichiers disjoints. Vérifier que les rôles nommés
sont disponibles ; sinon transmettre leur fiche comme mission à un agent générique,
sans prétendre avoir invoqué un rôle enregistré.

La configuration Codex locale propose Sol/medium pour le travail courant. Luna
convient aux tâches mécaniques bornées ; Astra/high aux décisions difficiles.
Ces choix sont des points de départ à comparer, pas des gains mesurés. Ne pas
changer de modèle pendant un run verrouillé ni ignorer un choix explicite de l'utilisateur.

Le dépôt et ses artefacts vérifiables font autorité. Les mémoires automatiques et
résumés de conversation sont des pointeurs, pas une seconde source d'avancement.
À la fin d'un jalon ou lors d'une reprise, mettre à jour le document d'état existant
avec date, branche/commit, preuves, limites et prochaine action. Séparer code
implémenté, tests synthétiques, exécution réelle et résultat scientifique validé.
Ne pas créer de résumé concurrent si un état ou journal existe déjà.

Utiliser les skills locaux `project-teacher`, `project-graph` et `project-graphify`
dans `.agents/skills/teacher`, `graph` et `graphify`. Les variantes globales restent
présentes pour StarThink ; leurs déclencheurs et procédures ne s'appliquent pas à
ce projet. Ces consignes locales remplacent les obligations globales de cours en
six parties, de dessin en anglais, de demande systématique de destination et de
Graphify avant toute question sur du code. Ne pas charger les anciens routeurs ni
le skill de mémoire globale pour ajouter des étapes à une tâche déjà autorisée.
Pour `/graphify`, lire ici `.agents/skills/graphify/SKILL.md` en priorité. `Cours:` demande un enseignement ;
une demande explicite de fichiers/correction demande une réalisation. Un `(graph)`
affirmatif ou une demande de schéma utilise la langue et le dossier déjà convenus.
`/graphify` reste disponible explicitement ; aucune reconstruction automatique.
Le renvoi global `@RTK.md` ne rend pas RTK obligatoire : ne le supposer ni installé
ni injecté par un hook. Utiliser les outils réellement disponibles.

Les hooks Codex hérités de routage et de mémoire sont désactivés dans ce périmètre
au profit de ces workflows explicites. Cela ne remplace ni les tests du projet ni
ses commandes de validation. Les réglages globaux et StarThink restent inchangés.
