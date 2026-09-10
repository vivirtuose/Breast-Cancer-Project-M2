# Cours — Segmentation de lésions mammaires sur CBIS-DDSM

Quatre modules. Chacun suit la même structure : **Ancrage → Intuition → Mécanisme → Sur tes données → Là où ça casse → Contrôle**.

Les questions de contrôle sont ouvertes et sans corrigé dans le fichier. Réponds-y, je corrige — en disant d'abord ce qui est faux.

| Module | Sujet | Statut |
|---|---|---|
| [01](01_maladie_et_mammographie.md) | La maladie et sa détection en mammographie | fait, corrigé le 2026-09-09 |
| [02](02_la_donnee.md) | La donnée : du film scanné au tenseur | à faire |
| [03](03_les_metriques.md) | Les métriques : segmentation et radiologie | à faire |
| [04](04_les_modeles.md) | Le paysage des modèles | à faire |

## Étiquetage épistémique

Utilisé dans tout le cours :

- **[établi]** — définition ou résultat standard, vérifiable.
- **[consensus]** — pratique dominante du domaine, pas un théorème.
- **[inférence]** — mon raisonnement, pas une source.
- **[incertain]** — je ne sais pas, et je le dis plutôt que de combler.

## Le fil rouge

Un seul énoncé traverse les quatre modules, décliné à chaque niveau :

> **La marge de la lésion est une propriété de forme locale, et c'est elle qui porte le diagnostic.**

- Module 1 — pourquoi, cliniquement : un contour spiculé signe l'infiltration.
- Module 2 — ce qui la détruit dans la donnée : redimensionnement, JPEG 8 bits, fenêtrage.
- Module 3 — pourquoi le Dice ne la voit pas, et quelle métrique la voit.
- Module 4 — quelles architectures la préservent, et lesquelles la lissent.

Si tu ne devais retenir qu'une chose de ces quatre modules, c'est celle-là.

## Vocabulaire à ne jamais confondre

| | Densité mammaire | Marge |
|---|---|---|
| Porte sur | le sein entier (le fond) | la lésion (local) |
| Échelle | globale, une valeur par image | locale, morphologique |
| Codage | BI-RADS densité 1–4 | circonscrite → spiculée |
| Rôle | difficulté du cas | **signe diagnostique** |

Et : la mammographie est une **projection**, jamais une coupe. Il n'y a pas de reconstruction 3D dans CBIS-DDSM. La modalité qui produit des coupes est la tomosynthèse (DBT), absente de ce jeu de données.

## Sources du cours

- Lee R.S., Gimenez F., Hoogi A., Miyake K.K., Gorovoy M., Rubin D.L., « A curated mammography data set for use in computer-aided detection and diagnosis research », *Scientific Data* 4:170177 (2017). DOI 10.1038/sdata.2017.177 — copie locale : `../sdata2017177.pdf`.
- Fiche de collection TCIA pour CBIS-DDSM.
- Shen L., Margolies L.R., Rothstein J.H., Fluder E., McBride R., Sieh W., « Deep Learning to Improve Breast Cancer Detection on Screening Mammography », *Scientific Reports* 9:12495 (2019). DOI 10.1038/s41598-019-48995-4.
- Grille d'évaluation MLB : `../Evaluation.pdf`.

Schémas associés : `../schemas/cbis_ddsm_project_map_fr.svg`.
