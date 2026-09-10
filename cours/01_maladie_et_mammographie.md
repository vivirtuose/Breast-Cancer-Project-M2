# Module 1 — La maladie et sa détection en mammographie

*Niveau : praticien. Séance du 2026-09-08, corrigée le 2026-09-09.*

---

## 1. Ancrage

Le dépistage cherche à repérer un cancer **avant** qu'il ne se voie ou se palpe, sur une image où la tumeur a presque exactement la même apparence que le tissu sain qui l'entoure.

---

## 2. Intuition

Une mammographie est une radiographie du sein comprimé. Deux tissus dominent : la **graisse**, qui laisse passer les rayons X et apparaît sombre, et le **tissu fibroglandulaire** (les glandes et leur charpente), qui les arrête et apparaît clair. Une tumeur est faite de cellules densément tassées : elle arrête les rayons X, donc elle apparaît **claire elle aussi**.

Chercher un cancer sur une mammographie, c'est donc chercher du blanc dans du blanc. L'analogie qui tient : **un ours polaire dans une tempête de neige**. Et la « quantité de neige » a un nom clinique — la **densité mammaire**. Sur un sein graisseux, l'ours se détache sur fond noir. Sur un sein dense, il disparaît.

C'est pour cette raison que la densité n'est pas une variable parmi d'autres dans ce projet : c'est *la* covariable qui gouverne la difficulté, pour le radiologue comme pour ton réseau.

---

## 3. Le mécanisme

### 3.1 La physique

L'atténuation d'un faisceau de rayons X suit la loi de Beer-Lambert [établi] :

$$I = I_0\,e^{-\mu x}$$

où $I_0$ est l'intensité incidente, $x$ l'épaisseur de tissu traversée, et $\mu$ le coefficient d'atténuation linéique, propre au tissu et dépendant de l'énergie du faisceau.

Le contraste entre graisse et tissu glandulaire vient d'un écart de $\mu$ minuscule. D'où le choix d'une énergie très basse en mammographie, typiquement **25–35 kVp** [établi], là où une radiographie du thorax tourne autour de 120 kVp. Basse énergie = plus de contraste entre tissus mous, mais aussi plus de dose absorbée par un organe radiosensible : tout le compromis de la mammographie tient dans cet arbitrage.

### 3.2 La compression

Le sein est écrasé entre deux plaques. Ce n'est pas gratuit :

- réduit l'épaisseur traversée, donc la dose ;
- limite le **rayonnement diffusé**, qui voile l'image et détruit le contraste ;
- **étale les tissus**, ce qui réduit les superpositions.

### 3.3 Les deux incidences — et pourquoi ce ne sont pas des coupes

Un sein est un volume ; une mammographie est une **projection** de tout ce volume sur un plan. Il n'y a **aucune reconstruction 3D, aucune tranche**. C'est le point conceptuel le plus important de cette section.

Conséquence : deux structures normales situées à des profondeurs différentes peuvent se superposer sur la projection et produire une tache claire qui ressemble à une masse. C'est un **artefact de sommation**.

D'où l'acquisition systématique de deux vues sous des angles différents :

- **CC** — craniocaudale, de haut en bas ;
- **MLO** — médio-latérale oblique, en diagonale, incluant le muscle pectoral.

Une anomalie retrouvée sur les deux vues, à des positions géométriquement cohérentes, est un objet réel. Une anomalie présente sur une seule vue est suspecte d'être un artefact [établi].

> C'est la justification **médicale** de la fusion CC + MLO listée comme décision ouverte sur le schéma du projet. Ce n'est pas une astuce d'ingénierie ML : c'est la façon dont un radiologue raisonne. Et les deux vues ne s'**additionnent** pas comme des coupes adjacentes — elles se **corroborent**. Corroboration, pas accumulation.

La modalité qui produit réellement des coupes existe — la **tomosynthèse** (DBT) — mais elle est postérieure au DDSM et absente de CBIS-DDSM.

### 3.4 Ce qu'on cherche : deux signes radicalement différents

C'est la raison pour laquelle CBIS-DDSM est livré en **deux jeux séparés**, avec des CSV distincts.

#### (a) Les masses

Une lésion qui occupe de l'espace, de l'ordre du centimètre. Le radiologue la décrit par :

- sa **forme** : ronde, ovale, lobulée, irrégulière ;
- ses **contours** (la « marge ») : circonscrits, microlobulés, masqués, indistincts, **spiculés**.

**La marge porte l'essentiel du diagnostic** [établi] :

| Marge | Interprétation |
|---|---|
| Circonscrite, nette, régulière | lésion qui *repousse* les tissus → probablement bénigne |
| Indistincte, masquée | indéterminée |
| **Spiculée** — épines rayonnantes | *infiltration* du tissu de soutien → hautement évocatrice de malignité |

**Conséquence directe pour ton projet : la frontière du masque est diagnostique, pas décorative.** Une métrique qui ne mesure qu'un recouvrement de surface est aveugle à exactement le signe qui compte. Le module 3 démonte ce point.

#### (b) Les microcalcifications

Des dépôts de calcium de 0,1 à 1 mm. Le calcium a un numéro atomique élevé : elles apparaissent **très** blanches et très contrastées — paradoxalement plus faciles à voir qu'une masse, mais minuscules.

On les décrit par :

- leur **type** : amorphes, grossières hétérogènes, pléomorphes fines, linéaires fines ramifiées… ;
- leur **distribution** : diffuse, régionale, groupée, linéaire, segmentaire.

Des microcalcifications **pléomorphes fines en distribution segmentaire ou linéaire** sont la signature du **CCIS** — carcinome canalaire *in situ*, un cancer au stade le plus précoce — et en sont souvent le **seul** signe visible [établi]. C'est précisément ce que le dépistage vise à attraper.

> Ordre de grandeur à garder en tête : une masse ≈ 2 cm, une microcalcification ≈ 0,5 mm. **Un facteur 40.** Ce n'est pas la même tâche de vision.

### 3.5 Le vocabulaire commun : BI-RADS

Un lexique standardisé du Collège américain de radiologie. **Attention : le même sigle recouvre deux choses différentes dans tes CSV** [établi].

**(a) La catégorie de densité, 1 à 4 :**

1. presque entièrement graisseux
2. densités fibroglandulaires éparses
3. dense hétérogène
4. extrêmement dense

**(b) L'évaluation finale, 0 à 5 :**

| Code | Signification |
|---|---|
| 0 | incomplet — bilan complémentaire nécessaire, on rappelle la patiente |
| 1 | normal |
| 2 | bénin |
| 3 | probablement bénin — < 2 % de malignité, surveillance à 6 mois |
| 4 | suspect — biopsie indiquée |
| 5 | hautement évocateur de malignité — ≥ 95 % |

### 3.6 La vérité ne vient pas du radiologue

Le label bénin / malin de CBIS-DDSM est **anatomopathologique** : il vient de l'analyse au microscope du tissu prélevé par biopsie. C'est ce qui fait la valeur scientifique de ce jeu de données.

Le BI-RADS, lui, n'est qu'une **opinion** — expertisée, mais une opinion. Retiens cette asymétrie, elle décide de ce qui peut servir d'entrée à un modèle et de ce qui ne le peut pas.

---

## 4. Sur tes données

Voici comment ce vocabulaire atterrit dans les fichiers que tu vas charger.

Les **noms de colonnes** sont ceux documentés par Lee et al. 2017 [établi]. Les **vocabulaires de valeurs** sont le lexique BI-RADS et restent **à vérifier sur le CSV réel**, que tu n'as pas encore téléchargé [incertain sur les chaînes exactes].

| Colonne | Ce que c'est vraiment | Section ci-dessus |
|---|---|---|
| `breast_density` | 1–4 | l'épaisseur de la tempête de neige (3.5a) |
| `left or right breast` | latéralité | — |
| `image view` | CC / MLO | les deux projections (3.3) |
| `abnormality id` | index dans l'image | un cas peut porter plusieurs anomalies |
| `mass shape` | ronde, ovale, irrégulière… | forme de la lésion (3.4a) |
| `mass margins` | circonscrits → spiculés | **le signe diagnostique clé** (3.4a) |
| `calc type` | amorphes, pléomorphes… | morphologie (3.4b) |
| `calc distribution` | groupée, segmentaire… | signature du CCIS (3.4b) |
| `assessment` | BI-RADS 0–5 | l'**opinion** du radiologue (3.5b) |
| `pathology` | BENIGN / BENIGN_WITHOUT_CALLBACK / MALIGNANT | **la vérité, par biopsie** (3.6) |
| `subtlety` | 1–5 | difficulté cotée à la main |

Quatre fichiers :

```
mass_case_description_train_set.csv
mass_case_description_test_set.csv
calc_case_description_train_set.csv
calc_case_description_test_set.csv
```

La séparation masse / calcification est **structurelle**, pas cosmétique — c'est la section 3.4 matérialisée en fichiers.

### Effectifs

| | Bénin | Malin |
|---|---|---|
| Calcif. — train | 329 cas (552 anomalies) | 273 cas (304 anomalies) |
| Calcif. — test | 85 cas (112 anomalies) | 66 cas (77 anomalies) |
| Masses — train | 355 cas (387 anomalies) | 336 cas (361 anomalies) |
| Masses — test | 117 cas (135 anomalies) | 83 cas (87 anomalies) |

*Source : Table 3 de Lee et al. 2017 [établi].* Un « cas » = une anomalie vue sur CC et/ou MLO. Les deux colonnes diffèrent parce que certains cas portent plusieurs anomalies.

---

## 5. Là où ça casse

**1. Les deux BI-RADS.** `breast_density` (1–4) et `assessment` (0–5) sont deux colonnes sans rapport. Les confondre, ou empiler « BI-RADS » comme une seule feature, est l'erreur d'entrée classique.

**2. `BENIGN_WITHOUT_CALLBACK` n'est pas `BENIGN`.** C'est « bénin, et on n'a même pas jugé utile de rappeler la patiente » : les cas les plus faciles du jeu. Les fusionner avec `BENIGN` pour binariser est légitime [consensus], mais **change la difficulté de ton problème**. Tes chiffres ne sont comparables à un article que si tu déclares ce choix explicitement.

**3. Utiliser `assessment` comme feature d'entrée est une fuite.** Le BI-RADS est déjà le jugement d'un expert humain sur cette image. Un modèle qui le consomme n'apprend pas à voir un cancer : il apprend à recopier un radiologue, et affiche un score flatteur qui ne veut rien dire [inférence].

Pire dans ce jeu précis : les auteurs notent eux-mêmes qu'il est inhabituel d'attribuer un BI-RADS 4 ou 5 à une image de **dépistage**, et que ces évaluations ont probablement été mises à jour **après** le bilan complémentaire [établi, Lee et al. 2017]. C'est-à-dire : après qu'on savait. Le BI-RADS est un **stratificateur pour l'analyse**, jamais une entrée.

**4. Segmenter des masses et des microcalcifications n'est pas la même tâche.** Facteur 40 en taille. À la résolution où un U-Net voit confortablement une masse, une microcalcification tient sur moins d'un pixel. Un modèle unique à une seule échelle ne peut pas faire les deux [inférence]. C'est une raison technique — en plus de la qualité des masques — de commencer par les masses.

**5. Densité élevée = double peine.** Le tissu dense **masque** les lésions *et* constitue un **facteur de risque** de cancer [établi]. Un modèle qui sous-performe sur les seins denses échoue donc précisément sur la population la plus à risque. C'est le résultat le plus intéressant à montrer à l'oral — et celui que la moyenne globale écrase.

---

## 6. Contrôle

**a)** Un modèle atteint 0,86 de Dice sur les seins de densité 1 et 0,61 sur ceux de densité 4. Un camarade propose de rééquilibrer l'entraînement en suréchantillonnant les seins denses. Qu'est-ce que cette solution suppose implicitement sur la cause du problème — et quelle autre cause, vue dans ce module, rendrait ce remède inefficace ?

**b)** Pourquoi le fait que les contours spiculés soient le signe de malignité le plus fort a-t-il une conséquence directe sur le **choix des métriques**, et pas seulement sur le choix du modèle ?

**c)** On te propose d'entraîner un classifieur bénin/malin en donnant en entrée l'image **et** les colonnes `mass shape` et `mass margins`. Est-ce la même faute que d'utiliser `assessment` ? Justifie — la réponse n'est pas évidente dans les deux sens.

---

## Correction — séance du 2026-09-09

<details>
<summary>Corrigé (déplié : contient les réponses)</summary>

### a) — Faux, sur trois points distincts

**Contresens physique.** Un sein dense est une situation de **faible** contraste : la lésion claire est noyée dans un fond clair. C'est le sens de l'ours polaire dans la tempête. Le sein *graisseux* est le cas à fort contraste, et c'est le cas facile.

**Confusion de niveau.** Densité mammaire = propriété du **fond**, du sein entier, cotée 1–4 pour toute l'image. Marge = propriété de la **lésion**, locale, décrite en **netteté et régularité de forme**, pas en densité. « Un contour à forte densité » n'existe pas dans le lexique.

**Réponse attendue.** Suréchantillonner suppose que le modèle échoue **parce qu'il a vu trop peu** de seins denses — donc que c'est un problème de *représentation dans l'échantillon*, réparable en changeant la distribution d'entraînement.

L'autre cause, qui rend le remède inopérant : **la vérité terrain est moins bonne sur les seins denses**. Les masques viennent d'un level-set automatique dont la précision se dégrade quand la densité augmente [établi, Lee et al. 2017]. Suréchantillonner des cas mal annotés **amplifie le bruit**. Le score de validation peut même monter tout en s'éloignant de la réalité clinique, puisque validation et entraînement partagent le même défaut d'annotation.

Seconde cause, physique et irréductible : sur un sein extrêmement dense, une partie de l'information **n'est pas dans l'image**. Aucune stratégie d'échantillonnage ne restaure un signal non acquis.

> **Précision sur le schéma du projet.** Les valeurs 0,904 / 0,886 / 0,749 / 0,808 y sont présentées en face des quatre catégories de densité. C'est trop affirmatif : ce sont les Dice de **quatre images d'exemple** (une par catégorie, Figure 2 de l'article), pas des moyennes par catégorie. La tendance « la précision baisse quand la densité monte » est bien l'affirmation des auteurs [établi], mais ces quatre nombres sont des cas isolés, n = 1 chacun. **Ne pas les citer comme des moyennes à l'oral.**

### b) — Hors sujet, plus une erreur conceptuelle

La question portait sur le **choix des métriques**. La réponse donnée parlait de choix de modèle et de fusion de vues.

**Erreur conceptuelle à corriger en priorité :** « la somme des relevés d'anomalie dans les **coupes** ». Il n'y a pas de coupes — voir 3.3. CC et MLO ne s'additionnent pas, elles se corroborent.

**Réponse attendue.** Dice et IoU mesurent un **recouvrement de surface**. Deux masques peuvent avoir exactement le même Dice en portant des diagnostics opposés : une masse ronde et lisse d'un côté, la même masse hérissée de spicules de l'autre. Les spicules sont fins : ils pèsent très peu en aire, donc les rater ne coûte presque rien au Dice.

Le signe le plus prédictif de malignité est donc **invisible à la métrique principale**. C'est une conséquence sur les métriques, et pas seulement sur le modèle, pour deux raisons :

1. **Ce que tu rapportes.** Un Dice de 0,82 peut recouvrir un modèle qui lisse systématiquement les contours — donc qui efface le signe de malignité. Sans métrique de frontière (HD95, ASSD, boundary F-score), l'évaluation ne peut pas voir cet échec.
2. **Ce que tu optimises.** Une *loss* purement volumique (Dice loss, BCE) **récompense activement le lissage** : arrondir un contour spiculé fait baisser la loss. Choix de métrique et choix de loss sont le même problème vu deux fois.

### c) — Conclusion fausse, mais l'objection réelle a été pressentie

Réponse donnée : « c'est la même faute que `assessment` ». C'est faux, et l'argument (« c'est de la connaissance externe ») est trop grossier — il condamnerait aussi `pathology`, qui est la cible.

**La distinction manquante : descripteur / conclusion.**

- `assessment` (BI-RADS 0–5) est une **conclusion** : la réponse du radiologue à la question exacte qu'on pose. Et dans ce jeu elle est contaminée — révisée après le bilan complémentaire [établi]. Elle encode partiellement l'issue. **Fuite au sens strict.**
- `mass shape` et `mass margins` sont des **descripteurs** : des observations morphologiques, en amont du diagnostic. Les utiliser est légitime en principe — c'est ce que fait toute la tradition CADx à base de features [consensus], et c'est pourquoi ces colonnes existent.

**La vraie objection**, double, et qui n'est pas de la fuite :

1. **Déploiement.** Sur une nouvelle patiente, qui remplit `mass margins` ? Un radiologue. Un modèle qui consomme cette colonne n'est pas un détecteur autonome : c'est un outil d'aide qui *présuppose* qu'un expert a déjà caractérisé la lésion. Cela ne fausse pas l'entraînement — cela restreint ce qu'on a le droit d'**affirmer**. On ne peut plus se comparer à Shen 2019, qui part de l'image nue.
2. **Circularité, spécifique à ce projet.** Ces descripteurs sont donnés **par anomalie** : les avoir suppose que l'anomalie a déjà été localisée. Or localiser l'anomalie *est la tâche*. Pour un projet de segmentation, ils ne sont pas une entrée : ils sont une conséquence de la sortie cherchée.

**Le bon usage** de `shape`, `margins`, `assessment`, `subtlety` et `density` est identique et unique : **des axes de stratification pour analyser les résultats**, jamais des entrées du modèle.

### Bilan de séance

a) faux · b) hors sujet + confusion projection/coupe · c) conclusion fausse, mais l'objection de déploiement a été trouvée seule.

Le fil non attrapé : **la marge est une propriété de forme locale, pas de densité, et c'est elle qui porte le diagnostic.** Les trois questions tournaient autour de ça.

</details>
