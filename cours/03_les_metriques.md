# Module 3 — Les métriques : segmentation et radiologie

*Niveau : praticien. Prérequis : modules 1 et 2.*

---

## 1. Ancrage

Une métrique n'est pas un score qu'on calcule à la fin : c'est la **définition de ce que « réussir » veut dire**, et elle décide de ce que ton modèle va apprendre à faire.

---

## 2. Intuition

Deux personnes regardent la même sortie de modèle et n'ont pas la même question.

L'**ingénieur en vision** demande : *« Les pixels que tu as coloriés sont-ils les bons ? »* → une métrique de **recouvrement**, le Dice.

Le **radiologue** demande : *« Combien de fois par examen vas-tu me faire perdre mon temps, et combien de cancers vas-tu laisser passer ? »* → une métrique de **coût opérationnel**, la sensibilité à un taux de faux positifs donné.

Ces deux questions ne sont pas des variantes l'une de l'autre, et un bon score à la première ne dit presque rien de la seconde. L'analogie : un correcteur orthographique. Le Dice mesure « quel pourcentage des lettres fautives as-tu souligné ». Le FROC mesure « combien de mots corrects soulignes-tu par page avant que je désinstalle le logiciel ». Un correcteur qui souligne tout le texte a un rappel parfait et est inutilisable.

**L'idée centrale du module, à retenir avant toute formule :** la même prédiction produit des nombres complètement différents selon l'**unité d'agrégation** — le pixel, la lésion, l'image, la patiente. Choisir l'unité est un acte plus lourd que choisir la formule.

---

## 3. Le mécanisme

### 3.1 La matrice de confusion, et sur quoi on la compte

Quatre quantités, définies pour une unité donnée :

- **TP** (vrai positif) : positif prédit, positif en vérité ;
- **FP** (faux positif) : positif prédit, négatif en vérité — *fausse alerte* ;
- **FN** (faux négatif) : négatif prédit, positif en vérité — *cancer manqué* ;
- **TN** (vrai négatif) : négatif prédit, négatif en vérité.

Tout ce qui suit n'est qu'un arrangement de ces quatre nombres. Ce qui change tout, c'est l'unité sur laquelle on les compte :

| Unité | TP = | Utile pour |
|---|---|---|
| **pixel** | un pixel de lésion correctement marqué | segmentation |
| **lésion** | une lésion réelle touchée par une détection | détection (CADe) |
| **image** | une image contenant un cancer, signalée | classification (CADx) |
| **patiente** | une femme porteuse d'un cancer, rappelée | décision clinique |

### 3.2 Pourquoi l'exactitude est inutilisable ici

$$\text{Exactitude} = \frac{TP + TN}{TP + TN + FP + FN}$$

Ordre de grandeur, au pixel [inférence, calcul d'ordre de grandeur] : une image de 3000 × 5000 ≈ 15 millions de pixels ; une masse de 2 cm à ~50 µm/pixel fait ~400 pixels de diamètre, soit une aire d'environ $\pi \cdot 200^2 \approx 1{,}25 \times 10^5$ pixels. Rapport : **moins de 1 %**.

Un modèle qui prédit « tout est du fond » obtient donc plus de 99 % d'exactitude, avec TP = 0. La métrique est dominée par les **TN**, qui sont ici sans intérêt : personne ne félicite un radiologue d'avoir correctement ignoré du tissu normal.

**D'où la propriété qui sauve le Dice : il n'utilise pas les TN.**

### 3.3 Dice et IoU

$$\mathrm{Dice}(A,B) = \frac{2|A \cap B|}{|A| + |B|} = \frac{2TP}{2TP + FP + FN}$$

$$\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$

où $A$ est le masque prédit et $B$ la vérité terrain, $|\cdot|$ le nombre de pixels.

Le Dice est exactement le **F1-score calculé sur les pixels** : moyenne harmonique de la précision $TP/(TP+FP)$ et du rappel $TP/(TP+FN)$ [établi].

Les deux sont liés par une bijection croissante [établi] :

$$\mathrm{Dice} = \frac{2\,\mathrm{IoU}}{1 + \mathrm{IoU}}, \qquad \mathrm{IoU} = \frac{\mathrm{Dice}}{2 - \mathrm{Dice}}$$

Donc : **ils classent les modèles dans le même ordre**, et rapporter les deux n'apporte aucune information. Mais ils ne donnent pas les mêmes chiffres — $\mathrm{Dice} \ge \mathrm{IoU}$ toujours. Un Dice de 0,80 correspond à un IoU de 0,67. Comparer ton Dice au IoU d'un article, c'est se donner 13 points gratuits.

### 3.4 Le point aveugle du Dice, et les métriques de frontière

C'est le fil rouge du module 1 qui revient, et c'est le cœur de ce module.

Le Dice compte des **aires**. Un spicule est fin : il pèse presque rien en aire. Deux masques peuvent donc avoir le même Dice en portant des diagnostics opposés — une masse lisse d'un côté, la même hérissée de spicules de l'autre. Le signe le plus prédictif de malignité est **invisible à la métrique principale**.

Les métriques qui le voient mesurent des **distances entre frontières**.

**Distance de Hausdorff** [établi] :

$$H(A,B) = \max\big(h(A,B),\, h(B,A)\big), \qquad h(A,B) = \sup_{a \in \partial A}\ \inf_{b \in \partial B}\ d(a,b)$$

où $\partial A$ est le contour de $A$ et $d$ la distance euclidienne. En clair : *pour chaque point du contour prédit, la distance au point le plus proche du contour vrai ; on garde la pire*. Le $\max$ la rend maximalement sensible : **un seul pixel aberrant fait exploser la valeur**.

D'où l'usage du **HD95** : le 95ᵉ percentile de ces distances au lieu du maximum. On garde la sensibilité à la forme, on jette la queue de distribution [consensus].

**ASSD** (*average symmetric surface distance*) : la moyenne — et non le max — de toutes les distances contour-à-contour dans les deux sens. Plus stable, moins sensible aux détails fins.

**Unité.** Ces trois métriques sont des **distances**. En pixels elles ne veulent rien dire d'une image à l'autre — il faut multiplier par `PixelSpacing` pour obtenir des millimètres. Voir module 2, § 3.5.

### 3.5 Le vocabulaire du radiologue

Sur une unité *image* ou *patiente* :

$$\text{Sensibilité} = \frac{TP}{TP+FN}\quad\text{(rappel)} \qquad \text{Spécificité} = \frac{TN}{TN+FP}$$

$$\text{VPP} = \frac{TP}{TP+FP}\quad\text{(précision)} \qquad \text{VPN} = \frac{TN}{TN+FN}$$

La distinction qui compte, et qui est la source de l'erreur la plus coûteuse du projet :

- **Sensibilité et spécificité sont conditionnées par la vraie classe.** Elles ne dépendent **pas** de la prévalence. Elles se transportent d'une population à l'autre.
- **VPP et VPN dépendent de la prévalence.** Elles ne se transportent pas.

**L'arithmétique du dépistage**, à faire une fois dans sa vie [établi, arithmétique ; taux de détection ≈ 5 ‰ en dépistage : consensus] :

> 1 000 femmes dépistées, 5 cancers.
> Modèle : sensibilité 90 %, spécificité 95 %.
>
> - vrais positifs : $5 \times 0{,}90 = 4{,}5$
> - faux positifs : $995 \times 0{,}05 \approx 50$
> - **VPP $= 4{,}5 / 54{,}5 \approx 8~\%$**

Plus de **neuf alertes sur dix sont fausses**, avec un modèle dont les chiffres semblent excellents. C'est la réalité du dépistage, et c'est la raison pour laquelle les CADe sont jugés en **faux positifs par image** et non en spécificité.

### 3.6 ROC, AUC, et pourquoi la détection a besoin d'autre chose

Un modèle sort un score continu ; chaque **seuil** donne un couple (sensibilité, 1 − spécificité). La **courbe ROC** est le lieu de ces couples, l'**AUC** son aire.

Interprétation exacte [établi] : l'AUC est la probabilité qu'un positif tiré au hasard reçoive un score supérieur à celui d'un négatif tiré au hasard. Elle est **indépendante du seuil et de la prévalence** — sa force, et son piège : une AUC de 0,91 ne dit rien du nombre de fausses alertes à un point de fonctionnement utilisable.

> Dans la littérature CAO ancienne, l'AUC est notée **Az** [établi]. Les tableaux 1 et 2 de Lee et al. 2017 l'utilisent. Ne bute pas dessus.

**La PR-AUC** (précision-rappel) dépend, elle, de la prévalence, et donne une image plus honnête quand les positifs sont rares [consensus].

**Pourquoi la ROC ne marche pas pour la détection.** En détection, il n'y a pas de TN. Un « vrai négatif » serait *un endroit de l'image où il n'y a pas de lésion et où le modèle n'a rien marqué* — mais le nombre de tels endroits est arbitraire : il dépend de la façon dont tu découpes l'image. La spécificité n'est pas définie, donc la ROC non plus.

**Le FROC** remplace la spécificité par une quantité bien définie :

$$\mathrm{FPPI} = \frac{\text{nombre total de faux positifs}}{\text{nombre d'images}}$$

La courbe FROC porte la sensibilité **par lésion** en ordonnée et le FPPI en abscisse — abscisse **non bornée**, ce n'est pas une probabilité. Le résumé usuel est le **CPM**, moyenne des sensibilités à 1/8, 1/4, 1/2, 1, 2, 4 et 8 FPPI [établi, convention issue des challenges de détection pulmonaire type LUNA16].

**Le critère de touche.** Quand considère-t-on qu'une lésion est détectée ? Trois conventions courantes, incompatibles :

1. IoU entre boîtes ou masques $> \tau$ ;
2. le centroïde prédit tombe dans le masque vrai ;
3. le centroïde vrai tombe dans le masque prédit.

**Ce choix déplace la courbe FROC.** Il doit être écrit noir sur blanc dans ton rapport, sinon tes chiffres ne sont comparables à rien [consensus].

---

## 4. Sur tes données

### 4.1 Ton jeu de test est enrichi — et ça détruit la VPP

Effectifs du test officiel [établi, Table 3 de Lee et al.] :

| | Bénin | Malin | Total | Prévalence |
|---|---|---|---|---|
| Masses | 117 | 83 | 200 | **41,5 %** |
| Calcifications | 85 | 66 | 151 | **43,7 %** |

Compare à la prévalence du dépistage réel : ~0,5 %. **Facteur ~80.**

Reprends le modèle de la section 3.5 — sensibilité 90 %, spécificité 95 % — et applique-le à ton test masses :

- vrais positifs : $83 \times 0{,}90 = 74{,}7$
- faux positifs : $117 \times 0{,}05 = 5{,}9$
- **VPP $= 74{,}7 / 80{,}6 \approx 93~\%$**

Le **même modèle** rapporte une VPP de **93 % sur CBIS-DDSM** et de **8 % en dépistage réel**. Un facteur 11.

**Conclusion opérationnelle** [inférence, mais conséquence directe de l'arithmétique] : sur ce jeu, la sensibilité et la spécificité sont interprétables ; **la VPP, la VPN et l'exactitude ne le sont pas** comme grandeurs cliniques. Si tu les rapportes, tu dois écrire à côté que le jeu est enrichi. C'est un des meilleurs points possibles pour ton oral : il montre que tu as compris la différence entre un benchmark et une population.

### 4.2 Tes intervalles de confiance seront larges

83 cas malins dans le test masses. Une sensibilité estimée à 86 % a un écart-type d'échantillonnage :

$$\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0{,}86 \times 0{,}14}{83}} \approx 0{,}038$$

Soit un intervalle à 95 % d'environ **[78,5 % ; 93,5 %]** — ±7,5 points [établi, approximation normale ; préférer Wilson ou un bootstrap, l'approximation normale étant mauvaise près des bornes].

Traduis : **une différence de 3 points entre deux de tes modèles n'est pas une différence.** Rapporter « U-Net 0,79 vs Attention U-Net 0,82, donc Attention U-Net gagne » sur ce jeu est une conclusion que les données ne portent pas. Un bootstrap sur les cas (pas sur les pixels) coûte dix lignes et rend ton oral incontestable.

### 4.3 Te comparer à Shen 2019 demande un travail supplémentaire

Shen et al. rapportent une **AUC par image de 0,88 (modèle unique) / 0,91 (ensemble de 4)**, pour une tâche de **classification** [établi]. Toi tu produis des **masques**.

Un Dice ne se compare pas à une AUC. Pour te positionner, il faut agréger ta carte de pixels en un score de malignité par image — par exemple le maximum de la carte de probabilité, ou la somme des probabilités sur la composante connexe la plus forte — puis calculer une AUC **sur le même jeu de test officiel**. C'est ce pont qui rend ton travail comparable à la littérature [inférence].

---

## 5. Là où ça casse

**1. L'exactitude au pixel.** > 99 % pour un modèle vide. Ne la rapporte jamais.

**2. Dice moyen ≠ Dice global.** Deux agrégations, deux nombres [établi] :
- *Dice moyen* : un Dice par image, puis moyenne → chaque **image** pèse pareil, donc une lésion minuscule pèse autant qu'une énorme ;
- *Dice global* : on cumule TP, FP, FN sur tout le jeu, puis un seul Dice → chaque **pixel** pèse pareil, donc les grosses lésions dominent.

Sur un jeu à tailles très variables comme le tien, l'écart entre les deux est important. Beaucoup d'articles ne précisent pas lequel ils rapportent — c'est une source majeure de fausses comparaisons.

**3. Le cas vide.** Si vérité et prédiction sont toutes deux vides, le Dice vaut $0/0$. Il faut une convention explicite (1, ou exclusion). Si la vérité est vide et la prédiction non, le Dice vaut 0 quel que soit le nombre de faux positifs — la métrique ne distingue plus une petite fausse alerte d'une catastrophe.

**4. HD95 en pixels.** Sans `PixelSpacing`, ce n'est pas une distance physique, et les images de ton jeu n'ont pas toutes la même résolution.

**5. Comparer un Dice à un IoU.** 13 points d'écart gratuits à 0,80.

**6. Rapporter une VPP sur un jeu enrichi sans le dire.** Section 4.1.

**7. Une AUC sans point de fonctionnement.** « AUC 0,93 » ne dit pas si le système est utilisable. Il faut donner au moins un couple (sensibilité, FPPI) sur la courbe.

**8. Changer de critère de touche entre deux expériences.** Les FROC deviennent incomparables, y compris avec les tiennes de la semaine précédente.

**9. Optimiser une loss volumique et rapporter une métrique de frontière.** Cohérent seulement si c'est assumé : tu montres alors que ton modèle est bon en aire et mauvais en contour, ce qui est un **résultat**, pas un accident. C'est l'inverse qui est malhonnête — optimiser le Dice et ne rapporter que le Dice, en concluant sur la qualité diagnostique.

---

## 6. Contrôle

**a)** Deux modèles sur ton test masses. Le modèle A : Dice moyen 0,78, HD95 médian 4,1 mm. Le modèle B : Dice moyen 0,81, HD95 médian 9,6 mm. Ton binôme retient B parce que « le Dice est la métrique principale ». En te servant du module 1, explique pourquoi ce choix peut être exactement le mauvais, et dis quel **troisième chiffre** — que ni A ni B ne fournit ici — permettrait de trancher pour de bon.

**b)** Tu rapportes une sensibilité de 88 % et une spécificité de 91 % sur le test masses, puis tu écris en conclusion : « appliqué au dépistage, ce système signalerait à tort moins d'une femme sur dix ». Cette phrase contient une erreur et une seule. Identifie-la, dis laquelle des quatre grandeurs (sensibilité, spécificité, VPP, VPN) tu as implicitement invoquée sans le savoir, et calcule ce que donnerait réellement ce modèle sur 1 000 femmes à une prévalence de 5 ‰.

**c)** Tu passes de la segmentation à la détection et tu veux tracer une courbe FROC. Ton modèle produit une carte de probabilité continue sur l'image entière. Décris la chaîne d'opérations qui mène de cette carte à **un point** de la courbe FROC — il y a au moins quatre décisions à prendre, dont deux ne sont pas des seuils. Puis dis laquelle de ces décisions rendrait tes résultats incomparables à ceux d'un article si tu ne la documentais pas.
