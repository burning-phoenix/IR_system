# When Models Manipulate Manifolds: The Geometry of a Counting Task

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** C-geometry — Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah, Batson (Anthropic). A full worked account of a model computing on a low-dimensional manifold — the closest published analogue to 'what is the encoder actually doing'.
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/2601.04480.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

# **When Models Manipulate Manifolds: The Geometry of a Counting Task** 

**Wes Gurnee,**<sup>_∗_</sup> **Emmanuel Ameisen,**<sup>_∗_</sup> **Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson**<sup>_∗†_</sup> 

Anthropic 

> _∗_ Equal contribution. _†_ Corresponding author: `joshb@anthropic.com` 

## **Abstract** 

Language models can perceive visual properties of text despite receiving only sequences of tokens—we mechanistically investigate how Claude 3.5 Haiku accomplishes one such task: linebreaking in fixed-width text. We find that character counts are represented on low-dimensional curved manifolds discretized by sparse feature families, analogous to biological place cells. Accurate predictions emerge from a sequence of geometric transformations: token lengths are accumulated into character count manifolds, attention heads “twist” these manifolds to estimate distance to the line boundary, and the decision to break the line is enabled by arranging estimates orthogonally to create a linear decision boundary. We validate our findings through causal interventions and discover “visual illusions”—character sequences that hijack the counting mechanism. Our work demonstrates the rich sensory processing of early layers, the intricacy of attention algorithms, and the importance of combining feature-based and geometric views of interpretability.<sup>1</sup> 

## **1 Introduction** 

Intelligent systems need perception to understand, predict, and navigate their environment. These sensory capabilities reflect what’s useful for survival in a specific environment: bats use echolocation, migratory birds sense magnetic fields, Arctic reindeer shift their UV vision seasonally. But when your world is made of text, what do you see? Language models encounter many text-based tasks that benefit from visual or spatial reasoning: parsing ASCII art, interpreting tables, or handling text wrapping constraints. Yet their only “sensory” input is a sequence of integers representing tokens. They must learn perceptual abilities from scratch, developing specialized mechanisms in the process. 

In this work, we investigate the mechanisms that enable Claude 3.5 Haiku to perform a natural perceptual task which is common in pretraining corpora and involves tracking position in a document. We find learned representations of position that are in some ways quite similar to the biological neurons found in mammals who perform analogous tasks (“place cells” and “boundary cells” in mice), but in other ways unique to the constraints of the residual stream in language models. We study these representations and find dual interpretations: we can understand them as a family of discrete features or as a one-dimensional “feature manifold”/“multidimensional feature” [58, 57, 21, 17].<sup>2</sup> In the first interpretation, position is determined by which features activate and how strongly; in 

> 1This is an archival version of a paper originally published on October 21, 2025 at Transformer Circuits, which we recommend for its improved interactivity and styling. 

> 2All features have a magnitude dimension; so a discrete feature is a one-dimensional ray, and a onedimensional feature manifold is the set of all scalings of that manifold, contracting to the origin. See What is a Linear Representation? What is a Multidimensional Feature? 

Archival Preprint. 



<!-- Start of picture text -->
LLMsof text, perceive despite visual only propertiesseeing featuresWe can understand and attribution this graphs using ‘computationAnalyzing thegeometryclarifies mechanismsof<br>alist<br>of numbers<br>ject chose the chemical. 47 WN Ua<br>13, also called (aluminim) 51 H i -<br>Wit stumioum: fe PreviouslineJ | currentline mooie 555<br>WAT THE HODEL SEES: i Qk twists manifold to<br>3 AAD| Sense endottine<br>47425 1810 3356 249175 2193764153862 528 2986 yi | aetence,limit. a)oF<br>167 14 93 26 41 2 68 35 71 (2) =a<br>No, it exceeds the ine limit ie = oe)<br>ome cOMLETION i i; -} (ext‘too long) word cen hes3% || Secldesttorovertownear boundary<br>a — AD we<br><!-- End of picture text -->

Figure 1: Left: a description of the linebreaking task. Center: example feature families we study. Right: a highlight of the geometry of representations and computations. 

the latter interpretation, it’s determined by angular movement on the feature manifold. Similarly, computation has two dual interpretations, as discrete circuits or geometric transformations. 

The task we study is linebreaking in fixed-width text. When training on source code, chat logs, email archives, scanned articles, or judicial rulings that have line width constraints, how does the model learn to predict when to break a line?<sup>3</sup> Human visual perception lets us do this almost completely subconsciously – when writing a birthday card, you can see when you are out of room on a line and need to begin the next – but language models see just a list of integers. In order to correctly predict the next _token_ , in addition to selecting the next _word_ , the model must somehow count the characters in the current line, subtract that from the line width constraint of the document, and compare the number of characters remaining to the length of the next word. As a concrete example, consider the below pair of prompts with an implicit 50-character line wrapping constraint.<sup>4</sup> When the next word fits, the model says it; when it does not, the model breaks the line: 



<!-- Start of picture text -->
A language model needs to predict whether words fit in lines<br>Either<br>the next word fits Orit does not<br>After gallium was ruled out due to melting point,» 49| | After gallium was ruled out due to melting point, 49<br>elementthe engineerswith atomicon thenumberproject58,chosealsothecalledchemical~tin 46)47| | the(elementengineerswith atomicon the numberproject13,chosealsothecalledchemicalaluminum 4751<br>Correct completion: (2) Correct completion:<br>...Without the benefit of visual alignment<br>‘Tokens the model sees<br>(TT.<br>| 3-42 16 56 24.91 37 15 62 8 29 47 5[10]33 75.21 9 64 38 52 86{16]7 14 99 26 41 OTT)2 68 35.71 2 |<br><!-- End of picture text -->

Figure 2: A pair of prompts highlighting the difficulty of the linebreaking task. 

> 3Michaud _et al._ looked for “quanta” of model skills by clustering gradients [46]. Their Figure 1 shows that predicting newlines in fixed-width text formed one of the top 400 clusters for the smallest model in the 70m Pythia model. 

> 4The wrapping constraint is implicit. Each newline gives a lower bound (the previous word did fit) and an upper bound (the next word did not). We do not nail down the extent to which the model performs optimal inference with respect to those constraints, rather focusing on how it approximately uses the length of each preceding line to determine whether to break the next. There are also many edge cases for handling tokenization and punctuation. A model could even attempt to infer whether the source document used a non-monospace font and then use the pixel count rather than the character count as a predictive signal! 

2 

To orient ourselves to the stages of the computation, we first studied the model using discrete dictionary features. In this frame, we can understand computation as an “attribution graph” [2] where a cascade of features excite or inhibit each other.<sup>5</sup> 



<!-- Start of picture text -->
sntnnteuntcmanes a<br>Aavfichecconpuondet Mr colleen or stat——"> prerace ‘too long<br>cosh wat co hen predict aeancel<br>‘m),t Ru (rutheniue), Os (osmium), Aialuminum),¥ Cy aluminum: Timit<br>It As rather: for us. tobe herelMleate@RONR great. task. position in<br>Ancreased devotion to that cause fori gave the. current line<br>previouswan oF (tion,Sat geermcnuneer cody eshalt e,have ofaepresse ond.<br>ine! token length<br><!-- End of picture text -->

Figure 3: Attribution Graph for Claude 3.5 Haiku’s prediction of a newline in the aluminum prompt. We see features relating to “width of the previous line” and “position in the current line” which together activate features for “distance from line limit”. Combined with features for the planned next word, these features activate “predict newline” features. 

The attribution graph shows how the model performs this task by combining features that represent different concepts it needs to track: 

1. Features for the current position in the line (the character count) as well as features for the total line width (the constraint) are computed by accumulating features for individual token lengths. 

2. The model then combines these two representations — current position and line width — to estimate the distance from the end of the line, leading to “characters remaining” features. 

3. Finally, the model uses this estimate of characters remaining along with features for the planned next word to determine if the next word will fit on the line or not. 

The attribution graph provides a kind of execution trace of the algorithm, showing on this prompt which variables are computed and from what. After finding large feature families involved in representing these quantities across a diverse dataset, we suspected a simpler lens might be provided in terms of lower-dimensional feature manifolds interacting geometrically. We found geometric perspectives on the following questions: 

**How does the model represent different counts?** The number of characters in a token, the number of characters in the current line, the overall line width constraint, and the number of characters remaining in the current line are each represented on 1-dimensional feature manifolds embedded with high curvature in low-dimensional subspaces of the residual stream. These manifolds have a dual interpretation in terms of discrete features, which tile the manifold in a canonical way, providing approximate local coordinates. Manifolds with similar geometry arise for a variety of ordinal concepts, and a ringing pattern we see in the embedded geometry in all these cases is optimal with respect to a simple physical model (§Representing Character Count).<sup>6</sup> 

**How does the model detect the boundary?** To detect an approaching line boundary, the model must compare two quantities: the current character count and the line width. We find attention 

> 5We actually first tried to use patching and probing without looking at the graph as a kind of methodological test of the utility of features, but did not make much progress. In hindsight, we were training probes for quantities different than the ones the model represents cleanly, e.g., a fusion of the current token position and the line width. 

> 6Ringing, in the manifold perspective, corresponds to interference in the feature superposition perspective. 

3 



<!-- Start of picture text -->
heen 7 ow ar tterentchareter counts represented? Hom isthe boundary detects?<br>Secttristattechose theCuicetiiniag)chesscal’ $73 [ie HNHAN -<br>to's Wo 7521 9 es 0 30a \ A a<br>of ae 98 120 38 NOD<br>“Lerten py Seransorate annetteAAgas\ erencQe at Dott e c<br>Wat the nextword ft? ow ae representations constructed?<br>;<br>oon = Nn<br>to fi)<br>pe is ied Y i ae<br>ii j  eitN om iy a J ae an|  ~:: *%dan<br>Ltt ae Si a<br>‘apreoonanions tobe testy soporte Shoreumcreetese‘tncount mening en = e<br><!-- End of picture text -->

Figure 4: Key steps in the linebreaking behavior can be described in terms of the construction and manipulation of manifolds. 

heads whose QK matrix rotates one counting manifold to align it with the other at a specific offset, creating a large inner product when the difference of the counts falls within a target range. Multiple heads with different offsets work together to precisely estimate the characters remaining (§Sensing the Line Boundary). 

**How does the model know if the next word fits?** The final decision — whether to predict a newline — requires combining the estimate of characters remaining with the length of the predicted next word. We discover that the model positions these counts on near-orthogonal subspaces, creating a geometric structure where the correct linebreak prediction is linearly separable (§Predicting the Newline). 

**How does the model construct these curved geometries?** The curvature in the character count representation manifold is produced by many attention heads working together, each contributing a piece of the overall curvature. This distributed algorithm is necessary because individual components cannot generate sufficient output variance to create the full representation (§A Distributed Character Counting Algorithm). 

We validate these interpretations through targeted interventions, ablations, and “visual illusions” — character sequences that hijack specific attention mechanisms to disrupt spatial perception (§Visual Illusions). 

Zooming out, we take several broader lessons from this mechanistic case study: 

**When Models Manipulate Manifolds** . For representing a scalar quantity (e.g., integer counts from 1 to _N_ ), it is inefficient to use _N_ orthogonal dimensions, and not expressive enough to use just one<sup>7</sup> . Instead models learn to represent these quantities on a feature manifold with intrinsic dimension 1 (the count) embedded in a subspace with extrinsic dimension 1 _< d ≪ N_ (e.g., [21, 17, 48]), in which the curve “ripples”. Such rippled manifolds optimally trade off capacity constraints (roughly, dimensionality) with maintaining the distinguishability of different scalar values (curvature). Our work demonstrates the intricate ways in which these manifolds can be manipulated to perform _computation_ and show how this can require distributing computation across multiple model components. 

**Duality of Features and Geometry** . Dictionary features provide an unsupervised entry point for discovering mechanisms, and attribution graphs surface the important features for any particular prediction. Sometimes, discrete features (and their interactions) can be equivalently described using continuous feature manifolds (and their transformations). In cases where it is possible to explicitly 

> 7Orthogonal dimensions would also not be robust to estimation noise. 

4 

parameterize the manifold (as with the various integer counts we study), we can directly study the geometry, making some operations clearer (e.g., boundary detection). But this approach is expensive in researcher time and potentially limited in scope: it’s straightforward when studying _known_ continuous variables but becomes difficult to execute correctly for more complex, difficult-to-parametrize concepts. 

**Complexity Tax** . While unsupervised discovery is a victory in and of itself, dictionary features fragment the model into a multitude of small pieces and interactions – a kind of complexity tax on the interpretation. In cases where a manifold parametrization exists, we can think of the geometric description as reducing this tax. In other cases, we will need additional tools to reduce the interpretation burden, like hierarchical representations [12] or macroscopic structure in the global weights [54]. We would be excited to see methods that extend the dictionary learning paradigm to unsupervised discovery of other kinds of geometric structures (e.g., those found in prior work [31, 11, 7, 73, 59, 42, 74, 32, 48]). 

**Natural Tasks** . The crispness of the representations and circuits we found was quite striking, and may be due to how well the model does the task. Linebreaking is an extremely natural behavior for a pretrained language model, and even tiny models are capable of it given enough context. Studying tasks which are natural for pretrained language models, instead of those of more theoretical interest to human investigators, may offer promising targets for finding general mechanisms. 

### **1.1 Preliminaries** 

To enable systematic analysis, we created a synthetic dataset using a text corpus of diverse prose where we (1) stripped out all newlines and (2) reinserted newlines every _k_ characters to the nearest word boundary _≤ k_ for _k_ = 15 _,_ 20 _, . . . ,_ 150. As an example, here is the opening sentence of the Gettysburg Address, wrapped to _k_ = 40 characters, with the newlines shown explicitly. 

```
Fourscoreandsevenyearsagoour[newline]
fathersbroughtforthonthiscontinent,[newline]
anewnation,conceivedinLiberty,and[newline]
dedicatedtothepropositionthatall[newline]
menarecreatedequal.
```

Claude 3.5 Haiku is able to adapt to the line length for every value of _k_ , predicting newlines at the correct positions with high probability by the third line (see Appendix). 

All features in the main text of this paper are from a 10 million feature Weakly Causal Crosscoder (WCC) dictionary [43] trained on Claude 3.5 Haiku. Feature activation values are normalized to their max throughout. 

## **2 Representing Character Count** 

We define the _line character count_ (or _character count_ ) at a given token in a prompt to be the total number of characters since the last newline, including the characters of the current token. 

A natural thing to check is if the model linearly represents the character count as a quantitative variable: that is, can we predict character count with high accuracy via linear regression on the residual stream? Yes: a linear probe fit on the residual stream after layer 1 has an _R_<sup>2</sup> of 0.985. This success does not mean, however, that the model actually represents the character count along a single line. 

Instead, we find a multidimensional representation of the character count that we will analyze from four perspectives: 

1. Sparse crosscoder features.<sup>8</sup> 

> 8Each feature has an encoder, which acts as a linear + (Jump)ReLU probe on the residual stream, and a decoder. Ten features _f_ 1 _, . . . , f_ 10 are associated with line character count. The model’s estimate of the character count, given a residual stream vector _x_ , is summarized by the set of activities of each of the 10 features _{fi_ ( _x_ ) _}_ . 

5 

2. A low-dimensional subspace.<sup>9</sup> 

3. A continuous 1-dimensional manifold contained in that low-dimensional subspace.<sup>10</sup> 

4. A set of 150 logistic probes (corresponding to values of line character count from 1 to 150).<sup>11</sup> 

Each of these perspectives provides a complementary view of the same underlying object. The feature perspective is valuable for getting oriented, the subspace is perfect for causal intervention, the manifold is helpful for understanding how the representation is constructed and then manipulated to detect boundaries, and the logistic probes are useful for analyzing the OV and QK matrices of the individual attention heads involved. 

### **2.1 Character Count Features** 

We begin with the features. In layers one and two, we found features that seemed to activate based on a token’s character position within a line. For example, in the attribution graph for the aluminum prompt, there were two features active on the final word “called” that seemed to fire when the line character count was between 35–55 and 45–65, respectively. To find more such features, we computed the mean activation of each feature binned by line character count. There were ten features with smooth profiles and large between-character-count variance, shown below: 



<!-- Start of picture text -->
LUne Character Count vs Average Feature Activation<br>os Feature<br>CT et os,<br>054 | | |tst'ttSot tie seusseg al ot svetett ov = steam theSSQSHeRBR sesso.se = tcczateas<br>sine sti hres oe is<br>g al’ | (Gtteermcnt he pet, cs teen, — teesco<br>2 — tees,<br>Foo = ter<br>i ‘Feature Splitting i)<br>\<br>aoo A AQ x —~ >< ‘a<br>a Fy % rs r io io rr<br>Lin character Count<br><!-- End of picture text -->

Figure 5: A family of features representing the current character count in a line of text. The tuning curve of the features’ activity increases at larger line character counts. 

We find these features especially interesting as they are quite analogous to curve-detector features in vision models [6, 22] and place cells in biological brains [51]. In all three of these cases, a continuous variable is represented by a collection of discrete elements that activate for particular value ranges. Moreover, we also observe _dilation_ of the receptive fields (i.e., subsequent features activate over increasingly large character ranges) which is a common characteristic of biological perception of numbers (e.g., [13, 62]). 

In the Appendix, we show these features are universal across dictionaries of different sizes, but that some feature splitting occurs with respect to the line width constraint. 

### **2.2 The Model Represents Character Count on a Continuous Manifold** 

We observe that character count feature activations rise and fall at an offset, with two features being active at a time for most counts. This pattern suggests that the features are reconstructing a curved continuous manifold, locally parametrized by the activity of the two most active features. Given that 

> 9The model’s estimate of the character count is summarized by the projection _π_ ( _x_ ) of _x_ onto that subspace. Two datapoints have similar character counts if their projections are close in that subspace. 

> 10The model’s estimate of the character count is summarized by the nearest point on the manifold to the projection of _x_ into the subspace, and its confidence in that estimate by the magnitude of _π_ ( _x_ ). 

> 11The model’s estimate of the character count is summarized by the probability distribution given by the softmax of the probe activities, softmax( _Px_ ). 

6 



<!-- Start of picture text -->
° °<br><!-- End of picture text -->

Figure 6: Character count is represented on a manifold in a 6 dimensional subspace (jagged line). This manifold can be approximately locally parametrized by the features we identified (crosses). 

their joint activation profiles follow a sinusoidal pattern, we expect reconstructions to lie on a curve between adjacent feature decoders. 

To visualize this, we first compute the average layer 2 residual stream for each value of line character count on our synthetic dataset. We compute the PCA of these 150 vectors, and find that the top 6 components capture 95% of the variance; we project data to that 6 dimensional subspace which we call the “character count subspace” (top 3 PCs on the left below, next 3 PCs on the right). We observe the data form a twisting curve, resembling a helix from the perspective of PCs 1–3 and a more complex twist from the perspective of PCs 4–6. 

We also reconstruct the residual stream for each datapoint using only the 10 character count features identified above, and compute the average _reconstructed_ residual stream. We project the resulting curve, along with the feature decoders, into the same subspace. We find that the average line character count vectors are quite closely approximated by the feature reconstruction, though with mild kinks near the feature vectors themselves, reminiscent of a spline approximation of a smooth curve. While the 10 feature vectors discretize the curve, interpolating between the 2–3 neighboring features which are active at a time allows for a high-quality reconstruction of 150 data points. 

### **2.3 Validation: The Character Count Subspace is Causal** 

To validate our interpretation of the character count subspace, we perform a coarse-grained ablation and a fine-grained intervention. 

**Ablation Experiment.** For our ablation experiment, we zero ablate (from a single early layer) a _k_ -dimensional subspace corresponding to the top _k_ principal components of the per–character count mean activations and compare to a baseline of ablating a random _k_ -dimensional subspace. Below we measure the loss effect, broken down by newlines and nonnewlines.<sup>12</sup> 

**Intervention Experiment.** As a more surgical intervention, we perform an experiment to modify the perceived character count at the end of the aluminum prompt (originally 42 characters). Specifically, we sweep over character counts _c_ , and substitute the mean activation across all tokens in our dataset with count _c_ . That is, _a_ patched = _a_ original _− µ_ original + _µc_ for activation _a_ and average activation matrix 

> 12Note, in general one should not assume that a subspace spanned by features (or a PCA) is dedicated to those features because it could be in superposition with many other features. However, because in this case the character count subspace is densely active (and therefore less amenable to being in superposition), this experimental design is more justified. 

7 



<!-- Start of picture text -->
Loss Ablations of Different Subspaces<br>Character Count Subspace<br>0.4) ~® Next is newline<br>Character Count Subspace<br>° Next NOT newline<br>8038 °°) = RandomNext is newlineSubspace<br>£ Random Subspace<br>8 ® Next NOT newline<br>302<br>2<br>2<br>g<br>201<br>0.0 wo = = = - ~~ - 8 --- 8<br>1 2 3 4 5 6 7 8 9 10<br>Subspace Dimension<br><!-- End of picture text -->

Figure 7: Ablating the character count subspace has a large effect only when the next token is a newline. 

_µ_ . We perform this intervention for three adjacent early layers and the last two tokens for both the entire mean vector and within the 6 dimensional PCA space of the mean vectors.<sup>13</sup> 



<!-- Start of picture text -->
Aluminum Prompt Intervention Results.<br>0.0<br>-0.5 <S=SSSS> = = ?<br>8 -1.0 Token Identity: :<br>Fa — newline :<br>2-154 — aluminum 5<br>3 ‘Subspace: EN<br>2-20) dmodel i<br>3 == rank 6 : .<br>z 25) original Line y, : SY<br>Char Count :<br>= 30<br>3 y : =<br>3.5 Resse ees :<br>20 25 30 35 40 45 50<br>Patched Line Count<br><!-- End of picture text -->

Figure 8: Intervening on a rank 6 subspace is sufficient to change the model’s linebreaking behavior. 

### **2.4 The Probe Perspective** 

We also train supervised logistic regression probes to predict character count.<sup>14</sup> Probes trained after layer 1 achieve a root mean squared error of 5, indicating some intrinsic noise in the character count representation — which is consistent with our features having relatively wide receptive fields. Performing PCA on the 150 probe weight vectors, we find that 6 components capture 82% of the variance. 

When we look at the average responses of each probe to tokens with different line character counts, we see a striking pattern. In addition to a diagonal band (probes, like the sparse features, have increasingly wide receptive fields), we see two faint off-diagonal bands on each side! The response 

> 13The attribution graph has several positional features and edges on both the last token (“called”) as well as the second-to-last token (“also”). We change the “also” count representation to be 6 characters prior to that for the final token, to maintain consistency. 

> 14as a 150-way multiclass classification problem 

8 



<!-- Start of picture text -->
Average Probe Activation by Line Character Count<br>x<br>100<br>40 07m §<br>5 oso §<br>a<br>260 o2 =<br>3 ooo $<br>Hs% 80 0.25-050 3&3<br>100 0.75 2<br>-1.00<br>0<br>a Line eeCharacter Count<br><!-- End of picture text -->

Figure 9: Response curve of Line Character Count probes as a function of Line Character Count show widening receptive fields and a “ringing” pattern of off-diagonal stripes. 

curve of each probe is not monotonically decreasing away from its max, but rebounds. This “ringing” turns out to be a natural consequence of embedding a “rippled” manifold into low dimensions. 

### **2.5 Rippled Representations are Optimal** 

We note that the cosine similarities of the mean activation vector (which form the helix-like curve visualized in PCA space above), the linear probe vectors, and feature decoder vectors all exhibit similar ringing patterns to the above figure.<sup>15</sup> Note that not only are neighboring features not orthogonal, features further away have negative similarities, and then those even further away have positive ones again. 



<!-- Start of picture text -->
Paris Cosine Smarty of Diterent Character Count Vectors<br>ean Activations Vectors Urea bes Feature Decoders<br>5<br>Es 2” 7 cas §<br>g: E£i oe ow<br>386<br>‘<br><!-- End of picture text -->

Figure 10: The pairwise cosine similarity of different character count vectors. (Left) mean activation vectors, (center) linear probes, (right) feature decoders. 

This structure turns out to be a natural consequence of having the desired pattern of similarity, trivially achievable in 150 dimensions, projected down to low dimensions. As a toy model of this, suppose that we wish to have a discretized circle’s worth of unit vectors, each similar to its neighbors but orthogonal to those further away. This can be realized by a symmetric set of unit vectors in 150 dimensions with cosine similarity matrix _X_ pictured below (left). Projecting this to its top 5 eigenvectors yields a 5-dimensional embedding of the same vectors with cosine similarity matrix (below right) exhibiting ringing. We also plot the curve these vectors form in the top 3 eigenvectors. We can think of the original 150-dimensional embedding of the circle as being highly curved, and 

> 15We use the term “ringing” in the sense of signal processing, a transient oscillation in response to a sharp peak, such as in the Gibbs Phenomenon). 

9 

the resulting 5-dimensional embedding as retaining as much of that curvature as possible. This manifests as ripples in the embedding of the circle when viewed in a 3D projection. A relationship of this construction to Fourier features is discussed in the appendix. 



<!-- Start of picture text -->
:<br><!-- End of picture text -->

Figure 11: Left panel shows an ideal similarity matrix for vectors representing points along a circle. Middle panel shows the optimal (PCA) approximation possible when embedding the points in 5 dimensions. Right panel shows the resulting projection of the circle to the top 3 dimensions, exhibiting rippling. 

Alternatively, one can view the ringing from the perspective of sparse feature decoders as a kind of interference weight [56]. With no capacity constraints, the model might use orthogonal vectors to represent the quantitative response of each feature, with its own receptive field, to the input data. Forced to put them into lower dimensional superposition, the similarity matrix picks up both a wider diagonal stripe and the upper/lower diagonal ringing stripes. 

Finally, we also construct a simple physical model showing that the rippling and ringing arise even when the solution is found dynamically, whenever many vectors are packed into a small number of dimensions. We show the result of a simulation in which 100 points confined to a 6-dimensional hypersphere are subjected to attractive forces to their 6 closest neighbors on each side (matching the RMSE error of our probes) and repulsive forces to all other points. (To avoid boundary conditions, we use the topology of a circle instead of an interval.) On the right below is a heatmap exhibiting two rings, and on the left is a 3-dimensional projection of the 6-dimensional curve. There exists an interactive version in the web version of this paper here, and the reader is encouraged to experiment with reinitializing the points, switching the ambient dimension, and modifying the width of the attractive zone. Decreasing the attractive zone or increasing the embedding dimension both increase curvature (and the amount of ringing), and vice versa.<sup>16</sup> As the number of points on the curve grows and the attractive zone width shrinks (in relative terms), the curvature grows quite extreme, approaching a space-filling curve in the limit. 

Of particular interest is the result from setting the ambient dimension to 3<sup>17</sup> : the result is a curve similar to the seams of a baseball (below left, circle), which matches the topology observed for three intrinsically one-dimensional phenomena observed in [48, 17], of colors by hue, dates of the year, and years of the 20th century (which also exhibit dilation). Similar ripples were predicted to occur by Olah [58] and then observed by Gorton [21] in curve detector features in Inception v1. One of the earliest observations of ringing in a cosine similarity plot and rippled spiral/helix shape in a lowdimensional embedding was of the learned positional embeddings of tokens in GPT2 [75, 76]. We also find similar structure in other representations, which we study in More Sensory and Counting Representations in the appendix. 

> 16The simulation can sometimes find itself in local minima. Increasing the width of the attractive zone before decreasing it again usually solves this issue. 

> 17Optimization in dimension 3, unlike in higher dimensions, admits bad local minima, because a generic curve on the surface of a sphere self-intersects. To avoid this, either increase the zone width until you get a great circle, then decrease it, or do the optimization in 4D, then select 3D. 

10 



<!-- Start of picture text -->
rai Smltian race vsaton fe Dimensions: 30,4050 B70 80) Topoogy: NE ries!<br>... newex 6 —0 sont °<br>s  %<br>° %<br>%, 3<br>"Fe 9902" :<br>°o :<br>° :<br>°6 s<br>Peece®<br><!-- End of picture text -->

Figure 12: Screenshot of our interactive particle simulation. 



<!-- Start of picture text -->
Toy model English names or colours Years of the 20th century Dates ofthe yeor<br>fo,@ mye.it -SE = wy ayRe ayjar<br>Modell et al. (2025)<br><!-- End of picture text -->

Figure 13: Left curve is a locally optimal high-curvature embedding of the circle onto the 2-sphere. Right figures, reproduced with permission from _Modell et al._ [48], show 3-dimensional PCA projections of data or features related to colours, years, and dates. 

## **3 Sensing the Line Boundary** 

We now study how the character counting representations are used to determine if the current line of text is approaching the _line boundary_ . To detect the line boundary, the model needs to (1) determine the overall _line width_ constraint and (2) compare the current character count with the line width to calculate the characters remaining. 

### **3.1 Twisting with QK** 

We find that newline tokens have their own dedicated character counting features that activate based on the width of the line, counting the number of characters between adjacent newlines. 

To better understand how these representations are related, we train 150 probes for each possible value of “Line Width” like we did for “Character Count”. Using the attribution graph, we identify an attention head which activates boundary detection features. We visualize both sets of counting representations directly using the first 3 components of their joint PCA in the residual stream (left) and in the reduced QK space of this _boundary head_ (right).<sup>18</sup> 

> 18Specifically we multiply the line width probes through _WK_ and the character count probes through _WQ_ , and plot the points in the 3D PCA basis of their joint embedding. 

11 



<!-- Start of picture text -->
‘Atormen Between Character Count Probes an Line With Probes<br>ine esa soa seena x ce<br>at<br>ont, ws ||<br>| *<br>|I} \<br>ot<br>eeiYite 4h ahomtee .“ahaa,me . uf of p a.<br>aar af eeePe~All]in ai,<br>"1 " Stee. oo a<br>\ ee<br>POPCuAnt ane | b ANS Ss<br>Wyaa<br>© eamcata {seein ate 0 cemomewg  dntnee<br><!-- End of picture text -->

Figure 14: Boundary heads twist the representation of line width and character count to detect the line boundary. Left: Joint PCA of character count and line width probes. Right: Same after multiplying them through the corresponding QK weights of the boundary head. Range is from 40 (dark) to 150 (light). 

We find that this attention head “twists” the character count manifold such that character count _i_ is aligned with line width _k_ = _i_ + _ϵ_ . This causes the head to attend to the newline when the character count is just a bit _less_ than the line width, thereby indicating that the boundary is approaching. This algorithm is quite general, and enables this head to detect approaching line boundaries for arbitrary line widths!<sup>19</sup> 



<!-- Start of picture text -->
Probes Probes Cosine Sim through Qk Probes Cosine Sim through Qk<br>on. Cosine Sim ° (Boundary Head) on (Random Head) 1.00<br>204 20 ‘iy of 075<br>gol OX 0 i, oof! US om<br>5 i 0 ty, \ es NY oa §<br>B i00 N 100 100: N -025 &<br>320 K 120 220: NX one<br>Mo Nao 140. aN “07s<br>0 20 40 60 80 100 120 140 0 20 40 60 80 100 120 140 0 20 40 60 80 100 120 140 200<br>Previous Line Length Previous Line Length Previous Line Length<br><!-- End of picture text -->

Figure 15: Cosine similarity of previous line width and character count probes through different transforms. (Left) the identity map, (Center) QK of boundary head, (Right) QK of random head in the same layer. Boundary heads align the probes but with a small offset. 

This plot shows that 

- In the residual stream, probes for character count _i_ are maximally aligned with probes line width probes _k_ when _i_ = _k_ , but are not highly aligned in absolute terms – the maximum cosine sim is _∼_ 0.25. 

- In the QK space of the boundary head, the probes are maximally aligned on the offdiagonal _i < k_ , and are almost perfectly aligned in absolute terms – the maximum cosine sim is _≈_ 1. 

> 19This algorithm also generalizes to arbitrary kinds of separators (e.g., double newlines or pipes), as the QK circuit can handle the positional offset independently of the OV circuit copying the separator type. 

12 

- In the QK space of a random head, there is almost no structure between the probes. 

As a consequence of the ringing in the character count representations, we also observe ringing in the inner products (see Rippled Representations are Optimal above). The model is robust to these off-diagonal interference terms via the softmax applied to attention scores. 

### **3.2 Leveraging Multiple Boundary Heads** 

We find that the model actually uses multiple boundary heads, each twisting the manifolds by a different offset to implement a kind of “stereoscopic” algorithm for computing the number of characters remaining.<sup>20</sup> We attach more visualizations of boundary heads in the Appendix. 



<!-- Start of picture text -->
° Boundary Head HO (r=7.71) ° Boundary Head H1 (r=2.93) °. Boundary Head H2 (r=-2.71)<br>20 , / 20 Why 2 hy,<br>& ol \ | ae Mi. WS Nyy<br>5 Np My ' i<br>‘ i 60 “” « A,ih<br>*<br>5<br>B<br>8 100 100 100<br>oj | 120 120<br>40 - h 40 ] 140<br>0 7 © 6 60 10 120 140 0 70 4 @ a0 100 120 140 0 20 4 G0 m0 100 120 140<br>Previous Line Length Previous Line Length Previous Line Length<br><!-- End of picture text -->

Figure 16: Cosine similarity of line width and character count probes through three different boundary heads in the same layer with different amounts of twisting. Green line indicates the argmax for each row, and is used to calculate the average offset reported in the subtitles. 

To better understand each boundary head’s output, we train a set of probes for each value of characters remaining in the line (i.e., the line width _k_ minus the character count _i_ , restricted to _k − i <_ 40). For each boundary head, we show the proportion of attention on the newline, as well as the norm of each head’s output projected onto the probe space as a function of characters remaining. 

As predicted by our weights based analysis, we observe that boundary heads have distinct but overlapping response curves that “tile” the possible values of characters remaining. 



<!-- Start of picture text -->
Azenon Haina vs Caracas hams sorta Hand Oupus vs harcars araing<br>soars v2] tonto<br>= Srormes = Serie<br>==:Xo ig | emi ‘><br>3z<br>H ——<br>eB CharactersRemainng Characters Remaining<br><!-- End of picture text -->

Figure 17: Each boundary head’s response curve peaks at a different distance from the end of the line. 

It’s worth understanding why the model needs multiple boundary heads rather than just one. If the model relied only on boundary head 0, it couldn’t distinguish between 5 characters remaining and 

> 20There are also multiple sets of boundary heads at multiple layers that usually come in sets of _∼_ 3 with similar relative offsets (so not actually “stereo”). 

13 

17 characters remaining—both would produce similar outputs. By having each head’s output vary most significantly in different ranges, their sum achieves high resolution across the entire relevant range of “Characters Remaining” values. 

We can see this more clearly by plotting each head’s output in the first two principal components of the characters remaining space (which captures 92% of the variance). Head 0 shows large variance in the [0, 10] and [15, 20] ranges, Head 1 varies most in the [10, 20] range, and Head 2 varies most in the [5, 15] range. While no single head provides high resolution across the entire curve, their sum produces an evenly spaced representation that covers all values effectively. 



<!-- Start of picture text -->
Boundary Head Outputs in First Principal Components of Characters Remaining Space<br>© Heoso<br>@<br>©© HeadHead?Sum ot1 Key Heads ° ° 2° oe a<br>° °<br>ao<br>° ° s °<br>° . °<br>a ° °<br>S<br>2 ° 08 on<br>°° By BP %0 °<br>9 eo<br>a °° °° a<br>o. °°<br>“a °<br>2°<br>°e<br>°<br>°.<br>RET<br><!-- End of picture text -->

Figure 18: Each head’s output as a function of characters remaining, and their sum in the PCA basis. Individual head outputs are almost one-dimensional, while the sum is a two-dimensional curve. 

We validate the causal importance of this two-dimensional subspace by performing an ablation and intervention experiment. Specifically, we conduct the same experiments as before: ablate the subspace and measure its effect on loss by token (left) and precisely modulate the characters remaining estimate on the last token in the aluminum prompt by substituting mean activation vectors. 



<!-- Start of picture text -->
Loss Ablations of Different Subspaces Aluminum Prompt Intervention Results<br>00 :<br>20 508 i es<br>FHé a “hats Remaining Subspace | © —h° PK tiatantn<br>zSis Nextisnewline A3 i/XH ew<br>4 CharsNext RemainingNOT newline Subspace |=3-20>> :H K-_ auminom:Dewars<br>g Random Subspace 2 j Se pee<br>3° “e No centne Fas j soe<br>8 Random Subspace 8 : ~ “Subspace<br>Sos5 ® NextNOT newline B-30 fi Sbte<br>Bess] V7A?I —== d.modelrank2<br>0.0| t--—e-——~~“ -- === =| -40 :: ese RemainingOfiginal Chars<br>? 2 3 @ §&§ 6 @ 0 5m w 2 2 30<br>Subspace Dimension Patched Characters Remaining to Boundary<br><!-- End of picture text -->

Figure 19: Characters remaining subspace can be causally intervened upon. (Left) Ablating the subspace has a large effect only when the next token is a newline. (Right) We surgically intervene on the characters remaining space to modulate the prediction of the newline by subtracting the true characters remaining mean activation and adding in a patched characters remaining activations. Note that the completion “ aluminum.” requires ten characters to fit. 

### **3.3 The Role of the Extra Dimensions** 

We are now in a position to understand two distinct but related questions: (1) why these counting representations are multidimensional and (2) why multiple attention heads are required to compute these multidimensional representations. 

14 

**Geometric Computations** – A multi-dimensional representation enables the model to rotate position encodings using linear transformations—something impossible with one-dimensional representations. For instance, to detect an approaching line boundary, the model can rotate the position manifold to align with line width, then use a dot product to identify when only a few characters remain. With a 1D encoding, linear operations reduce to scaling and translation, so comparing position against line width would just multiply the two values, producing a monotonically increasing result with no natural threshold. Higher dimensions beyond 2D allow the manifold to pack more information through additional curvature. 

**Resolution** – For character counting, the model must distinguish between adjacent counts for a large range of character positions, as this determines whether the next word fits. In a one-dimensional representation, positions would be arranged along a ray, with each position separated by some constant _δ_ . To reliably distinguish adjacent positions above noise, we need _||v_ 42 _− v_ 41 _||_ = _δ_ to exceed some threshold. But with 150+ positions to represent, this creates an untenable choice: either use enormous dynamic range ( _||v_ 150 _|| ≫||v_ 1 _||_ ), which is problematic for transformer computations, or sacrifice resolution between adjacent positions. (Normalization blocks only exacerbate this effect: while points can be spaced far away on a ray if their norms get large enough, there is at most _π_ worth of angular distance along the projection of that ray onto the unit hypersphere.) Embedding the curve into higher dimensions solves this: positions maintain similar norms while being well-separated in the ambient space, achieving fine resolution without norm explosion (See Rippled Representations are Optimal above.) For counting the characters remaining, the dynamic range is smaller, and so the model is able to embed the representation in a smaller subspace as a result. 

To achieve the curvature for necessary high resolution, multiple attention heads are needed to cooperatively construct the curved geometry of the counting manifold. An individual attention head’s output is a linear combination of its inputs (weighted by attention and transformed by the OV circuit), and thus is fundamentally constrained by the curvature already present in those inputs. In the absence of MLP contributions to the counting representation, if the output manifold needs to exhibit substantial curvature, multiple attention heads need to coordinate—each contributing a piece of the overall geometric structure. We will see another example of distributed head computation in the section on the Distributed Character Counting Algorithm. 

### **3.4 A Discovery Story** 

How did we originally find this boundary detection mechanism? When we first computed an attribution graph, we saw several edges from the previous newline features and embedding to predictnewline features. QK attributions showed that the top key feature was a “the previous line was 40–60 characters long” feature and the top query feature was “the current character count is 35–50” feature. At any one time there were often multiple counting features active at different strengths, suggesting that these features might be discretizing a manifold. 

The boundary heads cause a family of boundary detecting features to activate in response to how close the current line is to the global line width. That is, they sense the approaching line boundary or the reverse index of the line count. Investigating these three sets of feature families led us to the count manifolds which they sparsely parametrize, and investigating the relevant attention heads let us find the boundary heads. 

Finally, we note that these boundary-sensing representations parallels a well-studied phenomenon in neuroscience: boundary cells [67], which activate at specific distances from environmental boundaries (e.g., walls). Both the artificial features and biological cells come in families with varied receptive fields and offsets. 

## **4 Predicting the Newline** 

The final step of the linebreak task is to combine the estimate of the line boundary with the prediction of the next word to determine whether the next word will fit on the line, or if the line should be broken. 

15 



<!-- Start of picture text -->
Attribution Graph View .<br>A newline Boundary i<br>in wrapped Head near line<br>text boundary<br>line width is<br>40-60 chars " | 7 35-50into charsline<br>line width is ENXK\ el\zl \n 25-45 chars<br>30-50 chars iy]y NA a \ AALSY \o into line<br>| ee a element with atomic number 13, also called<br><!-- End of picture text -->

Figure 20: Attribution graph representation of the boundary detection subcircuit. 

In the attribution graph for the aluminum prompt, we see exactly this merging of paths. The most influential feature<sup>21</sup> in the entire graph is a late feature that activates in contexts where the next word would cause the current line to exceed the overall line width. For our prompt, this feature upweights the probability of newline and downweights the probability of “aluminum.” The top two inputs to this break predictor feature are a “say aluminum” feature and “boundary detecting” feature that gets activated by the aforementioned boundary head. 



<!-- Start of picture text -->
omorphic with\ShBeked 2exceptions, J © aluminum<br>disk controller toWOFK correctly.) The) 7 a.<br>la, resulta directal§indirectament e |<br>lng too farin thelledirection.22 |n-nnnnnnnnnnunnnnmmnnnnnnnnnnnl (nextPredict word 4<br>ies for all intents\and purposes, andfi ‘too long<br>s frais, ayant unfiitte 2alcoomét<br>ise that infact hinders©> performance)<br>predict predict<br>aluminum newline<br>5-10 chars<br>aluminum away from<br>line limit<br>+++ @ element with atomic number 13 , also called |<br><!-- End of picture text -->

Figure 21: Attribution graph representation of the final linebreaking decision. 

While the boundary detector activates regardless of the next token length, break predictor features activate only if the next token will exceed the length of the current line (as in the Aluminum prompt), 

> 21Influence in the sense of influence on the logit node, as defined in Ameisen et al. [2] 

16 

and hence upweight the prediction of a newline.<sup>22</sup> We also see break _suppressor_ features, which only activate if the next token would just barely fit on the line, and hence downweight the prediction of a newline. Both break predictors and suppressors come in larger feature families, which we display in the Appendix. 



<!-- Start of picture text -->
soundary Detector Break editor bras Supcessor<br>i’ “yt ron om<br>pe om pe ye cn<br>ie ow fe > eae oe<br>a 3 ome 0s<br>Fro . E fw \ om Ee<br><!-- End of picture text -->

Figure 22: Average activations of three features based on true next token character length and the characters remaining in a line (line width - character count). 

### **4.1 Joint Geometry Enables Easy Computation** 

What is the geometry underlying the model’s ability to determine if the next token will fit on the line? Put another way, how is the break predictor feature above constructed from the boundary detector and next-word features? 

To study this, we compute the average activations at the end of the model ( _∼_ 90% depth) across all tokens for all values of characters remaining _i_ and next token lengths _j_ .<sup>23</sup> By performing a PCA on the combination of mean vectors, we see that the two counts are arranged in orthogonal subspaces with only moderate curvature. Note, this lower dimensional geometry may suffice here because the dynamic range of the count is much smaller. 



<!-- Start of picture text -->
se rd ag CtCroomReig Reps on tn Dichn Byer tng"ieSensingeySee<br>+<br>| } 2<br>20%<br>1 é £3334 % “s<br>¢ 33 3 8 FBX<br>i °<br>© mening 0 onemnteey © mete Semmens © spin tnt<br><!-- End of picture text -->

Figure 23: Low dimensional projections of next token character length and characters remaining counting manifolds for 1 (dark) to 15 (light) characters. (Left) The PCA of their union. (Right) The PCA of all their pairwise combinations. The orthogonal representations make the correct newline decision linearly separable. 

> 22These features also sometimes activate on zero-width _modifier_ tokens (e.g., a token which indicates the first letter of the following token should be capitalized) that need to be adjacent to the modified token, and the modified token is sufficiently long to go over the line limit (e.g. for “Aluminum” instead of “aluminum”). 

> 23We use the true next non-newline token as the label. This is an approximation because it assumes that the model perfectly predicts the next token. 

17 

Now consider the pairwise sum of each possible character-remaining vector _i_ and next-token-length vector _j_ .<sup>24</sup> Since these counts are arranged orthogonally, the decision to break the line _i − j ≥_ 0 corresponds to a simple separating hyperplane. In other words, the prediction to break the line is made trivial by the underlying geometry! 

When we use the separating hyperplane from the PCA of these average embeddings on real data, we achieve an AUC of 0.91 on the ground truth of whether the next token should be a newline. This reflects both the error of the three dimensional classifier _and_ the error from Haiku’s estimates of the next token. 

If the length of the most likely next word is linearly represented, this scheme would allow the model to predict newlines when that word is longer than the length remaining in the line. One could imagine a more general mechanism where the model comprehensively redirects the probability mass from all words that exceed the line limit to the newline. Claude 3.5 Haiku does not seem to leverage such a mechanism: when we compare the predicted distribution of tokens at the end of a line to the distribution on an identical prompt with the newlines stripped, we find them to be quite different. 

## **5 A Distributed Character Counting Algorithm** 

Having described how the various character counting representations are _used_ , the last big remaining question is: how are they _computed_ ? 

We will show how Haiku uses many attention heads across multiple layers to cooperatively compute an increasingly accurate estimate of the character count. This turned out to be the most complicated mechanism we studied, though there are many similarities with the boundary detection mechanism. 



<!-- Start of picture text -->
!f _<br><!-- End of picture text -->

Figure 24: Comparison of Layer 0 (left) vs Layer 1 (right) average attention outputs in the PCA basis of the character count probes from 1 (dark) to 150 (light) characters. In each layer, the outputs from each head tile the space. In Layer 0, each head output is almost 1-dimensional, while in Layer 1 heads display more curvature (which they got from Layer 0!). 

To get an intuitive understanding of the behavior of the heads important for counting, we project their outputs into the PCA space of the line character count probes.<sup>25</sup> Layer 0 heads each write along what appears as a ray when visualized in the first 3 principal components—it is their sum that generates a curved manifold. Layer 1 heads instead output curves which combine to produce an increasingly complex manifold. They appear responsible for sharpening the Layer 0 representation 

> 24This sum is principled because both sets of vectors are marginalized data means, so collectively have the mean of the data, which we center to be 0. 

> 25We display the average outputs over many prompts. 

18 

and thus the estimate of the count. We find that the _R_<sup>2</sup> for the character count prediction<sup>26</sup> of the 5 key Layer 0 heads is 0.93, compared to 0.97 using 11 heads in the first two layers. 

### **5.1 Embedding Geometry** 

To understand how the character count is computed, we start at the very beginning: the embedding matrix. 

As before, we can train probes or compute the average weights for every distinct token length in the embedding. We visualize the _token_ character count probes for character length 1–14 and visualize their top principal components. Using the first 3 principal components, which capture 70% of the variance, we see that embedding character counts are arranged in a circular pattern (PC1 vs PC2) with an oscillating component (PC3). This pattern is consistent with the ones observed in Rippled Representations are Optimal. 



<!-- Start of picture text -->
PCA of We averaged by Character Length (Var: 82.6%)<br>é 14<br>4 12 3s<br>10 8<br>2 3<br>=85 é Pid us 6 o8o<br>&z= =<br>é . 3 5<br>oS ry a8<br>2<br>Pe:"2 0oj3)<br>26%) ge x<br><!-- End of picture text -->

Figure 25: PCA of embedding vectors in _WE_ averaged by token character length. 

As with all of the counting manifolds, we also find features that discretize this space into overlapping notions of short, medium, and long words. 

### **5.2 Attention Head Outputs Sum To Produce the Count** 

To understand the counting mechanism, we will work backwards from the summed attention outputs to the embedding. Notably, we: 

- **Ignore MLPs** – The attention head outputs affect the character count representation 4 _×_ more than the MLPs, so we restrict our focus to attention; 

- **Focus on First Two Layers** – Even after layer 0, counting probes have reasonable accuracy and there are coarse positional features. Therefore, we focus on how attention transforms the embeddings into the count and how layer 1 further refines this representation. 

We can decompose the sum above into the contribution from the output of each individual head in layer 0.<sup>27</sup> Under this lens, we see each head performing a relatively low rank computation akin to a classification. 

How do individual heads implement this behavior? We can break down the behavior of an individual head by analyzing its QK circuit (where it attends) and OV circuit (the linear transformation from the embeddings to the output) [15]. 

> 26The prediction is the argmax of the head outputs projected on the character count probes. 

> 27We omit a previous token head for visual presentation. 

19 



<!-- Start of picture text -->
100 Lo Head Sumco7 PoePeco samaLo Head'  Sum vs Tue Char;7i  Count 711<br>oo), oo i i i<br>cy tee | | |<br>z€<br>=0 8 oo} | / i i<br>” aor<br>7 oe<br>ol Mtl i :<br>° % +»contest postion % (token)© 5% © cr ‘content potion ake)<br><!-- End of picture text -->

Figure 26: The summed output of 5 important Layer 0 heads on one prompt by token. (Left) The inner product of the summed attention outputs and the character counting probes; (Right) How the argmax of this product compares to the true line count. Context position starts at the first newline, with newlines denoted with dashes. 



<!-- Start of picture text -->
i oHo Head OutputsLoin projected on Line CharactervorCount Probes Low3<br>100 | 100 Ti Ti ' 100 100 iHi F * =<br>ge 80 iy 5 !|} 20 20 |l '<br>iH of | ' |e| oy i aj<br>#40 40 J~ 4 fl<br>3 , oF é<br>20 20 j 4 2 2 i | i ;<br>of 0 HhoBo. o i 1 + =<br>° conte2  poston© token) © "9%center postion» token) @ $2conte poston # token) © "9 content poston0 token)<br><!-- End of picture text -->

Figure 27: The individual outputs of 4 important Layer 0 heads on one prompt projected onto the character count probes. 



<!-- Start of picture text -->
Token Position vs Attention to Newline<br>wo — head0<br>°<br>& teasa<br>z — Head2<br>Zoe —— Head3<br>2<br>8<br>ie<br>Sos<br>2<br>i<br>B02:<br>E5<br>Fa<br>oo<br>° i ry 5 3 %<br>‘Token Position Within Line<br><!-- End of picture text -->

Figure 28: The average attention to the previous newline as a function of the token index in the line. Like boundary heads, these counting heads specialize with different positional offsets. 

20 

**QK Circuit** . Each head _h_ uses the previous newline as an “attention sink,” such that for some number of tokens after the newline ( _sh_ ), the head just attends to the newline. After _sh_ tokens, the head begins to smear its attention over its receptive field, which goes up to a maximum of _rh_ tokens. 

**OV Circuit** . The OV circuit coordinates with the QK circuit to create a heuristic estimate based on the number of tokens in the line multiplied by the average token length ( _µc ≈_ 4), with an additional length correction term. When attending to the newline, each head upweights the average token length multiplied by the head’s sink size: _sh × µc_ characters. If no attention is paid to the newline, then from the perspective of the head, the current token must be at least _sh_ + _rh_ tokens into the line and should upweight ( _sh_ + _rh_ ) _× µc_ character outputs. Finally, the OV circuit applies an additional correction depending on whether the tokens in the receptive field are above or below average in length. 

Below, we include a detailed walkthrough of L0H1. 



<!-- Start of picture text -->
LOH1 Head Analysis<br>=m | H !<br>| | | 7<br>- J<br>oe<br>CASES What is the head attendingto?<br>, i © Newine<br>.<br>@ Average Tokens +<br>p © short Tokens =<br>© Long Tokens - °<br><!-- End of picture text -->

Figure 29: The QK and OV circuit of counting head L0H1. Top left: the head output projected onto the character counting probes for 64 tokens of a single prompt (truncated to the first newline). Bottom left: the attention pattern (transposed of the canonical ordering). Top right: the average embedding vectors projected onto the character count probes via the OV matrix. Bottom right: a summary of the overall computation. 

For a more detailed analysis of each head, see The Mechanics of Head Specialization. Layer 1 attention heads perform a similar operation, but additionally leverage the initial estimate of the character count (see Layer 1 Head OVs). 

### **5.3 Computing the Line Width** 

To compute the line width, the model seems to use a similar distributed counting algorithm to count the characters between adjacent newlines. However, one subtlety that we do not address in this work is how the line width is actually aggregated. It is possible that the model computes a global line width by taking the max over all line lengths in the document or uses an exponentially weighted moving average of the last several line lengths. We do note that the line width uses a partially disjoint set of heads, likely because the “attend to previous newline as a sink” mechanism needs modification when the _current_ token is also a newline. 

21 

## **6 Visual Illusions** 

Humans are susceptible to “visual illusions” in which contextual cues can modulate perception in seemingly unexpected ways. Famous examples include the M¨uller-Lyer illusion, in which arrows placed on the ends of a line can alter the perceived length of the line [33]; the Ponzo and Sander illusions which also modulate perceived line length [77]; and others [64]. 



<!-- Start of picture text -->
Miller-Lyer Ponzo Sander<br><!-- End of picture text -->

Figure 30: Classic visual illusions in which perception of line-length is modulated. 

Can we use our understanding of the character counting mechanism to construct a “visual illusion” for language models? 

To get started, we took the important attention heads for character counting and investigated what other roles they perform on a wider data distribution. We identified instances in which heads that normally attend from a newline to the previous newline would instead attend from a newline to the two-character string <mark>`@@`</mark> . This string occurs as a delimiter in git diffs, a circumstance in which you might want to start your line count at a location other than the newline: 

```
[Enter]@@-14,30+31,24@@exportinterfaceClaudeCodeIAppTheme\{[Enter]
```

But what happens when this sequence appears outside of a git diff context—for instance, if we insert <mark>`@@`</mark> in the aluminum prompt without changing the line length? 



<!-- Start of picture text -->
Original Prompt Insert @@<br>After gallium was ruled out due to melting point, After gallium was ruled out due 00 melting point,<br>the engineers on the project chose the chemical the engineers @@ the project chose the chemical<br>element with atomic nunber 13, also called element with atomic number 13, also called<br>Original Prediction New Prediction<br>2) 0.79 (a) 0.22<br>aluminum) 0.12 aluminum) 0.66<br><!-- End of picture text -->

Figure 31: Prediction comparison for original aluminum prompt (left) with visual illusion insert (right). 

We find that it does modulate the predicted next token, disrupting the newline prediction! As predicted, the relevant heads get distracted: whereas with the original prompt, the heads attend from newline to newline, in the altered prompt, the heads also attend to the <mark>`@@`</mark> . 

How specific is this result: does any pair of letters nonsensically inserted into the prompt fully disrupt the newline prediction? We analyzed the impact of inserting (at the same two positions) 180 different two-character sequences, half of which were a repeated character. We found that while most inserted sequences moderately impact the probability of predicting a newline, newline usually remains the top prediction. There was also no clear difference between sequences consisting of the same or different characters. However, a few sequences substantially disrupted newline prediction, most of which appeared to be related to code or delimiters: <mark>`’’ >>`</mark> _}}_ `;|` <mark>`|| ’, @@` .</mark> 

We further analyzed the extent to which there was a relationship between ‘distraction’ of the important attention heads and the impact on the newline prediction. Indeed we found that many of the sequences with potent modulation of newline probability—and especially code-related character pairs—also exhibited substantial modulation of attention patterns. 

While in the aluminum prompt the task is implicit, this illusion generalizes to settings where the comparison task is made explicit. These direct comparisons are perhaps more analogous to the Ponzo, Sander, and M¨uller-Lyer illusions, where the perception and comparison is more direct. 

22 



<!-- Start of picture text -->
LiHO original Pattern LaHo Pattern with @@ Insertion<br>° Key . Key<br><!-- End of picture text -->

Figure 32: Insertion of <mark>`@@`</mark> ‘distracts’ an attention head which normally attends from _\_ n back to the previous _\_ n. (left) Original attention pattern (truncated). (right) Attention pattern (truncated) with <mark>`@@`</mark> insertion. Now it also attends back to the <mark>`@@`</mark> . 



<!-- Start of picture text -->
12 6)<br>S<br>ee)<br>2cge we<br>gs 08 7<br>22EoaS wePy 0?eeee e<br>38 06 rr<br>Bg “ OS Seen Pee os<br>Es . oF eee 8<br>ae 0.4 22,285 Oy ed .<br>& wees<br>£ © 08 PR eees<br>0.2 o<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Newline probability<br><!-- End of picture text -->

Figure 33: Insertion of most pairs of characters only moderately impacts the probability of predicting a newline. A subset of pairs, most of which appear related to code or delimiters, substantially disrupt newline prediction. The impact on newline prediction (originally 0.79) is correlated with how much inserted tokens ‘distracts’ character counting attention heads. 

23 



<!-- Start of picture text -->
Original Prompt Insert @@<br>(Human: Answer with one letter. Which of the Human: Answer with one letter. Which of the<br>following sentences is shorter (in characters)? following sentences is shorter (in characters)?<br>(a) The dog barked loudly and chased the car (a) The dog barked loudly and chased the car.<br>(b) The bird flew away quickly to avoid the hawk. (b) The bird @@ew away quickly to avoid the hawk.<br>Assistant: ( ) (Assistant: (<br>Original Prediction New Prediction<br>@ 080 @ 015<br>® 0.09 ® 078<br><!-- End of picture text -->

Figure 34: Prediction for explicit length comparison task with normal prompt (left) and with visual illusion insert (right). 

These effects are robust to multiple choice orderings. Moreover, if the length of the text following the <mark>`@@`</mark> exceeds that of the alternative choice, the alternative choice is selected as being shorter. 

While we are not claiming any direct analogy between illusions of human visual perception and this alteration of line character count estimates, the parallels are suggestive. In both cases we can see the broader phenomena of contextual cues, and the application of learned priors about those cues, modulating estimates of object properties of entities. In the human case, priors such as threedimensional perspective can influence perception of object size, or color constancy can influence estimates of luminance (such as in the checker shadow illusion). Here, one possible interpretation of our results is that mis-application of a learned prior, including the role of cues such as <mark>`@@`</mark> in git diffs, can also modulate estimates of properties such as line length. 

## **7 Related Work** 

**Objective** . This work is at the intersection of LLM “biology” (making empirical observations about what is going on inside models; e.g. [44, 63]) and low level reverse engineering of neural networks (attempting to fully characterize an algorithm or mechanism; e.g. [55, 72, 52, 41]). Methodologically, our work makes heavy use of attribution graphs [2, 20, 14] with QK attributions [36] built on top of crosscoders [43]. 

**Linebreaking** . Michaud _et al._ [46] identified linebreaking in fixed-width text as one of the top 400 “quanta” of model behavior in the smallest model (70m parameters) in the Pythia suite. 

**Position.** Prior interpretability work on positional mechanism has largely focused on _token_ position (e.g., [75, 76, 71, 10, 27]). These works have shown that there exist MLP neurons [71, 27], SAE features [10], and learned position embeddings [75] with periodic structure encoding absolute token position. Our work illustrates how a model might also want to construct non-token based position schemes that are more natural for many downstream prediction tasks. 

Others have also studied, even going back to LSTMs, the existence of mechanisms in language models for controlling the length of output responses [66, 49], as well as performed more theoretical analyses of the space of counting algorithms [69, 8]. 

**Geometry and Feature Manifolds.** Beyond position, there has been extensive work in understanding the geometric representation of numbers, especially in toy models (e.g., [52, 78, 50]) and in the context of arithmetic in LLMs (e.g., [68, 79, 53, 37, 34]). Collectively, these works have shown that both real LLMs and toy transformers learn periodic representations [79, 37, 34], with numbers arranged in a helix to enable certain matrix multiplication based addition algorithms [52, 37], and that these representations are provably optimal in certain settings [50]. In our context, we similarly observe helical representations [37], numeric dilation [1], and distributed algorithms across components that collectively implement a correct computation [28, 34]. 

Multidimensional features with clear geometric structure have been found in more natural contexts [23, 17, 48], like in the representation and computation of certain ordinal relationships (e.g., months of the year). In vision models, curve detector neurons [5] and features [22] have been especially 

24 

well studied and closely resemble the kind of discretization we observe with the families of character counting features. Many other topics have received interpretability analysis of the underlying geometry, such as grammatical relations [31, 11], multilingual representations [7], truth [45], binding [18], refusal [74], features [32, 42], and hierarchy [59], though more conceptual research is needed [73]. 

Perhaps most relevant is recent work from Modell et al. [48], who provide a more formal notion of a feature manifold, and propose that cosine similarity encodes the intrinsic geometry of features. When testing their theory, they observe highly structured and interpretable data manifolds that have ripples and dilation, similar to our counting manifolds. These observations raise a methodological challenge in how to best capture data with different structure (see e.g. [32, 47, 35]), but also the exciting hypothesis that many naturally continuous variables (e.g., [30, 25]) exist in more organized manifolds. 

**Biological Analogues.** The geometric and algorithmic patterns we observe have suggestive parallels to perception in biological neural systems. Our character count features are analogous to place cells on a 1-D track [51] and our boundary detecting features are analogous to boundary cells [67]. These features exhibit dilation—representing increasingly large character counts activating over increasingly large ranges—mirroring the dilation of number representations in biological brains [13, 62]. Moreover, the organization of the features on a low dimensional manifold is an instance of a common motif in biological cognition (e.g., [61]). While the analogies are not perfect, we suspect that there is still fruitful conceptual overlap from increased collaboration between neuroscience and interpretability [70, 29, 40]. 

## **8 Discussion** 

In this paper, we studied the steps involved in a large model performing a naturalistic behavior. The linebreaking task, frequently encountered in training, requires the model to represent and compute a number of scalar quantities involving position in character count units that are not explicit in its input or output<sup>28</sup> , then integrate those values with the outputs of complex semantic circuits (that predict the next proper word) to predict the next token. We found sparse features corresponding to each important step of the computation, and for those steps involving scalar quantities, we were able to find a geometric description that significantly simplified the interpretation of the algorithm used by the model. We now reflect on what we learned from that process: 

**Naturalistic Behavior and Sensory Processing** . Deep mechanistic case studies benefit from choosing behaviors that the model performs consistently well, as these are more likely to have crisper mechanisms. This means prioritizing tasks that are natural in pretraining over tasks that seem natural to human investigators, and ideally, that are easily supervisable. As in biological neuroscience, perceptual tasks are often both natural and easy to supervise for interpretability (e.g., it is easy to modify the input in a programmatic way). Although we sometimes describe the early layers of language models as responsible for “detokenizing” the input [16, 26, 19, 38], it is perhaps more evocative to think of this as perception. The beginning of the model is really responsible for _seeing_ the input, and much of the early circuitry is in service of sensing or perceiving the text similar to how early layers in vision models implement low level perception [55, 39]. 

**The Utility of Geometry.** Many of the representations and computations we studied had elegant geometric interpretations. For example, the counting manifolds are the result of an optimal tradeoff between capacity and resolution, with deep connections to space-filling curves and Fourier features. The boundary head twist was especially beautiful, and after discovering one such head, we were able to correctly predict that there would need to be additional heads to provide curvature in the output. The distributed character counting algorithm was more complex, but we were still able to clarify our view by studying linear actions on these manifolds. For other computations, like the final breaking decision, the linear separation was clearly a part of the story but there must be some additional complexity we were not able to see yet to handle multitoken outputs. For the more semantic operations, we purely relied on the feature view. Of course, describing any behavior in full is immensely complicated, and there is a long list of possible subtleties we did not study: how the model accounts for uncertainty in its counting, its mechanism for estimating the line width given multiple prior lines of 

> 28Tokens do not come annotated with character counts, and there are no vertical bars on the page showing the line width. 

25 

text, how it adapts to documents with variable line width, how it handles multiple plausible output tokens of different lengths or multitoken words, or various special cases (e.g., a L<sup>A</sup> TEX _\_ footnote _{}_ or a markdown link). For the inspired, we share transcoder attribution graphs for a fixed-width line break prompt on Gemma 2 2B and Qwen 3 4B, using the new neuronpedia interactive interface. 

**Unsupervised Discovery** It likely would not have been possible to develop this clarity if it were not for the unsupervised sparse features. In fact, when we started this project, we attempted to just probe and patch our way to understanding, but this turned out poorly. Specifically, we did not understand what we were looking for (e.g. we didn’t know to distinguish line width vs. character count), where to look for it (e.g., we didn’t expect line width to only be represented on the newline), or how to look for it (we started by training 1-D linear regression probes). However, after identifying some relevant features but before spending substantial effort systematically characterizing their activity profiles, we were also confused by what they were representing. We saw dozens of features that were vaguely about newlines and linebroken text, but their differences were not obvious from flipping through the activating examples. Only after we tested these features on synthetic datasets did their role in the graph and the underlying computation become clear. We suspect better automatic labels [3, 60, 24] enhanced with agentic workflows [65, 4] would accelerate this work, especially in less verifiable domains. 

**Feature-Manifold Duality.** The discrete feature and geometric feature-manifold perspectives offer dual lenses on the same underlying object. For example, in this work the model’s representation of character count can be completely described (modulo reconstruction error) by the activities of the features we identified, where the action of the boundary heads is described by virtual weights that expand out the feature interactions via attention head matrices. The same character count representation can be described by a 1-dimensional feature manifold – a curve in the residual stream parametrized by the character count variable – where linear action of the boundary heads is described by continuous “twisting” of the manifold. In general, geometric structures learned by the model will likely admit both global parametrizations and local discrete approximations. 

**The Complexity Tax.** Despite this duality, the descriptions produced by the two perspectives differ in their simplicity. The discrete features shatter the model into many pieces, producing a complex understanding of the computation. This seems like a general lesson. It seems like discrete features and attribution graphs may provide a true description of model computation, which can be found in an automated way using dictionary learning. Getting any true, understandable description of the computation is a very non-trivial victory! However, if we stop there, and don’t understand additional structure which is present, we pay a _complexity tax_ , where we understand things in a needlessly complicated way. In the line breaking problem, constructing the manifold paid down this tax, but one could imagine other ways of reducing the interpretation burden. 

**A Call for Methodology.** Armed with our feature understanding, we were able to directly search for the relevant geometric structures. This was an existence proof more than a general recipe, and we need methods that can automatically surface simpler structures to pay down the complexity tax. In our setting, this meant studying feature manifolds, and it would be nice to see unsupervised approaches to detecting them. In other cases we will need yet other tools to reduce the interpretation burden, like finding hierarchical representations [12] or macroscopic structure [54] in the global weights [2]. 

**A Call for Biology.** The model must perform other elegant computations. We can find these by starting with a specific task the model performs well, study this from multiple perspectives, develop methodology to answer the remaining questions, and relentlessly attempt to simplify our explanations. Because the investigation is grounded in specific examples of a behavior, it provides a fast feedback loop, can shed light on weaknesses of existing methods and inspire new ones, and can sharpen our conceptual language for understanding neural networks. We would be excited to see more deep case studies that adopt this approach. 

## **References** 

- [1] HV AlquBoj, Hilal AlQuabeh, Velibor Bojkovic, Tatsuya Hiraoka, Ahmed Oumar El-Shangiti, Munachiso Nwadike, and Kentaro Inui. Number representations in llms: A computational parallel to human perception. _arXiv preprint arXiv:2502.16147_ , 2025. 

26 

- [2] Emmanuel Ameisen, Jack Lindsey, Adam Pearce, Wes Gurnee, Nicholas L. Turner, Brian Chen, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah, and Joshua Batson. Circuit tracing: Revealing computational graphs in language models. _Transformer Circuits_ , 2025. URL `https://transformer-circuits.pub/2025/attribution-graphs/methods.html` . 

- [3] Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models, 2023. URL `https://openaipublic.blob.core.windows.net/ neuron-explainer/paper/index.html` . 

- [4] Trenton Bricken, Rowan Wang, Sam Bowman, Euan Ong, Johannes Treutlein, Jeff Wu, Evan Hubinger, and Samuel Marks. Building and evaluating alignment auditing agents. `https: //alignment.anthropic.com/2025/automated-auditing/` , July 2025. 

- [5] Nick Cammarata, Gabriel Goh, Shan Carter, Ludwig Schubert, Michael Petrov, and Chris Olah. Curve detectors. _Distill_ , 2020. URL `https://distill.pub/2020/circuits/ curve-detectors` . 

- [6] Nick Cammarata, Gabriel Goh, Shan Carter, Chelsea Voss, Ludwig Schubert, and Chris Olah. Curve circuits. _Distill_ , 2021. URL `https://distill.pub/2020/circuits/ curve-circuits` . 

- [7] Tyler A Chang, Zhuowen Tu, and Benjamin K Bergen. The geometry of multilingual language model representations. _arXiv preprint arXiv:2205.10964_ , 2022. 

- [8] Yingshan Chang and Yonatan Bisk. Language models need inductive biases to count inductively. _arXiv preprint arXiv:2405.20131_ , 2024. 

- [9] David Chanin, James Wilken-Smith, Tom´aˇs Dulka, Hardik Bhatnagar, and Joseph Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. _arXiv preprint arXiv:2409.14507_ , 2024. URL `https://arxiv.org/pdf/2409.14507` . 

- [10] Bilal Chughtai and Yeu-Tong Lau. Understanding positional features in layer 0 SAEs, 07 2024. URL `https://bilalchughtai.co.uk/pos-sae/` . Blog post. 

- [11] Andy Coenen, Emily Reif, Ann Yuan, Been Kim, Adam Pearce, Fernanda Vi´egas, and Martin Wattenberg. Visualizing and measuring the geometry of bert. _Advances in Neural Information Processing Systems_ , 32, 2019. URL `https://proceedings.neurips.cc/paper_files/ paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf` . 

- [12] Val´erie Costa, Thomas Fel, Ekdeep Singh Lubana, Bahareh Tolooshams, and Demba Ba. From flat to hierarchical: Extracting sparse representations with matching pursuit. _arXiv preprint arXiv:2506.03093_ , 2025. 

- [13] Stanislas Dehaene. The neural basis of the weber–fechner law: a logarithmic mental number line. _Trends in cognitive sciences_ , 7(4):145–147, 2003. 

- [14] Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. Transcoders find interpretable llm feature circuits. _Advances in Neural Information Processing Systems_ , 37:24375–24410, 2025. URL `https://arxiv.org/abs/2406.11944` . 

- [15] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. URL `https://transformer-circuits.pub/2021/framework/index.html` . 

27 

- [16] Nelson Elhage, Tristan Hume, Catherine Olsson, Neel Nanda, Tom Henighan, Scott Johnston, Sheer ElShowk, Nicholas Joseph, Nova DasSarma, Ben Mann, Danny Hernandez, Amanda Askell, Kamal Ndousse, Andy Jones, Dawn Drain, Anna Chen, Yuntao Bai, Deep Ganguli, Liane Lovitt, Zac Hatfield-Dodds, Jackson Kernion, Tom Conerly, Shauna Kravec, Stanislav Fort, Saurav Kadavath, Josh Jacobson, Eli Tran-Johnson, Jared Kaplan, Jack Clark, Tom Brown, Sam McCandlish, Dario Amodei, and Christopher Olah. Softmax linear units. _Transformer Circuits Thread_ , 2022. URL `https://transformer-circuits.pub/2022/solu/ index.html` . 

- [17] Joshua Engels, Eric J Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not all language model features are one-dimensionally linear. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL `https://openreview.net/forum?id= d63a4AM4hb` . 

- [18] Jiahai Feng and Jacob Steinhardt. How do language models bind entities in context? _arXiv preprint arXiv:2310.17191_ , 2023. URL `https://arxiv.org/pdf/2310.17191` . 

- [19] Javier Ferrando and Elena Voita. Information flow routes: Automatically interpreting language models at scale. _arXiv preprint arXiv:2403.00824_ , 2024. URL `https://arxiv.org/pdf/ 2403.00824` . 

- [20] Xuyang Ge, Fukang Zhu, Wentao Shu, Junxuan Wang, Zhengfu He, and Xipeng Qiu. Automatically identifying local and global circuits with linear computation graphs. _arXiv preprint arXiv:2405.13868_ , 2024. URL `https://arxiv.org/pdf/2405.13868` . 

- [21] Liv Gorton. Curve detector manifolds in inceptionv1, 8 2024. URL `https://livgorton. com/curve-detector-manifolds/` . 

- [22] Liv Gorton. The missing curve detectors of inceptionv1: Applying sparse autoencoders to inceptionv1 early vision. _arXiv preprint arXiv:2406.03662_ , 2024. URL `https://arxiv. org/pdf/2406.03662` . 

- [23] Rhys Gould, Euan Ong, George Ogden, and Arthur Conmy. Successor heads: Recurring, interpretable attention heads in the wild, 2023. URL `https://arxiv.org/pdf/2312.09230` . 

- [24] Yoav Gur-Arieh, Roy Mayan, Chen Agassy, Atticus Geiger, and Mor Geva. Enhancing automated interpretability with output-centric feature descriptions. _arXiv preprint arXiv:2501.08319_ , 2025. 

- [25] Wes Gurnee and Max Tegmark. Language models represent space and time, 2024. URL `https://arxiv.org/pdf/2310.02207` . 

- [26] Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. _arXiv preprint arXiv:2305.01610_ , 2023. URL `https://arxiv.org/pdf/2305.01610` . 

- [27] Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, and Dimitris Bertsimas. Universal neurons in gpt2 language models. _arXiv preprint arXiv:2401.12181_ , 2024. URL `https://arxiv.org/pdf/2401.12181` . 

- [28] Michael Hanna, Ollie Liu, and Alexandre Variengien. How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. _Advances in Neural Information Processing Systems_ , 36:76033–76060, 2023. URL `https://proceedings.neurips.cc/paper_files/paper/2023/file/ efbba7719cc5172d175240f24be11280-Paper-Conference.pdf` . 

- [29] Zhonghao He, Jascha Achterberg, Katie Collins, Kevin Nejad, Danyal Akarca, Yinzhu Yang, Wes Gurnee, Ilia Sucholutsky, Yuhan Tang, Rebeca Ianov, et al. Multilevel interpretability of artificial neural networks: leveraging framework and methods from neuroscience. _arXiv preprint arXiv:2408.12664_ , 2024. 

- [30] Benjamin Heinzerling and Kentaro Inui. Monotonic representation of numeric properties in language models. _arXiv preprint arXiv:2403.10381_ , 2024. 

28 

- [31] John Hewitt and Christopher D Manning. A structural probe for finding syntax in word representations. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4129–4138, 2019. doi: 10.18653/v1/N19-1419. URL `https://aclanthology.org/N19-1419.pdf` . 

- [32] Sai Sumedh R Hindupur, Ekdeep Singh Lubana, Thomas Fel, and Demba Ba. Projecting assumptions: The duality between sparse autoencoders and concept geometry. _arXiv preprint arXiv:2503.01822_ , 2025. 

- [33] Catherine Q Howe and Dale Purves. The m¨uller-lyer illusion explained by the statistics of image–source relationships. _Proceedings of the National Academy of Sciences_ , 102(4):1234– 1239, 2005. 

- [34] Xinyan Hu, Kayo Yin, Michael I Jordan, Jacob Steinhardt, and Lijie Chen. Understanding in-context learning of addition via activation subspaces. _arXiv preprint arXiv:2505.05145_ , 2025. 

- [35] Xinting Huang and Michael Hahn. Decomposing representation space into interpretable subspaces with unsupervised learning. _arXiv preprint arXiv:2508.01916_ , 2025. 

- [36] Harish Kamath, Emmanuel Ameisen, Isaac Kauvar, Rodrigo Luger, Wes Gurnee, Adam Pearce, Sam Zimmerman, Joshua Batson, Thomas Conerly, Chris Olah, and Jack Lindsey. Tracing attention computation through feature interactions. _Transformer Circuits Thread_ , 2025. URL `https://transformer-circuits.pub/2025/attention-qk/index.html` . 

- [37] Subhash Kantamneni and Max Tegmark. Language models use trigonometry to do addition, 2025. URL `https://arxiv.org/pdf/2502.00873` . 

- [38] Vedang Lad, Jin Hwa Lee, Wes Gurnee, and Max Tegmark. The remarkable robustness of llms: Stages of inference? _arXiv preprint arXiv:2406.19384_ , 2024. 

- [39] Michael Lepori, Alexa Tartaglini, Wai Keen Vong, Thomas Serre, Brenden M Lake, and Ellie Pavlick. Beyond the doors of perception: Vision transformers represent relations between objects. _Advances in Neural Information Processing Systems_ , 37:131503–131544, 2024. 

- [40] Anna Leshinskaya, Taylor Webb, Ellie Pavlick, Jiahai Feng, Gustaw Opielka, Claire Stevenson, and Idan A Blank. Cognitively inspired interpretability in large neural networks. In _Proceedings of the Annual Meeting of the Cognitive Science Society_ , volume 47, 2025. 

- [41] Belinda Z Li, Zifan Carl Guo, and Jacob Andreas. (how) do language models track state? _arXiv preprint arXiv:2503.02854_ , 2025. 

- [42] Yuxiao Li, Eric J Michaud, David D Baek, Joshua Engels, Xiaoqing Sun, and Max Tegmark. The geometry of concepts: Sparse autoencoder feature structure. _Entropy_ , 27(4):344, 2025. 

- [43] Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, and Christopher Olah. Sparse crosscoders for cross-layer features and model diffing, 2024. URL `https://transformer-circuits.pub/2024/crosscoders/index.html` . 

- [44] Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah, and Joshua Batson. On the biology of a large language model. _Transformer Circuits Thread_ , 2025. URL `https: //transformer-circuits.pub/2025/attribution-graphs/biology.html` . 

- [45] Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. _arXiv preprint arXiv:2310.06824_ , 2023. URL `https://arxiv.org/pdf/2310.06824` . 

29 

- [46] Eric J Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL `https://openreview.net/forum?id=3tbTw2ga8K` . 

- [47] Eric J. Michaud, Liv Gorton, and Tom McGrath. Understanding sparse autoencoder scaling in the presence of feature manifolds. 2025. URL `https://arxiv.org/abs/2509.02565` . 

- [48] Alexander Modell, Patrick Rubin-Delanchy, and Nick Whiteley. The origins of representation manifolds in large language models. _arXiv preprint arXiv:2505.18235_ , 2025. 

- [49] Sangjun Moon, Dasom Choi, Jingun Kwon, Hidetaka Kamigaito, and Manabu Okumura. Length representations in large language models. _arXiv preprint arXiv:2507.20398_ , 2025. 

- [50] Depen Morwani, Benjamin L Edelman, Costin-Andrei Oncescu, Rosie Zhao, and Sham Kakade. Feature emergence via margin maximization: case studies in algebraic tasks. _arXiv preprint arXiv:2311.07568_ , 2023. 

- [51] Edvard I. Moser, Emilio Kropff, and May-Britt Moser. Place cells, grid cells, and the brain’s spatial representation system. _Annual review of neuroscience_ , 31:69–89, 2008. URL `https: //api.semanticscholar.org/CorpusID:16036900` . 

- [52] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. _arXiv preprint arXiv:2301.05217_ , 2023. URL `https://arxiv.org/pdf/2301.05217` . 

- [53] Yaniv Nikankin, Anja Reusch, Aaron Mueller, and Yonatan Belinkov. Arithmetic without algorithms: Language models solve math with a bag of heuristics, 2024. URL `https:// arxiv.org/pdf/2410.21272` . 

- [54] Chris Olah. Interpretability dreams. Transformer Circuits Thread, May 2023. URL `https: //transformer-circuits.pub/2023/interpretability-dreams/index.html` . 

- [55] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. _Distill_ , 2020. doi: 10.23915/distill.00024.001. URL `https://distill.pub/2020/circuits/zoom-in` . https://distill.pub/2020/circuits/zoom-in. 

- [56] Chris Olah, Nicholas L. Turner, and Tom Conerly. A toy model of interference weights, July 2025. URL `https://transformer-circuits.pub/2025/interference-weights/ index.html` . 

- [57] Christopher Olah. What is a linear representation? what is a multidimensional feature?, 2024. URL `https://transformer-circuits.pub/2024/july-update/index.html# linear-representations` . 

- [58] Christopher Olah and Josh Batson. Feature manifold toy model, 2023. URL `https:// transformer-circuits.pub/2023/may-update/index.html#feature-manifolds` . 

- [59] Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. The geometry of categorical and hierarchical concepts in large language models. _arXiv preprint arXiv:2406.01506_ , 2024. 

- [60] Gonc¸alo Paulo, Alex Mallen, Caden Juang, and Nora Belrose. Automatically interpreting millions of features in large language models. _arXiv preprint arXiv:2410.13928_ , 2024. URL `https://arxiv.org/pdf/2410.13928` . 

- [61] Matthew G Perich, Devika Narain, and Juan A Gallego. A neural manifold view of the brain. _Nature Neuroscience_ , pages 1–16, 2025. 

- [62] Manuela Piazza, V´eronique Izard, Philippe Pinel, Denis Le Bihan, and Stanislas Dehaene. Tuning curves for approximate numerosity in the human intraparietal sulcus. _Neuron_ , 44(3): 547–555, 2004. 

30 

- [63] Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A primer in bertology: What we know about how bert works. _Transactions of the Association for Computational Linguistics_ , 8:842–866, 2020. doi: 10.1162/tacl ~~a 0~~ 0349. URL `https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/ A-Primer-in-BERTology-What-We-Know-About-How-BERT` . 

- [64] Odelia Schwartz, Anne Hsu, and Peter Dayan. Space and time in visual context. _Nature Reviews Neuroscience_ , 8(7):522–535, 2007. 

- [65] Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, and Antonio Torralba. A multimodal automated interpretability agent. In _Forty-first International Conference on Machine Learning_ , 2024. 

- [66] Xing Shi, Kevin Knight, and Deniz Yuret. Why neural translations are the right length. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pages 2278–2282, 2016. 

- [67] Trygve Solstad, Charlotte N Boccara, Emilio Kropff, May-Britt Moser, and Edvard I Moser. Representation of geometric borders in the entorhinal cortex. _Science_ , 322(5909):1865–1868, 2008. 

- [68] Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. _arXiv preprint arXiv:2305.15054_ , 2023. URL `https://arxiv.org/pdf/2305.15054` . 

- [69] Mirac Suzgun, Sebastian Gehrmann, Yonatan Belinkov, and Stuart M Shieber. Lstm networks can perform dynamic counting. _arXiv preprint arXiv:1906.03648_ , 2019. 

- [70] Martina G Vilas, Federico Adolfi, David Poeppel, and Gemma Roig. Position: An inner interpretability framework for ai inspired by lessons from cognitive neuroscience. _arXiv preprint arXiv:2406.01352_ , 2024. 

- [71] Elena Voita, Javier Ferrando, and Christoforos Nalmpantis. Neurons in large language models: Dead, n-gram, positional. _arXiv preprint arXiv:2309.04827_ , 2023. 

- [72] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. _arXiv preprint arXiv:2211.00593_ , 2022. URL `https://arxiv.org/pdf/2211.00593` . 

- [73] Martin Wattenberg and Fernanda B Vi´egas. Relational composition in neural networks: A survey and call to action. _arXiv preprint arXiv:2407.14662_ , 2024. 

- [74] Tom Wollschl¨ager, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan G¨unnemann, and Johannes Gasteiger. The geometry of refusal in large language models: Concept cones and representational independence. _arXiv preprint arXiv:2502.17420_ , 2025. URL `https://arxiv.org/pdf/2502.17420` . 

- [75] Adam Yedidia. GPT-2’s positional embedding matrix is a helix, 07 2023. URL `https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/ gpt-2-s-positional-embedding-matrix-is-a-helix` . LessWrong post. 

- [76] Adam Yedidia. The positional embedding matrix and previous-token heads: how do they actually work? _Alignment Forum_ , August 2023. URL `https://www.alignmentforum.org/posts/zRA8B2FJLtTYRgie6/ the-positional-embedding-matrix-and-previous-token-heads-how` . 

- [77] Gizem Y Yildiz, Irene Sperandio, Christine Kettle, and Philippe A Chouinard. A review on various explanations of ponzo-like illusions. _Psychonomic Bulletin & Review_ , 29(2):293–320, 2022. 

- [78] Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. The clock and the pizza: Two stories in mechanistic explanation of neural networks. _Advances in neural information processing systems_ , 36:27223–27250, 2023. URL `https://proceedings.neurips.cc/paper_files/paper/2023/file/ 56cbfbf49937a0873d451343ddc8c57d-Paper-Conference.pdf` . 

31 

- [79] Tianyi Zhou, Deqing Fu, Vatsal Sharan, and Robin Jia. Pre-trained large language models use fourier features to compute addition. _arXiv preprint arXiv:2406.03445_ , 2024. URL `https: //arxiv.org/pdf/2406.03445` . 

32 

## **A Citation Information** 

For attribution in academic contexts, please cite with this BibTeX citation 

- `@article{gurnee2025when,` 

```
author={Gurnee,WesandAmeisen,EmmanuelandKauvar,IsaacandTarng,Juliusand
Pearce,AdamandOlah,ChrisandBatson,Joshua},
```

```
title={WhenModelsManipulateManifolds:TheGeometryofaCountingTask},
journal={TransformerCircuitsThread},
year={2025},
```

```
url={https://transformer-circuits.pub/2025/linebreaks/index.html}
```

```
}
```

## **B Acknowledgments** 

We would like to thank the following people who reviewed an early version of the manuscript and provided helpful feedback that we used to improve the final version: Owen Lewis, Tom McGrath, Eric Michaud, Alexander Modell, Patrick Rubin-Delanchy, Nicholas Sofroniew, and Martin Wattenberg. We are also thankful to all the members of the interpretability team for their helpful discussion and feedback, especially Doug Finkbeiner for discussions of rippling and ringing, Jack Lindsey on framing, Tom Henighan for feedback on clarity, Brian Chen for improving the design of the figures and line edits of the text, and the team who built the attribution graph [2] and QK attribution infrastructure [36]. 

## **C Haiku Task Performance** 

Haiku is able to adapt to the line length for every value of _k_ , predicting newlines at the correct positions with high probability by the third line. Of course, some error is to be expected even with a perfect estimate of line length, as the model may incorrectly predict the next semantic token. Below is the mean log-prob and accuracy for newline prediction of Haiku on 200 prose sequences that were synthetically wrapped to have lines of character length _k_ , for _k_ = 20 _,_ 40 _, . . . ,_ 140. 



<!-- Start of picture text -->
°<br>Z os<br>32<br>3 Foe<br>3 Line Length ik) § | Line Length (k)<br>3 —2» For} | — 2<br>3-6A —o— r38 //| —— 4060<br>100 02) 100<br>Ea -8 — 80120 ] — 80120<br>— 140 — 140<br>00<br>oo: 4 6 8 10 W 1 16 o 2 4 6 8 0 W 18 6<br>Line Number Line Number<br><!-- End of picture text -->

Figure 35: Task performance of Haiku (as measured in mean log prob of newline and top-1 accuracy) on the linebreaking task as a function of number of lines in the context. 

## **D Feature Splitting and Universality** 

It is natural to ask if the character counting features are fundamental, or simply one discretization of the space among many. We found that dictionaries of different sizes learn features with very similar receptive fields, so this featurization – including the slowly dilating widths – is in some sense _canonical_ . We hypothesize that this canonical structure emerges from boundary constraint: positions near zero (start of line) create a natural anchoring point for feature development. 

33 



<!-- Start of picture text -->
Line Count Features Across Different Dictionary Sizes<br>08 =-- am HN ay FANNY iN i\ # Aw ey A<br>§ TEOMAHa, y Wyteiit MYWA i{ ifai Woit vy<br>i iff WE WE VAR OWA hy i VY<br>ed ear | Wi A; iX., :<br>g |—wee fff MA LM [hy A hy J AX<br>Stee Hf MA fi [ai iy ih \ As<br>Foalé — veewes HP WPMfi fake fryfil hoofVif j fi \\i jiHON.\<br>s [owe HE ATER iN \i ff j , ff ‘<br>2)2)— ca i ftAOU VyAY Wiia \\ \iVe?wf j VfAf \‘.<br>3 Fy ry @ Line Character Countr To io te<br><!-- End of picture text -->

Figure 36: Feature response curves from three differently sized dictionaries, suggesting that this particular sparse parameterization is canonical. 



<!-- Start of picture text -->
[i 1M Features Feature Decoder Cosine SimSMresturesacross Dictionary Sizes (layer 2) 10M Features vn<br>i [oooa 2a osoons<br>2 2 2 5<br>#3£ oa 3 00 H<br>Bs 4 4 i<br>@ 025%<br>-030<br>“075<br>? ? ?<br>-1.00<br>bitsisesFeature nde bitsitesfeature nde bitsFestuca nde bee<br><!-- End of picture text -->

Figure 37: Pairwise cosine similarity of feature decoders across different dictionary sizes. 

The geometry of the decoder directions is also fairly consistent between the dictionaries, showing characteristic ringing. 

However, we do see some evidence of feature splitting. For example, below are three character count feature which activate on the same interval ( _∼_ 20–45 characters in the line), but differentially activate for lines of different widths: LCC2.a activates on all line widths, LCC2.b preferentially activates on long line widths, and LCC2.c preferentially activates when close to the line width boundary. 



<!-- Start of picture text -->
ons tes. Lcce.b eeze<br>=e ie ja)<br>p=' j=: ios=u APN on Ama \ oa<br>joof=S2 |>3 =om Ai oo \\r\<br>¥ 000) — 6 — us \ 02<br>ee oe 4<br>Foo » = | ,<br>ve ALINf we ZN‘ os<br><!-- End of picture text -->

Figure 38: Response curves for three different features that all activate from the to 20-50 character for all possible line widths. 

Recent work has raised the possibility that feature dictionaries could behave pathologically where there exist feature manifolds [47], because a dictionary could allocate an increasing number of features in a finer tiling of the space. However, our observation that cross-coders of varying size tile this feature manifold in a canonical way suggest that this behavior does not occur in this setting. 

34 

## **E Line Width Features** 

Line width features tile the space similar to character count features. 



<!-- Start of picture text -->
Line Width Feature Activations<br>festre<br>7 7 wre<br>a6 /" /\ = wwe<br>i \ /\ = ws<br>/ \ / \ — wre<br>5 os / } \ ws<br>Boa / \ wre<br>i / — wer<br>H || LN == wreows<br>:°a}| \ = wo<br>a )<br>oo Ww Z<br>3 7 wo Line witt(characters betweenrs acer newts)wo io lo<br><!-- End of picture text -->

Figure 39: Feature response curves for line width features. 

## **F Dynamical System Model** 

We simulate _N_ = 100 points on the unit ( _n −_ 1)-sphere in R<sup>_n_</sup> ( _n ∈{_ 3 _, . . . ,_ 8 _}_ ) with pairwise forces: **F** _ij_ = _−_<sup>min</sup> _r_<sup><u>(</u></sup> _ij_<sup>5</sup><sup>_<u>,</u>_1</sup><sup>_<u>/rij</u>_</sup> ˆ **r**<sup><u>)</u></sup> _ij_ when _dij ≤ w_ � 1 _−_ <u>(</u> _dij −rij_ 1) _<u>/</u>_ 2 ˆ **r** _ij_ when _dij > w_<sup>,where</sup><sup>_rij_=</sup><sup>_∥_</sup><sup>**x**</sup><sup>_j−_</sup><sup>**x**</sup><sup>_i∥_, ˆ</sup><sup>**r**</sup><sup>_ij_=(</sup><sup>**x**</sup><sup>_j−_</sup><sup>**x**</sup><sup>_i_)</sup><sup>_/rij_,</sup> 

_w_ is the attractive zone width parameter, and _dij_ = min( _|j − i|, |j − i_ + _N |, |j − i − N |_ ) is the index distance (for the circular topology; for the interval it is just _dij_ = _|j − i|_ ). Evolution follows **v** ˙ _i_ =<sup>�</sup> xy _j̸_ = _i_<sup>**F**</sup><sup>_ij−_0</sup><sup>_._05</sup><sup>**v**</sup><sup>_i_and</sup><sup>**x**˙</sup><sup>_i_=</sup><sup>**v**</sup><sup>_i_withsphereconstraint</sup><sup>**x**</sup><sup>_i←_</sup><sup>**x**</sup><sup>_i/∥_</sup><sup>**x**</sup><sup>_i∥_enforcedaftereach</sup> timestep (∆ _t_ = 0 _._ 01, damping _α_ = 0 _._ 95). 

## **G Analytic Construction of Ringing and Fourier Modes** 

We explore a deeper connection between the ringing observed in the character count feature manifold, and a connection to Fourier analysis in an analytical construction. 

Suppose that we wish to have a discretized circle’s worth of unit vectors, each similar to its neighbors but orthogonal to those further away. Then the cosine similarity matrix _X_ of these will be the circulant matrix of a narrow-peaked function _f_ (left, below). The columns of the square root of _X_ are _n_ vectors _v_ 1 _, . . . , vn ∈_ R<sup>_n_</sup> , where _n_ is the number of discrete points on the circle, whose inner products reproduce the similarity matrix.<sup>29</sup> Now suppose we would like to find vectors in a lower-dimensional space whose similarity matrix approximates _X_ , like the model does for character counts. The best _k_ -dimensional approximation of this similarity, in an _L_<sup>2</sup> sense, is given by taking the eigendecomposition of _X_ and truncating it to the top _k_ eigenvectors; the square root of the result will provide _n_ vectors in _k_ -dimensions with the corresponding similarity pattern. If _πk_ is the projector onto the span of the top _k_ eigenvectors, then the images _πkvi_ , which live in a _k_ - dimensional subspace, are precisely those vectors. We can see below that the resulting low-rank matrix has ringing (colab notebook). Finally, because the discrete Fourier transform diagonalizes circulant matrices, the Fourier coefficients of _f_ are in fact the principal values of _X_ ; the low-rank approximation consists of truncating the small fourier coefficients of _f_ ; and the resulting rows of _X_ exhibit ringing. 

One essential feature of the representation of line character counts is that the “boundary head” twists the representation, enabling each count to pair with a count slightly larger, indicating that the boundary is close. That is, there is a linear map QK which slides the character count curve along itself. Such an action is not admitted by generic high-curvature embeddings of the circle or the 

> 29The entire continuous circle embeds into the infinite-dimensional Hilbert space _L_ 2S⊮ via this construction. 

35 



<!-- Start of picture text -->
100 Ideal Similarity Matrix 100 Rank-5eeApproximation 1.00<br>075<br>80 20<br>050<br>fo co 025<br>i i 100<br>° ~025<br>-050<br>20 20<br>0.75<br>° ° -1.00<br>o 2 4Column 6 # 100 Cr a aColumn a eT)<br><!-- End of picture text -->

Figure 40: Pairwise cosine similarity of the ideal similarity matrix and its rank-5 approximation, reproduces ringing behavior. 

interval like the ones in the physical model we constructed. But it is present in both the manifold we observe in Haiku and, as we now show, in the Fourier construction. First, note permutating the coordinates of R<sup>_n_</sup> by taking _ei �→_ aa _ei_ +1 has the effect of mapping _vi �→_ Lad _vi_ +1. That is, the (linear) action of this permutation _ρ_ on R<sup>_n_</sup> acts by a rotation of the embedded circle with respect to its intrinsic geometry. Because conjugation by _ρ_ fixes the circulant matrix _X_ , it therefore respects its eigendecomposition, and thus commutes with the projection _πk_ onto the vector space spanned by its top _k_ eigenvectors. The restriction of _ρ_ to that subspace, _<u>ρ</u>_ := _πk ◦ ρ ◦ πk_ , acts by rotation on the lower-dimensional vectors _<u>ρ</u>_ ~~-~~ : _πkvi �→_ od _πkvi_ +1. Thus we have found _n_ vectors in a _k_ -dimensional space, whose similarity is as close as possible to _X_ (and has ringing), with a linear action of R<sup>_k_</sup> that rotates the vectors along a rippled embedded circle. 

We evaluate whether a Fourier decomposition of the character count curve is optimal, and find that it is quite close given that it does not account for dilation. Fourier components explain at most 10% less variance than an equivalent number of PCA components, which are optimal for capturing variance. 



<!-- Start of picture text -->
orm ofeach Foure frequency ‘Variance Explained: SVD vs Top Fore (1 pair = 2 SVD components)<br>ome<br>— eee = pt fer conanpon<br>Tree (yeesate umber of congoers 0)<br><!-- End of picture text -->

Figure 41: Fourier results on character counting curve. 

Finally, we note that as one moves through layers, the representation becomes more peaked. This sharpening of the receptive field is useful to the model to better estimate character counts, and corresponds to higher curvature in the embedding and, as predicted by the model above, more pronounced ringing. Below we show cross-sections (at character count 30, 60, 90, 120) of the cosine similarity matrix of probes trained after layers 0, 1, 2, and 3. With each subsequent layer, the graphs get more tightly peaked and secondary rings go higher. 

36 



Figure 42: Cross sections of character count pairwise similarity. 

## **H Geometry of Twisting** 

Different heads access and manipulate the space in different ways. Below, we show the cosine similarity of both probe sets through QK for three heads: one which keeps them aligned, one which shifts character count to align better with later line widths, and one which does the opposite. 



Figure 43: Pairwise cosine similarity of probes through different boundary detection heads. 

We can also look at this transformation by visualizing the Singular Value Decomposition of each set of probes in a joint basis after passing them through QK. Once more, the alignment, left offset, and right offset can be read directly from the components. 



Figure 44: Loading of different PCA components as a function of count for three different boundary detection heads. 

We can directly plot the first 3 components of the joint probe space after passing them through each QK. Doing so shows that one head keeps the representations aligned, while the others twist them either clockwise or counterclockwise. 

## **I Break Predictor Features** 

Boundary detector features (at about _∼_ 1/3 model depth) do not take into account the length of the next token. 

37 





Later in the model, there exist features which incorporate both the number of characters remaining _and_ the length of the most likely next token. These features only activate when the most likely next token is longer than the number of characters remaining (i.e. below the red diagonal below), as is the case in our aluminum prompt. 



We also found features for the converse: features which suppress the newline because the predicted next token is _shorter_ than the number of characters remaining in the line. 

Both break prediction and suppression features sometimes also have interpretable logit effects on the output of all tokens, not just the newline. For instance, the features below respectively excite and suppress the newline as their top effect, but also systematically suppress tokens with more characters. This is because if the model is wrong about the value of the next token (and whether it’s a newline), the token must at least be short enough to fit on the line. 

## **J Representing Token Lengths** 

We find layer 0 features that activate as a function of the character count of individual tokens. 

38 



<!-- Start of picture text -->
Break Suppression Features Sensitive to Next Token Length<br>sto srt ose<br>H ono E w= 4 oo<br>be re won 36 “<br>2 \ am 3 om 8 aoe<br>sr sea est<br>ie NN re a > fe aoe<br>Pw NS om Ee \ Sie cal<br><!-- End of picture text -->



<!-- Start of picture text -->
BPF2 Direct Effects BSF3 Direct Effects<br>OAS fomnennnnnnnnnnnnnnnncnncnncnnnennnnccccccanee O15 === Neulne Token Effect<br>10 bonded 010 +} $<br>5 piatptads: so pit<br>g= 0.00 — % 00 —<br>$2 -005 3-0052 i<br>¥ -o20 orl ys © 020 ar rn rer) Lad<br>01s cas} ttt<br>dizsiséetvesuunngwe orisasesesmnvoues<br>Token Character Length Token character Length<br><!-- End of picture text -->



<!-- Start of picture text -->
tofore-* SL o ngmervileSSRASGRISH.Word Feature BEBE nefMedium Wordupon 668. Feature/BRBIBEE {Bethe additionalShort Word Featureburden BRI re<br>B2 _nAccommodationeéifically for bad knee SABER SSSR| ‘Chicago|nS,and corresponding There canbe ratesbutfro 0: HRRIBR>oliticsilRREIEWA expected private conversationof8 continue<br>S  rectly states thati'7/S0SGRREEA =——ad not been FONE. HeBFE<br>J uncon eatitutional ER<br>3 hocomecterseespeciaizar with heEATERS —to hoRESRTAUR ofABBE HB firth Sermon fi faieth,<br>‘Desenvolvimentor1em GSSRNOMIGAE of the near the Mis UHI thereaftertwice ices shat1<br>‘ut based on WTO non SiSDesenvolvimento+ Batand theirdoneeHFRIAEI the B IE E R.S of t © T he  oaFSGUBE BBWBaid outheld dailyCoroner BB errssides sell!<br>10 0.20<br>os 03<br>§ oas<br>S06 oo<br>g oa0<br>B04<br>éFor OL 0.05,<br>00 00 0.00<br>° token5characterFalength Fe 0 token3characteryolength Fey 0 token3character 10length 8<br><!-- End of picture text -->

39 

These features are overlapping (e.g. there are tokens for which the long word and medium word features are both active) and non-exhaustive (none of them fire on some common tokens, where we suspect the representation of character length is partially absorbed [9] into features which just activate for that token). 

## **K The Mechanics of Head Specialization** 

Heads collaborate to generate the count manifold, but _how_ does each head aggregate counts? 

As a toy model, consider the following construction for character counting with a single attention head: 

- The head uses the previous newline token as a “sink” where it defaults all of its attention to (i.e., attention 1) 

- Each token since the newline gets _α_ attention, such that after _j_ tokens the newline has 1 _−αj_ attention. Note this limits the construction to only work on lines of up to 1 _/α_ tokens, but the model could use multiple heads at different offsets to count for longer sequences. The model could also dedicate attention to tokens proportionally to token length 

- The output of the head on the newline is 0, and the output of each non-newline token is a vector with the same direction but magnitude proportional to the character count of the token. 

This produces a ray with total length proportional to the character count of the line. 



<!-- Start of picture text -->
Toy Attention Pattern Counting Ray<br><!-- End of picture text -->

In practice, we observe that individual attention heads do indeed use the newline as an attention sink, but at different offsets. As an example, we visualize the attention patterns of 4 important Layer 0 heads on several prompts with different line widths (starting from the first newline in the sequence). 

To characterize the mechanism more precisely, we compute the average attention as a function of the number of tokens since the previous newline and also as a function of the character length of individual tokens. 

Similar to boundary detection, individual attention heads specialize in particular offsets to tile the space. Moreover, we observe that most of these attention heads have a bias towards attending to longer tokens. 

In addition to QK, a head can change its output based on the OV circuit [15]. We study this by analyzing the pairwise interaction of probes as mediated by the OV matrix. Specifically, for the averaged token embedding vectors for each token length _Et_<sup>30</sup> , our line length probes _Pc_ , and the weight matrices for the attention output of each head _WOV_ , we compute _Pc_<sup>_TWOVEt_.</sup> 

> 30That is, for each token character length _i_ , we compute the average embedding vector in _WE_ . We also prepend this with the newline embedding vector to make the plot below. 

40 



Figure 45: Attention patterns of four important Layer 0 heads (columns) for 3 different prompts (rows) prompt showing head specialization. Patterns start from the first newline with red dashes indicating linebreaks. 



Figure 46: Normalized attention as a function of tokens since newline (left) and token character length (right) for four heads. Each head specialized in a different offset similar to boundary heads. 

41 



<!-- Start of picture text -->
toro Product of (LO line charactertorn  count probes) x OV x (WeLore average by token length)tors<br>100 100 100 100<br>3” 0 0 %<br>gie5 ° ° ° ° i<br>3i<br>20 « « «<br>2°2°2°I - 2<br>aa |<br>Taken charactercoint (We) tokencharacer oie) taken character cont Wi)” tokenchrater cau)<br><!-- End of picture text -->

Figure 47: Inner product of line character count and token character count through OV for 4 important layer 0 heads. The OV responses reflect the difference in attention pattern biases. 

The output of each head can be thought of as having two components: (1) a character offset from the newline driven by the attention pattern and (2) an adjustment based on the actual character length of the tokens. Note that the average character count of a token is approximately 4.5 (and the median is 4), so we can interpret these effects shifting from a mean response (i.e. the transition point is always around count 4). 

To walk through a head in action, consider the perspective of L0H1, which attends to the newline for the first _∼_ 4 tokens and then spreads out attention over the previous _∼_ 4–8 tokens: 

- While attending to the newline, L0H1 writes to the 5–20 character count (5–20CC) directions and suppresses the 30–80CC directions. This makes sense as an approximation because attending to the newline implies that the line is currently at most 3 tokens long (newline has no width) and on average, any 3 token span is _∼_ 15 characters (and is unlikely to be ¿30). 

- While _not_ attending to the newline at all, L0H1 defaults to predicting CC40, since not attending to the newline implies that there are _∼_ 8 tokens in the line with _∼_ 5 characters each on average (including spaces). Then, there is an additional correction applied depending on how long the tokens are: 

- (1) If tokens being attended to are _short_ (¡4 chars), then upweight 10–35CC and downweight ¿40CC. 

- (2) If tokens being attended to are _long_ ( _≥_ 5 chars) then do the opposite. 

- In cases with some attention on newlines and nonnewlines, linearly interpolate the above predictions. 

Other heads perform a similar operation, except with different offsets depending on their newline sink behavior. Layer 1 heads also perform a similar operation, though they also can leverage the character count estimate of the Layer 0 heads (see Layer 1 Head OVs). 

## **L Layer 1 Head OVs** 

Similar to the OVs of the Layer 0 attention heads, Layer 1 heads write to the character count features in accordance to how long the tokens they attend to are. 

However, in addition to the token character length, Layer 1 heads also use the initial line length estimate constructed in Layer 0 to create a more refined estimate of the character count. 

These repeated computations appear responsible for implementing the sharpening of representations. 

42 



<!-- Start of picture text -->
Product of (line character count probes) x OV x (token character count probes)<br>Lawo ua ue ue<br>100 100 100 100<br>zg” rs rs 0<br>ze © © © ° i<br>fw ” ” 0 5<br>20 20 20 20<br>°°°<br>atoken character our (0) token character our (0) token character oun (0) token character our (0)<br><!-- End of picture text -->



<!-- Start of picture text -->
Product of (L1 line character count probes) x OV x (LO line character count probes)<br>Lavo ua uve ue<br>eo © © ©<br>:d00 ” ” ” a3<br>Bx 20 — 20 20 5<br>°. —— » o. oO.<br>tne charcer out (0) tne charcer out (0) tne charcer out (0) tne charter out (0)<br><!-- End of picture text -->



<!-- Start of picture text -->
Head Outputs vs True Character Count<br>=] 10 -<br>100 Lo Important Head Sum Argmax (5 heads) Lo and L2 Important Head Sum Argmax (21 heads)<br>bienxay i (er= 0.5926) fst newine) eeonay = bienxay (R= erst 0972) newine) + al<br>H + tt ea Seace, Peer oe<br>i tu, POET] F . xeren*<br>5 te.nepeee &Sw}. . get.<br>ix*:<br>3Z00 Ee zw .<br>3 = eee 3<br>= 2 r+ shaapelate © 3= 20 i *<br>ate °<br>3 2 ‘We inry Character@Count ra wo 8 2 ‘We ryLine characterryCount rs 100<br><!-- End of picture text -->

## **M Full Layer 0 Attention Results** 

Below, we show head sums for 3 different prompts with different line widths. 

As before, we can look at their decomposition. 

## **N More Sensory and Counting Representations** 

While in this work we carefully studied the perception of line lengths and fixed width text, there are many tasks which language models must perform which benefit from a positional, visual, or spatial representation of the underlying text. In the course of our investigation, we came across several other feature families and representations for these behaviors and report several below. 

43 



<!-- Start of picture text -->
(Lo) Sum of moran Head Outputs projected on ine har Count be<br>- Lo Head Sum4 teed Sum 0) Lo Head Sum 100)z :<br>4] w<br>0. | | 1 i 1<br>a a i | | o ¥! i iY SIF E e ij<br>aj—teacce/| | | dal]| | ] ] 1<br>of | | = |<br>ojol |<br>i | 1 | | |<br>: | «| } |<br>: | |<br>i” . | | *°<br>xf |<br>x | »<br>ro} |<br>| |<br><!-- End of picture text -->



<!-- Start of picture text -->
Head Outputs projected on Line Character Count Probes<br>LOHO (keto) LOH (k=0) Low? (kao) LOH tkedo)<br>© od Be da) oo || Zoe rem<br>Keo of | 0. oy<br>Fao of | © ol i<br>* 10 20 ry] ue 20 iia Fa<br>0- . » © @ ©. "os “ » o © 0. “S »m © © 0. S it «©oe @<br>LOHO (k=60) LOH (k=60) LOH? (k=60) LOH (k=60)<br>© % © % iii<br>:goo “ “0 6 1 1<br>Fo © “0 ra ;<br>* 10 20 20 20. i poi<br>of ° ° ° i 1<br>om © © "o » © °S wm © © S * «© @<br>LoH0 (ke100) L0H (k=100) Loi? (k=100) Low (k=100)<br>3kory © ra %©<br>Fe “0 “0 “0<br>* 10 20: 20- 20-<br>ols ° ° °<br>o content0  poston© @ "o context~ poston@ © °S conte~ poston© © S conte #poston @<br>= = ES = = a<br>Ho ouput projected onto probes, ep project onto probes a euput projected onto probes 3 outa projected ont roves<br><!-- End of picture text -->

44 

### **N.1 What Follows an Early Linebreak?** 

In addition to line width for tracking the absolute character length of a full line of text, there also exist features that are sensitive to lines which have ended early (i.e, lines where the character count is substantially shorter than the line width _k_ ). While these features are less useful for linebreaking, they enable the model to better predict the token following a linebreak. Specifically, if a line ends _c_ characters before the line limit _k_ , the next word should be at least _c_ characters, otherwise it would have been able to fit in the previous line. 



<!-- Start of picture text -->
ont banking ae[Mcaracterizedby _>ages wouldbe badlylundersampled _andverdure patternswashable;up but must presume to fda,what itd<br>|BBsE ratpubisoyoufonstantyrunintitieaWGandotharanguage/oolsy esarethepolcelemenstatealti splewhoarenot truyinterested who: sentstaavalableat arelopportunityRussiatat tine ponever and?we ensa.t2pted. AujouretThe peoplewhallleonstitutetrou sons<br>oremetteurest Mlstpendanta stbecause<br>tcién,includasolffinancaciénye! tementduriveaude[ffinancement,si} withightysecure dataffthese emplo ordinance<br>i they otterinterestinginsi _ghercost traveloptions Apple Storex ter had told him that with 15,000<br>‘derithsago.tthelpedffignificantly, one hour Ourffrevolutionarybuturb _vatlonelopedon (oratleastnotflencouragingits\_Ingitover alate 2020ifftimetablewith s+ shand.wilsetto Amazonup acalidiscuss S3because yourtse he#8, notcomprehensive proby any sufrom infange c hiffperson,fheeed Sonands. com as<br>oe<br>os<br>§ a oe as020 oo<br>502 a2 oxo ox0<br>Fon oa 0.05, 0.05<br>ry ¢ 3» b » oo o 3 w & m% 00 6 3 w % wm .00 6 3 wo ws w<br><!-- End of picture text -->

Figure 48: A feature family for how many characters were remaining in a line after it was broken. 

It is worth emphasizing that the role of these features, like others in this work, is not obvious from a typical workflow of quickly looking at dataset examples. It might be tempting to ignore these as “newline” features, but careful analysis yields quite clear behavior. 

### **N.2 Markdown Table Representations** 

In addition to prose, language models must parse other kinds of more structured data like tables. Accurate prediction of a table’s content requires careful integration of row and column information (e.g. is this a column of text or numbers?). To facilitate this, we use a synthetic dataset of 20 markdown tables to find feature families which activate on separator tokens, specialized to particular rows or columns. Visualizing feature activations on each of these 20 tables (arranged by location in the table) showed clear patterns. 



<!-- Start of picture text -->
Fl F2 3 Fa FS<br><!-- End of picture text -->

Figure 49: A feature family for the row index in markdown tables. Activations are shown for 20 tables on the ”—” token in the first column of the nth row. 



<!-- Start of picture text -->
FI F2. F3 FA<br>fsE:2 arie<br><!-- End of picture text -->

Figure 50: A feature family for the column index in markdown tables. Activations are shown for 20 tables on the ”—” tokens in the nth column. 

45 

On a synthetic dataset of larger tables, we also observe counting representations for the column and row index that resemble the character counting representations. Specifically, we see ringing in the pairwise probe cosine similarities and the characteristic “baseball seam” in the PCA basis. 



<!-- Start of picture text -->
Representations of Row and Column Indices in Markdown Tables.<br>°DBRow Index Probes: Cosine Similarity Matrix100 a o0 °ColumnOaindex Probes: Cosine Similarity100Matrix et a0<br>ars as<br>2 20<br>aso aso<br>, oo eB oo §<br>i -oasd § -025 8<br>80: -030 ° 60 -050<br>100 “0754609 “075<br>Row Index Probe -1.00 Column Index Probe -1.00<br>Row Index Probes: PCA Projection (Var: 44.1%) Column index Probes: PCA Projection (Var: 38.0%)<br>pos 100 100<br>fy Se E |feo ‘Pe F [feo<br>é. .. 3 z (ars = z<br>oa eo he & fae 8 ifo &<br>96. oes 20 2<br>PO aes &J ° P2235, eeGiTay °<br><!-- End of picture text -->

Figure 51: Representations of markdown table row indices (left) and column indices (right). (Top) pairwise inner products of probes trained to predict the index; (bottom) probes projected into a 3D PCA basis. 

## **O Rejected Titles** 

- A General Language Assistant as a Laboratory for A-line-ment 

- A-line-ment Science: The Geometry of Textual Perception 

- The Geometry of Textual Perception: How Models Stay Aligned 

- The Geometry of Counting: How Transformers Perceive and Manipulate Spatial Structure in Text 

- The Mechanistic Basis of Alignment 

- Reading between the lines: The Perception of Linebreaks 

- Linebreaking: More than you wanted to know 

- The Line Must Be Drawn Here! Character Counting in Neural Networks 

- Newline, Who Dis? Attention Heads and Their Distractible Nature 

- The End of the Line: How Language Models Count Characters 

- How I Learned to Stop Worrying and Love the Carriage Return 

- Breaking down the Linebreak Mechanism: The Geometry of Text Perception 

- We Found a GPS Inside a GPT 

46 

