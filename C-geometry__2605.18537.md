# Probing for Representation Manifolds in Superposition

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** C-geometry — Modell. Probing adapted to the manifold case — the direct methodological upgrade path from W3's linear/RBF SVMs.
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/2605.18537.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

# **Probing for Representation Manifolds in Superposition** 

### **Alexander Modell** 

Department of Mathematics Imperial College London, U.K. `a.modell@imperial.ac.uk` 

## **Abstract** 

This paper introduces the Manifold Probe, a supervised method for discovering representation manifolds in superposition. The method generalizes linear regression probes by learning the space of features of a concept that can be linearly predicted from the representations, and then learning the directions used to encode them. We demonstrate the probe on representations of time and space in Llama 2-7b, finding manifolds which linearly represent an interpretable set of features in each case. In the case of time, we show that by steering along the manifold, we can influence the model’s completions about the years in which famous songs, movies and books were released, providing evidence that the Manifold Probe can discover manifolds which are causally involved in model behaviour. 

- `alexandermodell/maniprobe` 

## **1 Introduction** 

The ability to interpret the representation geometry of large language models is a fundamental goal in a larger scientific effort to understand AI systems as a whole (Bereska and Gavves, 2024). 

A key hypothesis in this effort is the linear representation hypothesis (Nanda et al., 2023b): that neural networks organize their internal representations so as to make semantically important features accessible via linear projections. A related hypothesis is that of superposition (Smolensky, 1990; Mikolov et al., 2013; Elhage et al., 2022): the idea that representations of distinct concepts combine additively to produce representations of their joint semantics. 

Initial work around these hypotheses focused on understanding representations of simple binary concepts, which are considered to be either present or absent (Elhage et al., 2022). In this setting, concepts are hypothesised to be represented by almost-orthogonal directions in representation space and the presence of multiple features is represented by summing the corresponding directions. These hypotheses motivate many contemporary interpretability methodologies, such as linear probes (Alain and Bengio, 2017; Nanda et al., 2023b), sparse autoencoders (Elhage et al., 2022; Bricken et al., 2023; Cunningham et al., 2024), and steering vectors (Li et al., 2023; Marks and Tegmark, 2024; Rimsky et al., 2024; Turner et al., 2025). 

More recently, there has been a push towards understanding the representation geometry of more complex, continuous concepts, which don’t fit in to the binary framework. Examples include numerics, time, space, colour, and more abstract concepts such as emotion, ideaology and phylogeny (Gurnee and Tegmark, 2024; Olah, 2024; Engels et al., 2025; Modell et al., 2025; Gurnee et al., 2025; Pearce et al., 2025; Savietto et al., 2026; Choi and Weber, 2026; Sofroniew et al., 2026; Sun et al., 2026). There is growing empirical evidence that continuous concepts are represented on manifolds which bend and twist through multiple dimensions of the representation space (Gorton, 2024; Modell et al., 2025; Yocum et al., 2025; Gurnee et al., 2025; Karkada et al., 2026). The presence of such multi-dimensional manifolds is compatible with both the linear representation and superposition hypotheses. In particular, their shape directly determines the information about the concept which can be accessed via linear projections. 

Preprint. 

The problem of isolating multidimensional representations of concepts, and discovering the geometry of representation manifolds in superposition is relatively unexplored. For the former, we are only aware of Engels et al. (2025) who propose to do this by clustering dictionary vectors in sparse autoencoders. For the latter, Yocum et al. (2025) and Gurnee et al. (2025) propose fitting a family of linear classifier probes to a discretization of the concept space, and Modell et al. (2025) propose approximating representation manifolds with neighbour graphs. 

In this paper, we propose the Manifold Probe, a supervised probing methodology to discover representation manifolds which are represented in superposition with other semantic information. 

The probe is fitted in two stages. The first stage is to learn the space of features _f_ ( _z_ ) of the concept values _z_ which are well-predicted by a linear function _w_<sup>_⊤_</sup> _x_ + _b_ of the representations _x_ . While a standard linear regression probe would consider a fixed feature as its target, our probe learns the features at the same time as the regression parameters. We show that under a generic statistical model of a representation manifold in superposition, these learned features approximate the geometry of the manifold with respect to some unknown basis. The second stage learns this basis using linear regression. The main methodological contribution of this paper is the formulation and optimization of the first step of this procedure. 

To demonstrate the Manifold Probe, we use it to explore residual-stream representations of time and space in Llama 2-7b from probing datasets curated in Gurnee and Tegmark (2024). While Gurnee and Tegmark (2024) show that the concept values themselves are linearly represented, our probe brings to light many more features which are too, some of which are represented more precisely than the concept values themselves. 

We show how applying factor analysis to the learned features can help us interpret them. Employing a Varimax rotation which aims to make the features approximately sparse reveals that the time manifold linearly separates decades, while the space manifold linearly separates many states in the U.S.A.. 

Finally, we show that the time manifold we find is not only present in the residual stream, but is used by the model. We perform an intervention experiment where, at a given layer, we steer the residual stream representations by adding steering vectors which trace the manifold. By doing this, we can influence the model to complete a prompt about the release date of a song, movie or book with a year that we target. 

## **2 Setup and background** 

### **2.1 Concepts and representation manifolds** 

A _concept_ is a topological space _Z_ which we can attach some real world meaning to. The simplest example might be a binary concept which is considered to be either present or absent. Continuous concepts include time (a line _Z_ = R), space (a plane _Z_ = R<sup>2</sup> or a sphere _Z_ = S<sup>2</sup> ), colour (a cylinder _Z_ = S _×_ R<sup>2</sup> or cube _Z_ = R<sup>3</sup> ) and can include more abstract concepts such as emotion with an appropriate mathematical model (such as the valence-arousal-dominance model). 

We say that any injective map _ϕ_ : _Z →_ R<sup>_p_</sup> _represents_ the concept _Z_ . If _ϕ_ is also continuous, then its image _M_ = _ϕ_ ( _Z_ ) is a _representation manifold_ embedded in R<sup>_p_</sup> which, under some mild conditions<sup>1</sup> , is topologically equivalent to the concept _Z_ . We’ll write _U ⊆_ R<sup>_p_</sup> to denote the smallest subspace containing _M_ , and _d_ to denote its dimension. 

For example, if _Z_ is an interval, then _M_ is a curve; if _Z_ is a circle, then _M_ is a loop; and if _Z_ is a rectangle, then _M_ is a sheet, all of which might bend and twist to occupy more dimensions in the representation space than might be implied by the intrinsic dimensionality of the concept itself. 

### **2.2 Semantics and superposition** 

We now turn to the question of how multiple concepts might be represented together. 

To this end, we will consider an abstract topological space _S_ which we refer to as the _semantic space_ , which we assume encodes the semantics of any input to the neural network. We will assume _S_ can be 

> 1such as _Z_ being compact. 

2 

factorized into a set of interpretable concepts _Z_ 1 _, · · · , Zm_ , and a set of other semantics Ξ, so that _S_ = _Z_ 1 _× · · · × Zm ×_ Ξ _._ 

We will be interested in hypothesizing about, and making inferences about the structural form of a map _x_ : _S →_ R<sup>_p_</sup> which represents _S_ . 

A key hypothesis in mechanistic interpretability is that of _superposition_ : the idea that representations of concepts combine additively to produce representations of their joint semantics. **Definition 1.** _We say that a map x_ : _S →_ R<sup>_p_</sup> _represents the concepts Z_ 1 _, · · · , Zm in superposition if there exists maps ϕi_ : _Zi →Ui ⊆_ R<sup>_p_</sup> _for i_ = 1 _, · · · , m, and a map η_ : Ξ _→V ⊆_ R<sup>_p_</sup> _such that_ 



_for all s_ = ( _z_ 1 _, . . . , zm, ξ_ ) _∈S._ 

In the special case that concept representations are one-dimensional, this superposition hypothesis has been studied extensively in the mechanistic interpretability literature. The general setting presented above, in which concept representations are allowed to occupy multidimensional subspaces, has received comparatively little attention, with the notable exception of Engels et al. (2025). It also presents an additional inference problem: not only is it of interest to estimate the subspace which the concept representation occupies, but also the geometry of the representation _within_ that subspace. 

In this paper, we will be concerned with developing methodology to discover the representation _ϕ_ := _ϕ_ 1 of a single target concept _Z_ := _Z_ 1. From hereon, we will absorb any additional concepts in Ξ, and assume that _S_ = _Z ×_ Ξ. 

### **2.3 Probing** 

In order to discover the representation _ϕ_ of the target concept _Z_ , we will employ the probing methodology (Alain and Bengio, 2016). The idea behind probing is to construct a dataset of representation-concept values pairs _D_ = _{_ ( _xi, zi_ ) _}_<sup>_n_</sup> _i_ =1<sup>, and to use this in a supervised fashion to fit a</sup> statistical model which elucidates the representation geometry of interest. 

We assume that each representation-concept value pair ( _xi, zi_ ) _∼_ P in the probing dataset _D_ is sampled independently by sampling a semantic value _si_ = ( _zi, ξi_ ) from a distribution P _S_ , and then constructing the representation _xi_ = _x_ ( _si_ ) according to the superposition equation (1) in Definition 1. While we observe the representation-concept value pairs ( _xi, zi_ ); the nuisance semantics _ξi_ , and the functional form of the maps _ϕ_ and _η_ are unobserved. We assume that the concept value _zi_ and nuisance semantics _ξi_ are independent, i.e. P _S_ = P _Z ×_ PΞ. 

Our statistical objective is to use the probing dataset _D_ to learn two maps which estimate _ϕ_ ( _z_ ) from either the concept value _z_ , or a corresponding representation _x_ : 

1. a smooth non-linear map _ϕ_<sup>ˆ</sup> : _Z → M ⊂_<sup>ˆ</sup> R<sup>_p_</sup> from the concept values _Z_ to a manifold _M_<sup>ˆ</sup> in some _d_ -dimensional subspace _U_<sup>ˆ</sup> _⊂_ R<sup>_p_</sup> of the representation space. 

2. a linear (affine) map Ψ : R<sup>_p_</sup> _→ U_<sup>ˆ</sup> _⊂_ R<sup>_p_</sup> from the representation space R<sup>_p_</sup> to the subspace _U_ ˆ _⊂_ R<sup>_p_</sup> . 

We point out at this stage that the maps _ϕ_ and _η_ in the superposition equation (1) are only defined up to translation, and so for the purpose of estimation, we will assume without loss of generality that E[ _ϕ_ ( _z_ )] = 0. 

### **2.4 Manifold estimation as regression** 

Before discussing how we might estimate the representation map _ϕ_ from a finite probing dataset _D_ , it is useful to consider how we might obtain _ϕ_ given access to the true underlying population distribution P. 

The following lemma, which we prove in Section C of the appendix, tells us how. 

**Lemma 2.** _There exists a basis u_ 1 _, . . . , ud ∈U and a set of mean-zero, orthonormal features f_ 1 _, . . . , fd_ : _Z →_ R _such that_ 



3 

_which also solve the following sequential population regression problems:_ 



_where expectations are taken with respect to_ ( _x, z_ ) _∼_ P _, and f ⊥ g means_ E( _fg_ ) = 0 _._ 

Lemma 2 suggests the shape of a representation manifold is intimately connected to space of features which it linearly represents. This dual interpretation is key to our finite-sample estimation procedure, and also provides a lens through which to interpret the manifold geometry. 

## **3 Methodology** 

This section is dedicated to developing a sample-based estimation procedure for estimating _ϕ_ from the probing data _D_ based on the population regression problems in Lemma 2. 

In order to fit a feature _f_ , we parametrize it in some basis _h_ 1 _, . . . , hm_ which we treat as known, so that it can be written as 



for some unknown scalar parameters _β_ := ( _β_ 1 _, . . . , βm_ ). We denote the space of functions of the form (5) as _H_ . An appropriate choice of basis depends on the nature of the concept space _Z_ . In our examples, we use cubic B-splines, or tensor products thereof, however our method can accomodate any choice of basis. 

In order to avoid overfitting the function _f_ , we need some way to control its complexity. The standard approach in functional regression is to choose an overly flexible basis, and then to control the complexity of _f_ via a penalty function _J_ ( _f_ ) which we add to our loss function. The advantage of this approach is that it allows us to choose the level of permitted complexity using the data. 

In this paper, we will assume that the chosen penalty function _J_ is quadratic which allows us to write it as a quadratic form _J_ ( _f_ ) = _β_<sup>_⊤_</sup> _Sβ_ in the basis coefficients _β_ . In our examples, we use the integrated, squared second derivative penalty 



which is usually considered a default choice. However, our method is flexible enough to accomodate quadratic penalty, and in Section B of the appendix, we discuss how our method can be adapted to accomodate non-quadratic penalties, such as _ℓ_ 1 and mixed _ℓ_ 1 and _ℓ_ 2-type penalties. 

With _H_ and _J_ defined, we are ready to write down our probing procedure for estimating _ϕ_ ( _z_ ). 

**Definition 3.** _We write ϕ,_<sup>ˆ</sup> Ψ = _ManifoldProbe_ ( _D, d_ ; _λw, λf_ ) _if_ 



_with gk_ ( _x_ ) = _w_ ˆ _k_<sup>_⊤x_+ ˆ</sup><sup>_bk,where for k_=1</sup><sup>_, . . . , d,_( ˆ</sup><sup>_fk,w_ˆ</sup><sup>_k,_ˆ</sup><sup>_bk_)</sup><sup>_solves the sequential optimization_</sup> _problem:_ 



_and_ (ˆ _uk,_ ˆ _ck_ ) _solves the optimization problem:_ 



4 

For fixed regularization parameters _λw, λf_ , the Manifold Probe has a closed-form solution. 

We’ll define the centered model matrices _X ∈_ R<sup>_n×p_</sup> and _H ∈_ R<sup>_n×m_</sup> with rows _Xi,_ : = _xi − x_ ¯ and _Hi,_ : = _h_ ( _zi_ ) _− h_<sup>¯</sup> respectively, where _x_ ¯ = (1 _/n_ )<sup>�</sup><sup>_n_</sup> _i_ =1<sup>_xi_and ¯</sup><sup>_h_= (1</sup><sup>_/n_) �</sup><sup>_n_</sup> _i_ =1<sup>_h_(</sup><sup>_zi_).</sup> 

The closed-form solution that we present requires that the matrix _H_ has full-column rank, so that all of the coefficients _β_ can be uniquely estimated. If this is not the case (which is likely given the centering), we can linearly reparametrize the basis so that it does<sup>2</sup> . From hereon, we will assume that the basis is parametrized such that _H_ has full-column rank. 

**Proposition 4.** _Let_ ( _f_<sup>ˆ</sup> _k, w_ ˆ _k,_<sup>ˆ</sup> _bk_ ) _, be the solutions to the optimization problem_ (6) _. Then, f_<sup>ˆ</sup> _k_ ( _z_ ) = _β_ ˆ _k_<sup>_⊤h_(</sup><sup>_z_)</sup><sup>_whereβ_ˆ1</sup><sup>_, . . . ,β_ˆ</sup><sup>_dare an orthonormal set of eigenvectors corresponding to the d smallest_</sup> _eigenvalues ν_ ˆ _m, . . . ,_ ˆ _νm−d of the generalized eigenvalue problem_ 



_where_ 



A proof of Proposition 4 is given in Section D of the appendix. 

### **3.1 Fitting the regularization parameters** 

While Proposition 4 tells us how to fit the Manifold Probe for a fixed pair of regularization parameters _λw, λf_ , it doesn’t tell us anything about how we should choose them. In practice, we will want to choose them using the data, and we will want to choose different regularization parameters for each sequentially fitted feature. 

One approach is to directly apply _k_ -fold cross-validation to the objective, searching over candidate parameter values using, for example, grid search. 

In Section B of the appendix, we present an alternative approach which we favour in practice. We show that (6) can be optimized by solving a sequence of alternating (generalized) ridge regression problems, which we prove converge to the global minimizer under very mild conditions. We then propose to select the regularization parameters at each iteration using a closed-form criterion appropriate for ridge regression. In practice, we use either Generalized Cross-Validation (Craven and Wahba, 1978; Wood, 2004) or Restricted Maximum Likelihood (Bartlett, 1937; Wood, 2011), which have closed-forms and can be optimized very quickly using Newton’s method, without refitting the model. 

## **4 Discovering time and space manifolds in large language models** 

In this section, we use the Manifold Probe to discover hidden time and space manifolds in the residual stream of Llama 2-7b (Touvron et al., 2023), an open-weights large language model. We demonstrate how we can use the probe both as an interpretability tool, to discover features which are linearly represented in the residual stream; and as a steering tool, to causally influence the model’s behaviour. 

We make use of two probing datasets collected by Gurnee and Tegmark (2024). The first dataset contains the names and creators of popular songs, movies and books alongside their corresponding release dates (represented as a decimal year); and the second contains the names and geographic coordinates of places in the U.S.A.. After some filtering, we have 29,503 works released in _Z_ `time` = [1950 _,_ 2020], and 17,381 places with coordinates in _Z_ `space` = [24 _._ 5 _,_ 49 _._ 5] _×_ [ _−_ 125 _._ 0 _, −_ 66 _._ 5] (the bounding box of mainland U.S.A.). In both cases, we consider a 50-50 train/test split. 

For each entity, we construct a string such as “Queen’s Bohemian Rhapsody” or “Lake of the Ozarks” which we feed into the language model, and record the last token residual stream activations at each 

> 2for example, using its singular value decomposition. 

5 



<!-- Start of picture text -->
Release dates of songs, movies and books<br><!-- End of picture text -->



<!-- Start of picture text -->
years<br><!-- End of picture text -->



<!-- Start of picture text -->
Fone<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature 1 Feature 2 Feature 3 Feature 4 Feature 5<br>18601580 2000 2020 1960 1580 2000 7020 1960 18e0 2000 2020 960 1680 2000 2020 i960 1980 2000 2070<br>seo<br>1580 2000 2020 1960 1se0 2000 2020 1960 1580 2000 2020 1960 1880 2000 2020 1960 1980 2000 2020<br>‘ear ‘ear veer veer veer<br>Test #2 07676<br>Test#2:05208 Test #203391 TestR?: 02079 TestR: 0.1355<br><!-- End of picture text -->

Figure 1: A representation manifold (top left) and linear prediction (top right) from a Manifold Probe fitted the release dates of songs, books and movies from layer 16 residual stream activations of Llama 2-7b. _Below:_ the first five fitted features (top row), corresponding linear predictions (bottom row) for representations in the test set, and test _R_<sup>2</sup> coefficients. 

layer. To train the manifold probe, we parametrize time features using cubic B-splines with 280 knots, and parametrize space features using a tensor product of two cubic B-splines with 40 and 80 knots respectively. We use the fitting procedure described in Section B, and use the REML criterion to select the regularization parameters. 

### **4.1 Interpretability: exploring linearly-represented features** 

The bottom row of plots of Figures 1 and 2 show the first few features _f_<sup>ˆ</sup> 1 _, f_<sup>ˆ</sup> 2 _, . . ._ fitted by the probe to the layer 16 representations from the two training sets, and the corresponding linear predictions _g_ 1( _x_ ) _, g_ 2( _x_ ) _, . . ._ of these features from representations _x_ in the test sets. We report the _R_<sup>2</sup> coefficients of these test predictions, which measure the extent to which the features are linearly represented. Perfect predictions have an _R_<sup>2</sup> coefficient of one, predicting the feature mean always has an _R_<sup>2</sup> coefficient of zero, and predictions which are worse than predicting the mean have a negative _R_<sup>2</sup> coefficient. The top left plots show three dimensions of the estimated manifolds _M_<sup>ˆ</sup> and the manifold predictions Ψ( _x_ ) from the representations _x_ in the test set, with respect to the first three fitted basis vectors _u_ ˆ1 _,_ ˆ _u_ 2 _,_ ˆ _u_ 3. 

Since the features _f_ 1 _, . . . , fd_ in the decomposition (2) are only defined up to rotations of the basis _u_ 1 _, . . . , ud_ , it can be informative to apply factor analysis to the learned features to rotate them into a basis in which they are more easily interpretable. We take the top 5 time features, and the top 32 space features, and apply Varimax rotation (Kaiser, 1958; Rohe and Zeng, 2023) which aims to make the rotated features approximately sparse. The resulting features are shown in Figure 5 in the appendix. It is of particular note that rotated space features localize on many U.S. states, showing that they are 

6 



<!-- Start of picture text -->
Geographic coordinates of places in the U.S.A.<br><!-- End of picture text -->



<!-- Start of picture text -->
eftgd<br>o| Ae sere<br>\aow ofFS ®Se<br>coe<br><!-- End of picture text -->



<!-- Start of picture text -->
2 Lite<br>*| es Be.<br>| . URGES<br>wae<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature 1<br>oe fy<br>Cae ee<br>Y<br>Test R?: 0.6502<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature 2<br>ERR ae 4<br>eC < Mn<br>Y<br>Test R?: 0.6835<br><!-- End of picture text -->



<!-- Start of picture text -->
Feature 3<br>FERS, Es Ls<br>Y<br>Test R?: 0.6526<br><!-- End of picture text -->

Figure 2: A representation manifold (top left) and linear prediction (top right) from a Manifold Probe fitted to the geographic coordinates of places in the U.S.A. from layer 16 residual stream activations of Llama 2-7b. _Below:_ the first three fitted features (top row), corresponding linear predictions (bottom row) for representations in the test set, and test _R_<sup>2</sup> coefficients. Feature values are given by colour. 

approximately linearly separated in the representations. We can also interpret from the rotated time features that the decades from the 1950s to the 2010s are approximately linearly separated. 

To get an idea for how much information about the release dates and locations is linearly represented in the residual stream at each layer, we fitted features until the corresponding test _R_<sup>2</sup> coefficients were continually below zero. In Figure 3, we plot the value of the _k_ th ranked test _R_<sup>2</sup> coefficient for each layer where this is above zero. In both datasets, the predictabilities of features increase in predictability throughout the first half of the layers before levelling off. The location representations consistently contain three features which are considerably more predictable that the rest. 

We include the test _R_<sup>2</sup> coefficients of a ridge regression fit directly to the release dates (dotted line) from the songs, movies and books representations; and to the latitude (dotted line) and longitude (dashed line) from the U.S. places representations. These were reported in Gurnee and Tegmark (2024) as evidence that language models linearly represent space and time. In the time representations, we find that for all layers the most linearly predictable feature is very close to the identity feature, and so the test _R_<sup>2</sup> coefficient for our highest ranked feature and the direct year track very closely. In the time representations, the highest ranked feature we find has a higher test _R_<sup>2</sup> coefficient than the latitude and longitude features. 

### **4.2 Steering: causally influencing the model’s understanding of time** 

In this section, we demonstrate that we can use the representation manifold learned by the Manifold Probe to causally influence the model’s internal belief about the release dates of songs, movies and books. 

7 



<!-- Start of picture text -->
Release dates of songs, movies and books<br><!-- End of picture text -->



<!-- Start of picture text -->
08<br>06:<br>Soa<br>2<br>02<br>0.0 7 <><br>0 5 1 1 2 2 30<br>Layer<br><!-- End of picture text -->



<!-- Start of picture text -->
Geographic coordinates of places in the U.S.A.<br><!-- End of picture text -->



<!-- Start of picture text -->
o7<br>06 ry miei<br>08 WiLo g G-FFST k=™—=[]!HS XEN<br>= 04 Wr [IIPS<br>8 hi VieV/ Ee<br>Fos 5] rxYj<br>02 1) TOYS<br>/7 | GS><br>NAYY//0.0 KY V V7<br>5 10 15 20 2 30<br>layer<br><!-- End of picture text -->

Figure 3: Ranked test _R_<sup>2</sup> values for features fitted using the probing datasets described in Section 4 at each layer of Llama 2-7b. _Left:_ the dotted line shows the test _R_<sup>2</sup> coefficient of a ridge regression fit directly to the release dates from the songs, movies and books representations. _Right:_ the dotted and dashed lines show the test _R_<sup>2</sup> coefficients of ridge regression fits directly to the latitude and longitude from the U.S. places representations respectively. 

To do this, we treat the learned concept representations _ϕ_<sup>ˆ</sup> ( _z_ ) as an infinite continuum of steering vectors which trace the manifold _M_<sup>ˆ</sup> . To steer the model’s internal belief about the value of a concept to a particular value _z_ , we propose to intervene on a representation _x_ by adding a scalar multiple of _ϕ_ ( _z_ ) to it, i.e. setting 



for some _α >_ 0. 

From the probing dataset of songs, movies and books, we fit three-dimensional manifolds _ϕ_<sup>ˆ</sup> _l_ to the last-token residual stream activations at each layer of the model as in the previous section. We then select a stratified sample of 1,400 works from the test set, with 200 from each decade, to use for our steering experiment. For each song, movie, or book, we construct a prompt of the form 



which we feed into the model. Assuming standard temperature-one sampling, we record the probability of the model completing the prompt each year between 1945 and 2025. 

We do this for a clean run, and for each layer _l_ and year _z_ between 1950 and 2020, we do this again but intervene by steering the residual stream activations of the last token of the work’s title at layer _l_ by the steering vector _ϕ_<sup>ˆ</sup> _l_ ( _z_ ) and _α_ = 100. This results in a total of 32 _×_ 70 = 2240 interventions per prompt<sup>3</sup> . To measure efficacy of an intervention, we report the probability that the model completes the prompt with a year that is within two years of the target _z_ . 

The top plots in Figure 4 show the mean efficacy of the interventions for each layer, grouped in the left-hand plot by the decade of the song, movie or book, and in the right-hand plot by the decade of the target year _z_ . From the left-hand plot, we see that the efficacy of the interventions peak at layers 8 and 14, depending on the release decade of the work, and that this drops sharply after layer 15, and dropping to baseline performance at layer 20 and beyond. From the right-hand plot, we see that steering efficacy and the optimal layer for intervention depends quite heavily on the target year. 

Figure 6 in the appendix shows the probability that the model completes the prompt with a year at all. With the exception of steering to years in the 1950s, we see that these interventions have very little effect on the model’s ability to meaningfully complete the prompt. 

For layers 3 to 20, the bottom plot in Figure 4 shows the mean probability of each completion for each steering target. The dark diagonal streaks indicate that the interventions are having some success at influencing the model to complete the prompt with a given target year. Figure 7 in the appendix shows the standard deviations. 

> 3The full experiment took approximately 100 GPU hours on Nvidia RTX 4090s. 

8 



<!-- Start of picture text -->
Grouped by release decade Grouped by target decade<br>scr<br>t = is60s<br>LyA os \ bad — 1970s<br>a2 pk Viees = 1500s<br>Peo. BN, j oo \ — 19905<br>BE AS Ne = 20005<br>an BAR , ' a i f toto:<br>ae s Kr 02 ss \ eal<br>23%) Bx Ww KA /\ ne steered<br>uefA Si pesessseeses | °! extabl oleNa| sss Clean<br>00 oF 4 6 & iowa ie Wo 10 20 22 24 26 8 30 00 0 2 4 6 8 io ia is Wo We 20 2 te 6 00 30<br>layer layer<br><!-- End of picture text -->



<!-- Start of picture text -->
Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8<br>1960 i} (<br>B 1985<br>e 2010 | “ \<br>Layer 9 Layer 10 Layer 11 Layer 12 Layer 13 Layer 14<br>1960-4 | Vy, , li<br>:B ses wm Me eS<br>&<br>2010<br>Layer 15 Layer 16 Layer 17 Layer 18 Layer 19 Layer 20<br>veo I<br>* My, \<br>B oes \\<br>e Me ‘<br>1960Completion1965 2010 1960‘completion1985 2010 1960Completion1985 2010 1960‘Completion1985 2010 1960‘Completion1985 2010 1960Completion1985 2010<br><!-- End of picture text -->

Figure 4: Steering experiment. _Top:_ the mean probability a completion is within two years of the target year it was steered to at each layer, grouped by release decade (left) and target decade (right). Clean baselines are shown with dashed lines. _Bottom:_ colour intensity (capped at 0.1) indicates the mean probability of a completion given the steering target. 

## **5 Discussion** 

In this work, we introduced the Manifold Probe, a supervised method for discovering representation manifolds in superposition. We hope that our probe will serve as a useful new tool for the mechanistic interpretability community. We see potential applications as a data-driven way to discover mechanisms of continuous computation such as counting (Gurnee et al., 2025; Wu et al., 2025) and modular arithmetic (Nanda et al., 2023a; Zhong et al., 2023), and as a tool to map out abstract concepts such as emotion (Sofroniew et al., 2026; Sun et al., 2026). There are also potential implications for scientific discovery, for example, in interpreting recently-discovered phylogenetic and hematopoietic representation manifolds in biological foundation models (Pearce et al., 2025; Wu et al., 2025). 

One limitation of our framework is that it implicitly assumes that the number of training samples is large relative to the dimension of the representation space, so that the sample estimates concentrate around their population counterparts. For state-of-the-art foundation models, this might require in the order of tens of thousands of examples. With smaller probing datasets, for good statistical performance it may be necessary to perform a preliminary principal component analysis to reduce the dimension of the representation space before fitting the probe. 

Finally, while interpretability tools such as this one might be used to develop mechanistic guardrails or to steer model behaviour to serve the goals of safety, alignment and security, they might be used to learn to bypass safety guardrails or produce intentionally harmful behaviour, which as a community, we must be mindful of as we progress our scientific understanding of AI systems. 

9 

## **Acknowledgements** 

The author would like to thank Jacob Davies, Jake Yukich, Can Rager, David Chanin, Nathalie Kirch, Patrick Rubin-Delanchy and Nick Whiteley for enlightening discussions on the topics of this paper. 

## **References** 

- Alain, G. and Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. _arXiv preprint arXiv:1610.01644_ . 

- Alain, G. and Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. In _5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings_ . OpenReview.net. 

- Bartlett, M. S. (1937). Properties of sufficiency and statistical tests. _Proceedings of the royal society of london. series a-mathematical and physical sciences_ , 160(901):268–282. 

- Bereska, L. and Gavves, E. (2024). Mechanistic interpretability for ai safety - a review. _Transactions on Machine Learning Research_ . Survey Certification, Expert Certification. 

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. _Transformer Circuits Thread_ . 

- Choi, B. J. and Weber, M. (2026). Latent structure of affective representations in large language models. _arXiv preprint arXiv:2604.07382_ . 

- Craven, P. and Wahba, G. (1978). Smoothing noisy data with spline functions: estimating the correct degree of smoothing by the method of generalized cross-validation. _Numerische mathematik_ , 31(4):377–403. 

- Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. (2024). Sparse autoencoders find highly interpretable features in language models. In _The Twelfth International Conference on Learning Representations_ . 

- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., and Olah, C. (2022). Toy Models of Superposition. _Transformer Circuits Thread_ . 

- Engels, J., Michaud, E. J., Liao, I., Gurnee, W., and Tegmark, M. (2025). Not All Language Model Features Are One-Dimensionally Linear. In _The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025_ . OpenReview.net. 

- Gorton, L. (2024). Curve Detector Manifolds in InceptionV1. 

- Gurnee, W., Ameisen, E., Kauvar, I., Tarng, J., Pearce, A., Olah, C., and Batson, J. (2025). When models manipulate manifolds: The geometry of a counting task. _Transformer Circuits Thread_ . 

- Gurnee, W. and Tegmark, M. (2024). Language models represent space and time. In _The Twelfth International Conference on Learning Representations_ . 

- Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. _Psychometrika_ , 23(3):187–200. 

- Karkada, D., Korchinski, D. J., Nava, A., Wyart, M., and Bahri, Y. (2026). Symmetry in language statistics shapes the geometry of model representations. _arXiv preprint arXiv:2602.15029_ . 

- Li, K., Patel, O., Viégas, F., Pfister, H., and Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. _Advances in Neural Information Processing Systems_ , 36:41451–41530. 

10 

- Marks, S. and Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In _First Conference on Language Modeling_ . 

- Mikolov, T., Yih, W.-t., and Zweig, G. (2013). Linguistic regularities in continuous space word representations. In _Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies_ , pages 746–751. 

- Modell, A., Rubin-Delanchy, P., and Whiteley, N. (2025). The origins of representation manifolds in large language models. _arXiv preprint arXiv:2505.18235_ . 

- Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. (2023a). Progress measures for grokking via mechanistic interpretability. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net. 

- Nanda, N., Lee, A., and Wattenberg, M. (2023b). Emergent linear representations in world models of self-supervised sequence models. _arXiv preprint arXiv:2309.00941_ . 

- Olah, C. (2024). What is a Linear Representation? What is a Multidimensional Feature? _Transformer Circuits Thread_ . 

- Pearce, M., Simon, E., Byun, M., and Balsam, D. (2025). Finding the tree of life in evo 2. _Goodfire_ . Correspondence to michael@goodfire.ai. 

- Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. (2024). Steering llama 2 via contrastive activation addition. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 15504–15522. 

- Rohe, K. and Zeng, M. (2023). Vintage factor analysis with varimax performs statistical inference. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 85(4):1037–1060. 

- Savietto, D., Campbell, D., Panisson, A., Nurisso, M., Petri, G., Cohen, J. D., and Perotti, A. (2026). The geometry of representational failures in vision language models. _arXiv preprint arXiv:2602.07025_ . 

- Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. _Artificial intelligence_ , 46(1-2):159–216. Publisher: Elsevier. 

- Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., and Lindsey, J. (2026). Emotion concepts and their function in a large language model. _Transformer Circuits Thread_ . 

- Sun, L., Yan, L., Lu, X., Lee, A., Zhang, J., and Shao, J. (2026). Valence-arousal subspace in llms: Circular emotion geometry and multi-behavioral control. _arXiv preprint arXiv:2604.03147_ . 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ . 

- Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. (2025). Steering language models with activation engineering. In _The Thirteenth International Conference on Learning Representations_ . 

- Wood, S. N. (2004). Stable and efficient multiple smoothing parameter estimation for generalized additive models. _Journal of the American Statistical Association_ , 99(467):673–686. 

- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. _Journal of the Royal Statistical Society Series B: Statistical Methodology_ , 73(1):3–36. 

- Wu, W., Zhou, Z., Riley, R., Abdulqader, M., Song, X., Kautsar, S., Egan, R., Hofmeyr, S., Liu, G., Goldhaber-Gordon, S., Yu, M., Ho, H., Liu, Y., Steindorff, A. S., Liu, F., Chen, F., Morgan-Kiss, R., Shi, L., Liu, H., and Wang, Z. (2025). Uncovering the genomic manifold via scalable learning from the global microbiome. _bioRxiv_ . 

11 

- Yocum, J., Allen, C., Olshausen, B., and Russell, S. (2025). Neural manifold geometry encodes feature fields. In _NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations_ . 

- Zhong, Z., Liu, Z., Tegmark, M., and Andreas, J. (2023). The clock and the pizza: Two stories in mechanistic explanation of neural networks. _Advances in neural information processing systems_ , 36:27223–27250. 

12 

## **A Additional figures** 



<!-- Start of picture text -->
Release dates of songs, movies and books<br><!-- End of picture text -->



<!-- Start of picture text -->
“19601980 2000 2020<br>TestA: 04444<br><!-- End of picture text -->



<!-- Start of picture text -->
—_—S<br>1960 1980 2000 2020<br>TestRi; 0.4230<br><!-- End of picture text -->



<!-- Start of picture text -->
1960 1980 2000 2020<br>TestA; 03008<br><!-- End of picture text -->



<!-- Start of picture text -->
1960 1980 2000 2020<br>TestA; 0.3688<br><!-- End of picture text -->



<!-- Start of picture text -->
1960 1980 2000 2020<br>TestRi: 03380<br><!-- End of picture text -->



<!-- Start of picture text -->
Geographic coordinates of places in the U.S.A.<br><!-- End of picture text -->



<!-- Start of picture text -->
Tesh? is<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?=<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? es<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?*<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? ss<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? in<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? fi<br><!-- End of picture text -->



<!-- Start of picture text -->
&<br><!-- End of picture text -->



<!-- Start of picture text -->
Tesh? is<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?i<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? ee<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?:<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? ic<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?a<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?= :<br><!-- End of picture text -->



<!-- Start of picture text -->
es,<br><!-- End of picture text -->



<!-- Start of picture text -->
Tesh?e<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?i<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? as<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? i;<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?5<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? i“ ‘<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? zs<br><!-- End of picture text -->



<!-- Start of picture text -->
&<br><!-- End of picture text -->



<!-- Start of picture text -->
Tesh?i<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? is<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?=<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? es<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? ts<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A?*<br><!-- End of picture text -->



<!-- Start of picture text -->
Test A? i?<br><!-- End of picture text -->



<!-- Start of picture text -->
®<br><!-- End of picture text -->

Figure 5: The top 5 time features, and top 32 space features from layer 16 of Llama 2-7b after applying a Varimax rotation to make them approximately sparse. The rotation makes the features interpretable. In particular, many of the space features localize on particular U.S. states, and the time features separate the decades from the 1950s to the 2010s. 

13 



<!-- Start of picture text -->
to Grouped by release decade to Grouped by target decade<br>= 160s,<br>Bos os = ioe<br>i<br>306<br>[= Bm<br>aa == 180s,ros<br>Fos os<br>é em steered<br>z sees Clean<br>302 02<br>oo of 4 6 8 10 Wis 16 1B 20 2 te 20 7 30 00 of 4 8 8 WW is Wo We 70 22 te W678 30<br>layer layer<br><!-- End of picture text -->

Figure 6: The mean probability that the model completes the prompt with a valid year in the steering experiment, grouped by release decade (left) and target decade (right). The dashed line shows the mean probability for the clean runs. The interventions have very little effect on the model’s ability to meaningfully complete the prompt, with the exception of steering to years in the 1950s. 



<!-- Start of picture text -->
Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8<br>1960 | i<br>B 1985 k<br>2010<br>& | My<br>Layer 9 Layer 10 Layer 11 Layer 12 Layer 13 Layer 14 oe<br>2200 4) ] r<br>B aves<br>&<br>2010 I}<br>Layer 15 Layer 16 Layer 17 Layer 18 Layer 19 Layer 20 on<br>7 7 1 a TT<br>|<br>% 1260 4 |, ] Wl I WT<br>aB ives JP Hy! wi! h<br>2010 L HAN Aivw biidiiit °<br>1960Completion1985 2010 1960‘Completion1985 2010 1960‘Completion1985 2010 1960‘Completion1985 2010 1960‘Completion1985 2010 1960Completion1985 2010<br><!-- End of picture text -->

Figure 7: Colour intensity indicates the standard deviation of the probability of a completion given the steering target in the steering experiment. 

14 

## **B An efficient algorithm to fit the regularization parameters** 

In this section, we discuss an optimization strategy which allows us to efficiently optimize the Manifold Probe objective in (6) while also selecting the regularization parameters _λw_ and _λf_ using a closed-form criterion such as GCV or REML which apply to linear predictors. 

Instead of directly employing the closed-form solution in Proposition 4, we propose the Alternating Least Squares procedure detailed in Algorithm 1. Here, we have used the notation _∥α∥_ Σ = _√α_<sup>_⊤_</sup> Σ _α_ and write _α ⊥_ Σ _β_ to mean _α_<sup>_⊤_</sup> Σ _β_ = 0. 

### **Algorithm 1:** Alternating least squares optimization of (6). 

**Input:** inital parameters _β_ 1<sup>(0)</sup><sup>_, . . . , β_</sup> _d_<sup>(0);</sup> **for** _k_ = 1 _, . . . , d_ **: for** _t_ = 1 _,_ 2 _, . . ._ **:** _w_ **-update** : _w_<sup>(</sup><sup>_t_+1)</sup> _←−_ argmin _∥y − Xw∥_<sup>2</sup> 2<sup>+ ˜</sup><sup>_λw∥w∥_</sup> 2<sup>2,</sup> _y_ = _Hβ_<sup>(</sup><sup>_t_)</sup> ; _w β_ **-update** : _β_<sup>(</sup><sup>_t_+1</sup><sup>_/_2)</sup> _←−_ argmin _∥y − Hβ∥_<sup>2</sup> 2<sup>+ ˜</sup><sup>_λfβ⊤Sβ_,</sup> _y_ = _Xw_<sup>(</sup><sup>_t_+1)</sup> ; _β⊥_ Σ _β_<sup>ˆ</sup> _k−_ 1 _,...,β_<sup>ˆ</sup> 1 _β_<sup>(</sup><sup>_t_+1)</sup> _←− β_<sup>(</sup><sup>_t_+1</sup><sup>_/_2)</sup> _/∥β_<sup>(</sup><sup>_t_+1</sup><sup>_/_2)</sup> _∥_ Σ; 

While it may seem needlessly inefficient to optimize (6) using Algorithm 1 rather than the closedform solution in Proposition 4, when we formulate the power-iteration procedure used to solve the generalized eigenvalue problem, we see that this is exactly equivalent to the alternating least squares procedure. Power-iteration is known to converge to the global solution under mild conditions on the initial value, and therefore we can guarantee that Algorithm 1 converges to the global solution under the same conditions. 

**Lemma 5.** _Suppose that ν_ ˆ _k > ν_ ˆ _k_ +1 _and wk_<sup>(0)</sup> _⊥ w_ ˆ _k for all k_ = 1 _, . . . , d, then for some λ_<sup>˜</sup> _w, λ_<sup>˜</sup> _f , Algorithm 1 converges to the global minimizer of_ (6) _. I.e._ 



The distinct eigenvalue condition is not strictly necessary and can be relaxed to simply _ν_ ˆ _d > ν_ ˆ _d_ +1, allowing repeated eigenvalues. The stricter condition is stated for simplicity. We provide a proof based on the power-iteration argument in Section B.2. 

By framing the optimization in this way, we can apply closed-form criteria designed for linear predictors such as GCV or REML to select the regularization parameters _λ_<sup>˜</sup> _w_ and _λ_<sup>˜</sup> _f_ at each iteration of the alternating least squares procedure. 

Viewed this way, we are also not restricted to quadratic penalties, and can use non-quadratic penalties such as the _ℓ_ 1 or elastic-net-type penalties, provided we have an efficient off-the-shelf regression solver which accomodates it. 

### **B.1 Efficient parametrization of the ridge regression problems** 

To efficiently perform the required computations in Algorithm 1, we perform some reparametrizations and matrix decompositions which allow us reduce each iteration to simple matrix multiplcations of size _p × p_ , removing the time-dependence on the number of samples and avoiding performing any matrix inversions. 

To do this, we first reparametrize the _β_ -problem to enforce the linear orthogonality constraints, and then reparametrize it again so that _S_ becomes the identity matrix. If _S_ is rank-deficient, we simply set its zero eigenvalues to some small positive constants to make it positive-definite to allow the 

15 

reparametrization. We are then left with a standard ridge regression problem. We note that from here on we parametrize and solve the _w_ -problem in exactly the same way, so we won’t discuss it separately. 

We next compute the singular value decomposition of _H_ as _H_ = _UDV_<sup>_⊤_</sup> , where all diagonal entries of _D_ are positive, and reparametrize the problem again to make _H_ = _UD_ . This ensures that _H_ has full-column rank and avoids some unnecessary matrix multiplications down the line. It is then straightforward to show that 



where y = _U_<sup>_⊤_</sup> _y_ and _r_ = _∥y∥_<sup>2</sup> 2<sup>_−∥_y</sup><sup>_∥_</sup> 2<sup>2isaconstantwhichdoesnotdependon</sup><sup>_β_.Notethat</sup> y = _U_<sup>_⊤_</sup> _Xw_ so as long as _U_<sup>_⊤_</sup> _X_ is pre-computed, this multiplication does not depend on the number of samples _n_ . The solution to the ridge regression problem (7) is 



which given y can be computed in _O_ ( _p_ ) time for any _λ_ . The GCV or REML criterion and their gradients and Hessians can also be computed efficiently using this reparametrization, allowing us to very efficiently select the regularization parameters at each iteration using Newton’s method. 

### **B.2 Proof of Lemma 5** 

We begin by showing the convergence of _β_<sup>(</sup><sup>_t_)</sup> for _k_ = 1. Once this is established, the convergence of _w_<sup>(</sup><sup>_t_)</sup> is trivial. We’ll consider the case that _k_ = 1 and note that the subsequent cases follow by a deflation argument. 

We recall that the ridge updates have closed forms 



and 

_β_<sup>(</sup><sup>_t_+1</sup><sup>_/_2)</sup> = ( _H_<sup>_⊤_</sup> _H_ + _λ_<sup>˜</sup> _f S_ )<sup>_−_1</sup> _H_<sup>_⊤_</sup> _Xw_<sup>(</sup><sup>_t_+1)</sup> = ( _H_<sup>_⊤_</sup> _H_ + _λ_<sup>˜</sup> _f S_ )<sup>_−_1</sup> _H_<sup>_⊤_</sup> _X_ ( _X_<sup>_⊤_</sup> _X_ + _λ_<sup>˜</sup> _wI_ )<sup>_−_1</sup> _X_<sup>_⊤_</sup> _Hβ_<sup>(</sup><sup>_t_)</sup> _._ We define the matrix _T_ = _LH_<sup>_⊤_</sup> _AH_ where _L_ = ( _H_<sup>_⊤_</sup> _H_ + _λ_<sup>˜</sup> _f S_ )<sup>_−_1</sup> and _A_ = _X_ ( _X_<sup>_⊤_</sup> _X_ + _λ_<sup>˜</sup> _wI_ )<sup>_−_1</sup> _X_<sup>_⊤_</sup> , so that 



A full _β_ -update is then given by 



where _∥a∥_ Σ = _√a_<sup>_⊤_</sup> Σ _a_ with Σ = _H_<sup>_⊤_</sup> _H/n_ . This shows that the sequence is a power-iteration for the matrix _T_ , and therefore as long as its largest eigenvalue is unique, and the initial value _β_<sup>(0)</sup> is not orthogonal to the corresponding eigenvector, it converges to the leading eigenvector of _T_ by a standard argument<sup>4</sup> . It remains to show that the leading eigenvector of _T_ is the same as the eigenvector _β_<sup>ˆ</sup> with the smallest eigenvalue _ν_ ˆ of the generalized eigenvalue problem 



Plugging in _M_ and Σ and rearranging we obtain 



and setting _λ_<sup>˜</sup> _f_ = _λf /_ (1 _− ν/n_ ˆ ), we have 



Multiplying both sides on the left by _L_ , we have 



which shows that _β_<sup>ˆ</sup> is the leading eigenvector of _T_ , which completes the proof. 

> 4see, for example, `https://en.wikipedia.org/wiki/Power_iteration` . 

16 

## **C Proof of Lemma 2** 

### **C.1 Proof of** (3) 

Let _f_ 1<sup>_⋆, . . . , f_</sup> _d_<sup>_⋆_beanysetofmean-zero,orthonormalfeatureswhichsatisfy(2),andlet</sup><sup>_F_=</sup> span _{f_ 1<sup>_⋆, . . . , f_</sup> _d_<sup>_⋆}_.We will show that span</sup><sup>_{f_1</sup><sup>_, . . . , fd}_=</sup><sup>_F_.We begin with the case</sup><sup>_k_= 1.By the</sup> law of iterated expectation, (3) can be written as 

therefore 

Now and so 



_f_ 1( _z_ ) = _w_<sup>_⊤_</sup> [ _ϕ_ ( _z_ ) + ¯ _η_ ]+ _b_ = _w_<sup>_⊤_</sup> ( _f_ 1<sup>_⋆_(</sup><sup>_z_)</sup><sup>_u_1+</sup><sup>_· · · f_</sup> _d_<sup>_⋆_(</sup><sup>_z_)</sup><sup>_ud_) = (</sup><sup>_w⊤u_1)</sup><sup>_f_</sup> 1<sup>_⋆_(</sup><sup>_z_)+</sup><sup>_· · ·_+(</sup><sup>_w⊤ud_)</sup><sup>_f_</sup> _d_<sup>_⋆_(</sup><sup>_z_)+</sup><sup>_w⊤η_¯+</sup><sup>_b._</sup> Now, _f_ 1 is constrained so that E[ _f_ 1] = 0, and given that E[ _f_ 1<sup>_⋆_] =</sup><sup>_· · ·_= E[</sup><sup>_f_</sup> _d_<sup>_⋆_] = 0, this implies that</sup> _w_<sup>_⊤_</sup> _η_ ¯ + _b_ = 0. Therefore, _f_ 1 is a linear combination of _f_ 1<sup>_⋆, . . . , f_</sup> _d_<sup>_⋆_.Since E(</sup><sup>_f_2)=1, at least one</sup> coefficient must be non-zero and therefore _f_ 1 _∈F_ . Next, we suppose that _f_ 1 _, . . . , fk−_ 1 _∈F_ and _f_ 1 _⊥· · · ⊥ fk−_ 1 for some _k ∈{_ 1 _, . . . , d}_ . We will show that _fk ∈F_ . 

Minimizing (3) subject to the constraint _fk ⊥ fk−_ 1 _, . . . , f_ 1 is equivalent to minimizing (3) replacing _x_ ( _z_ ) with the deflation 



As before, this is minimized by 



Since this is a linear combination of functions in _F_ , and _fk_ is constrained to be non-trivial, this implies that _fk ∈F_ . Therefore, by induction, _f_ 1 _, . . . , fd ∈F_ . 

Now the functions _f_ 1 _, . . . , fd_ are constrained to be orthogonal, and therefore the span _F_ , so 



### **C.2 Proof of** (4) 

To find the optimal ( _uk, ck_ ) that minimizes the population least-squares error 



we start by taking the gradient with respect to _c_ and setting it equal to zero to obtain 



which gives that 



where _x_ ¯ = E[ _x_ ] and we have used the fact that E[ _fk_ ] = 0. Substituting this back into (4), taking the gradient with respect to _u_ and setting it equal to zero gives us 



which implies that 



Since E[ _fk_<sup>2] = 1, this gives us that</sup> 



17 

Now _x_ ¯ = _η_ ¯, and so 

_u_ = E[ _fk_ ( _z_ )( _ϕ_ ( _z_ ) + _η_ ( _ξ_ ) _− η_ ¯)] = E[ _fk_ ( _z_ ) _ϕ_ ( _z_ )] + E[ _fk_ ( _z_ ) _η_ ( _ξ_ )] _−_ E[ _fk_ ( _z_ )¯ _η_ ] = E[ _fk_ ( _z_ ) _ϕ_ ( _z_ )] 

where the final two terms are zero by the assumption of independence between _z_ and _ξ_ , and the fact that E[ _fk_ ] = 0. Now, substituting in the decomposition of _ϕ_ ( _z_ ) gives us 

_u_ = E[ _fk_ ( _z_ )( _u_ 1 _f_ 1( _z_ ) + _· · ·_ + _udfd_ ( _z_ ))] = _u_ 1E[ _fkf_ 1] + _· · ·_ + _ud_ E[ _fkfd_ ] Now by assumption E[ _fkfj_ ] = 0 for all _j̸_ = _k_ , and E[ _fk_<sup>2] = 1, so we have that</sup> 



as required. 

## **D Proof of Proposition 4** 

We begin by observing that since _f ∈H_ , it can be written as _f_ ( _z_ ) = _β_<sup>_⊤_</sup> _h_ ( _z_ ) and the constraint � _ni_ =1<sup>_f_(</sup><sup>_zi_)=0 implies that</sup><sup>_β⊤h_¯=0.Therefore, we can write evaluations of</sup><sup>_f_in terms on the</sup> centered model matrix _H_ : 



Now, to minimize (6) with respect to _b_ , we take the gradient with respect to _b_ and set it equal to zero to obtain 



Since<sup>�</sup> _i_ =1<sup>_f_(</sup><sup>_zi_) = 0, we obtain that</sup><sup>_b_=</sup><sup>_−w⊤x_¯.Therefore we can write</sup> 

_f_ ( _zi_ ) _− w_<sup>_⊤_</sup> _xi − b_ = _f_ ( _zi_ ) _− w_<sup>_⊤_</sup> _xi −_ ( _−w_<sup>_⊤_</sup> _x_ ¯) = _f_ ( _zi_ ) _− w_<sup>_⊤_</sup> ( _xi − x_ ¯) = ( _Hβ_ ) _i −_ ( _Xw_ ) _i._ 

Recalling that _J_ ( _f_ ) = _β_<sup>_⊤_</sup> _Sβ_ , this means we can write (6) in matrix form as minimizing 



subject to _β_<sup>_⊤_</sup> Σ _β_ = 1 and _β_<sup>_⊤_</sup> Σ _β_<sup>ˆ</sup> _j_ = 0 for all _j < k_ . 

For fixed _β_ , the optimal _w_ is given by the ridge regression solution 



With _A_ = _X_<sup>_⊤_</sup> ( _X_<sup>_⊤_</sup> _X_ + _λwI_ ) _X_<sup>_⊤_</sup> as in the proposition, we have 



Now, we define _K_ = ( _X_<sup>_⊤_</sup> _X_ + _λwI_ )<sup>_−_1</sup> , and noting that _X_<sup>_⊤_</sup> _X_ = _K_<sup>_−_1</sup> _− λwI_ , we observe that 



and 



18 

Therefore 



Therefore, the objective function (8) can be written as 



By the Rayleigh-Ritz theorem, minimizer of this with respect to _β_ , subject to _β_<sup>_⊤_</sup> Σ _β_ = 1 is given by the generalized eigenvector of (4) with the smallest eigenvalue. Continuing sequentially, the generalized eigenvector with the _k_ th smallest eigenvalue gives the solution to (8) subject to _β_<sup>_⊤_</sup> Σ _β_<sup>ˆ</sup> _j_ = 0 for all _j < k_ . This completes the proof. 

19 

