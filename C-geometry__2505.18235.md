# The Origins of Representation Manifolds in Large Language Models

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** C-geometry — Modell, Rubin-Delanchy, Whiteley. Why representations form manifolds rather than isolated directions; geodesic distance as the meaningful metric.
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/2505.18235.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

# **The Origins of Representation Manifolds in Large Language Models** 

**Alexander Modell Patrick Rubin-Delanchy Nick Whiteley** Department of Mathematics School of Mathematics School of Mathematics Imperial College London University of Edinburgh University of Bristol `a.modell@imperial.ac.uk prd@ed.ac.uk nick.whiteley@bristol.ac.uk` 

### **Abstract** 

There is a large ongoing scientific effort in mechanistic interpretability to map embeddings and internal representations of AI systems into human-understandable concepts. A key element of this effort is the linear representation hypothesis, which posits that neural representations are sparse linear combinations of ‘almostorthogonal’ direction vectors, reflecting the presence or absence of different features. This model underpins the use of sparse autoencoders to recover features from representations. Moving towards a fuller model of features, in which neural representations could encode not just the presence but also a potentially continuous and multidimensional value for a feature, has been a subject of intense recent discourse. We describe why and how a feature might be represented as a manifold, demonstrating in particular that cosine similarity in representation space may encode the intrinsic geometry of a feature through shortest, on-manifold paths, potentially answering the question of how distance in representation space and relatedness in concept space could be connected. The critical assumptions and predictions of the theory are validated on text embeddings and token activations of large language models. 

### **1 Introduction** 

There is a large, ongoing, scientific effort in mechanistic interpretability to map internal representations used by AI systems into human-understandable concepts (Lin, 2024; Templeton et al., 2024), with broad implications for humanity including safety, alignment, and the future role of AI in science (Bostrom, 2014; Soares and Fallenstein, 2017; Wang et al., 2023). 

A key element of this effort is the linear representation hypothesis (LRH), which posits that language models represent human-interpretable features as directions in representation space, and that model representations are (literally) a sparse linear combination of these directions (Smolensky, 1990; Arora et al., 2018; Elhage et al., 2022). The methodology of sparse autoencoders (SAEs) (Elhage et al., 2022; Bricken et al., 2023) employs ideas from sparse coding (Elad, 2010) to estimate a dictionary of these directions from representations. 

This model and methodology reflect a radical goal of breaking representations down into basic, irreducible, atomic concepts which are meaningfully only described as present or absent (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024). Commonly cited examples are features such as `floppy_ears` , `Eiffel_Tower` , or `is_Arabic` , the presence of which it would presumably be useful for an algorithm to infer (corresponding e.g. to cat/dog classification, the topic of a question, the language of a query). 

It is generally accepted that this breakdown of representation space into purely atomic features does not tell the whole story (Smith, 2024; Mendel, 2024; Bussmann et al., 2024; Olah, 2024; Engels 

Preprint. Under review. 

et al., 2025). There is overwhelming empirical evidence that neural networks represent complex features in structures which unfold across multiple directions in potentially continuous, nonlinear ways: examples of curves (Hanna et al., 2023; Chang et al., 2022), swiss-roll-like manifolds (Cai et al., 2021), loops (Engels et al., 2025; Gorton, 2024), tori (Chang et al., 2022), hierarchical trees (Park et al., 2024) in real language models; topologically circular representations of numbers in toy models trained to perform modular arithmetic (Liu et al., 2022; Nanda et al., 2023a; Zhong et al., 2023; He et al., 2024) or simulated angular data (Olah and Batson, 2024), fractal geometry in simulated hidden Markov models (Shai et al., 2024); and broader phenomenology from local finite-state-automata (Bricken et al., 2023), to spatial ‘brain-like’ modularity (Li et al., 2025), to behaviour, such as deception (Templeton et al., 2024). 

SAEs are not made defunct by these discoveries, and in fact have often facilitated them through recombination of SAE directions (Bussmann et al., 2025; Engels et al., 2025). The LRH has been extended to allow this more flexible interpretation of the output of SAEs: 

**Definition 1** (Multidimensional linear representation hypothesis) **.** _There exists a collection of features labeled_ `f` _∈_ `F` _and associated subspaces V_ `f` _∈_ R<sup>_D_</sup> _such that the functional relationship between an input x ∈X and its representation_ Ψ( _x_ ) _is_ 



_where ρ_ `f` ( _x_ ) _is a non-negative scaling denoting the presence of the feature_ `f` _in x, and_ `F` ( _x_ ) = _{_ `f` : _ρ_ `f` ( _x_ ) _>_ 0 _} is the set of features which are present in x._ 

The standard LRH corresponds to the case where _v_ `f` ( _x_ ) is constant in _x_ (and _V_ `f` one-dimensional), and the extension above is a slightly relaxed and reparametrised version of that which appears in Engels et al. (2025). 

Our paper concerns the representation of a feature `f` as a _manifold_ in _V_ `f` , a phenomenon which is widely observed and intensely deliberated in the mechanistic interpretability community (Olah and Batson, 2024; Olah, 2024; Gorton, 2024; Engels et al., 2025). Despite numerous accounts (cited above) of a manifold clearly corresponding to some underlying ground truth feature (which may even be known exactly, e.g. in simulated data), there is no general description of this correspondence. 

We provide what we believe is a minimum viable mathematical theory to do this. Our most substantial, novel result establishes that under plausible hypotheses, cosine similarity in representation space encodes the intrinsic geometry of a feature through shortest, on-manifold paths. We develop this insight using concepts from _metric geometry_ – the theory of length and shape in metric spaces (Burago et al., 2001). The widespread use of cosine similarity across data science could suggest many other applications for this result. 

More generally, our work provides a (hopefully) accessible explanation of why manifold structure might arise in representation space, how its topology and geometry might reflect a human conceptualisation of the feature, and suggests some simple diagnostic plots and statistical checks to explore critical assumptions and predictions of the theory. 

Related to the problem of mechanistic interpretability, there is enormous interest in the use of _text embeddings_ (Li et al., 2020; Gao et al., 2021), which several entities now provide as a service, to support, for example, receiver augmented generation, search, recommendation, visualisation and classification. Again, these are representations which are usually unit vectors, and about which the cosine similarity is said to provide an effective measure of semantic similarity. The bulk of our results are also applicable to this area; in what follows, Ψ( _x_ ) can be viewed as a generic representation of some input _x_ , and _v_ `f` ( _x_ ) some unit-norm representation of a feature in _x_ . 

### **2 The continuous correspondence hypothesis** 

#### **2.1 What is a feature anyway?** 

Before we begin our discussion on how features are geometrically represented in language models, we really ought to pin down what exactly we mean by _“a feature”_ . 

**Definition 2.** _A_ feature _, labeled_ `f` _, is a metric space_ ( _Z_ `f` _,_ d `f` ) _._ 

2 



<!-- Start of picture text -->
English names for colours<br><!-- End of picture text -->



<!-- Start of picture text -->
ee“Sie: atest”<br>Fre ~<br><!-- End of picture text -->



<!-- Start of picture text -->
adteds<br><!-- End of picture text -->



<!-- Start of picture text -->
Years of the 20th century<br><!-- End of picture text -->



<!-- Start of picture text -->
gpt2-small<br><!-- End of picture text -->



<!-- Start of picture text -->
 e,4<br><!-- End of picture text -->



<!-- Start of picture text -->
é. :<br><!-- End of picture text -->



<!-- Start of picture text -->
Dates of the year<br><!-- End of picture text -->



<!-- Start of picture text -->
se get eePe ae<br>< fs is S<br><!-- End of picture text -->



<!-- Start of picture text -->
bos tyes<br><!-- End of picture text -->

Figure 1: Representation manifolds in large language models: colours, years and dates. The first and third example show text embeddings obtained from OpenAI’s `text-embedding-large-3` model from prompts relating to English names for colours and dates of the year, respectivly. The second example shows token activations from layer 7 of `GPT2-small` , which were studied in Engels et al. (2025). The token activations were processed via an SAE to extract a feature corresponding to years of the twentieth century as in Engels et al. (2025), and normalized to have norm one. For each example, we perform principal component analysis (PCA) to reduce the dimension to three and display the resulting point clouds from two perspectives. The embeddings of English names for colours are displayed in their respective colour value. Years are coloured from blue (1900) through green to yellow (1999), and dates are coloured from white (1st Janurary) through blue to black (1st July) through red and back to white. 

A metric space is simply a set equipped with a distance, and we find that it provides a simple yet highly expressive formal mathematical framework for discussing the abstract notion of a feature or concept. In particular, it allows us to readily talk about: 

1. _Atomic features: Z_ `f` a singleton set. 

2. _Hierarchical features: Z_ `f` a discrete set, d `f` a tree distance. 

3. _Continuous features: Z_ `f` an interval (equipped with e.g. d `f` ( _x, y_ ) = _|x − y|_ ), a circle (equipped with e.g. arc-length distance), multi-dimensional (equipped with e.g. the Euclidean distance), etc. 

We find that this formalism strikes a balance between the less expressive Euclidean and hyperspherical models often assumed in the learning theory literature (e.g. Zimmermann et al., 2021; Hyvärinen et al., 2024; Reizinger et al., 2025), and the more complicated and less accessible models which are often assumed in the disentanglement literature, such as Riemannian manifolds equipped with group structure (e.g. Higgins et al., 2018; Pfau et al., 2020). 

For any input _x_ on which the feature `f` is present (i.e. for which _ρn_ ( _x_ ) _>_ 0), we assume the existence of a value _z_ `f` ( _x_ ) which the input takes in _Z_ `f` . For example, if the feature `colour` is present in an input _x_ , then _ρ_ `colour` ( _x_ ) _>_ 0 and _z_ `f` ( _x_ ) might take a value describing the precise hue, saturation and lightness of that colour. 

3 



<!-- Start of picture text -->
Months of the year<br><!-- End of picture text -->



<!-- Start of picture text -->
?°<br><!-- End of picture text -->



<!-- Start of picture text -->
5<br>at u<br>i]<br>N<br>—<br>a?<br><!-- End of picture text -->



<!-- Start of picture text -->
Days of the week<br><!-- End of picture text -->



<!-- Start of picture text -->
Ls. .*<br><!-- End of picture text -->



<!-- Start of picture text -->
a<br>a<br>4<br>e 7<br>a<br><!-- End of picture text -->

Figure 2: Representation manifolds in token activations from layer 8 of Mistral 7B, processed via an SAE to extract representations of ‘months of the year’ and ‘days of the week’, as in Engels et al. (2025). We normalise the representations to have norm one, and perform PCA into three dimensions. The top-down view of the first two principal components, which was shown in Engels et al. (2025), obscures manifold structure which weaves through the third principal component. 

As a final note, we will assume throughout this paper that each _Z_ `f` is a compact set or, loosely speaking, “closed and bounded”: a standard assumption in manifold learning which avoids considerable and possibly distracting theoretical complications. 

#### **2.2 The continuous correspondence hypothesis** 

Given the multidimensional linear representation hypothesis, and our definition of a feature, perhaps the most basic hypothesis that one can make is that there is _some_ way of matching the representation directions _v_ `f` ( _x_ ) to the abstract features _z_ `f` ( _x_ ). 

**Hypothesis 1 (continuous correspondence).** _The features z_ `f` ( _x_ ) _and representation directions v_ `f` ( _x_ ) _are in a continuous, one-to-one correspondence. Formally, there is a continuous invertible map from the metric space into the hypersphere, ϕ_ `f` : _Z_ `f` _→_ S<sup>_D−_1</sup> _, with image M_ `f` := _ϕ_ ( _Z_ ) _, such that v_ `f` ( _x_ ) = _ϕ_ `f` ( _z_ `f` ( _x_ )) _for all x ∈X ._ 

Our correspondence hypothesis, combined with the prior assumption that _Z_ `f` is compact, has an immediate implication for the topological relationship between _Z_ `f` and _M_ `f` . 

**Proposition 1.** _Under Hypothesis 1, the map ϕ_ `f` : _Z_ `f` _→M_ `f` _is a homeomorphism._<sup>1</sup> 

Proposition 1 tells us that under the continuous correspondence hypothesis, we should expect the representations directions _v_ `f` ( _x_ ) to live on a _manifold M_ `f` that is topologically identical to _Z_ `f` . So, if _Z_ `f` is an interval, _M_ `f` is a one-dimensional curve in R<sup>_D_</sup> . If _Z_ `f` is a circle, then _M_ `f` is a loop. If _Z_ `f` is a discrete set comprising _m_ values, _M_ `f` is a discrete set comprising _m_ points. More generally, a homeomorphism preserves connected components, holes, branching points, and more. 

> 1This is simply a restatement of the well-established fact that a continuous invertible map over a compact domain has a continuous inverse (Sutherland, 2009, Proposition 13.26). 

4 

#### **2.3 Representations reflect the topology of features in LLMs** 

Figure 1 gives an indiction of the plausibility of Hypothesis 1 in some examples. The first and third subfigures show text embeddings obtained from OpenAI’s `text-embedding-large-3` model, with inputs corresponding to colours<sup>2</sup> , and dates of the year<sup>3</sup> respectively. This model returns 3,072 dimensional unit-norm embeddings which we reduce to three dimensions using PCA. We show two perspectives of each plot. In both cases, we see that the embeddings are roughly arranged around a loop which, perhaps after some stretching and bending, could seem consistent with the abstract circular model we might have for such concepts, such as the “colour wheel” or the “yearly cycle”. In particular, observe that the colours are arranged in the same order as the standard colour wheel of hue: red, purple, blue, green, yellow, orange, and back to red. 

The second subfigure shows token activations of years of the twentieth century in layer 7 of `GPT2-small` . This example is taken from Engels et al. (2025) who use a sparse autoencoder to attempt to disentangle the feature representations from the full superposed representation (see Section 5 of their paper for addition details of this procedure). We subselect only tokens corresponding to the years in question, normalize each activation vector to have norm one and perform PCA into three dimensions on the resulting vectors. One observes a clear one-dimensional curve which weaves and bends through the dimensions of the space, again reminiscent up to some geometric distortion of the standard human concept of a “time line”. 

Given what we see, our innate understanding of these concepts, and Proposition 1, we might conjecture that, allowing for error of different kinds, the shapes are homeomorphic to the following metric spaces: 

- **colour:** _Z_ `colour` = [0 _,_ 2 _π_ ), d `colour` ( _x, y_ ) = min( _|x − y|,_ 2 _π −|x − y|_ ), these angles corresponding to hues 0: red, ..., _π/_ 3: blue, ..., 2 _π/_ 3: yellow. 

**years:** _Z_ `years` = [1900 _,_ 1999], d `year` ( _x, y_ ) = _|x − y|_ 

**dates:** _Z_ `dates` = [0 _,_ 365), d `dates` ( _x, y_ ) = min( _|x − y|,_ 365 _−|x − y|_ ) 

In the case of years, a simple statistic to assess the conjecture of homeomorphism presents itself: the _rank_ correlation between the years and their corresponding position _along the manifold_ . We approximate position along the manifold using a _K_ -nearest neighbour graph with _K_ = 10 (picked as small as possible subject to the graph being connected), and rank the points according to weighted graph distance from the (mean) representation of 1900. The Kendall and Spearman rank correlations are 0.97 and over 0.99, respectively, telling us that the representations occur in very close to true temporal order along the manifold. 

In these examples it would clearly not be reasonable to say the shapes resembled circles or straight lines without any sort of geometric distortion, and in the coming section we provide a mechanistic argument for the presence of this geometric distortion in neural networks. 

This effect could be missed in some earlier papers due to 2D projection. The left and middle of panels of Figure 1 of Engels et al. (2025) show circular arrangements of day-of-the-week and month representations, but these seem subject to significant geometric distortion once we view the data in 3D, as in Figure 2. 

#### **2.4 Manifold geometry and computation** 

Our investigations (see Figure 1) and those of many others (see e.g. Ansuini et al., 2019; Cai et al., 2021; Chang et al., 2022; Hanna et al., 2023), have found not only that representations tend to live on low-dimensional manifolds, but that these manifolds curve and bend to occupy higher dimensional spaces. Why might it be advantageous for a language model to embed a concept in a larger dimension than its intrinsic topology seems to require? 

> 2These inputs are of are of the form “The color of the object is <color>. What color is the object?”. Color names and hex-codes were obtained from the XKCD color survey (Munroe, 2010), from which we removed entries with low saturation ( _<_ 0 _._ 4), high brightness ( _>_ 0 _._ 8), and whose names did not obviously refer to a color, such as fruits and gemstones. Some additional outliers were removed. 

> 3These inputs are of the form “1st January”, ”2nd January”, _. . ._ “31st December”. 

5 

To answer this question, we shall briefly illustrate how the space of functions which can be computed as a linear projection of _ϕ_ `f` ( _z_ ) relates to its geometry. Since linear operations are crucial component in how one layer of a neural network maps to the next, it seems a sensible working hypothesis that they would arrange their representations as to maximize the expressivity of these linear computations. 

For the purpose of this discussion, consider the case that _Z_ `f` is a unit interval _Z_ `f` = [0 _,_ 1]. If one simply wanted to be able to read _z_ from _ϕ_ `f` ( _z_ ) using a linear projection, then it is sufficient to represent _Z_ `f` as an arc on S<sup>_D−_1</sup> . For example, to set _ϕ_ `f` ( _z_ ) = _b_ 0( _z_ ) _v_ 0 + _b_ 1( _z_ ) _v_ 1 where _v_ 0 _, v_ 1 _∈_ S<sup>_D−_1</sup> are orthogonal unit-vectors, _b_ 1( _z_ ) _∝ z_ , and _b_ 0( _z_ ) is a function which ensures that _∥ϕ_ `f` ( _z_ ) _∥_ 2 = 1. In this way, the identity operation id( _z_ ) := _z_ can be computed via a linear projection id( _z_ ) _∝ v_ 1 _· ϕ_ `f` ( _z_ ). 

If instead, one wanted to be able to represent a richer class of functions of _z_ by linear projections of _ϕ_ `f` ( _z_ ), say, polynomials of order _p_ , then one could do this by setting _ϕ_ `f` ( _z_ ) = _b_ 0( _z_ ) _v_ 0 + _· · · bp_ +1( _z_ ) _vp_ +1 with _b_ 1( _z_ ) _∝_ 1 _, b_ 2( _z_ ) _∝ z, b_ 3 _∝ z_<sup>2</sup> _,_ etc... Such a map represents the interval [0 _,_ 1] as a continuous path which weaves through a _p_ + 2-dimensional subspace of S<sup>_D−_1</sup> . 

**Superposition.** Under the Linear Representation Hypothesis (1), the language model cannot access _ϕ_ `f` ( _z_ `f` ( _x_ )) directly, but must do so via Ψ( _x_ ). There is a generally agreed upon explanation, known as the superposition hypothesis, for how an algorithm might nonetheless be granted approximate access to _ϕ_ `f` ( _z_ `f` ( _x_ )), with only limited interference from other _ϕ_ `f` _′_ ( _z_ `f` _′_ ( _x_ )): features occur only sparsely (i.e. _ρ_ `f` ( _x_ ) = 0 for most `f` _∈_ `F` ), and are represented in almost-orthogonal subspaces (Elhage et al., 2021, 2022), a hypothesis which, in particular, would be consistent with the total number of features being substantially greater than the available representation dimensions<sup>4</sup> . 

If we assume (for a real-valued feature) that the identity id( _z_ ) is among the collection of functions linearly readable from _ϕ_ `f` , the superposition hypothesis also explains the efficacy of _linear probes_ (Alain and Bengio, 2017; Gurnee and Tegmark, 2023; Nanda et al., 2023b; Leask et al., 2024): low interference allows the feature of interest to be approximately recovered from the representation using linear regression. A similar story holds for discrete features accessed via linear classifiers. 

### **3 The interpretation of distance on representation manifolds** 

There is an open question in the mechanistic interpretability community about the meaning of _distance_ in representation space, perhaps well-summarised in the commentary of Olah and Batson (2024): 

- “We suspect this idea that feature manifolds many be embedded in more complex ways than their topology suggests, in order to achieve a given distance metric, may actually be quite deep and important.” 

If we accept there is a correspondence between features and their representations (Hypothesis 1), arguably the next most basic question we can ask is whether cosine similarity in representation space somehow tells us about distance between the corresponding feature values. 

**Hypothesis 2 (cosine similarity reflects distance).** _Locally, the cosine similarity between feature representations and the distance their corresponding feature values are inversely related. Formally, there is some function g_ `f` _with continuous second derivatives and with g_ `f`<sup>_′_(0)</sup><sup>_<_0</sup><sup>_, and some ϵ >_0</sup><sup>_,_</sup> _such that_ 

CosSim ( _ϕ_ `f` ( _z_ ) _, ϕ_ `f` ( _z_<sup>_′_</sup> )) = _g_ `f` (d `f` ( _z, z_<sup>_′_</sup> )<sup>2</sup> ) _,_ 

_for all z, z_<sup>_′_</sup> _∈Z_ `f` _such that_ d `f` ( _z, z_<sup>_′_</sup> ) _≤ ϵ._ 

Strengthening just Hypothesis 1 to both Hypotheses 1 and 2 has formidable consequences: there is an intrinsic sense in which _a feature and its representation are geometrically indistinguishable_ . This statement is made precise in Theorem 1. 

Metric spaces allow for a natural definition of a _path_ which, loosely speaking, captures the idea of a continuous route from one point to another and there is an associated definition of the _length_ of a path, denoted _L_ , which generalises the usual Euclidean notion of length (Burago et al., 2001). Formally, a 

> 4see, for example, Theorem 1 in the appendix of Engels et al. (2025). 

6 



<!-- Start of picture text -->
English names for colours<br><!-- End of picture text -->



<!-- Start of picture text -->
10<br>rats<br>09 7<br>s<br>Zoe§<br>2 %<br>rd Bins Sins ie<br>gor pa<br>os<br>£=0.692<br>000 005 010 015 020 0.25<br>‘Squared distance between hues<br><!-- End of picture text -->



<!-- Start of picture text -->
M75 -<br>oe<br>150 es<br>212s a<br>S104 2 oe<br>Sons<br>2 oso) :<br>°<br>0.25 :<br>0.00 =<br>00 01 02 03 O4 OS<br>Distance between hues<br><!-- End of picture text -->



<!-- Start of picture text -->
Years of the 20th century<br><!-- End of picture text -->



<!-- Start of picture text -->
10°<br>4<br>£07 i.<br>| a Me<br>Boe ‘ RG<br>uN<br>0s pa<br>§=0.837<br>J<br>° i 2 3<br>Squared distance between<br>log(2019 - years)<br><!-- End of picture text -->



<!-- Start of picture text -->
4 4<br>¥<br>23<br>=<br>32<br>2<br>1<br>°<br>0005.10<br>Distance between log(2019-years)<br><!-- End of picture text -->



<!-- Start of picture text -->
Dates of the year<br><!-- End of picture text -->



<!-- Start of picture text -->
1.00<br>0.90 7<br>oe<br>Boss] 9 ‘|<br>=3 co) mmabeseBi onsoeSi i<br>2 Seema<br>a se<br>3 ors<br>0.70<br>065<br>© 10000 20000 30000<br>‘Squared distance between dates<br><!-- End of picture text -->



<!-- Start of picture text -->
1a<br>Fe.<br>12<br>gro ag<br>S08 :<br>Bos<br>2 4<br>o2 P<br>oo<br>0 90 100~—«150<br>Distance between dates<br><!-- End of picture text -->

Figure 3: Evidence for Hypothesis 2 and its implications in Theorem 1. For each pair of representations, we plot their cosine similarities (first row) and estimated manifold distances (second row) against their (squared) distance in a putative metric space. We report the Chatterjee ( _ξ_ ) and Pearson ( _ρ_ ) correlation coefficients, respectively. Colours correspond to the colourmaps described in Figure 1. 

path in _Z_ `f` is a continuous mapping _η_ from some interval [ _a, b_ ] to _Z_ `f` , and the length of such a path is 



where the supremum is over all _n ≥_ 1 and _T_ = ( _t_ 0 _, t_ 1 _, . . . , tn_ ) such that _t_ 0 = _a ≤ t_ 1 _≤· · · ≤ tn_ = _b_ . 

Given a path in _Z_ `f` , we can think of the image of this path when mapped through _ϕ_ `f` , which we call the corresponding path on _M_ `f` . Its length is defined similarly, with Euclidean distance in place of d `f` . **Theorem 1.** _Let η be a path on Z_ `f` _of finite length and, assuming Hypothesis 1, let γ be the corresponding path on M_ `f` _. Then, under Hypothesis 2,_ 



A proof of Theorem 1 is given in the appendix. Theorem 1 tells us that we can recover the intrinsic geometry of _Z_ `f` , even though we know (almost) nothing about _g_ `f` : shortest paths on _M_ `f` correspond to shortest paths on _Z_ `f` , and their lengths are equal, up to a choice of unit (reflected by ~~�~~ _−_ 2 _g_ `f`<sup>_′_(0))</sup> 

#### **3.1 Geodesic distances on representation manifolds of LLMs are meaningful** 

We now explore the plausibility of Hypothesis 2 in the same colour, year, date examples considered in Section 2.3 and Figure 1. In all cases, we find indications of isometry, with some important caveats. 

Given a putative metric space, Hypothesis 2 suggests two diagnostic tests. The first (direct) approach is to plot cosine similarity against squared distance, to check if the first appears to be a decreasing function of the second, around zero, up to noise. We quantify the global strength of functional dependence using Chatterjee’s correlation coefficient _ξ_ (Chatterjee, 2021), which would be 1 if the 

7 



<!-- Start of picture text -->
Years of the 20th century<br><!-- End of picture text -->



<!-- Start of picture text -->
gpt2-small<br><!-- End of picture text -->



<!-- Start of picture text -->
10<br>> :<br>Zoeg "<br>ce8 06) Mag My, Mill<br>os sim<br>0 2000 4000 6000 8000<br>Squared distance between years<br><!-- End of picture text -->



<!-- Start of picture text -->
° fit aa<br>5g, siiFant atali<br>5 | itllae<br>le<br>% 20 40 60 80<br>Distance between years<br><!-- End of picture text -->

Figure 4: Evidence _against_ isometry with respect to the metric space _Z_ `years` = [1900 _,_ 1999], d `year` ( _x, y_ ) = _|x − y|_ . There is no clear regular linear relationship between distances in this metric space and estimated distances on the representation manifold. The colours indicate that distances between more recent years are expanded on the manifold. 

cosine similarity was a deterministic function of distance. These experiments are shown in the first row of Figure 3. 

The second (indirect) approach is to test Theorem 1: geodesic distance on _M_ `f` <u>(shortest path length)</u> should be linear in the geodesic distance on _Z_ `f` , up to noise (the slope being � _−_ 2 _g_<sup>_′_</sup> (0)). We estimate geodesics on _M_ `f` by constructing the _K_ -nearest-neighbours graph over the representations, and reporting weighted graph distance, _k_ chosen as small as possible subject to the graph being connected. We quantify the strength of isometry using Pearson’s correlation _ρ_ , which would be 1 if the distances were in a perfect proportional relationship. These experiments are shown in the second row of Figure 3. 

Across our experiments, we have found that a low-dimensional projection tends to be necessary for the representations to plausibly show isometry with a simple metric space. For our text embeddings, we find that projecting onto the first few (uncentered) principal components works well. The routine “low-rank” explanation that the remaining components are mostly noise seems disputable; these components often show clear structure. Our best explanation is that semantic similarity is much richer than the rudimentary metric spaces to which they are being compared. It is likely that we could achieve a deeper understanding of semantic similarity through improved metric space design. In the example of years, the process of extracting feature representation via an SAE automatically yields low-dimensional representations, so no PCA is applied in this case. 

Recall that we conjectured the following metric space for the years example: _Z_ `years` = [1900 _,_ 1999], d `year` ( _x, y_ ) = _|x − y|_ . Although we found a rank correlation near 1, indicating homeomorphism, the evidence of the tests above is _against_ isometry. The clearest indication in this direction is possibly provided by the right panel of Figure 4, which does not show a regular linear relationship, the colours suggesting that distances between more recent years are expanded on the manifold. 

In light of this, we consider a modified representation _Z_ `years` = _{_ log(2019 _−_ year) : year _∈_ [1900 _,_ 1999] _}_ , d `year` ( _x, y_ ) = _|x − y|_ , 2019 being the year GPT-2 was released (Radford et al., 2019). Observe that the rank correlation as computed in Section 2.3 remains unchanged: the two representations are homeomorphic, and cannot be distinguished on purely topological criteria. The tests are now in much stronger support of isometry. The top-middle plot of Figure 3 shows a trend which is clearly decreasing at zero, and a globally high functional dependence, 0.84. The bottom-middle panel of Figure 3 shows a clear linear fit, achieving a Pearson correlation of 0.99. 

Isometry is also found to be plausible for dates, and weakly plausible for colours, both with the original circular metric spaces conjectured in Section 2.3. The bottom-right panel shows a clear linear fit, achieving a Pearson correlation of 0.97. Observe in this case that the cosine similarity appears to be inconsistent with the metric for large distances, illustrating the point that Hypothesis 2 only requires an inverse functional relationship to hold locally. 

8 

### **4 Discussion, limitations and future work** 

This work provides a formal mathematical framework to explain and interpret representation manifolds in large language models. By modeling features as metric spaces, we are able to accurately characterise the topological and geometric properties of their representations under some basic hypotheses. 

We perform some preliminary investigations on internal representations from `GPT2-small` and text embeddings from OpenAI’s `text-embedding-large-3` model, which validate our theory and provide a nuanced, quantitative view of the correspondence between human concepts and their representation at different levels of geometric fidelity, namely homeomorphism and isometry. 

We find hints that these models encode distances in ways which are sometimes unexpected: years of the twentieth century in `GPT2-small` appear to be encoded on a logarithmic scale, with larger distances between more recent years, and colours in `text-embedding-large-3` appear to be encoded in a cycle of hues, rather than representations that other systems might have chosen, such as RGB or wavelength. 

#### **4.1 Limitations** 

In the spirit of scientific investigation, we have opted for a hypothesis-driven approach to structure discovery, in which we put down a possible metric space as a hypothesis and then assess evidence in favour or against. 

This “manual” approach is clearly not scalable, and moreover relies on there being some reasonable starting hypothesis for the metric space, which could be difficult for many features (say, emotions), as is evident from prior research (Li et al., 2023; Nanda et al., 2023b). There is an unexplored alternative approach of _learning the metric_ , but we do not know exactly how this would proceed given that an interpretable solution would presumably remain a requirement. 

In our experiments on `text-embedding-large-3` , we use PCA with some success to isolate simple human-understandable distances. However, in reality, we expect that the true notions of distance used by the language model are more complex: we find additional structure in further principal components and think it is possible that a language model could encode distances in a way that is mechanistically useful, but does not correspond to any existing human understanding of the feature. 

Another limitation of our approach relates to the fundamental statistical difficulty of estimating manifolds in the presence of noise (Genovese et al., 2012). Here, we have opted for a simple and interpretable approach of using the _K_ -nearest-neighbour graph to approximate the manifold, but this is prone to short-circuits causing enormous errors in the estimated manifold distances. It is often the case that one has to manually prune the graph in order to achieve reasonable manifold distance estimates, and we believe that more robust methodology for manifold estimation would be required to scale up our approach. 

#### **4.2 Implications for mechanistic interpretability research** 

In mechanistic interpretability, one of the underlying motivations for understanding representation geometry is to be able to steer model outputs by making interventions on their internal representations. For features represented on manifolds, our insights suggest a path forward for doing this: learn the map _ϕ_ `f` which maps the feature _Z_ `f` onto its representation manifold _M_ `f` . 

Sparse autoencoders provide a potentially promising avenue for this. We conjecture that the sparsity penalty of a sparse autoencoder, trained on representation manifold, will encourage it to learn a collection of dictionary vectors which trace the manifold (see Section 4 of Engels et al. (2025) for an argument for this phenomenon on representation subspaces). We also conjecture that much observed feature splitting in SAEs is a result of this. We hope our work will encourage development of “manifold-aware” SAEs. 

Finally, mechanistic interpretability is a nascent field of research which is still developing a common language, and we hope that researchers will find the formalism of a feature as a metric space to be a useful possibility in future scientific discourse. 

9 

### **References** 

- Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In _5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings_ . OpenReview.net, 2017. URL `https://openreview. net/forum?id=HJ4-rAVtl` . 

- Alessio Ansuini, Alessandro Laio, Jakob H Macke, and Davide Zoccolan. Intrinsic dimension of data representations in deep neural networks. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. Linear Algebraic Structure of Word Senses, with Applications to Polysemy. _Transactions of the Association for Computational Linguistics_ , 6:483–495, December 2018. ISSN 2307-387X. doi: 10.1162/tacl_a_ 00034. URL `https://direct.mit.edu/tacl/article/43451` . 

- Nick Bostrom. _Superintelligence: Paths, Dangers, Strategies_ . Oxford University Press, Oxford, UK, 2014. ISBN 978-0-19-967811-2. 

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. _Transformer Circuits Thread_ , 2023. URL `https://transformer-circuits.pub/ 2023/monosemantic-features/index.html` . 

- Dmitri Burago, Yuri Burago, Sergei Ivanov, and others. _A course in metric geometry_ . American Mathematical Society, 2001. 

- Bart Bussmann, Michael Pearce, Patrick Leask, Joseph Bloom, Lee Sharkey, and Neel Nanda. Showing SAE Latents Are Not Atomic Using Meta-SAEs. _Less Wrong_ , 2024. URL `https://www.lesswrong.com/posts/TMAmHh4DdMr4nCSr5/ showing-sae-latents-are-not-atomic-using-meta-saes` . 

- Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. Learning Multi-Level Features with Matryoshka Sparse Autoencoders. _arXiv preprint arXiv:2503.17547_ , 2025. 

- Xingyu Cai, Jiaji Huang, Yuchen Bian, and Kenneth Church. Isotropy in the contextual embedding space: Clusters and manifolds. In _International conference on learning representations_ , 2021. 

- Tyler A. Chang, Zhuowen Tu, and Benjamin K. Bergen. The Geometry of Multilingual Language Model Representations. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022_ , pages 119–136. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.9. URL `https://doi.org/10.18653/v1/2022.emnlp-main.9` . 

- Sourav Chatterjee. A new coefficient of correlation. _Journal of the American Statistical Association_ , 116(536):2009–2022, 2021. Publisher: Taylor & Francis. 

- Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. _arXiv preprint arXiv:2309.08600_ , 2023. 

- Michael Elad. _Sparse and redundant representations: from theory to applications in signal and image processing_ . Springer Science & Business Media, 2010. 

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A Mathematical Framework for Transformer Circuits. _Transformer Circuits Thread_ , 2021. URL `https://transformer-circuits.pub/2021/framework/index.html` . 

10 

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy Models of Superposition. _Transformer Circuits Thread_ , 2022. URL `https://transformer-circuits.pub/2022/ toy_model/index.html` . 

- Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not All Language Model Features Are One-Dimensionally Linear. In _The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025_ . OpenReview.net, 2025. URL `https://openreview.net/forum?id=d63a4AM4hb` . 

- Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021_ , pages 6894–6910. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021. EMNLP-MAIN.552. URL `https://doi.org/10.18653/v1/2021.emnlp-main.552` . 

- Christopher R Genovese, Marco Perone-Pacifico, Isabella Verdinelli, and Larry Wasserman. Minimax manifold estimation. _The Journal of Machine Learning Research_ , 13(1):1263–1291, 2012. 

- Liv Gorton. Curve Detector Manifolds in InceptionV1, August 2024. URL `https://livgorton. com/curve-detector-manifolds/` . 

- Wes Gurnee and Max Tegmark. Language models represent space and time. _arXiv preprint arXiv:2310.02207_ , 2023. 

- Michael Hanna, Roberto Zamparelli, David Mareˇcek, and others. The functional relevance of probed information: A case study. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pages 835–848. Association for Computational Linguistics, 2023. 

- Tianyu He, Darshil Doshi, Aritra Das, and Andrey Gromov. Learning to grok: Emergence of incontext learning and skill composition in modular arithmetic tasks. _Advances in Neural Information Processing Systems_ , 37:13244–13273, 2024. 

- Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. Towards a definition of disentangled representations. _arXiv preprint arXiv:1812.02230_ , 2018. 

- Aapo Hyvärinen, Ilyes Khemakhem, and Ricardo Monti. Identifiability of latent-variable and structural-equation models: from linear to nonlinear. _Annals of the Institute of Statistical Mathematics_ , 76(1):1–33, 2024. Publisher: Springer. 

- Patrick Leask, Bart Bussmann, and Neel Nanda. Calendar feature geometry in GPT-2 layer 8 residual stream SAEs. _Less Wrong_ , 2024. URL `https://www.lesswrong.com/posts/WsPyunwpXYCM2iN6t/ calendar-feature-geometry-in-gpt-2-layer-8-residual-stream` . 

- Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. On the Sentence Embeddings from Pre-trained Language Models. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020_ , pages 9119–9130. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.EMNLP-MAIN.733. URL `https: //doi.org/10.18653/v1/2020.emnlp-main.733` . 

- Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. _ICLR_ , 2023. Publisher: ICLR. 

- Yuxiao Li, Eric J Michaud, David D Baek, Joshua Engels, Xiaoqing Sun, and Max Tegmark. The geometry of concepts: Sparse autoencoder feature structure. _Entropy_ , 27(4):344, 2025. Publisher: MDPI. 

11 

- Johnny Lin. Neuronpedia: Interactive reference and tooling for analyzing neural networks, 2024. URL `https://www.neuronpedia.org` . 

- Ziming Liu, Ouail Kitouni, Niklas S Nolte, Eric Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An effective theory of representation learning. _Advances in Neural Information Processing Systems_ , 35:34651–34663, 2022. 

- Jake Mendel. SAE feature geometry is outside the superposition hypothesis. _AI Alignment Forum_ , 2024. URL `https://www.alignmentforum.org/posts/MFBTjb2qf3ziWmzz6/ sae-feature-geometry-is-outside-the-superposition-hypothesis` . 

- Randall Munroe. XKCD Color Name Survey Results, 2010. URL `https://xkcd.com/color/ rgb/` . 

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023a. URL `https://openreview.net/forum?id=9XFSbDPmdW` . 

- Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. _arXiv preprint arXiv:2309.00941_ , 2023b. 

- Chris Olah. What is a Linear Representation? What is a Multidimensional Feature? _Transformer Circuits Thread_ , 2024. URL `https://transformer-circuits.pub/2024/july-update/ index.html#linear-representations` . 

- Chris Olah and Josh Batson. Feature Manifold Toy Model. _Transformer Circuits Thread_ , 2024. URL `https://transformer-circuits.pub/2023/may-update/index. html#feature-manifolds` . 

- Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. The geometry of categorical and hierarchical concepts in large language models. _arXiv preprint arXiv:2406.01506_ , 2024. 

- David Pfau, Irina Higgins, Alex Botev, and Sébastien Racanière. Disentangling by subspace diffusion. _Advances in Neural Information Processing Systems_ , 33:17403–17415, 2020. 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. 2019. 

- Patrik Reizinger, Alice Bizeul, Attila Juhos, Julia E. Vogt, Randall Balestriero, Wieland Brendel, and David A. Klindt. Cross-Entropy Is All You Need To Invert the Data Generating Process. In _The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 2428, 2025_ . OpenReview.net, 2025. URL `https://openreview.net/forum?id=hrqNOxpItr` . 

- Adam S. Shai, Lucas Teixeira, Alexander Gietelink Oldenziel, Sarah Marzen, and Paul M. Riechers. Transformers Represent Belief State Geometry in their Residual Stream. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, _Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024_ , 2024. URL `http://papers.nips.cc/paper_files/paper/2024/hash/ 8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html` . 

- Lewis Smith. The ‘strong’ feature hypothesis could be wrong. _AI Alignment Forum_ , 2024. URL `https://www.alignmentforum.org/posts/tojtPCCRpKLSHBdpn/ the-strong-feature-hypothesis-could-be-wrong` . 

- Paul Smolensky. Tensor product variable binding and the representation of symbolic structures in connectionist systems. _Artificial intelligence_ , 46(1-2):159–216, 1990. Publisher: Elsevier. 

- Nate Soares and Benya Fallenstein. Agent foundations for aligning machine intelligence with human interests: a technical research agenda. _The technological singularity: Managing the journey_ , pages 103–125, 2017. Publisher: Springer. 

12 

- Wilson A. Sutherland. _Introduction to Metric and Topological Spaces_ . Oxford University Press, 2009. 

- Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L. Turner, Callum McDougall, Monte MacDiarmid, Alex Tamkin, Esin Durmus, Tristan Hume, Francesco Mosconi, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Christopher Olah, and Tom Henighan. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. _Transformer Circuits Thread_ , 2024. URL `https://transformer-circuits.pub/2024/scaling-monosemanticity/` . 

- Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, Anima Anandkumar, Karianne Bergen, Carla P. Gomes, Shirley Ho, Pushmeet Kohli, Joan Lasenby, Jure Leskovec, Tie\-Yan Liu, Arjun Manrai, Debora Marks, Bharath Ramsundar, Le Song, Jimeng Sun, Jian Tang, Petar Veliˇckovi´c, Max Welling, Linfeng Zhang, Connor W. Coley, Yoshua Bengio, and Marinka Zitnik. Scientific discovery in the age of artificial intelligence. _Nature_ , 620(7972):47–60, August 2023. doi: 10.1038/ s41586-023-06221-2. URL `https://www.nature.com/articles/s41586-023-06221-2` . 

- Ziqian Zhong, Ziming Liu, Max Tegmark, and Jacob Andreas. The clock and the pizza: Two stories in mechanistic explanation of neural networks. _Advances in neural information processing systems_ , 36:27223–27250, 2023. 

- Roland S Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel. Contrastive learning inverts the data generating process. In _International conference on machine learning_ , pages 12979–12990. PMLR, 2021. 

13 

## **Appendix** 

### **A Code to reproduce the experiments in this paper** 

Code to reproduce the experiments in this paper is made available at the GitHub repository `https: //github.com/alexandermodell/Representation-Manifolds` . 

### **B Supporting definitions and proof of Theorem 1** 

Hypotheses 1 and 2, and the statement of the Theorem 1 concern the metric space ( _Z_ `f` _,_ d `f` ), and functions _ϕ_ `f` and _g_ `f` associated with some particular feature `f` . To de-clutter the proof we remove dependence on `f` from the notation, and write simply ( _Z,_ d), _ϕ_ and _g_ . 

We shall need the following definitions, informed by (Burago et al., 2001). A _path_ in _Z_ is a continuous mapping _η_ from some interval [ _a, b_ ] to _Z_ . The length of such a path is: 



where the supremum is over all _T_ = ( _t_ 0 _, t_ 1 _, . . . , tn_ ) such that _n ≥_ 1 and _t_ 0 = _a ≤ t_ 1 _≤· · · ≤ tn_ = _b_ . A path is said to be of _finite length_ , or equivalently called _rectifiable_ , if _L_ ( _η_ ) _< ∞_ . For [ _a_<sup>_′_</sup> _, b_<sup>_′_</sup> ] _⊆_ [ _a, b_ ] we write _L_ ( _η, a_<sup>_′_</sup> _, b_<sup>_′_</sup> ) for the length of the restriction of _η_ to [ _a_<sup>_′_</sup> _, b_<sup>_′_</sup> ]. 

Any rectifiable path _η_ : [ _a, b_ ] _→Z_ admits a _unit-speed parameterisation,_ meaning _η_ has the representation _η_ = _η_ ˜ _◦ φ_ where _η_ ˜ : [0 _, L_ ( _η_ )] _→Z_ is a path, _φ_ is a continuous, nondecreasing map from [ _a, b_ ] to [0 _, L_ ( _η_ )], and _L_ (˜ _η, s, t_ ) = _t − s_ (Burago et al., 2001, Prop. 2.5.9). Adopting this parameterisation does not change overall length of the path, in the sense that the stated properties of _η_ ˜ imply _L_ (˜ _η_ ) = _L_ (˜ _η,_ 0 _, L_ ( _η_ )) = _L_ ( _η_ ). 

We shall also consider paths on the unit hyper-sphere S<sup>_D−_1</sup> := _{x ∈_ R<sup>_D−_1</sup> : _∥x∥_ 2 = 1 _}_ . The length of such a path, i.e., a continuous mapping _γ_ : [ _a, b_ ] _→_ S<sup>_D−_1</sup> is: 



where again the supremum is over all _T_ = ( _t_ 0 _, t_ 1 _, . . . , tn_ ) such that _n ≥_ 1 and _t_ 0 = _a ≤ t_ 1 _≤· · · ≤ tn_ = _b_ . 

_Proof of Theorem 1._ Since the claim of the theorem depends on _η_ only through its length, we can assume w.l.o.g. that we are considering the unit-speed parameterisation of _η_ . That is [ _a, b_ ] = [0 _, L_ ( _η_ )] and _η_ : [0 _, L_ ( _η_ )] _→Z_ with _L_ ( _η, s, t_ ) = _t − s_ . Under Hypothesis 1, _ϕ_ is continuous, hence _γ_ : [0 _, L_ ( _η_ )] _→_ S<sup>_D−_1</sup> defined by _γt_ = _ϕ_ ( _ηt_ ) is a path on S<sup>_D−_1</sup> . 

For any _T_ = ( _t_ 0 _, t_ 1 _, , . . . , tn_ ) such that _n ≥_ 1 and _t_ 0 = 0 _≤ t_ 1 _≤· · · ≤ tn_ = _L_ ( _η_ ), introduce the notation: 



Fix any _δ >_ 0. We shall prove that there exists _Tδ_ such that : 







14 

which implies the claim of the theorem. We shall construct _Tδ_ in the form _Tδ_ := _Tδ_<sup>(1)</sup> _∪Tδ_<sup>(2)</sup> _∪Tδ_<sup>(3)</sup> , i.e., _Tδ_<sup>(</sup><sup>_i_)</sup> _⊆Tδ_ for _i_ = 1 _,_ 2 _,_ 3, where _Tδ_<sup>(</sup><sup>_i_)</sup> are defined in the remainder of the proof. 

We first consider a difference of the form ��� _S_ ( _γ, ·_ ) _−_ � _−_ 2 _g_<sup>_′_</sup> (0) _S_ ( _η, ·_ )��� as appears in (3). Noting that the mapping _ϕ_ by definition satisfies _∥ϕ_ ( _z_ ) _∥_ 2 = 1 for all _z_ , under Hypothesis 2, there exists _ϵ >_ 0 such that if d( _z, z_<sup>_′_</sup> ) _< ϵ_ , then _⟨ϕ_ ( _z_ ) _, ϕ_ ( _z_<sup>_′_</sup> ) _⟩_ 2 = _g_ (d( _z, z_<sup>_′_</sup> )<sup>2</sup> ). Let _C >_ 0 be any finite constant such that sup _r≤ϵ |g_<sup>_′′_</sup> ( _r_ ) _| ≤ C_ . Such a constant exists because _g_ is _C_<sup>2</sup> by assumption. 



Using the fact that _η_ is unit-speed parameterised, it follows that, for 1 _≤ i ≤ n_<sup>(2)</sup> , 



Now consider any _T_ = ( _t_ 0 _, t_ 1 _, , . . . , tn_ ) with _n ≥ n_<sup>(2)</sup> , _t_ 0 = 0 , _tn_ = _L_ ( _η_ ) such that _Tδ_<sup>(2)</sup> _⊆T_ . Unit-speed parameterisation of _η_ combined with _Tδ_<sup>(2)</sup> _⊆T_ implies: 



and it follows from the definition of length and the triangle inequality that d( _ηti, ηti−_ 1) _≤ L_ ( _η, ti, ti−_ 1) for all _i_ = 1 _, . . . , n_ . Therefore using (6), we have: 



Using _∥ϕ_ ( _z_ ) _∥_ 2 = 1 for all _z_ , _γt_ = _ϕ_ ( _ηt_ ), the upper bound by _ϵ_ in (7) to enable the substitution � _ϕ_ ( _ηti_ ) _, ϕ_ ( _ηti−_ 1)�2<sup>=</sup><sup>_g_</sup> �d( _ηti, ηti−_ 1)<sup>2�</sup> , and taking a Taylor expansion of _g_ about zero, we have: 



where _ci_ is some point in the interval [0 _,_ d( _ηti, ηti−_ 1)<sup>2</sup> ]. Under Hypothesis 2, we have _g_<sup>_′_</sup> (0) _<_ 0. Then using (9) and lemma 1 with _α_ = _∥γti − γti−_ 1 _∥_ 2 and _β_ = � _−_ 2 _g_<sup>_′_</sup> (0)d( _ηti, ηti−_ 1), 

so that 



15 

where the final inequality uses (7). In summary, we have shown that 



= Now consider _|L_ ( _γ_ ) _− S_ ( _γ, ·_ ) _|_ as appears in (2). By the definition of _L_ ( _γ_ ), there exists _Tδ_<sup>(1)</sup> ( _t_<sup>(1)</sup> 0 = 0 _, t_<sup>(1)</sup> 1<sup>_, . . . , t_(1)</sup> _n_<sup>(1)=</sup><sup>_L_(</sup><sup>_η_)) such that:</sup> 



By applying the triangle inequality to the summands in _S_ ( _γ, Tδ_<sup>(1)</sup> ) we have for any _T_ with _Tδ_<sup>(1)</sup> _⊆T_ , _S_ ( _γ, Tδ_<sup>(1)</sup> ) _≤ S_ ( _γ, T_ ) _≤ L_ ( _γ_ ), hence 



Now consider � _−_ 2 _g_<sup>_′_</sup> (0) _|S_ ( _η, ·_ ) _− L_ ( _η_ ) _|_ as appears in (4). By similar arguments to those used above do establish (11), there exists _Tδ_<sup>(3)</sup> such that 



With _Tδ_ := _Tδ_<sup>(1)</sup> _∪Tδ_<sup>(2)</sup> _∪Tδ_<sup>(3)</sup> , the implications (10), (11), (12) together tell us that the inequality (5) holds, and this completes the proof of the theorem. 

**Lemma 1.** _For any α, β ≥_ 0 _, |α − β| ≤|α_<sup>2</sup> _− β_<sup>2</sup> _|_<sup>1</sup><sup>_/_2</sup> _._ 

_Proof._ W.l.o.g., assume _α ≥ β_ . Using the triangle inequality for the Euclidean norm in R<sup>2</sup> , _α_ = ( _β_<sup>2</sup> + _α_<sup>2</sup> _− β_<sup>2</sup> )<sup>1</sup><sup>_/_2</sup> _≤ β_ + ( _α_<sup>2</sup> _− β_<sup>2</sup> )<sup>1</sup><sup>_/_2</sup> , i.e., _α − β ≤_ ( _α_<sup>2</sup> _− β_<sup>2</sup> )<sup>1</sup><sup>_/_2</sup> . 

16 

