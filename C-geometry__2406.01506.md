# The Geometry of Categorical and Hierarchical Concepts

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** C-geometry — Hierarchy represented as orthogonality and simplices — relevant to Tyler's nested class structure (2-class inside 3-class tasks).
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/2406.01506.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

Published as a conference paper at ICLR 2025 

# THE GEOMETRY OF CATEGORICAL AND HIERARCHICAL CONCEPTS IN LARGE LANGUAGE MODELS 

**Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch** University of Chicago 

## ABSTRACT 

The linear representation hypothesis is the informal idea that semantic concepts are encoded as linear directions in the representation spaces of large language models (LLMs). Previous work has shown how to make this notion precise for representing binary concepts that have natural contrasts (e.g., _{_ male _,_ female _}_ ) as _directions_ in representation space. However, many natural concepts do not have natural contrasts (e.g., whether the output is about an animal). In this work, we show how to extend the formalization of the linear representation hypothesis to represent features (e.g., is ~~a~~ nimal) as _vectors_ . This allows us to immediately formalize the representation of categorical concepts as polytopes in the representation space. Further, we use the formalization to prove a relationship between the hierarchical structure of concepts and the geometry of their representations. We validate these theoretical results on the Gemma and LLaMA-3 large language models, estimating representations for 900+ hierarchically related concepts using data from WordNet.<sup>1</sup> 

## 1 INTRODUCTION 

Understanding how high-level semantic meaning is encoded in the representation spaces of large language models (LLMs) is a fundamental problem in interpretability. A particularly promising avenue is the _linear representation hypothesis_ (e.g., Mikolov et al., 2013; Elhage et al., 2022; Nanda et al., 2023; Gurnee & Tegmark, 2024; Park et al., 2024). This is the informal hypothesis that semantic concepts are represented _linearly_ in the representation spaces of LLMs. To assess the validity of this hypothesis, and systematically build tools on top of it, we must make precise what it means for a concept to be linearly represented, and understand how semantics are encoded in (the geometry of) the representation spaces. 

Focusing on the final softmax layer, Park et al. (2024) give a formalization for the case of binary concepts that can be defined by counterfactual pairs of words. For example, the concept male _⇒_ female is formalized using the counterfactual pairs _{_ (“man” _,_ “woman”) _,_ (“king” _,_ “queen”) _, . . . }_ . They prove that such binary concepts have a well-defined linear representation as a direction in the representation space. They further connect semantic structure and representation geometry by showing that, under a suitable inner product, _causally separable_ concepts that can be freely manipulated (e.g., male _⇒_ female and french _⇒_ english) are represented by orthogonal directions. 

However, this formalization is limited: many natural concepts cannot be defined by counterfactual pairs of words. For example, simple binary features (e.g., is ~~a~~ nimal) or categorical concepts (e.g., _{_ mammal _,_ bird _,_ reptile _,_ fish _}_ ) do not admit such formalizations. Additionally, it is not clear how semantic relationships beyond causal separability are encoded in the representation space. In particular, we are interested in this paper in understanding how hierarchical relationships between concepts are encoded in the representation space. That is, what is the relationship between the representations of `animal` , `mammal` , and `dog` ? 

In this paper, we extend the linear representation hypothesis as follows: 

1. We show how to move from representations of binary concepts as _directions_ to representations as _vectors_ . As a straightforward consequence, this allows us to represent categorical 

> 1Code is available at github.com/KihoPark/LLM ~~C~~ ategorical ~~H~~ ierarchical ~~R~~ epresentations. 

1 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
Bird<br>Animal Fish Elephant<br>Organism Tiger<br>GD <a Plant SSDNS Mammal ZzZrLL Cd) Dog<br>Reptile<br>NSESS<br>Reptile Elephant<br>Animal<br>Organism<br>⊕ ⊕ ⊕<br>Bird Mammal<br>Plant<br>Fish Tiger Dog<br>(a) Pictorial depiction of the representation of hierarchically related concepts.<br>oo plant animal vs mammal bird<br>reptile<br>1s bagfos 4}3 aa:<br>so ‘6 10 N<br>ee 8 \<br>2s .R6\<br>a<br>aa st : Ege. morinat<br>i 2h bird sale es<br>5°) mma animalane Bes “ep4 Sihet:<br>-75| gam mammal 2, SS<br>he : a 8<br>~109,2,00 -15 50-25 00-25-5075 100 8toy ——@ vere 0 2<br>(b) Hierarchy is encoded as orthogonality in Gemma. (c) Categorical concepts are represented as polytopes in Gemma.<br><!-- End of picture text -->

Figure 1: In the representation spaces of LLMs, hierarchically related concepts (such as plant _⇒_ animal and mammal _⇒_ bird) live in orthogonal subspaces, while categorical concepts are represented as polytopes. The top panel illustrates the structure; the bottom panels show the measured representation structure in the Gemma LLM. See Section 5 and Appendix A for details. 

   - concepts (e.g., _{_ mammal _,_ bird _,_ reptile _,_ fish _}_ ) as polytopes where each vertex is the vector representation of one of the elements of the concept (e.g., is ~~b~~ ird). 

2. Using this result, we show that semantic hierarchy between concepts is encoded geometrically as orthogonality between representations, in a manner we make precise. 

3. Finally, we empirically validate these theoretical results on the Gemma (Mesnard et al., 2024) and LLaMA-3 (Dubey et al., 2024) LLMs. To that end, we extract concepts from the WordNet hierarchy (Miller, 1995), estimate their representations, and show that the geometric structure of the representations aligns with the semantic hierarchy of WordNet. 

The final structure is remarkably simple, and is summarized in Figure 1. 

## 2 PRELIMINARIES 

We begin with some necessary background. 

- 2.1 LARGE LANGUAGE MODELS 

For the purposes of this paper, we consider a large language model to consist of two parts. The first part is a function _λ_ that maps an input text _x_ to a vector _λ_ ( _x_ ) in a representation space Λ _≃_ R<sup>_d_</sup> . This is the function given by the stacked transformer blocks. We take _λ_ ( _x_ ) to be the output of the final layer at the final token position. The second part is an unembedding layer that assigns a vector _γ_ ( _y_ ) in an unembedding space Γ _≃_ R<sup>_d_</sup> to each token _y_ in the vocabulary. Together, these define a 

2 

Published as a conference paper at ICLR 2025 

sampling distribution over tokens via the softmax distribution: 



We note that the results that follow rely on the duality between the embedding and unembedding spaces, and the softmax link between them. Accordingly, we do not address the “internal” structure of the LLMs. However, we are optimistic that a clear understanding of the softmax geometry will shed light on this as well. 

### 2.2 CONCEPTS 

We formalize a concept as a latent variable _W_ that is caused by the context _X_ and causes the output _Y_ . That is, a concept is a thing that could—in principle—be manipulated to affect the output of the language model. In the particular case where a concept is a binary variable with a word-level counterfactual, we can identify the variable _W_ with the counterfactual pair of outputs ( _Y_ (0) _, Y_ (1)). Concretely, we can identify male _⇒_ female with ( _Y_ (0) _, Y_ (1)) _∈R {_ (“man” _,_ “woman”) _,_ (“king” _,_ “queen”) _,_ (“he” _,_ “her”) _, . . . }_ . We emphasize that the notion of a concept as a latent variable that affects the output is more general than the counterfactual binary case. 

Given a pair of concept variables _W_ and _Z_ , we say that _W_ is _causally separable_ with _Z_ if the potential outcome _Y_ ( _W_ = _w, Z_ = _z_ ) is well-defined for all _w, z_ . That is, two variables are causally separable if they can be freely manipulated—e.g., we can change the output language and the sex of the subject freely, so these concepts are causally separable. 

### 2.3 CAUSAL INNER PRODUCT AND LINEAR REPRESENTATIONS 

We are trying to understand how concepts are represented. At this stage, there are two distinct representation spaces: Λ and Γ. The former is the space of context embeddings, and the latter is the space of token unembeddings. We would like to unify these spaces so that there is just a single notion of representation. 

Park et al. (2024) show how to achieve this unification via a “Causal Inner Product”. This is a particular choice of inner product that respects the semantics of language in the sense that the linear representations of (binary, counterfactual) causally separable concepts are orthogonal under the inner product. Their result can be understood as saying that there is some invertible matrix _A_ and constant vector ¯ _γ_ 0 such that, if we transform the embedding and unembedding spaces as 



then the Euclidean inner product in the transformed spaces is the causal inner product, and the Riesz isomorphism between the embedding and unembedding spaces is simply the usual vector transpose operation. We can estimate _A_ as the whitening operation for the unembedding matrix. Following this transformation, we can think of the embedding and unembedding spaces as the same space, equipped with the Euclidean inner product.<sup>2</sup> 

Notice that the softmax probabilities (eq. (2.1)) are unchanged for any _A_ and _γ_ ¯0, so this transformation does not affect the model’s behavior. The vector _γ_ ¯0 defines an origin for the unembedding space, and can be chosen arbitrarily. We give a particularly convenient choice below. 

In this unified space, the linear representation of a binary concept _W ∈R {_ 0 _,_ 1 _}_ is defined as: 

**Definition 1.** A vector _ℓ_<sup>¯</sup> _W_ is a linear representation of a binary concept _W_ if for all contexts _ℓ_ , and all concept variables _Z_ that are causally separable with _W_ , we have, for all _α >_ 0, 





That is, the linear representation is a direction in the representation space that, when added to the context, increases the probability of the concept, but does not affect the probability of any off-target 

> 2We are glossing over some technical details here; see Park et al. (2024) for details. 

3 

Published as a conference paper at ICLR 2025 

concept. The representation is merely a direction because _αℓ_<sup>¯</sup> _W_ is also a linear representation for any _α >_ 0 (i.e., there is no notion of magnitude). In the case of concepts corresponding to counterfactual pairs of words, this direction can be shown to be proportional to the “linear probing” direction, and proportional to _g_ ( _Y_ (1)) _− g_ ( _Y_ (0)) for any counterfactual pair _Y_ (1) _, Y_ (0) that differ on _W_ . 

## 3 GENERAL CONCEPTS AND HIERARCHICAL STRUCTURE 

Our high-level strategy will be to build up from binary concepts to more complex structure. We begin by defining the basic building blocks. 

**Binary and Categorical Concepts** The most general concept we address in this paper is a categorical concept, which refers to any concept corresponding to a categorical latent variable. This includes binary concepts as a special case. We consider two kinds of binary concept: _binary features_ and _binary contrasts_ . A binary feature _W ∈R {_ not ~~w~~ _,_ is ~~w~~ _}_ is an indicator of whether the output has the attribute _w_ . For example, if the feature `is` ~~`a`~~ `nimal` is true, then the output will be about an animal. A binary contrast a _⇒_ b _∈R {a, b}_ is a binary variable that contrasts two specific attribute values. For example, the variable mammal _⇒_ bird is a binary contrast. In the particular case where the binary contrast can correspond to counterfactual pairs of words (e.g., male _⇒_ female), the concept matches the definition used in Park et al. (2024). 

**Hierarchical Structure** The next step is to define what we mean by a hierarchical relation between concepts. To that end, to each attribute _w_ , we associate a set of tokens _Y_ ( _w_ ) that have the attribute. For example, _Y_ (mammal) = _{_ “ dog” _,_ “ cats” _,_ “ Tiger” _, . . . }_ . Then, 

**Definition 2.** A value _z_ is _subordinate_ to a value _w_ (denoted by _z ≺ w_ ) if _Y_ ( _z_ ) _⊆Y_ ( _w_ ). We say a categorical concept _Z ∈R {z_ 0 _, . . . , zn−_ 1 _}_ is subordinate to a categorical concept _W ∈R {w_ 0 _, . . . , wm−_ 1 _}_ if there exists a value _wZ_ of _W_ such that each value _zi_ of _Z_ is subordinate to _wZ_ . 

For example, the binary contrast dog _⇒_ cat is subordinate to the binary feature _{_ is ~~m~~ ammal _,_ not ~~m~~ ammal _}_ , and the binary contrast parrot _⇒_ eagle is subordinate to the categorical concept _{_ mammal _,_ bird _,_ fish _}_ . On the other hand, dog _⇒_ eagle is not subordinate to bird _⇒_ mammal, and bird _⇒_ mammal and live ~~i~~ n ~~h~~ ouse _⇒_ live ~~i~~ n ~~w~~ ater are not subordinate to each other. 

**Linear Representations of Binary Concepts** Now we return to the question of how binary concepts are represented. A key desideratum is that if _ℓ_<sup>¯</sup> _W_ is a linear representation, then moving the context embedding in this direction should modify the probability of the target concept _in isolation_ . If adding _ℓ_<sup>¯</sup> _W_ also modified off-target concepts, it would not be natural to identify it with the target concept _W_ . In Definition 1, this idea is formalized by the requirement that the probability of causally separable concepts is unchanged when the representation is added to the context. 

We now observe that, when there is hierarchical structure, this requirement is not strong enough to capture ‘off-target’ behavior. For example, if _ℓ_<sup>¯</sup> animal captures the concept of animal vs not-animal, then moving in this direction should not affect the relative probability of the output being about a mammal versus a bird. If it did, then the representation would actually capture some amalgamation of the animal and mammal concepts. Accordingly, we must strengthen our definition: 

**Definition 3.** A vector _ℓ_<sup>¯</sup> _W_ is a linear representation of a binary concept _W_ if 





for all contexts _ℓ_ , all _α >_ 0, and all concept variables _Z_ that are either subordinate to or causally separable with _W_ . Here, if _W_ is a binary feature for an attribute _w_ , then _W_ = 1 denotes _W_ = is ~~w~~ . 

Notice that, in the case of binary contrasts defined by counterfactual pairs, this definition is equivalent to Definition 1, because such variables have no subordinate concepts. 

4 

Published as a conference paper at ICLR 2025 

## 4 REPRESENTATIONS OF COMPLEX CONCEPTS 

We now turn to how complex concepts are represented. The high-level strategy is to show how to represent binary features as vectors, show how geometry encodes semantic composition, and then use this to construct representations of complex concepts. 

### 4.1 VECTOR REPRESENTATIONS OF BINARY AND CATEGORICAL CONCEPTS 

To build up to complex concepts, we need to understand how to compose linear representations of binary features. At this stage, the representations are only _directions_ in the representation space— they do not have a natural notion of magnitude. In particular, this means we cannot use vector operations (such as addition) to compose representations. To overcome this, we now show how to associate a magnitude to the linear representation of a binary feature. The key is the following result, which connects binary feature representations and word unembeddings: 

**Theorem 4** (Magnitudes of Linear Representations) **.** _Suppose there exists a linear representation (normalized direction) ℓ_<sup>¯</sup> _W of a binary feature W for an attribute w. Then, there is a constant bw >_ 0 _and a choice of unembedding space origin_ ¯ _γ_ 0<sup>_win eq._(2.2)</sup><sup>_such that_</sup> 



_Further, if there exist d attributes {w_ 0 _, . . . , wd−_ 1 _} such that the linear representations of the binary features for these attributes are linearly independent, we can choose a canonical origin γ_ ¯0 _in eq._ (2.2) _._ 

All proofs are given in Appendix B. 

The theorem says that, if a (perfect) linear representation of the `animal` feature exists, then every token having the animal attribute has the _same_ dot product with the representation vector; i.e., “cat” is exactly as much `animal` as “dog” is. If this weren’t true, then increasing the probability that the output is about an animal would also increase the relative probability that the output is about a dog rather than a cat. In practice, such exact representations are unlikely to be found by gradient descent in LLM training. Rather, we expect _ℓ_<sup>¯</sup><sup>_⊤_</sup> _W_<sup>_g_(</sup><sup>_y_)tobeisotropicallydistributedaround</sup><sup>_bw_and0,with</sup> variances that are small compared to _bw_ (so that animal and non-animal words are well-separated.) 

With this result in hand, we can define a notion of vector representation for **binary features** : 

**Definition 5.** We say that a binary feature _W_ for an attribute _w_ has a _vector representation ℓ_<sup>¯</sup> _w ∈_ R<sup>_d_</sup> if _ℓ_<sup>¯</sup> _w_ satisfies Definition 3 and _∥ℓ_<sup>¯</sup> _w∥_ 2 = _bw_ in Theorem 4. If the vector representation of a binary feature is not unique, we say _ℓ_<sup>¯</sup> _w_ is the vector representation that maximizes _bw_ . 

We have now moved from representations as directions to representations as vectors. Definition 5 and Theorem 4 give a simple way of composing binary features into **binary contrasts** : 

**Corollary 6** (Binary Contrasts Are Vector Differences of Binary Features) **.** _Let w_ 0 _⇒ w_ 1 _be a binary contrast, and suppose there exist vector representations ℓ_<sup>¯</sup> _w_ 0 _and ℓ_<sup>¯</sup> _w_ 1 _for each attribute w_ 0 _and w_ 1 _. Then, the difference ℓ_<sup>¯</sup> _w_ 1 _−ℓ_<sup>¯</sup> _w_ 0 _is a linear representation ℓ_<sup>¯</sup> _w_ 0 _⇒w_ 1 _in the sense of Definition 3._ 

We can also apply ordinary vector space operations to construct representation of **categorical concepts** , e.g., _{_ mammal _,_ reptile _,_ bird _,_ fish _}._ There is now a straightforward way to define the representation of such concepts: 

**Definition 7.** The _polytope representation_ of a categorical concept _W_ = _{w_ 0 _, . . . , wk−_ 1 _}_ is the convex hull of the vector representations of the elements of the concept.<sup>3</sup> 

In Appendix D, we state and prove a generalization of Corollary 6 to categorical concepts (which proves Corollary 6 itself). 

> 3In Appendix C, we prove that the polytope representation for a “natural” categorical concept is a simplex. 

5 

Published as a conference paper at ICLR 2025 



Figure 2: Hierarchical semantics are encoded as orthogonality in the representation space (Theorem 8). The plots show the projection of the unembedding vectors onto 2D subspaces: span _{ℓ_<sup>¯</sup> animal _, ℓ_<sup>¯</sup> mammal _}_ (left; (a)), span _{ℓ_<sup>¯</sup> animal _, ℓ_<sup>¯</sup> bird _−ℓ_<sup>¯</sup> mammal _}_ (middle; (b)), and span _{ℓ_<sup>¯</sup> animal _− ℓ_ ¯plant _,_ ¯ _ℓ_ bird _− ℓ_ ¯mammal _}_ (right; (c)). Gray points indicate all 256K tokens in the vocabulary, and the colored points are the tokens in _Y_ ( _w_ ). The blue and red vectors are used to span the 2D subspaces. 

### 4.2 HIERARCHICAL ORTHOGONALITY 

Now, we turn to the question of how hierarchical relationships between concepts are encoded in the representation space. The core intuition is that manipulating the “animal” concept should not affect relative probabilities of the “mammal” and “bird” concepts, so we might expect the representations of `animal` and mammal _⇒_ bird to be orthogonal. The following result formalizes this intuition by connecting the vector and semantic structures. The result is illustrated in Figure 2. 

**Theorem 8** (Hierarchical Orthogonality) **.** _Suppose there exist the vector representations for all the following binary features. Then, we have that_ 

- _(a) ℓ_<sup>¯</sup> _w ⊥ ℓ_<sup>¯</sup> _z − ℓ_<sup>¯</sup> _w for z ≺ w;_ 

- _(b) ℓ_<sup>¯</sup> _w ⊥ ℓ_<sup>¯</sup> _z_ 1 _− ℓ_<sup>¯</sup> _z_ 0 _for Z ∈R {z_ 0 _, z_ 1 _} subordinate to W ∈R {not_ _~~w~~ , is_ _~~w~~ };_ 

- _(c) ℓ_<sup>¯</sup> _w_ 1 _− ℓ_<sup>¯</sup> _w_ 0 _⊥ ℓ_<sup>¯</sup> _z_ 1 _− ℓ_<sup>¯</sup> _z_ 0 _for Z ∈R {z_ 0 _, z_ 1 _} subordinate to W ∈R {w_ 0 _, w_ 1 _}; and_ 



Together, Definition 7 and Theorem 8 give the simple structure illustrated in Figure 1: hierarchical concepts are represented as direct sums of polytopes. This direct sum structure is immediate from Theorem 8. We emphasize that all of the results—involving differences of representations—are only possible because we have _vector_ representations (mere directions would not suffice). 

## 5 EXPERIMENTS 

We now turn to empirically testing the theoretical results. Here, we present our empirical results on the Gemma-2B model (Mesnard et al., 2024).<sup>4</sup> In Appendix F, we further present empirical results on the LLaMA-3-8B model (Dubey et al., 2024), for which our findings are largely analogous. 

### 5.1 SETUP 

**Canonical Representation** The results in this paper rely on transforming the representation spaces so that the Euclidean inner product is a causal inner product, aligning the embedding and unembedding representations. Following Park et al. (2024), we estimate the required transformation as: 



where _γ_ is the unembedding vector of a word sampled uniformly from the vocabulary. Centering by E[ _γ_ ] is a reasonable approximation of centering by ¯ _γ_ 0 defined in Theorem 4 because this makes the projection of a random _g_ ( _y_ ) on an arbitrary direction close to 0. This matches the requirement that the projection of a word onto a concept the word does not belong to should be close to 0. 

> 4Code is available at github.com/KihoPark/LLM ~~C~~ ategorical ~~H~~ ierarchical ~~R~~ epresentations. 

6 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
2.0 Original Unembeddings 2.0 Sh uffl ed  Unembeddings<br>1.5 1.5<br>1.0 1.0<br>0.5 0.5<br>0.0 0.0<br>0.5 0.5<br>1.0 1.0<br>0 100 200 300 400 500 600 0 100 200 300 400 500 600<br>Binary Features in WordNet  H ierar ch y<br>train test random<br><!-- End of picture text -->

Figure 3: Vector representations exist for most binary features in the WordNet noun hierarchy. For each synset _w_ (indexed on the _x_ -axis) we estimate the vector representation _ℓ_<sup>¯</sup> _w_ using a train subset of the vocabulary _Y_ ( _w_ ). The plot shows the projections ( _g_ ( _y_ )<sup>_⊤_</sup> _ℓ_<sup>¯</sup> _w_ ) _/∥ℓ_<sup>¯</sup> _w∥_ 2<sup>2of train (green), test (blue),</sup> and random (orange) words on estimated vector representations for each WordNet feature, using either the original (left) or shuffled (right) unembeddings. Our theory predicts that this value should be close to 1 when _y_ has the target feature, and close to 0 when it does not. The thick lines present the mean of the projections for each feature and the error bars indicate the standard deviation. As predicted, the projections of test words are near 1, and random words near 0 (left plot). Further, this structure does not hold when using the shuffled control without natural semantics (right plot). 

**WordNet** We define a large collection of binary concepts using WordNet (Miller, 1995). WordNet organizes English words into a hierarchy of synsets, where each synset is a set of synonyms. The WordNet hierarchy is based on word hyponym relations, and reflects the semantic hierarchy of interest in this paper. We take each synset as an attribute _w_ and define _Y_ ( _w_ ) as the collection of all words belonging to any synset that is a descendant of _w_ . For example, the synset mammal.n.01 is a descendant of animal.n.01, so both _Y_ (mammal.n.01) and _Y_ (animal.n.01) contain the word “dog”. We collect all noun and verb synsets, and augment the word collections by including plural forms of the nouns, multiple tenses of each verb, and capital and lower case versions of each word. We filter to include only those synsets with at least 50 words in the Gemma vocabulary. This leaves us with 593 noun and 364 verb synsets, each defining an attribute. 

For space, we report results on the noun hierarchy here and defer the verb hierarchy to Appendix F. 

**Estimation via Linear Discriminant Analysis** Now, we want to estimate the vector representation _ℓ_<sup>¯</sup> _w_ for each attribute _w_ . To do this, we make use of vocabulary sets _Y_ ( _w_ ). Following Theorem 4, the vector associated to the concept _w_ should have two properties. First, when the full vocabulary is projected onto this vector, the words in _Y_ ( _w_ ) should be well-separated from the rest of the vocabulary. Second, the projection of the unembedding vectors for _y ∈Y_ ( _w_ ) should be approximately the same value. Equivalently, the variance of the projection of the unembedding vectors for _y ∈Y_ ( _w_ ) should be small. To capture these requirements, we estimate the directions using a variant of Linear Discriminant Analysis (LDA), which finds a projection minimizing within-class variance and maximizing between-class variance. Formally, we estimate the vector representation of a binary feature _W_ for an attribute _w_ as 



where _gw_ is the unembedding vector of a word sampled uniformly from _Y_ ( _w_ ) and Cov( _gw_ )<sup>_†_</sup> is a pseudo-inverse of the covariance matrix. We estimate the covariance matrix Cov( _gw_ ) using the Ledoit-Wolf shrinkage estimator (Ledoit & Wolf, 2004), because the dimension of the representation spaces is much higher than the number of samples. 

### 5.2 WORDNET HIERARCHY IS LINEARLY REPRESENTED 

**Existence of Vector Representations for Binary Features** The first question is whether vector representations of binary features exist. To evaluate this, for each synset _w_ in WordNet we split _Y_ ( _w_ ) into train words (70%) and test words (30%), fit the LDA estimator to the train words, and examine the projection of the unembedding vectors for the test and random words onto the estimated vector 

7 

Published as a conference paper at ICLR 2025 

Table 1: Adding the linear representation of the parent concept to context embeddings does not affect the logit differences between token pairs in a child concept, and it substantially affects those in the parent concept. For each parent-child pair of binary concepts ( _w_ 0 _⇒ w_ 1 _, z_ 0 _⇒ z_ 1), we first add the normalized linear representation _ℓ_<sup>¯</sup> _W_ = _ℓ_<sup>¯</sup> _w_ 1 _− ℓ_<sup>¯</sup> _w_ 0 to the context embeddings. Then, we show the change in logit difference log P<sup>P</sup> (<sup><u>(</u></sup> _y_<sup>_<u>y</u>_</sup> 0<sup>1</sup> _|_<sup>_<u>|</u>_</sup> _ℓ_<sup>_ℓ_</sup> )<sup><u>)</u>between the pairs (</sup><sup>_y_0</sup><sup>_, y_1)</sup><sup>_∈Y_(</sup><sup>_w_0)</sup><sup>_×Y_(</sup><sup>_w_1) (parent; top row)</sup> and ( _y_ 0 _, y_ 1) _∈Y_ ( _z_ 0) _× Y_ ( _z_ 1) (child; bottom row). Notice that, for any context _x_ and any tokens _y_ 0 and _y_ 1, adding a linear representation _ℓ_<sup>¯</sup> _W_ for a binary contrast _W_ = _w_ 0 _⇒ w_ 1 manipulates the logit difference between the tokens from _ℓ_ ( _x_ )<sup>_⊤_</sup> ( _g_ ( _y_ 1) _− g_ ( _y_ 0)) to ( _ℓ_ ( _x_ ) + _ℓ_<sup>¯</sup> _W_ )<sup>_⊤_</sup> ( _g_ ( _y_ 1) _− g_ ( _y_ 0)), which implies the change in logit difference between the tokens is _ℓ_<sup>¯</sup><sup>_⊤_</sup> _W_<sup>(</sup><sup>_g_(</sup><sup>_y_1)</sup><sup>_−g_(</sup><sup>_y_0)), irrespective</sup> of _ℓ_ ( _x_ ). We show the mean and standard deviation of the change in logit differences over all token pairs. 

|_W_ =parent0 _⇒_parent1 &_Z_ =child0 _⇒_child1|Change in Logit Differences|
|---|---|
|_W_ =plant.n.02_⇒_animal.n.01|5_._1265_±_1_._1731|
|_Z_ =mammal.n.01_⇒_reptile.n.01|_−_0_._0600_±_1_._2190|
|_W_ =fluid.n.02_⇒_solid.n.01|9_._8296_±_1_._1099|
|_Z_ =crystal.n.01_⇒_food.n.02|0_._3770_±_1_._5410|
|_W_ =scientist.n.01_⇒_contestant.n.01|14_._4222_±_0_._9458|
|_Z_ =athlete.n.01_⇒_player.n.01|_−_0_._1545_±_1_._1426|



representation. The left plot in Figure 3 shows the mean and standard deviation of the projections, divided by the magnitude of each estimated _ℓ_<sup>¯</sup> _w_ . Following Theorem 4, if a vector representation exists for an concept, we would expect the values on the test set to be close to 1, and the values for random words to be close to 0. We see that this is indeed the case, giving evidence that vector representations do exist for these features. 

As a baseline, the right plot in Figure 3 shows the same analysis but with the unembedding vectors randomly shuffled. In this case, the test projections are close to 0, which indicates that there are no linear representations. Thus, the existence of the linear representations in the original unembeddings relies on the underlying semantic structure. 

**Intervention** Next, we validate that adding the estimated linear representations of a binary contrast changes the target concept without changing other off-target concepts, as required by Definition 3. Table 1 shows the mean and standard deviation of the changes in the logit differences between the pairs from parent or child binary contrasts, after adding the normalized linear representations of the parent binary contrast to context embeddings. The results show that the logit differences change significantly for the target concept, while the off-target concept changes very little. 

**Relationship Between the Vector Representations of Binary Features** We now turn to examining whether the similarity between the vector representations of binary features reflects their semantic relation. The direct sum structure predicts that concepts that are close in semantic hierarchy should have similar vector representations. For example, `mammal` and `bird` are close in the hyponym graph because they share a common parent ( `animal` ). Our theory predicts that _ℓ_<sup>¯</sup> bird and _ℓ_<sup>¯</sup> mammal share a common component _ℓ_<sup>¯</sup> animal (the representation for “bird” is the representation for “animal” plus an orthogonal component, and similarly for mammal). This implies that the cosine similarity between _ℓ_<sup>¯</sup> mammal and _ℓ_<sup>¯</sup> bird should be substantial. 

In Figure 4, we show the shortest distance between each feature in the (undirected) WordNet noun hyponym graph in the left panel and the cosine similarity between the estimated vector representations _ℓ_<sup>¯</sup> _w_ in the middle panel. It is clear that, as predicted, the cosine similarity reflects the WordNet structure. As a control, we apply the same analysis after randomly shuffling the embeddings and show the resulting cosine similarities in the right panel. Here, direct parent-child (or grandparentgrandchild) relationships are still reflected in the cosine similarity because the shuffled embeddings still respect set inclusion (the words assigned to “mammal” are still a subset of those assigned to “animal”). However, sibling relationships are not reflected once the semantic structure is removed by the shuffling. In this case, the representations of siblings are effectively pairs of random vectors, and are nearly orthogonal as we would expect in a high dimensional space. 

8 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
0 Pairwise Shortest Distance 0 cos( w, z ) (Original Unembeddings) 0 cos( w, z ) (Shuffled Unembeddings) 1.00<br>0.75<br>100 100 100<br>0.50<br>200 200 200<br>0.25<br>300 300 300 0.00<br>0.25<br>400 400 400<br>0.50<br>500 500 500<br>0.75<br>1.00<br>0 100 200 300 400 500 0 100 200 300 400 500 0 100 200 300 400 500<br><!-- End of picture text -->

Figure 4: Hierarchical semantics in WordNet are linearly represented in Gemma-2B. The left heatmap shows pairwise shortest distance matrix between features in the noun hierarchy graph as (1 + min ~~d~~ istance)<sup>_−_1</sup> (higher values indicate closeness, such as in child-parent or sibling relationships). The middle heatmap shows the cosine similarity between the vector representations _ℓ_<sup>¯</sup> _w_ . As predicted, this similarity reflects the WordNet structure. The right heatmap is a control where the embeddings are randomly shuffled (removing semantic structure). In this case, nearly everything is orthogonal, as expected in high-dimensional space (set inclusion relationships remain due to the estimation procedure). In Appendix F, we include zoomed-in versions of these heatmaps. 



<!-- Start of picture text -->
cos( w parent of w , parent of w ), Whole set cos( w parent of w , parent of w ), Training set<br>0.25 0.25<br>0.00 0.00<br>0.25 0.25<br>0.50 0.50<br>0.75 0.75<br>1.00 1.00<br>0 100 200 300 400 500 600 0 100 200 300 400 500 600<br>Binary Features in WordNet Hierarchy<br>Original Unembeddings Shuffled Unembeddings Original Unembeddings + Random Parent<br><!-- End of picture text -->

Figure 5: WordNet noun hierarchy is encoded in the orthogonal structure predicted by statement (a) in Theorem 8. We plot the cosine similarity between a child-parent vector and a parent vector for each feature in the hierarchy (blue). As predicted, this value is close to 0. The left plot uses all data for representation estimation, and the right plot uses only 70% independently selected for each synset. We include baselines where a randomly selected feature is used as the parent (orange) and where the embeddings are shuffled (green) as controls for the possibility that the orthogonality is a simple byproduct of high-dimensional geometry, or of the set inclusion relationships used in estimation—see main text for details. See Appendix F for an analogous plot for statement (d). 

**Hierarchical Orthogonality** Finally, we evaluate the prediction that hierarchical relations are encoded orthogonally as predicted in Theorem 8. Figure 5 shows the cosine similarity _ℓ_<sup>¯</sup> parent and _ℓ_ ¯child _− ℓ_ ¯parent for the WordNet features. As predicted, this value is close to 0 (blue curve). 

Now, a challenge here is that in high dimensional spaces even random vectors are nearly orthogonal. So, it may be difficult to differentiate whether orthogonality reflects semantic structure or is merely a consequence of nearly everything being orthogonal in high dimensions. As a control, we also show the cosine similarity when the parent vector is the representation of a randomly selected feature (orange). In this case, the cosine similarity is far from 0, suggesting that the observed orthogonality is not merely a byproduct of the high-dimensional geometry. 

We also include another baseline that estimates the cosine similarity between the child-parent vector and the parent vector for each feature using shuffled unembeddings (green). In the left plot, we see that we still have orthogonality, which could suggest that the orthogonality is a consequence of the set inclusion relations in WordNet (rather than actual semantic structure).<sup>5</sup> To test this, in the right plot we estimate the representations using only 70% of the tokens, independently selected for each 

> 5In Appendix G, we explain why set inclusion (before train/test split) leads to orthogonality. 

9 

Published as a conference paper at ICLR 2025 

synset. This breaks the set inclusion. In this case, we see the orthogonality is preserved for the original unembeddings, but is broken for the shuffled unembeddings. Note that a cosine of _−_ 0 _._ 2 is highly nontrivial in a high-dimensional space. This suggests the orthogonality does indeed reflect the semantic structure. 

## 6 DISCUSSION AND RELATED WORK 

We set out to understand how semantic structure is encoded in the geometry of representation space. We have arrived at an astonishingly simple structure, summarized in Figure 1. The key contributions are moving from representing concepts as directions to representing them as vectors (and polytopes), and connecting semantic hierarchy to orthogonality. 

**Related Work** The results here connect closely to the study of linear representations in language models (e.g., Mikolov et al., 2013; Pennington et al., 2014; Arora et al., 2016; Elhage et al., 2022; Burns et al., 2022; Tigges et al., 2023; Nanda et al., 2023; Moschella et al., 2022; Li et al., 2023; Gurnee et al., 2023; Wang et al., 2023; Jiang et al., 2024; Park et al., 2024). In particular, Park et al. (2024) formalize the linear representation hypothesis by unifying three distinct notions of linearity: word2vec-like embedding differences, logistic probing, and steering vectors. Our work relies on this unification, and just focuses on the steering vector notion. Our work also connects to work aimed at theoretically understanding the existence of linear representations. These include early work on word2vec-style embedding models (Arora et al., 2016; Gittens et al., 2017; Arora et al., 2018; Ethayarajh et al., 2018; Frandsen & Ge, 2019; Allen & Hospedales, 2019) as well as dynamic topic models (Blei & Lafferty, 2006; Rudolph et al., 2016). Jiang et al. (2024) connect the existence of linear representations in LLMs to the implicit bias of gradient descent. In this paper, we do not seek to justify the _existence_ of linear representations, but rather to understand their _structure_ if they do exist. Though, by empirically estimating vector representations for thousands of concepts, we add to the body of evidence supporting the existence of linear representations. Elhage et al. (2022) also empirically observe the formation of polytopes in the representation space of a toy model, and the present work can be viewed in part as giving an explanation for this phenomenon. 

There is also a growing literature studying the representation geometry of natural language (Mimno & Thompson, 2017; Reif et al., 2019; Volpi & Malag`o, 2021; Li et al., 2020; Chen et al., 2021; Chang et al., 2022; Liang et al., 2022; Jiang et al., 2023; Wang et al., 2023; Park et al., 2024; Valeriani et al., 2024). In terms of hierarchical structures, existing work focuses on connections to hyperbolic geometry (Nickel & Kiela, 2017; Ganea et al., 2018; Chen et al., 2021; He et al., 2024). We do not find such a connection in LLMs, but it is an interesting direction for future work to determine if more efficient LLM representations could be constructed in hyperbolic space. Jiang et al. (2023) hypothesize that very general ”independence structures” are naturally represented by partial orthogonality in vector spaces (Amini et al., 2022). The results here confirm and expand on this hypothesis in the case of hierarchical structure in language models. 

**Implications and Future Work** The results in this paper give a foundational understanding the structure of representation space in language models. Of course, the ultimate purpose of foundations is to build upon them. One immediate direction is to refine the attempts to interpret LLM structure to explicitly account for hierarchical semantics. As an example, there is currently significant interest in using sparse autoencoders to extract interpretable features from LLMs (e.g., Cunningham et al., 2023; Bricken et al., 2023; Kissane et al., 2024; Braun et al., 2024). This work searches for representations in terms of distinct binary features. Concretely, it hopes to find features for, e.g., `animal` , `mammal` , `bird` , etc. Based on the results here, these representations are strongly co-linear, and potentially difficult to disentangle. On the other hand, a representation in terms of _ℓ_<sup>¯</sup> animal, _ℓ_ ¯mammal _− ℓ_ ¯animal, _ℓ_ ¯bird _− ℓ_ ¯animal, etc., will be cleanly separated and equally interpretable. Fundamentally, semantic meaning has hierarchical structure, so interpretability methods should respect this structure. Understanding the geometric representation makes it possible to design such methods. 

In a separate, foundational, direction: the results in this paper rely on using the canonical representation space, and we estimate this using the whitening transformation of the unembedding layer. However, this technique only works for the final layer representation. It is an important open question how to make sense of the geometry of internal layers. 

10 

Published as a conference paper at ICLR 2025 

### ACKNOWLEDGMENTS 

This work is supported by ONR grant N00014-23-1-2591 and Open Philanthropy. 

## REFERENCES 

- Carl Allen and Timothy Hospedales. Analogies explained: Towards understanding word embeddings. In _International Conference on Machine Learning_ , pp. 223–231. PMLR, 2019. 

- Arash A Amini, Bryon Aragam, and Qing Zhou. A non-graphical representation of conditional independence via the neighbourhood lattice. _arXiv preprint arXiv:2206.05829_ , 2022. 

- Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A latent variable model approach to pmi-based word embeddings. _Transactions of the Association for Computational Linguistics_ , 4:385–399, 2016. 

- Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. Linear algebraic structure of word senses, with applications to polysemy. _Transactions of the Association for Computational Linguistics_ , 6:483–495, 2018. 

- David M Blei and John D Lafferty. Dynamic topic models. In _Proceedings of the 23rd international conference on Machine learning_ , pp. 113–120, 2006. 

- Dan Braun, Jordan Taylor, Nicholas Goldowsky-Dill, and Lee Sharkey. Identifying functionally important features with end-to-end sparse dictionary learning. _arXiv preprint arXiv:2405.12241_ , 2024. 

- Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_ , 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html. 

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. _arXiv preprint arXiv:2212.03827_ , 2022. 

- Tyler A Chang, Zhuowen Tu, and Benjamin K Bergen. The geometry of multilingual language model representations. _arXiv preprint arXiv:2205.10964_ , 2022. 

- Boli Chen, Yao Fu, Guangwei Xu, Pengjun Xie, Chuanqi Tan, Mosha Chen, and Liping Jing. Probing bert in hyperbolic spaces. _arXiv preprint arXiv:2104.03869_ , 2021. 

- Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. _arXiv preprint arXiv:2309.08600_ , 2023. 

- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. _arXiv preprint arXiv:2407.21783_ , 2024. 

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. _arXiv preprint arXiv:2209.10652_ , 2022. 

- Kawin Ethayarajh, David Duvenaud, and Graeme Hirst. Towards understanding linear word analogies. _arXiv preprint arXiv:1810.04882_ , 2018. 

- Abraham Frandsen and Rong Ge. Understanding composition of word embeddings via tensor decomposition. _arXiv preprint arXiv:1902.00613_ , 2019. 

11 

Published as a conference paper at ICLR 2025 

- Octavian Ganea, Gary B´ecigneul, and Thomas Hofmann. Hyperbolic entailment cones for learning hierarchical embeddings. In _International Conference on Machine Learning_ , pp. 1646–1655. PMLR, 2018. 

- Alex Gittens, Dimitris Achlioptas, and Michael W. Mahoney. Skip-gram - zipf + uniform = vector additivity. In _Annual Meeting of the Association for Computational Linguistics_ , 2017. 

- Wes Gurnee and Max Tegmark. Language models represent space and time. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/ forum?id=jE8xbmvFin. 

- Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. _arXiv preprint arXiv:2305.01610_ , 2023. 

- Yuan He, Zhangdie Yuan, Jiaoyan Chen, and Ian Horrocks. Language models as hierarchy encoders. _arXiv preprint arXiv:2401.11374_ , 2024. 

- Yibo Jiang, Bryon Aragam, and Victor Veitch. Uncovering meanings of embeddings via partial orthogonality. _arXiv preprint arXiv:2310.17611_ , 2023. 

- Yibo Jiang, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, and Victor Veitch. On the origins of linear representations in large language models. In _International Conference on Machine Learning_ , 2024. 

- Connor Kissane, Robert Krzyzanowski, Arthur Conmy, and Neel Nanda. Sparse autoencoders work on attention layer outputs. Alignment Forum, 2024. URL https://www. alignmentforum.org/posts/DtdzGwFh9dCfsekZZ. 

- Olivier Ledoit and Michael Wolf. A well-conditioned estimator for large-dimensional covariance matrices. _Journal of multivariate analysis_ , 88(2):365–411, 2004. 

- Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. On the sentence embeddings from pre-trained language models. _arXiv preprint arXiv:2011.05864_ , 2020. 

- Kenneth Li, Oam Patel, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. _arXiv preprint arXiv:2306.03341_ , 2023. 

- Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. _Advances in Neural Information Processing Systems_ , 35:17612–17625, 2022. 

- Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi`ere, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. _arXiv preprint arXiv:2403.08295_ , 2024. 

- Tom´aˇs Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In _Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies_ , pp. 746–751, 2013. 

- George A Miller. Wordnet: a lexical database for english. _Communications of the ACM_ , 38(11): 39–41, 1995. 

- David Mimno and Laure Thompson. The strange geometry of skip-gram with negative sampling. In _Conference on Empirical Methods in Natural Language Processing_ , 2017. 

- Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, and Emanuele Rodola. Relative representations enable zero-shot latent space communication. _arXiv preprint arXiv:2209.15430_ , 2022. 

- Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. _arXiv preprint arXiv:2309.00941_ , 2023. 

12 

Published as a conference paper at ICLR 2025 

- Maximillian Nickel and Douwe Kiela. Poincar´e embeddings for learning hierarchical representations. _Advances in neural information processing systems_ , 30, 2017. 

OpenAI. GPT-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 

- Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In _International Conference on Machine Learning_ , 2024. 

- Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pp. 1532–1543, 2014. 

- Emily Reif, Ann Yuan, Martin Wattenberg, Fernanda B Viegas, Andy Coenen, Adam Pearce, and Been Kim. Visualizing and measuring the geometry of bert. _Advances in Neural Information Processing Systems_ , 32, 2019. 

- Maja Rudolph, Francisco Ruiz, Stephan Mandt, and David Blei. Exponential family embeddings. _Advances in Neural Information Processing Systems_ , 29, 2016. 

- Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda. Linear representations of sentiment in large language models. _arXiv preprint arXiv:2310.15154_ , 2023. 

- Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, and Alberto Cazzaniga. The geometry of hidden representations of large transformer models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Riccardo Volpi and Luigi Malag`o. Natural alpha embeddings. _Information Geometry_ , 4(1):3–29, 2021. 

- Zihao Wang, Lin Gui, Jeffrey Negrea, and Victor Veitch. Concept algebra for (score-based) textcontrolled generative models. In _Advances in Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum?id=SGlrCuwdsB. 

13 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
2 fish reptile<br>10 toe et 14 Zi<br>for? Sa 2 he<br>8 ODER: 10 \<br>4 + ramet ® .<br>-| fae 4 . atta.<br>, os 0 |S Tox<br>o 4 mam a . Sr ' -2} bith alee 4 vex<br>° ‘ NUE igs / Piatie. . <3<br>4 ¥ Pra ae mr) 2 hash<br>510 on tS.sa2 0 °, 4 6 aes= 9 2 -4 6 76 -8<br>2 12 «0 8 6 810 8 © 4 2<br><!-- End of picture text -->

Figure 6: Categorical concepts are represented as polytopes. The plots show the projection of the unembedding vectors on the 3D subspaces: span _{ℓ_<sup>¯</sup> mammal _, ℓ_<sup>¯</sup> bird _, ℓ_<sup>¯</sup> fish _}_ (left) and span _{ℓ_<sup>¯</sup> bird _− ℓ_ ¯mammal _,_ ¯ _ℓ_ fish _− ℓ_ ¯mammal _,_ ¯ _ℓ_ reptile _− ℓ_ ¯mammal _}_ (right). The gray points indicate all 256K tokens in the vocabulary, and the colored points are the tokens in _Y_ (w). The left plot further shows the orthogonality between the triangle and the projection of _ℓ_<sup>¯</sup> animal (black arrow). 

## A VISUALIZATION OF `animal` 

As a concrete example for visualizations, we examine the theoretical predictions for the familiar concept `animal` . However, the categories in WordNet are not the most intuitive; for example, the subcategories under textttanimal.n.01 include terms like “chordate”, “aquatic vertebrate”, “invertebrate” instead of more familiar categories such as “fish”, “amphibian”, or “insect”. This makes it challenging to gather tokens corresponding to clear, high-level concepts _{_ mammal _,_ bird _,_ fish _,_ reptile _,_ amphibian _,_ insect _}_ . Therefore, we generated two sets of tokens _Y_ (animal) and _Y_ (plant) using ChatGPT-4 (OpenAI, 2023), and manually inspected them. _Y_ (animal) is further divided into six sets of tokens for each subcategory _{_ mammal _,_ bird _,_ fish _,_ reptile _,_ amphibian _,_ insect _}_ . 

Figure 2 illustrates the geometric relationships between various representation vectors. The main takeaway is that the semantic hierarchy is encoded as orthogonality in the manner predicted by Theorem 8. The figure also illustrates Theorem 4, showing that the projection of the unembedding vectors for _y ∈Y_ ( _w_ ) is approximately constant, while the projection of _y̸ ∈Y_ ( _w_ ) is zero. 

Figure 6 illustrates that the representation of a categorical concept is a polytope. and the projection of unembedding vectors onto the subspace for the polytope are concentrated on each vertex as predicted in Corollary 10. The left plot also shows that, as predicted, the polytope for _{_ fish _,_ mammal _,_ bird _}_ is orthogonal to the vector representation of `animal` . 

## B PROOFS 

B.1 PROOF OF THEOREM 4 

**Theorem 4** (Magnitudes of Linear Representations) **.** _Suppose there exists a linear representation (normalized direction) ℓ_<sup>¯</sup> _W of a binary feature W for an attribute w. Then, there is a constant bw >_ 0 _and a choice of unembedding space origin_ ¯ _γ_ 0<sup>_win eq._(2.2)</sup><sup>_such that_</sup> 



_Further, if there exist d attributes {w_ 0 _, . . . , wd−_ 1 _} such that the linear representations of the binary features for these attributes are linearly independent, we can choose a canonical origin γ_ ¯0 _in eq._ (2.2) _._ 

14 

Published as a conference paper at ICLR 2025 

_Proof._ For any _y_ 1 _, y_ 0 _∈Y_ ( _w_ ) or _y_ 1 _, y_ 0 _̸ ∈Y_ ( _w_ ), let _Z_ be a binary concept where _Y_ ( _Z_ = 0) = _{y_ 0 _}_ and _Y_ ( _Z_ = 1) = _{y_ 1 _}_ . Since _Z_ is subordinate to _W_ , eq. (3.2) implies that 



where _A_ is the invertible matrix in eq. (2.2). This means that _ℓ_<sup>¯</sup><sup>_⊤_</sup> _W_<sup>_Aγ_(</sup><sup>_y_) is the same for all</sup><sup>_y∈Y_(</sup><sup>_w_),</sup> and it is also the same for all _y̸ ∈Y_ ( _w_ ). 

Furthermore, for any _y_ 1 _∈Y_ ( _w_ ) and _y_ 0 _̸ ∈Y_ ( _w_ ), eq. (3.1) implies that 



Thus, by setting _b_<sup>0</sup> _w_<sup>=</sup><sup>_ℓ_¯</sup><sup>_⊤_</sup> _W_<sup>_Aγ_(</sup><sup>_y_) for any</sup><sup>_y̸∈Y_(</sup><sup>_w_), and</sup><sup>_bw_=</sup><sup>_ℓ_¯</sup><sup>_⊤_</sup> _W_<sup>_Aγ_(</sup><sup>_y_1)</sup><sup>_−ℓ_¯</sup><sup>_⊤_</sup> _W_<sup>_Aγ_(</sup><sup>_y_0)</sup><sup>_>_0 for any</sup> _y_ 1 _∈Y_ ( _w_ ) and _y_ 0 _̸ ∈Y_ ( _w_ ), we get 



Then, we can choose an origin as 



satisfying eq. (4.1). 

On the other hand, if there exist _d_ attributes _{w_ 0 _, . . . , wd−_ 1 _}_ such that the linear representations _ℓ_ ¯ _W_ 0 _, . . . ,_ ¯ _ℓWd−_ 1 of the binary features _W_ 0 _, . . . , Wd−_ 1 for these attributes are linearly independent, then the linear system 



has a unique solution ¯ _γ_ 0. We can choose this vector as the canonical origin ¯ _γ_ 0 in eq. (2.2), ensuring that eq. (4.1) is satisfied. 

### B.2 PROOF OF THEOREM 8 

**Theorem 8** (Hierarchical Orthogonality) **.** _Suppose there exist the vector representations for all the following binary features. Then, we have that_ 







_Proof._ (a) For _ℓ_<sup>¯</sup> _w_ and _ℓ_<sup>¯</sup> _z_ where _z ≺ w_ , by Theorem 4, we have 



When _w \ z_ denotes an attribute defined by _Y_ ( _w_ ) _\ Y_ ( _z_ ), _ℓ_<sup>¯</sup> _z − ℓ_<sup>¯</sup> _w_ can change the target concept _w \z ⇒ z_ without changing any other concept subordinate or causally separable to the target concept. Thus, _ℓ_<sup>¯</sup> _z − ℓ_<sup>¯</sup> _w_ is the linear representation _ℓ_<sup>¯</sup> _w\z⇒z_ . This concept means not ~~z~~ _⇒_ is ~~z~~ conditioned on _w_ , and hence it is subordinate to _w_ . 

Therefore, _ℓ_<sup>¯</sup> _w_ is orthogonal to the linear representation _ℓ_<sup>¯</sup> _w\z⇒z_ = _ℓ_<sup>¯</sup> _z − ℓ_<sup>¯</sup> _w_ by the property of the causal inner product. If they are not orthogonal, adding _ℓ_<sup>¯</sup> _w_ can change the other concept _w \ z ⇒ z_ , and it is a contradiction. 



15 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
w 2<br>w 0 w 1 w 2<br>w 0<br>w 1<br><!-- End of picture text -->

Figure 7: Illustration of the case _k_ = 3 in the proof of Proposition 9. 

- (c) Let’s say that _w_ 1 is _wZ_ defined in Definition 2. The binary contrast _z_ 0 _⇒ z_ 1 is subordinate to the binary feature for the attribute _w_ 0. By the property of the causal inner product, _ℓ_<sup>¯</sup> _w_ 0 is orthogonal to the linear representation _ℓ_<sup>¯</sup> _z_ 0 _⇒z_ 1 = _ℓ_<sup>¯</sup> _z_ 1 _− ℓ_<sup>¯</sup> _z_ 0 (by Corollary 10). Then, with the above result (b), we have ( _ℓ_<sup>¯</sup> _w_ 1 _− ℓ_<sup>¯</sup> _w_ 0)<sup>_⊤_</sup> ( _ℓ_<sup>¯</sup> _z_ 1 _− ℓ_<sup>¯</sup> _z_ 0). 

(d) By the above result (a), we have 



Then, 









## C NATURAL CATEGORICAL CONCEPTS AS SIMPLICES 

Polytopes are quite general objects. Definition 7 also includes representations of categorical variables that are semantically unnatural, e.g., _{_ dog _,_ sandwich _,_ running _}_ . We would like to make a more precise statement about the representation of “natural” concepts. One possible notion of a “natural” concept is one where the model can freely manipulate the output values. The next proposition shows such concepts have particularly simple structure: 

**Proposition 9** (Categorical Concepts are Represented as Simplices) **.** _Suppose that {w_ 0 _, . . . , wk−_ 1 _} is a collection of k mutually exclusive attributes such that for every joint distribution Q_ ( _w_ 0 _, . . . wk−_ 1) _there is some ℓi such that_ P( _W_ = _wi | ℓi_ ) = _Q_ ( _W_ = _wi_ ) _for every i. Then, the vector representations ℓ_<sup>¯</sup> _w_ 0 _, . . . , ℓ_<sup>¯</sup> _wk−_ 1 _form a_ ( _k −_ 1) _-simplex in the representation space. In this case, we take the simplex to be the representation of the categorical concept W_ = _{w_ 0 _, . . . , wk−_ 1 _}._ 

_Proof._ If we can represent arbitrary joint distributions, this means, in particular, that we can change the probability of one attribute without changing the relative probability between a pair of other attributes. Consider the case where _k_ = 3, as illustrated in Figure 7. If _ℓ_<sup>¯</sup> _w_ 0 _, ℓ_<sup>¯</sup> _w_ 1 _, ℓ_<sup>¯</sup> _w_ 2 are on a line, then there is no direction in that line (to change the value in the categorical concept) such that adding the direction can change the probability of _w_ 2 without changing the relative probabilities between _w_ 0 and _w_ 1. However, if _ℓ_<sup>¯</sup> _w_ 0 _, ℓ_<sup>¯</sup> _w_ 1 _, ℓ_<sup>¯</sup> _w_ 2 are not on a line, they form a triangle. Then, there exists a line 

16 

Published as a conference paper at ICLR 2025 

that is toward _ℓ_<sup>¯</sup> _w_ 2 and perpendicular to the opposite side of the triangle. Now adding the direction _ℓ_ ˜ can manipulate the probability of _w_ 2 without changing the relative probabilities between _w_ 0 and _w_ 1. That is, for any _α >_ 0 and context embedding _ℓ_ , 



Therefore, the vectors _ℓ_<sup>¯</sup> _w_ 0 _, ℓ_<sup>¯</sup> _w_ 1 _, ℓ_<sup>¯</sup> _w_ 2 form a 2-simplex. 

This argument extends immediately to higher _k_ by induction. For each _i ∈{_ 0 _, . . . , k −_ 1 _}_ , there should exist a direction that is toward _ℓ_<sup>¯</sup> _wi_ and orthogonal to the opposite hyperplane (( _k −_ 2)simplex) formed by the other _ℓ_<sup>¯</sup> _wi′_ ’s. Then, the vectors _ℓ_<sup>¯</sup> _w_ 0 _, . . . , ℓ_<sup>¯</sup> _wk−_ 1 form a ( _k −_ 1)-simplex. 

## D SUBSPACES FOR CATEGORICAL CONCEPTS 

By Theorem 4, polytope representations have the following property: 

**Corollary 10** (Polytope Representations) **.** _Let W_ = _{w_ 0 _, . . . , wk−_ 1 _} be a categorical concept, and suppose there exist vector representations ℓ_<sup>¯</sup> _wi for the binary features of each attribute wi (their convex hull is the polytope representation of W ). Let L_<sup>¯</sup> = � _ℓ_ ¯ _w_ 1 _− ℓ_ ¯ _w_ 0 _, . . . ,_ ¯ _ℓwk−_ 1 _− ℓ_ ¯ _w_ 0� _∈_ R<sup>_d×_(</sup><sup>_k−_1)</sup> _and let C_ ( _L_<sup>¯</sup> ) _be its column space. Then, there exist some vectors τwi ∈C_ ( _L_<sup>¯</sup> ) _, for each i, such that_ 



_where_<sup>�</sup> _L_ ¯<sup>=</sup><sup>_L_¯(¯</sup><sup>_L⊤L_¯)</sup><sup>_†_¯</sup><sup>_L⊤istheprojectionmatrixontoC_(¯</sup><sup>_L_)</sup><sup>_,and_(¯</sup><sup>_L⊤L_¯)</sup><sup>_†denotesapseudo-_</sup> _inverse of the matrix._ 

_Proof._ By Theorem 4, for _i_ = 1 _, . . . , k −_ 1, we have 



for any _y ∈Y_ ( _wi_ ). Similarly, we have 



for any _y ∈Y_ ( _w_ 0), while 



for any _y̸ ∈∪_<sup>_k_</sup> _i_ =0<sup>_−_1</sup><sup>_Y_(</sup><sup>_wi_).Therefore, we have eq. (D.1) with some constant vectors</sup><sup>_τw_</sup> _i_<sup>.</sup> 

In words, the unembeddings of tokens for the same attribute share the same projections on the subspace spanned by the differences between the vector representations. Therefore, adding any vector in the subspace _C_ ( _L_<sup>¯</sup> ) to the context embedding changes the probability of the target concept _W_ = _{w_ 0 _, . . . , wk−_ 1 _} without_ changing any other concept subordinate to or causally separable with the target concept. Finally, note that Corollary 10 implies Corollary 6. 

## E EXPERIMENT DETAILS 

We employ the Gemma-2B version of the Gemma model (Mesnard et al., 2024), which is accessible online via the huggingface library. Its two billion parameters are pre-trained on three trillion tokens. This model utilizes 256K tokens and 2,048 dimensions for the representation space. 

We always use tokens that start with a space (‘ _\_ u2581’) in front of the word, as they are used for next-word generation with full meaning. Additionally, like WordNet data we use, we include plural 

17 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
animal<br>chordate 7 young invertebrate domestic_animal<br>bird TO mammal aquatic_vertebrate reptile<br>gallinaceous_bird a1 passerine ungulate ory, carnivore<br>YN<br>even-toed_ungulate odd-toed_ungulate<br><!-- End of picture text -->

Figure 8: Subtree in WordNet noun hierarchy for descendants of animal. 



<!-- Start of picture text -->
Pairwise Shortest Distance cos( w, z ) (Original Unembeddings) cos( w, z ) (Shuffled Unembeddings) 1.00<br>animal<br>domestic_animal 0.75<br>young<br>chordate 0.50<br>invertebrate<br>aquatic_vertebrate 0.25<br>bird<br>reptile 0.00<br>mammal<br>passerine 0.25<br>gallinaceous_bird<br>carnivore 0.50<br>ungulate<br>odd-toed_ungulate 0.75<br>even-toed_ungulate 1.00<br><!-- End of picture text -->

Figure 9: Zoomed-in Heatmaps of the subtree for animal in Figure 8. 

forms, and both capital and lowercase versions of the words in _Y_ (animal) and _Y_ (plant) for visualization in Appendix A. 

In the WordNet synset data, each content of the synset mammal.n.01 indicates that ”mammal” is a word, ”n” denotes ”noun,” and ”01” signifies the first meaning of the word. In the WordNet hierarchy, if a parent has only one child, we combine the two features into one. Additionally, since the WordNet hierarchy is not a perfect tree, a child can have more than one parent. We use one of the parents when computing the _ℓ_<sup>¯</sup> _w − ℓ_<sup>¯</sup> parent of _w_ . 

Lastly, we use the vector representations estimated by the non-split collection of tokens for Figure 2, Table 1, Figure 4, Figure 6, Figure 14, and Figure 18. 

## F ADDITIONAL RESULTS 

**Zooming in on a Subtree of Noun Hierarchy** As it is difficult to understand the entire WordNet hierarchy at once from the heatmaps in Figure 4, we present a zoomed-in heatmap for the subtree (Figure 8) for the feature animal in Figure 9. The left heatmap displays the shortest distance between the nodes of the subtree in Figure 8. The middle heatmap shows that the cosine similarities between the vector representations _ℓ_<sup>¯</sup> _w_ reflect the child-parent or sibling relationships. The right heatmap demonstrates that the vector representations estimated by shuffled unembeddings only reflect the set inclusion relationships, and other pairs have cosine similarities close to 0. 

**Additional Results on Hierarchical Orthogonality** Analogous to Figure 5, we validate the statement (d) in Theorem 8. Figure 10 shows that a child-parent vector and a parent-grandparent vector for each feature estimated by the original unembedidngs (blue) are orthogonal. The random parent and grandparent vectors (orange) make the cosine similarity not close to 0, suggesting that the orthogonality is not merely a byproduct of the high-dimensional space. 

18 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
cos( w parent of w , parent of w grandparent of w ), Whole set cos( w parent of w , parent of w grandparent of w ), Training set<br>0.25 0.25<br>0.00 0.00<br>0.25 0.25<br>0.50 0.50<br>0.75 0.75<br>1.00 1.00<br>0 100 200 300 400 500 600 0 100 200 300 400 500 600<br>Binary Features in WordNet Hierarchy<br>Original Unembeddings Shuffled Unembeddings Original Unembeddings + Random Parent and Grandparent<br>Figure 10: WordNet noun hierarchy is encoded as the orthogonal structure predicted by statement (d)<br>in Theorem 8. The cosine similarity between a child-parent vector and a parent-grandparent vector<br>for each feature in the hierarchy estimated by original (blue) and shuffled (green) unembedding<br>vectors with the whole (left) and training (right) set of tokens. Another baseline (orange) is the<br>cosine similarity with random parent and grandparent vectors.<br>2.0 Original Unembeddings 2.0 Sh uffl ed  Unembeddings<br>1.5 1.5<br>1.0 1.0<br>0.5 0.5<br>0.0 0.0<br>0.5 0.5<br>1.0 1.0<br>0 100 200 300 400 500 600 0 100 200 300 400 500 600<br>Binary Features in WordNet  H ierar ch y<br>train test random<br><!-- End of picture text -->

Figure 11: Comparison of projection of train (green), test (blue), and random (orange) words on estimated **mean vector** for each WordNet feature, from the original unembeddings (left) and the shuffled unembeddings (right). The value for each word _y_ is ( _g_ ( _y_ )<sup>_⊤_</sup> _ℓ_<sup>¯</sup> _w_ ) _/∥ℓ_<sup>¯</sup> _w∥_ 2<sup>2thatweexpectto</sup> be 1 when _y_ has the target feature. The _x_ -axis indices denote all features in the noun hierarchy. The thick lines present the mean of the projections for each feature and the error bars indicate the standard deviation. The features are ordered by the hierarchy. 

Similar to the findings in Figure 5, the shuffled unembedding vectors (green) reflect the set inclusion relationships, and the cosine similarities are close to 0. When we violate the set inclusion by using only the 70% training set of tokens, the right plot in Figure 10 shows that the cosine similarities from shuffled (green) unembeddings are much smaller, while those from original (blue) unembeddings are still close to 0. This indicates that the hierarchical orthogonality is not a trivial consequence of the set inclusion, but rather reflects the semantic hierarchy. 

**The Mean Estimator for the Vector Representation Has High Variance** The mean vector E( _gw_ ) can serve as an estimator for the vector representation _ℓ_<sup>¯</sup> _w_ . However, Figure 11, which is analogous to Figure 3, shows that the projections of train and test words onto the mean vector have larger variances. Therefore, the LDA direction (eq. (5.2)) would be a more appropriate estimator for the vector representation. 

**Geometry of Hierarchical relations on the Euclidean Inner Product** In this paper, we transform the representation spaces to use the causal inner product. However, what if we employ the naive Euclidean inner product instead? To address this, we estimate the vector representations for the WordNet noun hierarchy estimated from the original Gemma unembedding vectors through centering alone, without applying whitening (which is necessary for the causal inner product). This approach still preserves the Euclidean inner product. Then, Figure 12 shows that the hierarchical orthogonality is not satisfied in the Euclidean inner product space, as evidenced by the cosine similarities not being close to 0. 

19 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
0.4 cos( w parent of w , parent of w ), Whole set<br>0.2<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0 100 200 300 400 500 600<br>Binary Features in WordNet Hierarchy<br>Euclidean Inner Product Causal Inner Product<br>Figure 12: WordNet noun hierarchy is not encoded as the orthogonal structure when we use the<br>naive Euclidean inner product, whereas it is encoded as orthogonality in the causal inner product.<br>2.0 Original Unembeddings 2.0 S huffled  U nembeddings<br>1.5 1.5<br>1.0 1.0<br>0.5 0.5<br>0.0 0.0<br>0.5 0.5<br>1.0 1.0<br>0 50 100 150 200 250 300 350 0 50 10 0 150 200 250 300 350<br>Binary Features in WordNet Hierarchy<br>train test random<br><!-- End of picture text -->

Figure 12: WordNet noun hierarchy is not encoded as the orthogonal structure when we use the naive Euclidean inner product, whereas it is encoded as orthogonality in the causal inner product. 

Figure 13: Vector representations exist for most binary features in the WordNet **verb** hierarchy. 

**WordNet Verb Hierarchy** In the same way as for the noun hierarchy, we estimate the vector representations for the WordNet verb hierarchy. Analogous to Figure 3, Figure 13 shows that the vector representations exist for most binary features in the WordNet verb hierarchy. Analogous to Figure 4, Figure 14 shows that the hierarchical semantics in WordNet are encoded in the Gemma representation space. Analogous to Figure 5 and Figure 10, Figure 15 and Figure 16 show that the hierarchical orthogonality is encoded in the Gemma representation space for the WordNet verb hierarchy. 

**WordNet Noun Hierarchy encoded in LLaMA-3 Model** We also validate the theoretical predictions in the LLaMA-3-8B model (Dubey et al., 2024) in the same way as for the Gemma model. The vector representations for most binary features in the WordNet noun hierarchy exist in the LLaMA-3 model (Figure 17). The hierarchical semantics in WordNet are encoded in the LLaMA-3 representation space (Figure 18). Lastly, the hierarchical orthogonality is encoded in the LLaMA-3 representation space for the WordNet noun hierarchy (Figure 19 and Figure 20). 

## G WHY DOES SET INCLUSION GIVE THE ORTHOGONALITY? 

The left panel of Figure 5 shows that the cosine similarity between the child-parent vector and the parent vector is also close to 0 when they are estimated by the shuffled unembedding vectors. One of the possible explanations for this phenomenon is that the set inclusion relationships (and the estimating process) give the orthogonality. For example, the set inclusion between the collections _Y_ ( _z_ ) = _{_ dog _,_ sandwich _,_ running _}_ for a child _z_ and _Y_ ( _w_ ) = _{_ dog _,_ sandwich _,_ running _,_ France _,_ scientist _,_ diamond _}_ for a parent _w_ can make the child-parent vector and the parent vector orthogonal. 

Since the LDA-based vector representation estimated by eq. (5.2) is similar to the mean vector E( _gw_ ), we briefly derive the orthogonality using the mean vector as the vector representation. _i.i.d. ∼_ Suppose that _a_ 1 _, . . . , aNa_ , _b_ 1 _, . . . , bNb N_ (0 _, Id_ ), which may correspond to the (centered and whitened) shuffled unembeddings. Here, _ai_ ’s are for the child _z_ , and both _ai_ ’s and _bi_ ’s are for the par- 

20 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
0 Pairwise Shortest Distance 0 cos( w, z ) (Original Unembeddings) 0 cos( w, z ) (Shuffled Unembeddings) 1.00<br>50 50 50 0.75<br>100 100 100 0.50<br>0.25<br>150 150 150<br>0.00<br>200 200 200<br>0.25<br>250 250 250<br>0.50<br>300 300 300<br>0.75<br>350 350 350<br>1.00<br>0 100 200 300 0 100 200 300 0 100 200 300<br>Figure 14: WordNet  verb  hierarchy is encoded in Gemma representation space.<br>cos( w parent of w , parent of w ), Whole set cos( w parent of w , parent of w ), Training set<br>0.25 0.25<br>0.00 0.00<br>0.25 0.25<br>0.50 0.50<br>0.75 0.75<br>1.00 1.00<br>0 50 100 150 200 0 50 100 150 200<br>Binary Features in WordNet Hierarchy<br>Original Unembeddings Shuffled Unembeddings Original Unembeddings + Random Parent<br><!-- End of picture text -->

Figure 15: WordNet **verb** hierarchy is encoded as the orthogonal structure predicted by statement (a) in Theorem 8. 

ent _w_ . Then, the parent vector is _vw_ = _Na_ +1 _Nb_<sup>(�</sup><sup>_ai_+�</sup><sup>_bi_), and the child vector is</sup><sup>_vz_=</sup> _N_ <u>1</u> _a_ � _ai_ . We know that _a_ 1 _, . . . , aNa_ , _b_ 1 _, . . . , bNb_ are mutually orthogonal and _∥ai∥_<sup>2</sup> = _∥bi∥_<sup>2</sup> = _d_ with high probability. Therefore, we have 







with high probability. 

In words, the orthogonality between the child-parent vector and the parent vector can be derived from the set inclusion relationships. We can say that the set inclusion is the hierarchical relation as defined in Definition 2, but natural semantic hierarchy is our main focus in this paper. In this context, the right panel of Figure 5 shows that the violation of the set inclusion relationships makes the cosine similarity not close to 0, suggesting that semantic hierarchy is encoded as orthogonality. 

21 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
cos( w parent of w , parent of w grandparent of w ), Whole set cos( w parent of w , parent of w grandparent of w ), Training set<br>0.25 0.25<br>0.00 0.00<br>0.25 0.25<br>0.50 0.50<br>0.75 0.75<br>1.00 1.00<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Binary Features in WordNet Hierarchy<br>Original Unembeddings Shuffled Unembeddings Original Unembeddings + Random Parent and Grandparent<br>Figure 16: WordNet  verb  hierarchy is encoded as the orthogonal structure predicted by statement (d)<br>in Theorem 8.<br>2.0 Original Unembeddings 2.0 Shuffled Unembeddings<br>1.5 1.5<br>1.0 1.0<br>0.5 0.5<br>0.0 0.0<br>0.5 0.5<br>1.0 1.0<br>0 100 200 300 400 0 100 200 300 400<br>Binary Features in WordNet  H ierarchy<br>train test random<br><!-- End of picture text -->

Figure 17: Vector representations for most binary features in the WordNet noun hierarchy exist in the **LLaMA-3** model. 



<!-- Start of picture text -->
0 Pairwise Shortest Distance 0 cos( w, z ) (Original Unembeddings) 0 cos( w, z ) (Shuffled Unembeddings) 1.00<br>50 50 50 0.75<br>100 100 100 0.50<br>150 150 150<br>0.25<br>200 200 200<br>0.00<br>250 250 250<br>0.25<br>300 300 300<br>0.50<br>350 350 350<br>0.75<br>400 400 400<br>1.00<br>0 100 200 300 400 0 100 200 300 400 0 100 200 300 400<br><!-- End of picture text -->

Figure 18: WordNet noun hierarchy is encoded in **LLaMA-3** representation space. 



<!-- Start of picture text -->
cos( w parent of w , parent of w ), Whole set cos( w parent of w , parent of w ), Training set<br>0.25 0.25<br>0.00 0.00<br>0.25 0.25<br>0.50 0.50<br>0.75 0.75<br>1.00 1.00<br>0 100 200 300 400 0 100 200 300 400<br>Binary Features in WordNet Hierarchy<br>Original Unembeddings Shuffled Unembeddings Original Unembeddings + Random Parent<br><!-- End of picture text -->

Figure 19: In **LLaMA-3** representation space, WordNet noun hierarchy is encoded as the orthogonal structure predicted by statement (a) in Theorem 8. 

22 

Published as a conference paper at ICLR 2025 



<!-- Start of picture text -->
cos( w parent of w , parent of w grandparent of w ), Whole set cos( w parent of w , parent of w grandparent of w ), Training set<br>0.25 0.25<br>0.00 0.00<br>0.25 0.25<br>0.50 0.50<br>0.75 0.75<br>1.00 1.00<br>0 100 200 300 400 0 100 200 300 400<br>Binary Features in WordNet Hierarchy<br>Original Unembeddings Shuffled Unembeddings Original Unembeddings + Random Parent and Grandparent<br><!-- End of picture text -->

Figure 20: In **LLaMA-3** representation space, WordNet noun hierarchy is encoded as the orthogonal structure predicted by statement (d) in Theorem 8. 

23 

