# The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?

**arXiv:** 2507.08802 · **Source:** https://arxiv.org/abs/2507.08802
**Route:** markdown copied from `~/Papers/Symbolic_Embeddings/corpus/markdown/neurosymbolic/2507.08802.md`. That library does not record which PDF converter produced it. Single-column source, so no column interleaving. Equations and numbers resolve against the PDF.
**Authors:** Denis Sutter, Julian Minder, Thomas Hofmann, Tiago Pimentel. NeurIPS 2025.
**Corpus section:** Tyler Wheeler/interpretability
**Cited by:** NOT CITED BY WHEELER. Added for the audit phase of thread 01 (research/thread-01-amortized-inference/statement.md).
**Why it is here:** Proves that with unrestricted non-linear alignment maps any network can be mapped to any algorithm, so causal-abstraction claims must restrict the map class and carry random-weight controls. The constraint on every causal claim in the thread-01 audit.
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. Method literature, almost all on language models.
**Copied:** 2026-09-25

---

# **The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?** 

**Denis Sutter, Julian Minder, Thomas Hofmann, Tiago Pimentel** ETH Zürich EPFL 



densutter@ethz.ch, julian.minder@epfl.ch, {thomas.hofmann, tiago.pimentel}@inf.ethz.ch densutter/non-linear-representation-dilemma 

## **Abstract** 

The concept of **causal abstraction** got recently popularised to demystify the opaque decision-making processes of machine learning models; in short, a neural network can be **abstracted** as a higher-level algorithm if there exists a function which allows us to map between them. Notably, most interpretability papers implement these maps as linear functions, motivated by the **linear representation hypothesis** : the idea that features are encoded linearly in a model’s representations. However, this linearity constraint is not required by the definition of causal abstraction. In this work, we critically examine the concept of causal abstraction by considering arbitrarily powerful alignment maps. In particular, we prove that under reasonable assumptions, _any_ neural network can be mapped to _any_ algorithm, rendering this unrestricted notion of causal abstraction trivial and uninformative. We complement these theoretical findings with empirical evidence, demonstrating that it is possible to perfectly map models to algorithms even when these models are incapable of solving the actual task; e.g., on an experiment using randomly initialised language models, our alignment maps reach 100% interchange-intervention accuracy on the indirect object identification task. This raises the **non-linear representation dilemma** : if we lift the linearity constraint imposed to alignment maps in causal abstraction analyses, we are left with no principled way to balance the inherent trade-off between these maps’ complexity and accuracy. Together, these results suggest an answer to our title’s question: causal abstraction is _not_ enough for mechanistic interpretability, as it becomes vacuous without assumptions about how models encode information. Studying the connection between this informationencoding assumption and causal abstraction should lead to exciting future work. 

## **1 Introduction** 

The increasing popularity of machine learning (ML) models has led to a surge in their deployment across various industries. However, the lack of interpretability in these models raises significant concerns, particularly in high-stakes applications where understanding the decision-making process is crucial (Goodman and Flaxman, 2017; Tonekaboni et al., 2019; Zhu et al., 2020; Gao and Guan, 2023). Unsurprisingly, this opacity has motivated a multitude of research on mechanistic (or causal) interpretability, which tries to analyse and understand the hidden algorithms that underlie these models (Olah et al., 2020; Elhage et al., 2021; Mueller et al., 2024; Ferrando et al., 2024; Sharkey et al., 2025). 

A promising approach to address this challenge is **causal abstraction** (Beckers and Halpern, 2019; Geiger et al., 2024a), which tries to map the behaviour of a model to a higher-level (and conceptually simpler) algorithm which solves the task. At the core of this concept is the idea that if an intervention is found to change a model’s behaviour in a way that aligns with a specific algorithm, then that algorithm can be considered implemented by the model. Recent research, however, has raised considerable issues with this approach (e.g., Makelov et al., 2024; Mueller, 2024; Sun et al., 2025). Among those, Méloux et al. (2025) notes that a model’s causal abstraction is not necessarily unique, showing that many algorithms can be aligned to the same neural network. Additionally, most work on causal 

39th Conference on Neural Information Processing Systems (NeurIPS 2025). 



<!-- Start of picture text -->
Neural True/False Classifier ψηΦ  2 ψ 2<br>ψ 2 Φ Acc.<br>Identity Φ -1 Identity 0.8<br>ψ 1 ψ 2 Linear 0.9<br>ψηΦ  1 ψ 1 Non-linear 1.0<br>ψ 1 Φ<br>Accuracy = 1.0 ψη 2 ψ 2<br>Causal<br>Abstraction Linear Φ -1<br>Hierarchical Equality Algorithm ψηΦ  1 ψ 1<br>(a==b)==(c==d) ψηΦ  2 ψ 2<br>ψ 1 ψ 2<br>η1  : a==b< η2  : c==d Non-linear Φ -1<br><a     b c     d Φ<br>ψη 1 ψ 1<br>Datapoints: Dataset: Interventions:<br>Datapoint with label True = (a=1,b=2,c=2,d=4)  = (a=9,b=9,c=7,d=7) = (a=1,b=1,c=0,d=5) = ([ ]ψηΦ1, [ ]ψΦη1) = ([ ]ψη1Φ , [ ]ψΦη1)<br>Datapoint with label False = (a=6,b=6,c=3,d=3) = (a=5,b=8,c=5,d=5) = (a=4,b=0,c=2,d=2) = ([ ]ψη1Φ , [ ]ψΦη1) = ([ ]ψηΦ1, [ ]ψΦη1)<br>Predict O Predict X<br><!-- End of picture text -->

Figure 1: A visualisation of what happens when analysing causal abstractions with increasingly complex alignment maps _ϕ_ . The more complex _ϕ_ is, the higher the intervention accuracy—and, consequently, the stronger the algorithm–DNN alignment. In Theorem 1, we show that given arbitrarily complex alignment maps, we can always find a perfect alignment (under reasonable assumptions). 

abstraction (Wu et al., 2023; Geiger et al., 2024b; Minder et al., 2025; Sun et al., 2025) implicitly assumes information is linearly encoded in models’ representations, relying on the **linear representation hypothesis** (Alain and Bengio, 2016; Bolukbasi et al., 2016). Linearity, however, is not required by the definition of causal abstraction (Beckers and Halpern, 2019) and increasing evidence suggests that not all representations may be linearly encoded (White et al., 2021; Olah and Jermyn, 2024; Mueller, 2024; Csordás et al., 2024; Engels et al., 2025a,b; Kantamneni and Tegmark, 2025). 

In this paper, we first prove that, once we drop the linearity constraint, _any_ model can be perfectly mapped to _any_ algorithm under relatively weak assumptions—e.g., hidden activation’s inputinjectivity and output-surjectivity, which we will define formally. This renders causal abstraction vacuous when used without constraints. If we restrict alignment maps to only consider, e.g., linear functions, this problem does not arise though. It follows that causal abstraction implicitly relies on strong assumptions about how features are encoded in deep neural networks (DNNs), and becomes trivial without such assumptions. This puts us at an impasse: we may want to rely on stronger notions of causal abstraction which may leverage non-linearly encoded information, but this may make our analyses vacuous; we call this the **non-linear representation dilemma** (schematised in Fig. 1). 

To empirically validate our theoretical results, we reproduce the original distributed alignment search (DAS) experiments (Geiger et al., 2024b), but while leveraging more complex alignment maps. We find that key empirical patterns they observed—such as the first layer being easier to map to the tested algorithms in a hierarchical equality task—vanish when we use more powerful maps. Additionally, we find that we can achieve over 80% interchange intervention accuracy (IIA) using non-linear alignment maps in randomly initialised models. Extending our experiments to language models from the Pythia suite (Biderman et al., 2023), we show that near-perfect maps can be found for randomly initialised models in the indirect object identification (IOI) task (Wang et al., 2023); notably, as training progresses, the complexity of the alignment maps needed to achieve perfect IIA in this task decreases. Overall, our results show that causal abstraction, while promising in theory, suffers from a fundamental limitation: without _a priori_ constraints on the used alignment maps, it becomes vacuous as a method for understanding neural networks. 

## **2 Background** 

In this section, we formally define algorithms (§2.1) and deep neural networks (§2.2). These will then be used to define a causal abstraction (§3). First, we formalise a task as a function `T` : _X →Y_ , where **x** _∈X_ represents a set of input features and **y** _∈Y_ denotes the corresponding output. 

### **2.1 Algorithms** 

Given a task `T` , we may hypothesise different ways it can be solved. We term each such hypothesis an algorithm<sup>1</sup> `A` , which we represent as a deterministic causal model—a directed acyclic graph that 

> 1“Algorithm” here need not match a formal definition as, e.g., the considered functions may be uncomputable. 

2 

implements a function _f_ `A` : _X →Y_ .<sup>2</sup> These causal models have a set of nodes **_η_** `all` which can be decomposed into three disjoint sets: (i) input nodes **_η_ x** representing elements in **x** , (ii) output nodes **_η_ y** representing elements in **y** , and (iii) inner nodes **_η_** `inner` representing intermediate variables used in the computation of _f_ `A` . As we focus on acyclic causal models, the edges in this graph induce a partial ordering on nodes **_η_ x** _≺_ **_η_** `inner` _≺_ **_η_ y** . Let _vη_ denote the value held by node _η_ , and let **v** **_η_** denote values taken by the set of nodes **_η_** . The set of incoming edges to a node _η_ represent a direct causal relationship between that node and its parents `parA` ( _η_ ), denoted: _vη_ = _f_<sup>_η_</sup> `A`<sup>(</sup><sup>**v**</sup> `parA` ( _η_ )<sup>).We can compute</sup> algorithm _f_ `A` by iteratively solving the value of its nodes while respecting their partial ordering: 



where we define **v** **_η_ x** to be the input value **x** , and take the value of the output nodes **v** **_η_ y** as our algorithm’s output. Importantly, for an algorithm `A` to represent a task `T` , its output under “normal” operation must be _f_ `A` ( **x** ) = `T` ( **x** ). For an example of a task and related algorithms, see App. I.1. 

These causal models, however, allow us to go beyond “normal” operations and investigate the behaviour of our algorithm under counterfactual settings. We can, for instance, investigate what its behaviour would be if we enforce a node _η_<sup>_′_</sup> ’s value to be a constant _vη′_ = _c_ , which we write as: 



Now, let _f_<sup>:</sup> `A`<sup>_η_(</sup><sup>**x**)representafunctionwhichrunsouralgorithmwithinput</sup><sup>**x**untilitreaches</sup> node _η_ , outputting its value _vη_ . We can use such interventions to investigate the behaviour of algorithm `A` under input **x** , when node _η_<sup>_′_</sup> is forced to assume the value it would have under **x**<sup>_′_</sup> as: _f_ `A` ( **x** _,_ ( _vη′ ← f_ `A`<sup>:</sup><sup>_η′_(</sup><sup>**x**</sup><sup>_′_))).Now,let</sup><sup>**I**</sup><sup>`A`beamulti-nodeintervention,e.g.,</sup><sup>**I**</sup><sup>`A`= (</sup><sup>**v**</sup><sup>**_η_**</sup><sup>_′←_</sup><sup>**c**</sup><sup>_′_)</sup> where **_η_**<sup>_′_</sup> = [ _η_<sup>_′_</sup> _, η_<sup>_′′_</sup> ] and **c**<sup>_′_</sup> = [ _f_ `A`<sup>:</sup><sup>_η′_(</sup><sup>**x**</sup><sup>_′_)</sup><sup>_, c_].Wecanobservehowourmodeloperatesunderthose</sup> interventions by running _f_ `A` ( **x** _,_ **I** `A` ). See App. B for a pseudo-code implementation. 

### **2.2 Deep Neural Networks** 

Deep neural networks (DNNs) are the driving force behind recent advances in ML and can be defined as a sequence of functions _f_<sup>_ℓ_</sup> `N`<sup>:</sup><sup>_H_</sup><sup>**_ψ_**</sup> _ℓ_<sup>_→H_</sup><sup>**_ψ_**</sup> _ℓ_ +1<sup>, where</sup><sup>**_ψ_**</sup><sup>_ℓ_denotes the set of neurons in layer</sup><sup>_ℓ_and</sup> _H_ **_ψ_** _ℓ_ is the corresponding vector space. A DNN `N` with _L_ layers can be specified as follows: 



where **h** **_ψ_** _ℓ_ denotes the vector of activations for neurons **_ψ_** _ℓ_ . We focus on DNNs with real-valued neurons and probabilistic outputs, so that _H_ **_ψ_** 0 = _X_ , _H_ **_ψ_** _ℓ_ = R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> and _H_ **_ψ_** _L_ +1 = ∆<sup>_|Y|−_1</sup> . We define _f_<sup>**_ψ_**</sup> `N`<sup>_′_(</sup><sup>**x**</sup><sup>_′_) as the function that computes the activations of the subset of neurons</sup><sup>**_ψ_**</sup><sup>_′_when the network</sup> is evaluated on input **x**<sup>_′_</sup> . In particular, _f_<sup>**_ψ_**</sup> `N`<sup>_ℓ_(</sup><sup>**x**) returns the activations at layer</sup><sup>_ℓ_for input</sup><sup>**x**.Thus, the</sup> standard computation of the DNN corresponds to evaluating _p_ `N` ( **y** _|_ **x** ) = _f_ `N`<sup>**_ψ_**</sup><sup>_L_+1</sup> ( **x** ). This formulation allows us to instantiate common architectures, such as multi-layer perceptrons (MLPs) or transformers, by specifying the form of each _f_<sup>_ℓ_</sup> `N`<sup>and the structure of the neuron sets</sup><sup>**_ψ_**</sup><sup>_ℓ_.The parameters of these</sup> models are typically optimised to minimise the cross-entropy loss. 

Notably, similarly to the algorithms above, a DNN’s architecture induces a partial ordering on its neurons, respecting the order in which they are computed **_ψ_** 0 _≺_ **_ψ_** _ℓ ≺_ **_ψ_** _L_ . We can thus analogously define a **DNN intervention** as follows: given a set of neurons **_ψ_** in the network and a corresponding set of values **c** **_ψ_** , we denote the intervention by _f_ `N`<sup>**_ψ_**</sup><sup>_L_+1</sup> � **x** _,_ ( **h** **_ψ_** _←_ **c** **_ψ_** )�. This notation means that, during the forward computation of the DNN, the activations of the neurons in **_ψ_** are fixed to the specified values **c** **_ψ_** , while the rest of the network operates as usual. 

## **3 Causal Abstraction** 

To define causal abstraction, we will base ourselves on the definitions in Beckers and Halpern (2019) and Geiger et al. (2024a). Let an **abstraction map** be defined as _τ_ : **_H_** _→_ **_N_** , where **_H_** and **_N_** are, respectively, the Cartesian products of the hidden state-spaces in a neural network `N` (i.e., `def def` **_ψ_** `int` = **_ψ_** 1:), and the node value-spaces in an algorithm `A` (i.e., **_η_** `int` = **_η_** `inner` _∪_ **_η_ y** ), both excluding the inputs **_ψ_** 0 and **_η_ x** . In words, an abstraction map translates the inner states of a neural network into an algorithms’ inner states. Now, consider the DNN intervention **I** `N` = **h** **_ψ_** _←_ **c** **_ψ_** and the algorithm 

2Geiger et al. (2024a) also considers cyclic deterministic causal models and Beckers and Halpern (2019) considers cyclic and stochastic causal models. We leave the expansion of our work to such models for future work. 

3 

intervention **I** `A` = **v** **_η_** _←_ **c** **_η_** . Further, let **_H_ h** **_ψ_** = **c** **_ψ_** be the set of states in a DNN for which **h** **_ψ_** = **c** **_ψ_** holds, and equivalently for **_N_ v** **_η_** = **c** **_η_** . Under abstraction map _τ_ , we can define an intervention map as: 



Intuitively, _ωτ_ maps a DNN intervention **I** `N` to an algorithmic one **I** `A` if the sets of states they induce on `N` and `A` , respectively, are the same. Further, let **_I_** `A` be a set of interventions **I** `A` which can be performed on algorithm `A` . We can use _ωτ_ to derive a set of equivalent DNN interventions as: 



Given these definitions, we now put forward a first notion of causal abstraction. 

**Definition 1** ( **from Beckers and Halpern, 2019** ) **.** _An algorithm_ `A` _is a_ **_τ -abstraction_** _of a neural network_ `N` _iff: τ is surjective;_ **_I_** `A` = _ωτ_ ( **_I_** `N` ) _;_<sup>3</sup> _and there exists a surjective τ_ **_η_ x** _such that:_ 



In words, the first condition in this definition enforces that all states in an algorithm are needed to abstract the DNN, while the second and third enforce that interventions in the algorithm have the same effect as interventions in the DNN. We further say that `A` is a **strong** **_τ_ -abstraction** of `N` if it is a _τ_ -abstraction and **_I_** `A` is maximal, meaning that any intervention is allowed on algorithm `A` . However, while strong _τ_ -abstractions give us a notion of equivalence between algorithms and DNNs, the maps _τ_ may be highly entangled and provide little intuition about the DNN’s behaviour. To ensure algorithmic information is disentangled in the DNN, we say _τ_ is a **constructive abstraction map** if there exists a partition of `N` ’s neurons _{_ **_ψ_** _η | η ∈_ **_η_** `int` _} ∪{_ **_ψ_** _⊥}_ —where **_ψ_** _η_ are non-empty—and there exist maps _τη_ such that _τ_ is equivalent to the block-wise application of _τη_ ( **h** **_ψ_** _η_ ). In other words, constructive abstraction maps compute the value _vη_ of each node _η ∈_ **_η_** `int` in `A` using non-overlapping sets of neurons **_ψ_** _η_ from `N` , with set **_ψ_** _⊥_ being left unused. We now define a second notion of causal abstraction. 

**Definition 2** ( **from Beckers and Halpern, 2019** ) **.** _An algorithm_ `A` _is a_ **_constructive abstraction_** _of a neural network_ `N` _iff there exists an τ : for which_ `A` _is a strong τ -abstraction of_ `N` _; and τ is constructive._ 

As we will deal with algorithm–DNN pairs which share the same input and output spaces, we will impose an additional constraint on _τ_ —one that is not present in Beckers and Halpern’s (2019) definition. Namely, we restrict: **_ψη_ x** to be the neurons in layer zero and _τ_ **_η_ x** to be the identity; and **_ψη_ y** to be the neurons in layer _L_ + 1 and _τ_ **_η_ y** to be the argmax operation.<sup>4</sup> 

### **3.1 Information Encoding in Neural Networks** 

The definition of constructive abstraction above maps non-overlapping sets of neurons in `N` , i.e., **_ψ_** _η_ , to nodes in `A` , i.e., _η_ . However, much research in ML interpretability highlights that concept information is not always neuron-aligned and that neurons are often polysemantic (Olah et al., 2017, 2020; Arora et al., 2018; Elhage et al., 2022). In fact, there is a large debate about how information is encoded in DNNs. We highlight what we see as the three most prominent hypotheses here. 

**Definition 3.** _The_ **_privileged bases hypothesis_** _(Elhage et al., 2023) argues that neurons form privileged bases to encode information in a neural network._ 

Most evidence in favour of this hypothesis comes from indirect evidence: i.e., the presence of neuron-aligned outlier features or activations in DNNs (Kovaleva et al., 2021; Elhage et al., 2023; He et al., 2024; Sun et al., 2024). Going back to 2015, Karpathy (2015) already showed that a single neuron in a language model could carry meaningful information. Importantly, this hypothesis is consistent with the notion of constructive abstraction above, as it argues each node _η_ ’s information should be encoded in separate, non-overlapping sets of neurons. Several researchers, however, question the special status of neurons assumed by this hypothesis, assuming instead that information is encoded in linear subspaces of the representation space, of which neurons are only a special case. 

**Definition 4.** _The_ **_linear representation hypothesis_** _(Alain and Bengio, 2016) argues that information is encoded in linear subspaces of a neural network._ 

> 3We overload function _ωτ_ here, with _ωτ_ ( **_I_** `N` ) simply applying _ωτ_ elementwise to the interventions in set **_I_** `N` . 

> 4We note that this implies algorithm `A` and network `N` must have the same outputs on the input set _X_ . 

4 

A large literature has developed, backed by the linear representation hypothesis, including: concept erasure methods (Ravfogel et al., 2020, 2022), probing methodologies (Elazar et al., 2021; Ravfogel et al., 2021; Lasri et al., 2022), and work on disentangling activations (Yun et al., 2021; Elhage et al., 2022; Huben et al., 2024; Templeton et al., 2024). Some, however, still question this idea that all information must be encoded linearly in DNNs: as neural networks implement non-linear functions, there is no a priori reason for why information should be linearly encoded in them (Conneau et al., 2018; Hewitt and Liang, 2019; Pimentel et al., 2020b,a, 2022). Further, recent research presents strong evidence that some concepts are indeed non-linearly encoded in DNNs (White et al., 2021; Pimentel et al., 2022; Olah and Jermyn, 2024; Csordás et al., 2024; Engels et al., 2025a,b; Kantamneni and Tegmark, 2025). 

**Definition 5.** _The_ **_non-linear representation hypothesis_** _(Pimentel et al., 2020b) argues that information may be encoded in arbitrary non-linear subspaces of a neural network._ 

### **3.2 Distributed Causal Abstractions** 

Following the discussion above, the definition of constructive abstraction may be too strict, as it assumes _τ_ must decompose across neurons—and thus that node information is encoded in non-overlapping neurons. With this in mind, Geiger et al. (2024a,b) proposed the notion of distributed interventions: they expose the subspaces where node information is encoded in a DNN `N` by applying a bijective function to its hidden states; this function’s output is then itself a constructive abstraction of algorithm `A` . Here, we make this notion a bit more formal. 

We define _τ_ as a **distributed abstraction map** if the following two conditions hold. First, there exists a bijective function _ϕ_ —termed here an **alignment map** —that maps the inner neurons **_ψ_** `int` of `N` block-wise to an equal-sized set of **latent variables** **_ψ_**<sup>_ϕ_</sup> `int`<sup>, in a manner that respects the partial</sup> ordering of computations in the network. Specifically, for each layer _ℓ_ , there exists a bijection _ϕℓ_ : R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _→_ R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> on its neurons such that _ϕ_ is defined as the concatenation of these layer-wise bijections. Similarly to the neurons’ activation **h** **_ψ_** , we will denote latent variables as **h** **_ψ_** _ϕ_ = _ϕ_ ( **h** **_ψ_** ). Second, there exists a partition _{_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_| η∈_</sup><sup>**_η_**</sup><sup>`int`</sup><sup>_} ∪{_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ}_oftheresultinglatentvariables</sup><sup>**_ψ_**</sup> `int`<sup>_ϕ_—</sup> where **_ψ_**<sup>_ϕ_</sup> _η_<sup>are non-empty—and a set of maps</sup><sup>_τη_such that</sup><sup>_τ_is equivalent to the block-wise application</sup> of _τη_ ( **h** **_ψ_** _ϕη_<sup>).In words, a distributed abstraction map computes the value</sup><sup>_vη_of each node</sup><sup>_η∈_</sup><sup>**_η_**</sup><sup>`int`in</sup> `A` using non-overlapping partitions of latent variables **_ψ_**<sup>_ϕ_</sup> _η_<sup>from</sup><sup>`N`, with partition</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_remaining unused.</sup> 

Given an alignment map _ϕ_ , we can perform distributed interventions: **h** **_ψ_** _ϕη_<sup>_←_</sup><sup>**c**</sup><sup>**_ψ_**</sup><sup>_ϕ_</sup> _η_<sup>.Theseinter-</sup> ventions are performed by first mapping the hidden state **h** **_ψ_** to the latent variables **h** **_ψ_** _ϕ_ = _ϕ_ ( **h** **_ψ_** ), intervening on a subset **_ψ_**<sup>_ϕ_</sup> _η_<sup>byreplacing</sup><sup>**h**</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>withdesiredvalues</sup><sup>**c**</sup><sup>**_ψ_**</sup><sup>_ϕ_</sup> _η_<sup>,andthenmappingthese</sup> intervened latent variables back to the original neuron base via **h**<sup>_′_</sup> **_ψ_**<sup>=</sup><sup>_ϕ−_1(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_).Thus, interventions</sup> are applied in the latent space defined by _ϕ_ , generalising privileged-bases interventions to arbitrary (possibly non-linear) subspaces. We are now in a position to define distributed abstractions. **Definition 6.** _An algorithm_ `A` _is a_ **_distributed abstraction_** _of a neural network_ `N` _iff there exists an τ : for which_ `A` _is a strong_ **_τ_** _-abstraction of_ `N` _; and τ is a distributed abstraction map._ 

Finally, we note that the set of all possible interventions **_I_** `A` and **_I_** `N` may be hard to analyse in practice. Geiger et al. (2024b) thus restrict their analyses to what we term here **input-restricted interventions** : the set of interventions which are themselves producible by a set of other input-restricted interventions. In other words, we restrict interventions **h** **_ψ_** _′ ←_ **c** **_ψ_** _′_ (where **_ψ_**<sup>_′_</sup> _⊆_ **_ψ_** `int` or **_ψ_**<sup>_′_</sup> _⊆_ **_ψ_**<sup>_ϕ_</sup> `int`<sup>) and</sup><sup>**v**</sup><sup>**_η_**</sup><sup>_′←_</sup><sup>**c**</sup><sup>**_η_**</sup><sup>_′_</sup> to **c** **_ψ_** _′_ and **c** **_η_** _′_ which are a product of other input-restricted interventions, e.g., **c** **_ψ_** _′_ = _f_ `N`<sup>**_ψ_**</sup><sup>_′_(</sup><sup>**x**</sup><sup>_′_) or</sup> **c** **_η_** _′_ = _f_<sup>:</sup> `A`<sup>**_η_**</sup><sup>_′_(</sup><sup>**x**</sup><sup>_′,_(</sup><sup>**v**</sup><sup>**_η_**</sup><sup>_′′←f_:</sup> `A`<sup>**_η_**</sup><sup>_′′_</sup> ( **x**<sup>_′′_</sup> ))). This leads to the definition of **input-restricted** **_τ_ -abstraction** : a weakened notion of strong _τ_ -abstraction, where intervention sets are restricted to input-restricted interventions. Finally, we define an analogous version of distributed abstraction, which is inputrestricted; this is the notion typically used in practice by machine learning practitioners. **Definition 7** ( **inspired by Geiger et al., 2024b** ) **.** _An algorithm_ `A` _is an_ **_input-restricted distributed abstraction_** _of a neural network_ `N` _iff there exists an τ : for which_ `A` _is an input-restricted_ **_τ_** _-abstraction of_ `N` _; and τ is a distributed abstraction map._ 

A visual representation of how these causal abstraction definitions are related is given in App. D as Fig. 6. Finally, we further introduce **input-restricted** **_V_ -abstractions** : input-restricted distributed abstractions for which we restrict alignment maps _ϕ_ to be in a specific variational family _V_ . The case of linear alignment maps, will be particular important here—as it relates to the linear representation hypothesis—and we will thus explicitly label it as **input-restricted linear abstraction** . 

5 

### **3.3 Finding Distributed Abstractions** 

How do we evaluate if an algorithm is an input-restricted distributed abstraction of a DNN? Geiger et al. (2024b) proposes an efficient method to answer this, called distributed alignment search (DAS). Before applying DAS, one must assume a partitioning _{_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|η∈_</sup><sup>**_η_**</sup><sup>`int`</sup><sup>_} ∪{_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ}_whichremains</sup> fixed during the method’s application; we term _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_the</sup><sup>**intervention size**.The principle behind</sup> DAS is then to leverage the constraint on _τ_ **_η_ y** , which is fixed as: **v** **_η_ y** = argmax **y** _∈Y p_ `N` ( **y** _|_ **x** ) since _p_ `N` ( **y** _|_ **x** ) = **h** **_ψ_** _L_ +1 . Given this constraint, we can initialise a parametrised function _ϕ_ , which we train to predict this equality under possible interventions; this is done via gradient descent, minimising the cross-entropy between the DNN and the algorithm. Specifically, we first select a set of nodes to be intervened **_<u>η</u>_** _∈P_ ( **_η_** `inner` ), where _P_ is a function that takes the powerset of a set, along with corresponding counterfactual inputs **x** _η ∈X_ for each _η ∈_ **_<u>η</u>_** and a base input **x** _∅ ∈X_ . We then define the following two interventions: 

**I** `N` = � **h** **_ψ_** _ϕη_ � _η∈_ **_<u>η</u>_**<sup>_←_</sup> � _f_ **_ψ_** `N`<sup>_ϕ_</sup> _η_<sup>(</sup><sup>**x**</sup><sup>_η_)</sup> � _η∈_ **_<u>η</u>_** and **I** `A` = � **v** _η_ � _η∈_ **_<u>η</u>_**<sup>_←_</sup> � _f_<sup>:</sup> `A`<sup>_η_(</sup><sup>**x**</sup><sup>_η_)</sup> � _η∈_ **_<u>η</u>_** (7) Finally, we run our algorithm under base input **x** _∅_ and intervention **I** `A` to get a ground truth output: **y** = _f_ `A` ( **x** _∅,_ **I** `A` ). Repeating this process _N_ times, we build a dataset _D_ = _{_ ( **x**<sup>(</sup> _∅_<sup>_n_)</sup><sup>_,_</sup><sup>**I**</sup> `N`<sup>(</sup><sup>_n_)</sup><sup>_,_</sup><sup>**y**(</sup><sup>_n_))</sup><sup>_}N_</sup> _n_ =1<sup>on</sup> which we can train the alignment map _ϕ_ such that the DNN matches the algorithm: _L_ = _−_ � log _p_<sup>_ϕ_</sup> `N`<sup>(</sup><sup>**y**(</sup><sup>_n_)</sup><sup>_|_</sup><sup>**x**(</sup> _∅_<sup>_n_)</sup><sup>_,_</sup><sup>**I**(</sup> `N`<sup>_n_))</sup><sup>_,_</sup> `where` _p_<sup>_ϕ_</sup> `N`<sup>(</sup><sup>**y**</sup><sup>_|_</sup><sup>**x**</sup> _∅_<sup>_,_</sup><sup>**I**</sup><sup>`N`) =</sup><sup>_f_</sup><sup>`N`(</sup><sup>**x**</sup> _∅_<sup>_,_</sup><sup>**I**</sup><sup>`N`)</sup> (8) ( **x**<sup>(</sup> _∅_<sup>_n_)</sup> _,_ **I** `N`<sup>(</sup><sup>_n_)</sup> _,_ **y**<sup>(</sup><sup>_n_)</sup> ) _∈D_ 

Notably, DAS mostly ignores how function _τ_ is constructed, relying solely on the assumed definition of _τ_ **_η_ y** . Finding a low-loss alignment map _ϕ_ is then assumed as sufficient evidence that `A` is an input-restricted distributed abstraction of `N` . 

## **4 Unbounded Abstractions are Vacuous** 

In this section, we provide our main theorem: that under reasonable assumptions, _any_ algorithm `A` can be shown to be an input-restricted distributed abstraction of _any_ DNN `N` , making this notion of causal abstraction vacuous. To show that, we need a few assumptions (for their formal definition, see App. F). Our first assumption (Assump. 1) is that we have a **countable input-space** _X_ . While this may not hold in general, it holds for common applications such as language modelling (where the input-space is the countably infinite set of finite strings) or computer vision (where the input-space is a countable union of pixels, which can assume a finite set of values). The second assumption (Assump. 2) is that DNNs are **input-injective in all layers** : i.e., _f_<sup>**_ψ_**</sup> `N`<sup>_ℓ_</sup> is injective for all layers. This guarantees that no information about a DNN’s input **x** is lost when computing the hidden states **h** **_ψ_** _ℓ_ . This assumption is also present in prior work (e.g., Pimentel et al., 2020b) and we show in App. G— assuming real-valued weights and activations—that this is almost surely true for transformers at initialisation.<sup>5</sup> Due to floating point precision and neural collapse (Papyan et al., 2020), it is likely not to hold fully in practice; however, it still seems to be well-approximated in many empirical settings (Morris et al., 2023, and App. H). The third assumption (Assump. 3) is **strict output-surjectivity in all layers** . This assumption guarantees that in each layer there is at least one choice of **h** **_ψ_** _ℓ_ that will produce the desired output. Notably, this assumption may not hold in theory, due to issues like the softmax-bottleneck (Yang et al., 2018). In practice, however, even with large vocabulary sizes, it seems that almost all outputs can still be produced by language models (Grivas et al., 2022) which is sufficient for these DNNs to be abstracted by many algorithms. Our fourth assumption (Assump. 4) is that the **algorithm** `A` **and DNN** `N` **have matchable partial-orderings** , meaning that there is a partitioning of neurons in `N` which would match the partial-ordering of nodes in `A` ; this is likely to be the case for most reasonable algorithms given the size of state-of-the-art deep neural networks. Finally, our last assumption (Assump. 5) is that the **DNN** `N` **solves the given task** `T` . We believe this assumption to be reasonable, as it would be impractical in practice to evaluate a neural network that does not perform the task correctly.<sup>6</sup> Given these assumptions, we can now present our main theorem. **Theorem 1.** _Given any algorithm_ `A` _and any neural network_ `N` _such that Assumps. 1 to 5 hold, we can show that_ `A` _is an input-restricted distributed abstraction of_ `N` _._ 

_Proof._ We refer to App. F for the proof. 

5Also see Nikolaou et al. (2025), who show almost sure injectivity holds for transformers throughout training. 

6If the model does not solve the task, perfect IIA is impossible since non-intervened inputs yield incorrect outputs. Thus, assuming the model solves the task is necessary. In practice, however, even when the DNN is imperfect, an alignment map could produce correct outputs for all intervened inputs, achieving near-perfect IIA scores. 

6 

## **5 Experimental Setup** 

Building on the previous section’s proof that alignment maps between DNNs and algorithms always exist, we now demonstrate their practical learnability and how increasingly complex alignment maps reveal various causal abstractions for different tasks, even on DNNs that do not solve them. 

**Alignment Maps.** To assess how complexity impacts causal-abstraction analyses, we explore three ways to parameterise _ϕ_ . First, we will consider the simplest **identity maps** : _ϕ_<sup>`id`</sup> ( **h** ) = **h** . This is the least expressive _ϕ_ we consider, and if we find that `A` abstracts `N` under this map, we can say that `A` is a constructive abstraction of `N` ; further, this map implicitly assumes the privileged bases hypothesis. For _ϕ_<sup>`id`</sup> , we greedily search for the optimal partition _{_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_| η∈_</sup><sup>**_η_**</sup><sup>`int`</sup><sup>_}_(instead of keeping it fixed) by</sup> iteratively adding neurons to them. For all **_η_** `inner` simultaneously, one neuron is added at a time for each **_ψ_**<sup>_ϕ_</sup> _η_<sup>, up to a maximum allowed intervention size; these neurons are chosen to minimise the loss in</sup> eq. (8). Second, we will consider **linear maps** : _ϕ_<sup>`lin`</sup> ( **h** ) = **W** `orth` **h** , where **W** `orth` _∈_ R<sup>_dℓ×dℓ_</sup> is an orthogonal matrix. This is the type of alignment map originally considered by Geiger et al. (2024b),<sup>7</sup> and implicitly assumes the linear representation hypothesis, evaluating input-restricted linear abstractions. Finally, we consider **non-linear maps** : _ϕ_<sup>`nonlin`</sup> ( **h** ) = `revnet` [ _L_ rn _, d_ rn]( **h** ), where `revnet` [ _L_ rn _, d_ rn] is a reversible residual network (RevNet; Gomez et al., 2017) with _L_ rn layers and hidden size _d_ rn. We can modulate the complexity of this final map by increasing _L_ rn and _d_ rn, assuming the non-linear representation hypothesis. We note that all three maps are bijective and easily invertible. 

**Evaluation Metric.** We evaluate the effectiveness of an alignment map _ϕ_ using the **interchange intervention accuracy** (IIA) metric proposed by Geiger et al. (2024b). For a held out test set _D_ `test` with the same structure as the training set _D_ defined in §3.3, we compute the accuracy of our model (i.e., argmax **y** _′∈Y p_ `N`<sup>_ϕ_(</sup><sup>**y**</sup><sup>_′|_</sup><sup>**x**(</sup> _∅_<sup>_n_)</sup><sup>_,_</sup><sup>**I**(</sup> `N`<sup>_n_)))whenpredictingtheintervened</sup><sup>**y**(</sup><sup>_n_)=</sup><sup>_f_</sup><sup>`A`(</sup><sup>**x**(</sup> _∅_<sup>_n_)</sup><sup>_,_</sup><sup>**I**(</sup> `A`<sup>_n_)).We</sup> compare this to the DNN’s accuracy on the test set _D_ `test` without interventions. 

### **5.1 Tasks, Algorithms, and DNNs.** 

**Hierarchical equality task (Geiger et al. (2024b)).** We will showcase our results primarily on this task. Let **x** = **x** 1 _◦_ **x** 2 _◦_ **x** 3 _◦_ **x** 4 be a 16-dimensional vector, and **x** 1 to **x** 4 each be 4-dimensional vectors, where _◦_ represents vector concatenation. Further, let _X_ = [ _−._ 5 _, ._ 5]<sup>16</sup> . This task consists of evaluating: **y** = ( **x** 1 == **x** 2) == ( **x** 3 == **x** 4). As our DNN `N` , we investigate a 3-layer multi-layer perceptron (MLP) with hidden size 16, trained to perform this task; we describe this DNN, and its training procedure in more detail in App. I.1. Finally, we explore three algorithms for this task. The both equality relations algorithm first computes the two equalities ( _vη_ 1 = ( **x** 1 == **x** 2) and _vη_ 2 = ( **x** 3 == **x** 4)) separately; it then determines whether they are equivalent as a second step. The left equality relation algorithm first computes the left equality ( _vη_ 1 = ( **x** 1 == **x** 2)), and then determines in a single step if this is equivalent to ( **x** 3 == **x** 4). Finally, the identity of first argument algorithm assumes we copy the first input to a node ( _vη_ 1 = **x** 1) and then compute the output directly. These three algorithms are more rigorously defined in App. I.1. 

**Indirect object identification (IOI) task.** In a second set of experiments, we explore this task, inspired by Wang et al. (2023) and using the dataset of Muhia (2022). This task is more realistic and relies on larger (language) models. Here, inputs **x** _∈X_ are strings where two people are first introduced, and later one of them assumes the role of subject ( `S` ), giving or saying something to the other, the indirect object ( `IO` ). The task is then to predict the first token of the `IO` , with the output set _Y_ containing the first token of each person’s name. E.g., **x** = “ `Friends Juana and Kristi found a mango at the bar` _._ `Kristi gave it to` ” and **y** = “ `Juana` ”. As our DNN, we use models from the Pythia suite (Biderman et al., 2023) across different sizes (from 31M to 410M parameters) and training stages. We evaluate the ABAB-ABBA algorithm where, given two names A and B, an inner node _vη_ 1 captures if the sentence structure is ABAB (e.g., “A and B ... A gave to B”) or ABBA (e.g., “A and B ... B gave to A”), and the algorithm outputs prediction B if _vη_ 1 is ABAB and A otherwise. This algorithm is more rigorously defined in App. I.2. 

## **6 Experiments and Results** 

We now proceed with our empirical study, applying alignment maps of varying complexity on both “toy” and real neural networks to evaluate their effects on the causal abstraction method DAS. 

> 7We note that, while Geiger et al. (2024b) describe the used _ϕ_ as a rotation, their pyvene (Wu et al., 2024a) implementation uses orthogonal matrices. This, however, makes no difference in the power of the alignment map. 

7 



<!-- Start of picture text -->
1.00 Both Equality Relations Left Equality Relation Identity of First Argument<br>linear identity<br>0.75 non-linear<br>0.50<br>0.25<br>1 2 8 1 2 8 1 2 8 1 2 8 1 2 8 1 2 8 1 2 8 1 2 8 1 2 8<br>L1 L2 L3 L1 L2 L3 L1 L2 L3<br>Accuracy<br><!-- End of picture text -->

Figure 2: IIA in the hierarchical equality task for causal abstractions trained with different alignment maps _ϕ_ . The figure shows results for all three analysed algorithms for this task. The bars represent the max IIA across 10 runs with different random seeds. The black lines represent mean IIA with 95% confidence intervals. The _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_denotes the intervention size per node.Without interventions, all</sup> DNNs reach almost perfect accuracy (>0.99). The used _ϕ_<sup>`nonlin`</sup> uses _L_ rn = 10 and _d_ rn = 16. 



<!-- Start of picture text -->
1 . 0<br>0 . 8<br>= 1<br>= 2<br>0 . 6 = 8<br>1 2 4 8 16<br>d rn<br>Accuracy<br><!-- End of picture text -->



<!-- Start of picture text -->
1 . 0<br>0 . 9<br>0 . 8 DNN<br>linear<br>0 . 7 L rn = 1 , d rn = 2 4<br>L rn = 5 , d rn = 2 4<br>0 . 6 L rn = 10 , d rn = 2 4<br>L rn = 10 , d rn = 2 7<br>0 . 5<br>0 1M Full<br>DNN Training Steps<br>Accuracy<br><!-- End of picture text -->

Figure 3: IIA of alignment between the both equality relations algorithm and an MLP, with interventions at layer 1. Left: Mean IIA over 5 seeds using _ϕ_<sup>`nonlin`</sup> ( _L_ rn = 1) on the trained DNN. Performance improves with larger hidden dimension _d_ rn and intervention size _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_.Right: Maximum</sup> IIA across 5 seeds using _ϕ_<sup>`lin`</sup> and _ϕ_<sup>`nonlin`</sup> with _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_= 8.Complex alignment maps achieve high IIA</sup> even with randomly initialised DNNs, while simpler maps gradually improve as training progresses. 

**Hierarchical equality task, main results.**<sup>8</sup> Fig. 2 presents IIA results across different alignment maps _ϕ_ for all three algorithms. As expected, the identity map _ϕ_<sup>`id`</sup> generally results in the worst performance. Using linear alignments ( _ϕ_<sup>`lin`</sup> ), we observe patterns consistent with Geiger et al. (2024b): IIA for both equality relations and left equality relation decreases substantially in the third layer, indicating information becomes difficult to manipulate using linear transformations at deeper layers. With the non-linear alignment ( _ϕ_<sup>`nonlin`</sup> ), this layer-dependent degradation vanishes, yielding near-optimal IIA across all layers. Consequently, while assuming linear representations seems to enable us to identify the location of certain variables in our DNN, many of these insights fail to generalise when more powerful non-linear alignment maps are employed. The identity of first argument algorithm’s IIA consistently hovers around 50% for _ϕ_<sup>`id`</sup> , _ϕ_<sup>`lin`</sup> and _ϕ_<sup>`nonlin`</sup> . Additional experiments (App. H) suggest this is caused by insufficient capacity of the used `revnet` model, as the identity of **x** 1 seems to be encoded in the model’s hidden states. 

**Hierarchical equality task, exploring** _ϕ_<sup>`nonlin`</sup> **’s complexity.** Fig. 3 (left) illustrates how varying the hidden size _d_ rn and intervention size _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_affects IIA with thebothequalityrelations</sup> algorithm on layer 1 of our MLP. Fig. 3 (right) shows IIA evolution as alignment complexity increases throughout the MLP’s training (evaluated on its layer 1). Remarkably, even with randomly initialised DNNs, we achieve over 80% IIA using the most complex alignment map. As training progresses, simpler alignment maps gradually attain higher IIA values. Additional results in App. I.1.3 extend these findings to other MLP layers, intervention sizes, and algorithms, consistently revealing similar patterns that reinforce our conclusion about the impact of alignment map complexity on IIA dynamics. 

**Indirect object identification task, main results.** Fig. 4 (left) presents the results of trying to find causal abstractions between the ABAB-ABBA algorithm and Pythia language models, exploring how model size affects alignment capabilities. Notably, despite only the larger models (160M and 410M parameters) successfully learning the IOI task, we can align the algorithm to models of all sizes— including the 31M and 70M parameter models that fail to learn the task. Further, and somewhat surprisingly, this alignment is perfect even for randomly initialised models across all sizes; smaller fully trained models (31M, 70M), though, show slightly reduced alignment accuracy. This reduction 

> 8As an additional task similar to hierarchical equality, we also explore the **distributive law task** in App. I.3. 

8 



<!-- Start of picture text -->
1 . 0 1 . 00<br>DNN<br>0 . 75 linear<br>L rn = 1<br>0 . 5 L rn = 2<br>DNN (Full) non-lin. (Full) lin. (Full) L rn = 4<br>DNN (Init.) non-lin. (Init.) lin. (Init.) 0 . 50 L rn = 8<br>31m(L3) 70m(L3) 160m(L6) 410m(L12) 0 1k 2k 3k 4k Full<br>DNN Size DNN Training Step<br>Accuracy Accuracy<br><!-- End of picture text -->

Figure 4: IIA of alignment between ABAB-ABBA algorithm and Pythia language models. Left: IIA across model sizes at initialisation (Init.) or after full training (Full), with intervention at the middle layer. Right: IIA with increasingly complex alignment maps during _Pythia-410m_ ’s training. Results show complex alignment maps yield near-perfect IIA. All _ϕ_<sup>`nonlin`</sup> use _d_ rn = 64. 

may stem from these smaller models saturating late in training (Godey et al., 2024), becoming highly anisotropic and making it harder for _ϕ_<sup>`nonlin`</sup> to access the information needed to match the algorithm. 

**Indirect object identification task, exploring** _ϕ_<sup>`nonlin`</sup> **’s complexity.** Fig. 4 (right) illustrates the interplay between model training progression and algorithmic alignment for the Pythia with 410M parameters. Notably, while this model begins to acquire task proficiency only around training step 3000 (as indicated by model accuracy), employing an 8-layer _ϕ_<sup>`nonlin`</sup> as alignment map yields nearperfect IIA across all training steps, including for randomly initialised models. This pattern partially extends to a 4-layers _ϕ_<sup>`nonlin`</sup> configuration; however, there is a noticeable dip in IIA at step 1000 for this configuration, which may be due to the model over-fitting to unigram statistics (Chang and Bergen, 2022; Belrose et al., 2024) at this point—thereby making context (and hidden states) be mostly ignored when producing model outputs. Interestingly, as training advances, even less complex alignment maps (1- and 2-layer _ϕ_<sup>`nonlin`</sup> ) eventually attain perfect alignment. In contrast, linear maps only approximate perfect alignment in the fully trained model, following a similar trend to the DNN’s performance. 

## **7 Discussion** 

Our results show that when we lift the assumption of linear representations, sufficiently complex alignment maps can achieve near-perfect alignment across all models—regardless of their ability to solve the underlying task. This provides compelling evidence for the non-linear representation dilemma, suggesting causal alignment may be possible even when the model lacks task capability. We now discuss our results in the context of prior literature, with additional related work in App. C. 

**Causal Abstraction is not Enough.** Causal abstraction (Geiger et al., 2024a) has gained traction as a theoretical framework for mechanistic interpretability, promising to overcome probing limitations by analysing DNN behaviour through interventions: if you intervene on a DNN’s representations and its behaviour changes in a predictable way, you have identified how the DNN “truly” encodes that feature (Elazar et al., 2021; Ravfogel et al., 2021; Lasri et al., 2022). Recent critiques of causal abstraction (e.g., Mueller, 2024) highlight practical shortcomings, including the non-uniqueness of identified algorithms (Méloux et al., 2025) and the risk of “interpretability illusions” (Makelov et al., 2024). Despite counterarguments to some of these critiques (Wu et al., 2024b; Jørgensen et al., 2025), concerns have emerged that methods based on causal abstraction may introduce new information rather than accurately reflect the behaviour of the DNN (Wu et al., 2023; Sun et al., 2025); as an example, causal abstraction methods applied to random models sometimes yield abovechance performance (Geiger et al., 2024b; Arora et al., 2024). By examining the implications of assuming arbitrary complex ways in which features may be encoded in a DNN, we show that nearly any neural network can be aligned to any algorithm. Together, our results thus suggest that the shift in interpretability research to causal abstractions does not, by itself, resolve the core challenge of understanding how representations are encoded. Additionally, we note that early causal abstraction methods (Geiger et al., 2021) implicitly rely on the privileged bases hypothesis, while recent advancements (Geiger et al., 2024b) rely on the linear representation hypothesis instead. 

**Balancing the Accuracy vs. Complexity of** _ϕ_ **.** Diagnostic probing was a previously popular method for interpretability research (Alain and Bengio, 2016), where a **probe** was applied to the hidden representations of a DNN and trained to predict a specific variable. Notably, the architecture chosen for this probe implicitly reflected assumptions about representation encoding, and the absence of a universally accepted model for representation encoding precluded a theoretically founded 

9 

choice of probe architecture (Belinkov, 2022). The debate regarding the trade-off between probing complexity and accuracy (Hewitt and Liang, 2019; Pimentel et al., 2020b,a; Voita and Titov, 2020) underscores the risk of complex probes merely memorising variable-specific relations, instead of revealing which information the DNN “truly” encodes and uses. In this paper, we revive this debate by showing a clear analogue in causal abstraction methodologies: the effect of _ϕ_ ’s complexity on IIA. Unfortunately, this debate was never solved by the probing literature, and solutions ranged from: controlling for the probe’s memorisation capacity (Hewitt and Liang, 2019),<sup>9</sup> explicitly measuring a probe’s complexity accuracy trade-off (Pimentel et al., 2020a), training minimum description length probes (Voita and Titov, 2020), or leveraging unsupervised probes (Burns et al., 2023).<sup>10</sup> 

**The Role of Generalisation.** We now highlight that Theorem 1 provides an existence proof for a perfect abstraction map (thus guaranteeing perfect IIA) between a DNN and an algorithm. This existence proof, however, leverages complex interactions between the intervened hidden states and the DNN’s structure, requiring perfect information about both and thus representing a form of extreme overfitting. Crucially, this theorem offers no guarantees regarding the learnability of the alignment map _ϕ_ from limited data or its generalisation to unseen inputs. This gap between theoretical existence and practical learnability becomes evident in practise. For instance, in an additional experiment on the IOI task (in App. I.2.3), we show that when training and test sets contain disjoint sets of names, the learned alignment map fails to generalise, resulting in low IIA on the test set. This suggests that generalisation should play a crucial role in causal abstraction analysis, as the ability to learn abstraction maps that transfer beyond training data seems fundamental to interpreting a model, distinguishing a genuine understanding about its inner workings from mere training pattern memorisation. 

**Investigating Representation Encoding in DNNs.** How neural networks encode variables/concepts is a long-standing question in interpretability, with three main hypotheses standing out: the privileged bases, linear representation, and non-linear representation hypotheses (see §3.1). One way to try to distinguish between these hypotheses is with causal abstraction analyses, but what can we learn about these hypotheses if our methods themselves rely on them as assumptions? One solution could be to compare results using _ϕ_ with different architectures. Our Fig. 4 (right), for instance, shows that while _ϕ_<sup>`nonlin`</sup> achieves consistently near-perfect results throughout model training, _ϕ_<sup>`lin`</sup> accompanies the actual DNN’s performance more closely. Intuitively, we may thus be inclined to support the linear representation hypothesis here. We (the authors), however, cannot make this intuition formal to justify why we believe this is the case. Furthermore, _ϕ_<sup>`lin`</sup> still manages to sometimes achieve IIA higher than the DNN’s accuracy, implying it may also “learn the task”. We expect future work will propose novel methodologies to analyse information encoding and try to answer these questions. 

## **8 Conclusion** 

This paper critically examines causal abstraction in machine learning, when no assumptions are imposed on how representations are encoded. We show that, under mild conditions, any algorithm can be perfectly aligned with any DNN, leading to the non-linear representation dilemma. Empirical validation through experiments on the hierarchical equality and the indirect object identification tasks corroborate our theoretical insights, demonstrating near perfect IIA even in randomly initialised DNNs. So, _what should you do if you want to perform a causal analysis of your DNN?_ We believe that it must be decided on a case-by-case basis. If you have reason to believe the linear representation hypothesis holds for the features you wish to extract, constraining _ϕ_ to linear functions may be advised. If you do not, however, you may face the non-linear representation dilemma, and be forced to investigate some kind of trade-off between _ϕ_ ’s accuracy and complexity. 

**Limitations.** Our proof that any algorithm can be aligned with any DNN (Theorem 1) relies on a form of overfitting. Yet, our experiments show that the learned alignment maps _ϕ_ generalise to unseen test data; studying the factors behind this generalisation would be valuable. Further, our theorem relies on two strong assumptions: input-injectivity (Assump. 2) and strict output-surjectivity (Assump. 3) in all layers. While we justify both, there are settings—related to, e.g., the softmax bottleneck—where they may fail; studying these failure modes could clarify our assumptions’ limitations. 

9Notably, this method was previously applied to causal abstraction analysis by Arora et al. (2024). 

10The complexity—accuracy trade-off in probing arises mainly in supervised settings, where more complex probes can extract richer features from model representations. Unsupervised probing avoids this, lacking the supervision that enables such “gerrymandered” mappings. 

10 

## **Contributions** 

Denis Sutter led the project, implemented the base version of the DAS code, conducted the MLP experiments and derived the base proof of Theorem 1 as well as the proof of Theorem 2. Julian Minder implemented and ran the language model experiments, produced all plots, and helped refine the proof of Theorem 2. Thomas Hofmann provided guidance throughout the project. Tiago Pimentel supervised the project, giving initial intuitions for the proofs in both Theorem 1 and Theorem 2, refining the proof of Theorem 1, and defining the main notation in the paper, integrating feedback from Denis and Julian. All authors wrote the paper together. 

## **Acknowledgments** 

This work was mostly done in the Data Analytics Lab at ETH Zürich. We would like to thank Pietro Lesci, Julius Cheng, Marius Mosbach, Chris Potts, and Atticus Geiger for their thoughtful feedback. We would also like to thank Frederik Hytting Jørgensen for bringing to our attention a mistake in our original Definition 1 and for his feedback on our manuscript. We thank Zhengxuan Wu and Kevin Du for early discussions related to the ideas presented here. We are grateful to the Data Analytics Lab at ETH for providing access to their computing cluster. Julian Minder is supported by the ML Alignment Theory Scholars (MATS) program. Denis Sutter gratefully acknowledges the financial support of his parents, Renate and Wendelin Sutter, throughout his graduate studies, during which this work was carried out, as well as the technical support of Urban Moser and Leo Schefer. 

## **References** 

- Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. _arXiv_ . 

- Aryaman Arora, Dan Jurafsky, and Christopher Potts. 2024. CausalGym: Benchmarking causal interpretability methods on linguistic tasks. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 14638–14663, Bangkok, Thailand. Association for Computational Linguistics. 

- Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. 2018. Linear algebraic structure of word senses, with applications to polysemy. _Transactions of the Association for Computational Linguistics_ , 6:483–495. 

- Sander Beckers and Joseph Y. Halpern. 2019. Abstracting causal models. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 33(01):2678–2685. 

- Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. _Computational Linguistics_ , 48(1):207–219. 

- Nora Belrose, Quintin Pope, Lucia Quirke, Alex Mallen, and Xiaoli Fern. 2024. Neural networks learn statistics of increasing complexity. In _Proceedings of the 41st International Conference on Machine Learning_ , ICML’24. JMLR.org. 

- Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, Usvsn Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. In _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 2397–2430. 

- Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. 2016. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. In _Advances in Neural Information Processing Systems_ , volume 29. 

- Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2023. Discovering latent knowledge in language models without supervision. In _The Eleventh International Conference on Learning Representations_ . 

- Tyler A. Chang and Benjamin K. Bergen. 2022. Word acquisition in neural language models. _Transactions of the Association for Computational Linguistics_ , 10:1–16. 

11 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sashank Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2023. PaLM: Scaling language modeling with pathways. _J. Mach. Learn. Res._ , 24(1). 

- Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. 2018. What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2126–2136, Melbourne, Australia. Association for Computational Linguistics. 

- Róbert Csordás, Christopher Potts, Christopher D Manning, and Atticus Geiger. 2024. Recurrent neural networks learn to store and generate sequences using non-linear representations. In _Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pages 248–262, Miami, Florida, US. Association for Computational Linguistics. 

- Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. 2021. Amnesic probing: Behavioral explanation with amnesic counterfactuals. _Transactions of the Association for Computational Linguistics_ , 9:160–175. 

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. 2022. Toy models of superposition. _Transformer Circuits Thread_ . 

- Nelson Elhage, Robert Lasenby, and Christopher Olah. 2023. Privileged bases in the transformer residual stream. _Transformer Circuits Thread_ , page 24. 

- Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ . 

- Joshua Engels, Eric J Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. 2025a. Not all language model features are one-dimensionally linear. In _The Thirteenth International Conference on Learning Representations_ . 

- Joshua Engels, Logan Riggs Smith, and Max Tegmark. 2025b. Decomposing the dark matter of sparse autoencoders. _Transactions on Machine Learning Research_ . 

- Javier Ferrando, Gabriele Sarti, Arianna Bisazza, and Marta R. Costa-jussà. 2024. A primer on the inner workings of transformer-based language models. _arXiv_ . 

- Lei Gao and Ling Guan. 2023. Interpretability of machine learning: Recent advances and future prospects. _IEEE MultiMedia_ , 30(4):105–118. 

- Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, and Thomas Icard. 2024a. Causal abstraction: A theoretical foundation for mechanistic interpretability. _arXiv_ . 

- Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. 2021. Causal abstractions of neural networks. In _Proceedings of the 35th International Conference on Neural Information Processing Systems_ , NIPS ’21, Red Hook, NY, USA. Curran Associates Inc. 

12 

- Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah Goodman, and Christopher Potts. 2022. Inducing causal structure for interpretable neural networks. In _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pages 7324–7338. PMLR. 

- Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah D. Goodman. 2024b. Finding alignments between interpretable causal variables and distributed neural representations. _arXiv_ . 

- Nathan Godey, Éric Villemonte de la Clergerie, and Benoît Sagot. 2024. Why do small language models underperform? Studying language model saturation via the softmax bottleneck. In _First Conference on Language Modeling_ . 

- Satvik Golechha and James Dao. 2024. Challenges in mechanistically interpreting model representations. _arXiv_ . 

- Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. 2017. The reversible residual network: Backpropagation without storing activations. In _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc. 

- Bryce Goodman and Seth Flaxman. 2017. European Union regulations on algorithmic decision making and a “right to explanation”. _AI Magazine_ , 38(3):50–57. 

- Andreas Grivas, Nikolay Bogoychev, and Adam Lopez. 2022. Low-rank softmax can have unargmaxable classes in theory but rarely in practice. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6738–6758, Dublin, Ireland. Association for Computational Linguistics. 

- Bobby He, Lorenzo Noci, Daniele Paliotta, Imanol Schlag, and Thomas Hofmann. 2024. Understanding and minimising outlier features in transformer training. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ . 

- John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 2733–2743, Hong Kong, China. 

- Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. 2024. Sparse autoencoders find highly interpretable features in language models. In _The Twelfth International Conference on Learning Representations_ . 

- Frederik Hytting Jørgensen, Luigi Gresele, and Sebastian Weichwald. 2025. What is causal about causal models and representations? _arXiv_ . 

- Subhash Kantamneni and Max Tegmark. 2025. Language models use trigonometry to do addition. _arXiv_ . 

- Andrej Karpathy. 2015. The unreasonable effectiveness of recurrent neural networks. 

- Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, and Anna Rumshisky. 2021. BERT busters: Outlier dimensions that disrupt transformers. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 3392–3405, Online. Association for Computational Linguistics. 

- Karim Lasri, Tiago Pimentel, Alessandro Lenci, Thierry Poibeau, and Ryan Cotterell. 2022. Probing for the usage of grammatical number. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8818–8831, Dublin, Ireland. Association for Computational Linguistics. 

- Aleksandar Makelov, Georg Lange, Atticus Geiger, and Neel Nanda. 2024. Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. In _The Twelfth International Conference on Learning Representations_ . 

13 

- Maxime Méloux, Silviu Maniu, François Portet, and Maxime Peyrard. 2025. Everything, everywhere, all at once: Is mechanistic interpretability identifiable? In _The Thirteenth International Conference on Learning Representations_ . 

- Julian Minder, Kevin Du, Niklas Stoehr, Giovanni Monea, Chris Wendler, Robert West, and Ryan Cotterell. 2025. Controllable context sensitivity and the knob behind it. In _The Thirteenth International Conference on Learning Representations_ . 

- John Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander Rush. 2023. Text embeddings reveal (almost) as much as text. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 12448–12460, Singapore. Association for Computational Linguistics. 

- Aaron Mueller. 2024. Missed causes and ambiguous effects: Counterfactuals pose challenges for interpreting neural networks. _arXiv_ . 

- Aaron Mueller, Jannik Brinkmann, Millicent Li, Samuel Marks, Koyena Pal, Nikhil Prakash, Can Rager, Aruna Sankaranarayanan, Arnab Sen Sharma, Jiuding Sun, Eric Todd, David Bau, and Yonatan Belinkov. 2024. The quest for the right mediator: A history, survey, and theoretical grounding of causal interpretability. _arXiv_ . 

Brian Muhia. 2022. ioi (revision 223da8b). 

- Vinod Nair and Geoffrey E. Hinton. 2010. Rectified linear units improve restricted Boltzmann machines. In _Proceedings of the 27th International Conference on International Conference on Machine Learning_ , ICML’10, page 807–814, Madison, WI, USA. Omnipress. 

- Giorgos Nikolaou, Tommaso Mencattini, Donato Crisostomi, Andrea Santilli, Yannis Panagakis, and Emanuele Rodolá. 2025. Language models are injective and hence invertible. _Preprint_ , arXiv:2510.15511. 

- Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. 2020. Zoom in: An introduction to circuits. _Distill_ . Https://distill.pub/2020/circuits/zoom-in. 

- Chris Olah and Adam Jermyn. 2024. What is a linear representation? What is a multidimensional feature? _Transformer Circuits Thread_ . 

- Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. 2017. Feature visualization. _Distill_ . Https://distill.pub/2017/feature-visualization. 

- Vardan Papyan, X. Y. Han, and David L. Donoho. 2020. Prevalence of neural collapse during the terminal phase of deep learning training. _Proceedings of the National Academy of Sciences_ , 117(40):24652–24663. 

- Tiago Pimentel, Naomi Saphra, Adina Williams, and Ryan Cotterell. 2020a. Pareto probing: Trading off accuracy for complexity. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 3138–3153, Online. Association for Computational Linguistics. 

- Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, and Ryan Cotterell. 2020b. Information-theoretic probing for linguistic structure. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4609–4622, Online. Association for Computational Linguistics. 

- Tiago Pimentel, Josef Valvoda, Niklas Stoehr, and Ryan Cotterell. 2022. The architectural bottleneck principle. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 11459–11472, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. _OpenAI blog_ . 

14 

- Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. 2020. Null it out: Guarding protected attributes by iterative nullspace projection. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 7237–7256, Online. Association for Computational Linguistics. 

- Shauli Ravfogel, Grusha Prasad, Tal Linzen, and Yoav Goldberg. 2021. Counterfactual interventions reveal the causal effect of relative clause representations on agreement prediction. In _Proceedings of the 25th Conference on Computational Natural Language Learning_ , pages 194–209, Online. Association for Computational Linguistics. 

- Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan D Cotterell. 2022. Linear adversarial concept erasure. In _Proceedings of the 39th International Conference on Machine Learning_ , volume 162 of _Proceedings of Machine Learning Research_ , pages 18400–18421. PMLR. 

- Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, Stella Biderman, Adria Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi Schoots, Joseph Miller, Eric J. Michaud, Stephen Casper, Max Tegmark, William Saunders, David Bau, Eric Todd, Atticus Geiger, Mor Geva, Jesse Hoogland, Daniel Murfet, and Tom McGrath. 2025. Open problems in mechanistic interpretability. _arXiv_ . 

- Jiuding Sun, Jing Huang, Sidharth Baskaran, Karel D’Oosterlinck, Christopher Potts, Michael Sklar, and Atticus Geiger. 2025. HyperDAS: Towards automating mechanistic interpretability with hypernetworks. _arXiv_ . 

- Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. 2024. Massive activations in large language models. In _First Conference on Language Modeling_ . 

- Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. 2024. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. _Transformer Circuits Thread_ . 

- Sana Tonekaboni, Shalmali Joshi, Melissa D McCradden, and Anna Goldenberg. 2019. What clinicians want: Contextualizing explainable machine learning for clinical end use. _arXiv_ . 

- Oskar van der Wal, Pietro Lesci, Max Müller-Eberstein, Naomi Saphra, Hailey Schoelkopf, Willem Zuidema, and Stella Biderman. 2025. PolyPythias: Stability and outliers across fifty language model pre-training runs. In _The Thirteenth International Conference on Learning Representations_ . 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Proceedings of the 31st International Conference on Neural Information Processing Systems_ , NIPS’17, page 6000–6010, Red Hook, NY, USA. Curran Associates Inc. 

- Elena Voita and Ivan Titov. 2020. Information-theoretic probing with minimum description length. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 183–196, Online. Association for Computational Linguistics. 

- Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. 2023. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _The Eleventh International Conference on Learning Representations_ . 

- Jennifer C. White, Tiago Pimentel, Naomi Saphra, and Ryan Cotterell. 2021. A non-linear structural probe. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 132–138, Online. Association for Computational Linguistics. 

- Zhengxuan Wu, Atticus Geiger, Aryaman Arora, Jing Huang, Zheng Wang, Noah Goodman, Christopher Manning, and Christopher Potts. 2024a. pyvene: A library for understanding and improving PyTorch models via interventions. In _Proceedings of the 2024 Conference of the North American_ 

15 

_Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)_ , pages 158–165, Mexico City, Mexico. Association for Computational Linguistics. 

- Zhengxuan Wu, Atticus Geiger, Jing Huang, Aryaman Arora, Thomas Icard, Christopher Potts, and Noah D. Goodman. 2024b. A reply to Makelov et al. (2023)’s “interpretability illusion” arguments. _arXiv_ . 

- Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, and Noah Goodman. 2023. Interpretability at scale: Identifying causal mechanisms in Alpaca. In _Advances in Neural Information Processing Systems_ , volume 36, pages 78205–78226. Curran Associates, Inc. 

- Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. 2018. Breaking the softmax bottleneck: A high-rank RNN language model. In _International Conference on Learning Representations_ . 

- Zeyu Yun, Yubei Chen, Bruno Olshausen, and Yann LeCun. 2021. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. In _Proceedings of Deep Learning Inside Out (DeeLIO): The 2nd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures_ , pages 1–10, Online. 

- Wenwu Zhu, Xin Wang, and Wen Gao. 2020. Multimedia intelligence: When multimedia meets artificial intelligence. _IEEE Transactions on Multimedia_ , 22(7):1823–1835. 

16 

## **A Reproducibility** 

We provide the code to reproduce our experiments in https://github.com/densutter/ non-linear-representation-dilemma. Refer to the README.md for instructions. 

## **B Pseudo-code for Running an Intervention on an Algorithm** 

In Fig. 5, we present pseudo-code that demonstrates how algorithms execute under our intervention framework. 



<!-- Start of picture text -->
1 def f( A ):<br>2 def f  A ( x , I A = None ):<br>3 v η x = x<br>4 for η in A .topological_sort( η inner  ∪ η y ):<br>5 if I A and η in I A :<br>6 vη = I A [ η ]<br>7 else :<br>8 vη = f η A ( v par A ( η ))<br>9 return v η y<br>10 return f  A<br><!-- End of picture text -->

Figure 5: Pseudo-code implementation of an algorithm with interventions, where interventions **I** `A` are specified as a Python dictionary mapping nodes to their intervened values. 

## **C Additional Related Work** 

The concept of causal abstraction was ported to deep neural networks by Geiger et al. (2024a), providing a generalised framework for understanding how neural networks can be abstracted to higher-level algorithms. Early work by Geiger et al. (2021) explored direct interventions on neuronaligned activations, laying the groundwork for more sophisticated approaches. Building on this, Geiger et al. (2024b) introduced distributed alignment search (DAS), which uses an alignment map to align distributed representations in neural networks with causal graphs. Several improvements to DAS have been proposed: Sun et al. (2025) developed HyperDAS, which automates the search for node information using hypernetworks, while Wu et al. (2023) introduced Boundless DAS, which automatically determines intervention size through gradient descent, scaling to larger models. 

However, recent work has raised important critiques of causal alignment methods. Méloux et al. (2025) demonstrated that multiple algorithms can be causally aligned with the same neural network, and conversely, a single algorithm can align with different network subspaces. Mueller (2024) identified fundamental limitations in counterfactual theories, showing they may miss certain causes and that causal dependencies in neural networks are not necessarily transitive. Makelov et al. (2024) showed that subspace interventions such as those used in DAS can lead to “interpretability illusions”— cases where manipulating a subspace changes the behaviour of the model through activating parallel pathways, rather than directly controlling the target feature. In their response, Wu et al. (2024b) argued these illusions may be artefacts of specific evaluation approaches rather than fundamental flaws, and that they depend on the definition of causality being used, a point also made by Jørgensen et al. (2025). 

Recent work has also raised significant challenges to the linear representation hypothesis. White et al. (2021) demonstrated that syntactic structure in language models is encoded non-linearly, showing that kernelised structural probes outperform linear ones while maintaining parameter count. Similarly, Csordás et al. (2024) found that recurrent neural networks use fundamentally non-linear representations for sequence tasks. Engels et al. (2025a) provided concrete examples of non-linear feature representations in language models, such as days of the week being encoded on a circular manifold. While Golechha and Dao (2024) argued that some language modelling behaviours may be represented linearly due to next-token prediction and LayerNorm folding, Mueller et al. (2024) advocated for exploring non-linear mediators to uncover more sophisticated abstractions. Additional evidence comes from Kantamneni and Tegmark (2025), who found that language models represent numbers on a helical manifold. Olah and Jermyn (2024) offered an important clarification: the linear representation hypothesis is not about dimensionality but rather about features behaving mathematically linearly through addition and scaling, allowing for multidimensional features with constrained geometry. This represents a relaxation of the strongest form of the hypothesis. 

17 

## **D Schematic of the Relation Between Notions of Causal Abstraction** 



<!-- Start of picture text -->
Restricting τ<br>Input-Restricted Constructive Abstraction Constructive Abstractionωτ<br>H τ N H τ N<br>ψη1 hψη1ψη←c2 ψhη1ψη2 ←cψη2 ωτ ηc vη η  ∈𝑃2η ∈←c2 Reachable by someInput+Interventionη2 η all e. g. {η1η1v, ηη12←c} η1 ψη1 hψψη1η2 ←chψψη1η2 ←cψη2 ηc vη η  ∈𝑃2η ∈ℝ←c2 η2 | η η all | e. g. {η1η1v, ηη12←c} η1<br>Input-Restricted Distributed Abstraction<br>H τ N Distributed Abstraction H τ N<br>ΦΦ -1 hΦψψη1Φη1←cψψΦη1Φη2 hΦψη2 ←cψΦη2 ωτ vη2η←c2 η2 vη1 ←cη1 ΦΦ -1 hΦψψη1Φη1←cψψΦη1Φη2 hΦψη2 ←cψΦη2 ωτ vη2η←c2 η2<br>η1 vη1 ←cη1<br>ηcη  ∈𝑃 ∈ Reachable by someInput+Intervention η all e. g. {η1, η2} ηcη  ∈𝑃 ∈ℝ | η η all | e. g. {η1η1, η2}<br>τ -Abstraction Strong τ -Abstraction<br>H τ N H τ N<br>ψ hψ ←hψ ωτ η vη  ← cη ψ hψ ←hψ ωτ η v η ← c η<br>η  ∈ Some Set ⊆𝑃( η all) η  ∈𝑃( η all)<br>cη  ∈ Some Set ⊆ℝ | η | cη  ∈ℝ | η |<br>Any Set of Interventions All Input-Restricted Interventions All Interventions<br>Constructive τ<br>Distributed τ<br>Unconstrained τ<br>Restricting<br>Interventions<br><!-- End of picture text -->

Figure 6: A schematic of the definitions of causal abstraction in §3. The axes represent an increase in how restricted the notion of causal abstraction is based on: _y_ -axis, constraints placed on _τ_ ; and _x_ -axis, constraints placed on the set of allowed interventions. Grey arrows symbolise a superset _→_ subset relationship: if an `A` - `N` pair fulfils the conditions in the subset, it also fulfils them in the superset. 

## **E DNN Definitions** 

### **E.1 MLP** 

A multi-layer perceptron (MLP) consists of a sequence of linear transformations interleaved with non-linear activation functions. 

**Submodule 1.** _We can define a_ **_multi-layer perceptron_** _(_ `mlp` _) by choosing:_ 



_where_ **W** 0 _∈_ R<sup>_|_</sup><sup>**_ψ_**1</sup><sup>_|×|_</sup><sup>**x**</sup><sup>_|_</sup> _,_ **W** _ℓ ∈_ R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ_+1</sup><sup>_|×|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _, and_ **b** _ℓ ∈_ R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ_+1</sup><sup>_|_</sup> _are trainable parameters, and σ is a non-linearity like ReLU. For this model, Hℓ_ = R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _for_ 0 _< ℓ< L and HL_ = R<sup>_|_</sup><sup>**y**</sup><sup>_|_</sup> _._ 

In this work, we focus on MLPs used for classification tasks, whose final layer includes a softmax transformation. 

**DNN 1.** _A_ **_classification multi-layer perceptron_** _(MLP) is defined like Submodule 1 but with a softmax on the last layer:_ 



_where_ **W** _L ∈_ R<sup>_|_</sup><sup>**y**</sup><sup>_|×|_</sup><sup>**_ψ_**</sup><sup>_L|_</sup> _is a trainable parameter._ 

### **E.2 Transformer Language Model** 

In this section, we provide a definition of decoder-only autoregressive language models (Radford et al., 2019; Vaswani et al., 2017). While many variations of transformer architectures have been developed, we focus on the original GPT-2 architecture (Radford et al., 2019). We highlight that the Pythia models explored in our experiments are slightly different from the original GPT-2 and use parallel attention (Chowdhery et al., 2023); however, we do not expect this change to strongly affect our results. We now define the different submodules that compose a transformer. 

18 

**Submodule 2.** _The_ **_Embedding_** _layer in a transformer maps input tokens to vectors:_ 



_In this equation,_ **e** _∈_ R<sup>_|X|×|_</sup><sup>**_ψ_**1</sup><sup>_|_</sup> _is a learned parameter matrix and_ **x** _indexes into its rows._<sup>11</sup> 

**Submodule 3.** **_Multi-Head Self-Attention_** _with H heads is defined as:_ 



_where each head operates in dimension d_ **_η_** _≪|_ **_ψ_** _ℓ| and computes:_ 



_with learned parameters_ **W**<sup>_O_</sup> _∈_ R<sup>_Hd_</sup><sup>**_η_**</sup><sup>_×|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _,_ **W** _i_<sup>_Q,_</sup><sup>**W**</sup> _i_<sup>_K,_</sup><sup>**W**</sup> _i_<sup>_V∈_R</sup><sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|×d_</sup><sup>**_η_**</sup><sup>_._</sup> 

**Submodule 4.** **_Layer Normalization_** _applies per-feature normalization:_ 



_where µ and σ are the mean and standard deviation across all features for a single input, and γ, β are learned parameters._ 

Using these submodules, we define a transformer block. 

**Submodule 5.** _A_ **_Transformer Block_** _chains together attention and MLP layers with residual connections:_ 



_where_ `mlp` _is applied to each token activations separatly as defined in Submodule 1._ 

Finally, we define the complete transformer language model. 

**DNN 2.** _A_ **_transformer language model_** _consists of an embedding layer, transformer blocks, and an output layer:_ 





_where_ `LN` ( **h** **_ψ_** _L_ ) _−_ 1 _selects the final token’s position after layernorm is applied. For this model, H_ **_ψ_** _ℓ_ = R<sup>_|_</sup><sup>**x**</sup><sup>_|×|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _for_ 1 _≤ ℓ ≤ L and H_ **_ψ_** _L_ +1 = ∆<sup>_|Y|−_1</sup> _._ 

## **F Proof of Theorem 1** 

In this section, we prove our main theorem. For notational simplicity, we write in this section: 



We start by formally stating our assumptions. 

**Assumption 1** (Countable input-space) **.** _We assume that the space of inputs (i.e., X ) is countable._ **Assumption 2** (Input-injectivity in all layers) **.** _We assume that f_<sup>:</sup> `N`<sup>_ℓis injective for all layers._</sup> 

**Assumption 3** (Strict output-surjectivity in all layers) **.** _We assume that the composition of f_<sup>_ℓ_</sup> `N`<sup>:</sup><sup>_and_</sup> _τ_ **_η_ y** _is strictly surjective for all layers (we define strict surjectivity in Definition 10)._ 

**Assumption 4** (Algorithm and DNN have matchable partial-orderings) **.** _We assume that there exists a partitioning {_ **_ψ_** _η | η ∈_ **_η_** `int` _} ∪{_ **_ψ_** _⊥} of_ `N` _’s neurons_ **_ψ_** `int` _—where_ **_ψ_** _η are single neurons—which respects the partial-ordering of algorithm_ `A` _, i.e., η ≺ η_<sup>_′_</sup> = _⇒_ **_ψ_** _η ≺_ **_ψ_** _η′ . Further, for each layer at least one neuron is left unused in this partitioning, i.e.,_ **_ψ_** _⊥ ∩_ **_ψ_** _ℓ̸_ = _∅._ 

> 11We ignore positional embeddings here for simplicity, as they do not affect our proofs of injectivity in App. G. Note that, in that section, we show injectivity on an entire layer’s activations **h** **_ψ_** _ℓ_ . Position embeddings would be needed to show injectivity on a single position, e.g., the last token’s position ( **h** **_ψ_** _ℓ_ ) _−_ 1; a property which we conjecture should also hold. Further, we note that position embeddings are used in our experiments. 

19 

**Assumption 5** (DNN solves the task) **.** _We assume that for any input_ **x** _∈X , the neural network solves the task correctly, satisfying_ `T` ( **x** ) = argmax **y** _∈Y p_ `N` ( **y** _|_ **x** ) _._ 

We provide a longer discussion about why we think these assumptions are reasonable in App. F.1. For convenience, we also put a self-contained version of Definition 7 (input-restricted distributed abstraction) in App. F.2. Now, we restate our theorem and present its proof. 

**Theorem 1.** _Given any algorithm_ `A` _and any neural network_ `N` _such that Assumps. 1 to 5 hold, we can show that_ `A` _is an input-restricted distributed abstraction of_ `N` _._ 

_Proof._ To show that an algorithm `A` is an input-restricted distributed abstraction of a neural network `N` , we must show (according to Definition 7) that there exists a _τ_ for which: `A` is an input-restricted **_τ_** -abstraction of `N` ; and _τ_ is a distributed abstraction map. For _τ_ to be a distributed abstraction map, we need a partition of hidden variables which allows us to independently compute it per node. Further, we need the partitioned hidden variables **_ψ_**<sup>_ϕ_</sup> `int`<sup>to be the output of an alignment map</sup><sup>_ϕ_which</sup> is layer-wise decomposable. We thus have: 



Therefore, to define a distributed abstraction map _τ_ , we must define the following three terms: (i) a set of layer-wise alignment maps _{ϕℓ}_<sup>_L_</sup> _ℓ_ =1<sup>(note that the alignment maps</sup><sup>_ϕ_0 and</sup><sup>_ϕL_+1 are fixed by</sup> definition); (ii) a partition of hidden variables **Ψ**<sup>_ϕ_</sup> ; and (iii) a set of per-node functions _{τη}η∈_ **_η_** `int` . To prove this theorem, then, we must show that there exists a way to define these terms while ensuring that `A` is an input-restricted **_τ_** -abstraction of `N` . 

We now note that—given Assump. 4 and independently of our choice of alignment map _ϕ_ —there exists at least one partition **Ψ**<sup>_ϕ_</sup> of the hidden variables **_ψ_**<sup>_ϕ_</sup> `int`<sup>in</sup><sup>`N`for which:</sup> 



where we define **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>as the latent variables given when applying</sup><sup>_ϕℓ_on</sup><sup>**_ψ_**</sup><sup>_ℓ_.To facilitate our proof, we</sup> choose one such partition **Ψ**<sup>_ϕ_</sup> which we will keep fixed independently of our choice of alignment map _ϕ_ . Given partition **Ψ**<sup>_ϕ_</sup> , we can assign each node _η_ to a specific layer _ℓ_ , as **_ψ_**<sup>_ϕ_</sup> _η_<sup>contains a single</sup> hidden variable and therefore trivially belongs to a single layer. We therefore can define **_η_** _ℓ_ as all nodes associated with layer _ℓ_ : 



We now consider the application of interventions on `N` as layer-wise on **_ψ_**<sup>_ϕ_</sup> _η_<sup>_⊆_</sup><sup>**_ψ_**</sup> _ℓ_<sup>_ϕ_for</sup><sup>_η∈_</sup><sup>**_η_**</sup><sup>_ℓ_.Let</sup> us therefore define **_I_**<sup>_ℓ_</sup> `N`<sup>as the set of all interventions on</sup><sup>**_ψ_**</sup><sup>_ϕ_</sup> _η_<sup>for</sup><sup>_η∈_</sup><sup>**_η_**</sup><sup>_ℓ_, where we note that</sup><sup>**_I_**</sup> `N`<sup>_ℓ_also</sup> includes an empty intervention (i.e., no intervention). For notational convenience, we will write the set of all interventions up to layer _ℓ_ as **_I_**<sup>:</sup> `N`<sup>_ℓ_, and the set of all nodes associated with those layers as</sup><sup>**_η_**:</sup><sup>_ℓ_:</sup> 



Finally, we get to an induction proof that will complete this theorem. We will iteratively construct abstraction and alignment maps for each layer such that it holds that: 



where int. stands for intervention. Note that if this holds for all layers, we have proven that `A` is an input-restricted _τ_ -abstraction of `N` , as we can perfectly reconstruct the behaviour of algorithm `A` from `N` ’s states under any intervention.<sup>12</sup> Also note, however, that our definition of abstraction map restricts _τ_ **_η_ y** ( **h** **_ψ_** _L_ +1) = argmax **h** **_ψ_** _L_ +1, so special care must be taken to guarantee that this last identity will 

> 12The attentive reader may note condition **1** only guarantees we can reconstruct the behaviour of algorithm `A` from pre-intervention hidden variables. Lemma 1 shows the same holds for post-intervention hidden variables. 

20 

be preserved. We thus also require an additional condition to hold at each step: 



This condition guarantees that information about previous nodes (i.e., _η ∈_ **_η_** : _ℓ−_ 1) is preserved in each layer’s non-intervened neurons (i.e., **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_).This final condition will be useful to guarantee</sup> conditions **1** and **2** are preserved in future layers. 

**Statement.** Conditions **1** , **2** , and **3** hold for all layers _ℓ_ in a DNN. 

**Base Case (** **_ℓ_ = 0).** For layer _ℓ_ = 0, we have **_η_** _ℓ_ = **_η_ x** . We also have both _ϕℓ_ and _τη_ as the identity function. Further, we consider **_I_**<sup>:</sup><sup>_−_1</sup> = _{∅}_ and **_I_**<sup>:0</sup> = _{∅}_ —where symbol _∅_ here denotes an empty intervention—and we consider _f_<sup>:0</sup> `N`<sup>to be the identity on</sup><sup>**x**.(Note that layer</sup><sup>_f_0</sup> `N`<sup>is not applied in</sup><sup>_f_:0</sup> `N`<sup>.)</sup> Now, it is easy to prove our base case: 

- **1** follows trivially, as _f_<sup>**_η_**</sup> `A`<sup>**x**(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`) =</sup><sup>**x**and</sup><sup>_f_:0</sup> `N`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`) =</sup><sup>**x**.</sup> 

- **2** follows from Assump. 5. 

- **3** follows trivially given **_η_** : _−_ 1 is an empty set. 

**Induction Step (given** **_ℓ −_ 1, then** **_ℓ_ ).** Now, due to the inductive hypothesis, we assume that **1** , **2** and **3** hold for layer ( _ℓ−_ 1). Given this, we must now prove that these conditions also hold for layer _ℓ_ . We will consider two cases: **_η_** _ℓ_ is either empty or not. Before doing so, however, we note that **1** and **3** hold for layer ( _ℓ−_ 1)’s pre-intervention hidden variables. In Lemmas 1 and 2, we show that the same applies for the post-intervention hidden variables. 

Let’s consider the **case where** **_ηℓ_ is empty** . 

In this case, we can simply define _ϕℓ_ as the identity map. Further, given an empty **_η_** _ℓ_ , we know that there are no interventions in this layer, i.e., **_I_**<sup>_ℓ_</sup> `N`<sup>=</sup><sup>_{∅}_, and, as such, we have that:</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ_=</sup><sup>**_I_**:</sup> `N`<sup>_ℓ−_1</sup> _×_ **_I_**<sup>_ℓ_</sup> `N`<sup>=</sup> **_I_** `N`<sup>:</sup><sup>_ℓ−_1</sup> . We can now prove the induction step for this case. 

- **1** is true trivially, since **_η_** _ℓ_ is empty. 



This shows **2** holds for layer _ℓ_ when **_η_** _ℓ_ is empty. 

- **3** follows using the inductive hypothesis. Let **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ−_1</sup> , **x** _∈X_ , and _η ∈_ **_η_** : _ℓ−_ 1. 

- Now, let **h**<sup>_′_</sup> `N`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`)),</sup><sup>**h**</sup><sup>_′_</sup> `N` ( **x** _,_ **I** `N` )), and **I** `A` = _ωτ_ ( **I** `N` ). Further, let **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>=</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup><sup>_ℓ_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>=</sup><sup>_ϕℓ−_1(</sup><sup>_f_:</sup><sup>_ℓ−_1</sup> 

21 

_gη_<sup>_ℓ−_1(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>) =</sup><sup>_f_:</sup> `A`<sup>_η_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`); we know such function exists due to the inductive hypothesis on</sup> **1** and **3** , together with Lemmas 1 and 2. Finally, since _f_<sup>:</sup> `N`<sup>_ℓ_is injective (by Assump. 2) and</sup> since _f_<sup>:</sup> `N`<sup>_ℓ_=</sup><sup>_f ℓ_</sup> `N`<sup>_−_1</sup> _◦ f_<sup>:</sup> `N`<sup>_ℓ−_1</sup> , we know that each **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>is mapped to a unique</sup><sup>**h**</sup> **_ψ_**<sup>_′ϕ_</sup> _ℓ_<sup>in the next</sup> layer. We can thus define function _ℓ_ `N` _−_ 1 on the domain formed by these hidden variables which, given a hidden variable **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>returns its “parent”</sup><sup>**h**</sup> **_ψ_**<sup>_′ϕ_</sup> _ℓ−_ 1<sup>; in other words,</sup> `N` _ℓ−_ 1 is an _ℓ−_ 1 partial inverse of _f_<sup>_ℓ_</sup> `N`<sup>_−_1</sup> defined only on its image. Defining now function _gη_<sup>_ℓ_=</sup><sup>_g_</sup> _η_<sup>_ℓ−_1</sup> _◦_ `N` , we can show: 



Let’s now consider the **case when** **_ηℓ_ is not empty** . 

To show **1** , **2** and **3** for layer _ℓ_ we need to find a suitable bijective _ϕℓ_ . We now show a careful way to construct this map which satisfies these conditions. To do so, we will again split this step of the proof into two parts. We will first take care of the case in which no interventions are applied to layer _ℓ_ , guaranteeing that the model behaves correctly in those cases. In that case, we must handle the set of **input-restricted pre-intervention hidden states** in layer _ℓ_ , which we define as: 



Notably, instead of defining the entire alignment map _ϕℓ_ at once, we will first define its behaviour only on those hidden states. We will denote this domain-restricted function as _ϕ_<sup>◆</sup> _ℓ_<sup>.Given this function,</sup> we will be able to define a set of **input-restricted pre-intervention hidden variables** in layer _ℓ_ as: 



where ◆ represents the non-intervened hidden states and variables, and we will use ❖ to represent the intervened instances. Note that _H_ **_ψ_**<sup>◆</sup><sup>_ϕ_</sup> _ℓ_<sup>is the set of representations output by alignment map</sup><sup>_ϕ_</sup> _ℓ_<sup>◆.</sup> 

The second case we will consider will then handle interventions on this layer, and will again guarantee that the model behaves as expected in those cases. We thus define the set of **input-restricted postintervention hidden variables** as: 



Notably, what an intervention on layer _ℓ_ does is re-combine the representations in _H_<sup>◆</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>.We can thus</sup> write _H_ **_ψ_** _ϕℓ_<sup>in terms of</sup><sup>_H_</sup> **_ψ_**<sup>◆</sup><sup>_ϕ_</sup> _ℓ_<sup>as:</sup> 



We further define the set of **input-restricted intervention-only hidden variables** as: 



22 

By carefully defining the behaviour of _ϕℓ_ on this set, we can guarantee the conditions above to hold. In particular, we will define this part of the function via its inverse _ϕ_<sup>❖</sup> _ℓ −_ 1, which maps these hidden variables back to hidden states. We therefore have _ϕℓ_ and its partial inverse defined as: 



We now define _ϕ_<sup>◆</sup> _ℓ_<sup>.</sup> 

**Definition 8.** _Partial map ϕ_<sup>◆</sup> _ℓ_<sup>:</sup><sup>_H_</sup> **_ψ_**<sup>◆</sup> _ℓ_<sup>_→_R</sup><sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|issomefixedfunctionthatisinjectiveoneach_</sup> _dimension, i.e., ∀i∈{_ 1 _,...,|_ **_ψ_** _ϕℓ_<sup>_|}∀_</sup><sup>**h**1</sup><sup>_,_</sup><sup>**h**2</sup><sup>_∈H_◆</sup> **_ψ_** _ℓ_<sup>:</sup><sup>**h**1</sup><sup>_̸_=</sup><sup>**h**2</sup><sup>_⇒ϕ_</sup> _ℓ_<sup>◆(</sup><sup>**h**1)</sup><sup>_i̸_=</sup><sup>_ϕ_</sup> _ℓ_<sup>◆(</sup><sup>**h**2)</sup><sup>_i._</sup> Such a function exists, because _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>is countable (Lemma 4) and R is uncountable.Further, its partial</sup> inverse _ϕ_<sup>◆</sup> _ℓ −_ 1, defined on the image _H_ ◆ _ℓ_<sup>is injective.</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>, exists because</sup><sup>_ϕ_◆</sup> 

We can now prove that conditions **1** and **3** hold. We also prove that **2** holds when **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ−_1</sup> , i.e., when there is no intervention in layer _ℓ_ . 

- **1** follows using the inductive hypothesis. Let **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ−_1</sup> , **x** _∈X_ , and _η ∈_ **_η_** _ℓ_ . Further, let 

- **h**<sup>_′_</sup> `N`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`)),</sup><sup>**h**</sup><sup>_′_</sup> `N` ( **x** _,_ **I** `N` )), and **I** `A` = _ωτ_ ( **I** `N` ). Now, note that there **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>=</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup><sup>_ℓ_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>=</sup><sup>_ϕℓ−_1(</sup><sup>_f_:</sup><sup>_ℓ−_1</sup> 

- exists a function _gη_ for which _vη_ = _gη_ ( **v** **_η_** : _ℓ−_ 1), as the parents of _η_ are a subset of **_η_** : _ℓ−_ 1. It now suffices to show that **h**<sup>_′_</sup> `A`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`)]</sup><sup>_η∈_</sup><sup>**_η_**</sup> : _ℓ−_ 1<sup>.By</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>encodes information about</sup><sup>**v**</sup><sup>**_η_**:</sup><sup>_ℓ−_1= [</sup><sup>_f_:</sup><sup>_η_</sup> 

- the inductive hypothesis on **1** and **3** , together with Lemmas 1 and 2, we know that **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1 

- encodes information about **v** **_η_** : _ℓ−_ 1; let _g_ **_η_** : _ℓ−_ 1 be a function that extracts this information, i.e., _g_ **_η_** : _ℓ−_ 1( **h**<sup>_′_</sup> `N`<sup>isinjective,and</sup><sup>_ϕ_◆</sup> _ℓ_<sup>isinjectiveoneach</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>)=</sup><sup>**v**</sup><sup>**_η_**:</sup><sup>_ℓ−_1.Now,since</sup><sup>_f_:</sup><sup>_ℓ_</sup> 

- output dimension, we know that **h**<sup>_′_</sup> **_ψ_** _ℓ_<sup>=[</sup><sup>_ϕ_</sup> _ℓ_<sup>◆(</sup><sup>_f ℓ_</sup> `N`<sup>_−_1</sup> ( _ϕ_<sup>_−_</sup> _ℓ−_<sup>1</sup> 1<sup>(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>)))]</sup><sup>**_ψ_**</sup> _η_<sup>_ϕ_containsthesame</sup> information as **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>.We can thus construct (partial) inverses</sup> `N` _ℓ−_ 1, _ϕℓ,_<sup>◆</sup><sup>_−_</sup> **_ψ_**<sup>1</sup><sup>_ϕ_</sup> _η_<sup>and define</sup><sup>_τη_as</sup> the composition _gη ◦ g_ : _η−_ 1 _◦ ϕℓ−_ 1 _◦ ℓ_ `N` _−_ 1 _◦ ϕ_<sup>◆</sup><sup>_−_1</sup> _ℓ,_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>, which concludes this step of the proof:</sup> 

- _τη_ ([ **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>]</sup><sup>**_ψ_**</sup> _η_<sup>_ϕ_) =</sup><sup>_τη_([</sup><sup>_ϕ_</sup> _ℓ_<sup>◆(</sup><sup>_f ℓ_</sup> `N`<sup>_−_1</sup> ( _ϕ_<sup>_−_</sup> _ℓ−_<sup>1</sup> 1<sup>(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>)))]</sup><sup>**_ψ_**</sup> _η_<sup>_ϕ_)</sup> definition of **h**<sup>_′_</sup> **_ψ_** _ℓ_<sup>(30a)</sup> = _gη_ ( _g_ : _η−_ 1( _ϕℓ−_ 1( `N` _ℓ−_ 1( _ϕℓ,_<sup>◆</sup><sup>_−_</sup> **_ψ_**<sup>1</sup><sup>_ϕ_</sup> _η_<sup>([</sup><sup>_ϕ_</sup> _ℓ_<sup>◆(</sup><sup>_f ℓ_</sup> `N`<sup>_−_1</sup> ( _ϕ_<sup>_−_</sup> _ℓ−_<sup>1</sup> 1<sup>(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>)))]</sup><sup>**_ψ_**</sup> _η_<sup>_ϕ_)))))</sup> (30b) = _gη_ ( _g_ : _η−_ 1( **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>))</sup> (30c) = _f_<sup>:</sup> `A`<sup>_η_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`)</sup> (30d) 



This shows **2** holds for layer _ℓ_ when there is no intervention in layer _ℓ_ . 

- **3** follows using the inductive hypothesis. Let **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ−_1</sup> , **x** _∈X_ , and _η ∈_ **_η_** : _ℓ−_ 1. 

- Now, let **h**<sup>_′_</sup> `N`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`)),</sup><sup>**h**</sup><sup>_′_</sup> `N` ( **x** _,_ **I** `N` )), and **I** `A` = _ωτ_ ( **I** `N` ). Further, let **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>=</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup><sup>_ℓ_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ−_ 1<sup>=</sup><sup>_ϕℓ−_1(</sup><sup>_f_:</sup><sup>_ℓ−_1</sup> 

23 



We have now proved **1** and **3** . We have also partially proved **2** for cases where there is no intervention in layer _ℓ_ .<sup>13</sup> We now finish our proof by considering cases where there is an intervention in this layer _ℓ_ . In the second case, we need to handle intervention-only representations _H_<sup>❖</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>.We will</sup> 



Where the first two conditions ensure the necessary bijectivity of _ϕℓ_ and the last characteristic ensures **2** . Now, let **x** _∈X_ be any input and **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ_any intervention.Further, let</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_** `N`<sup>_ϕ_</sup> _ℓ_<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`), and</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>=</sup><sup>_f_</sup> **I** `A` = _ωτ_ ( **I** `N` ). We now note that—given **1** and **3** , and Lemmas 1 and 2—the value _vη_ = _f_ `A`<sup>:</sup><sup>_η_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`)</sup> for all nodes _η ∈_ **_η_** : _ℓ_ are encoded in **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>.This is enough information to determine the algorithm’s</sup> output **y**<sup>_⋆_</sup> = _f_ `A` ( **x** _,_ **I** `A` ). Now, define a function _g_<sup>_⋆_</sup> which maps an element **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∈H_</sup> **_ψ_**<sup>❖</sup><sup>_ϕ_</sup> _ℓ_<sup>to the output</sup> algorithm `A` expects. Further, by Lemma 6 there exists an uncountably infinite set of hidden states _H_<sup>(</sup><sup>**y**</sup><sup>_⋆_)</sup> such that: **_ψ_** _ℓ_ 



We define _H_<sup>ˆ</sup> **_ψ_**<sup>(</sup><sup>**y**</sup> _ℓ_<sup>_⋆_)</sup> = _H_ **_ψ_**<sup>(</sup><sup>**y**</sup> _ℓ_<sup>_⋆_)</sup> _\ H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>, which—as</sup><sup>_H_</sup> **_ψ_**<sup>◆</sup> _ℓ_<sup>is countable—is still uncountably infinite.We can</sup> now map any **h** _∈H_ **_ψ_**<sup>❖</sup><sup>_ϕ_</sup> _ℓ_<sup>to an element in</sup><sup>_H_ˆ</sup> **_ψ_**<sup>(</sup><sup>_g_</sup> _ℓ_<sup>_⋆_(</sup><sup>**h**))</sup> fulfilling the third characteristic of Definition 9. That such a mapping exists adhering to the first and second characteristic of Definition 9 is ensured by the fact that _H_<sup>ˆ</sup> **_ψ_**<sup>(</sup><sup>**y**</sup> _ℓ_<sup>_⋆_)</sup> is uncountable and _H_ **_ψ_**<sup>❖</sup><sup>_ϕ_</sup> _ℓ_<sup>is countable (shown in Lemma 5).Further, as</sup><sup>_ϕ_</sup> _ℓ_<sup>❖</sup> _−_ 1 is injective, its partial inverse _ϕ_<sup>❖</sup> _ℓ_<sup>on its image</sup><sup>_H_</sup> **_ψ_**<sup>❖</sup> _ℓ_<sup>exists.</sup> 

> 13This also proves the result for any input **x** _∈X_ and intervention **I** `N` _∈_ **_I_** _ℓ_ where **h** _∈H_ **_ψ_** ◆ _ℓ_<sup>.By</sup> **3** at layer _ℓ_ , there exists **I**<sup>_′_</sup> `N`<sup>_∈_</sup><sup>**_I_**</sup><sup>_ℓ−_1—constructed by applying only the interventions from</sup><sup>**I**</sup> `N`<sup>on layers</sup><sup>_<ℓ_—such that</sup> _f_<sup>:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup> `N`<sup>) =</sup><sup>_f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>_′_</sup> `N`<sup>).Since both encode the same values on</sup><sup>**_η_**</sup> _<ℓ_<sup>according to</sup> **3** , which fully determine the output of _f_ `A` , and since **2** holds for ( **x** _,_ **I**<sup>_′_</sup> `N`<sup>) at layer</sup><sup>_ℓ_, it must also hold for (</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup> `N`<sup>).</sup> 

24 



Figure 7: Pseudo-code for `extract` _ `unused` _ `rep` . This function returns an unique element in ( _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>_∪H_</sup> **_ψ_**<sup>❖</sup> _ℓ_<sup>)</sup><sup>_\_(</sup><sup>_H_</sup> **_ψ_**<sup>◆</sup><sup>_ϕ_</sup> _ℓ_<sup>_∪H_</sup> **_ψ_**<sup>❖</sup><sup>_ϕ_</sup> _ℓ_<sup>) for each</sup><sup>**h**</sup><sup>_∈_(</sup><sup>_H_</sup> **_ψ_**<sup>◆</sup><sup>_ϕ_</sup> _ℓ_<sup>_∪H_</sup> **_ψ_**<sup>❖</sup><sup>_ϕ_</sup> _ℓ_<sup>)</sup><sup>_\_(</sup><sup>_H_</sup> **_ψ_**<sup>◆</sup> _ℓ_<sup>_∪H_</sup> **_ψ_**<sup>❖</sup> _ℓ_<sup>) ensuring bijectivity of</sup><sup>_ϕ_</sup> _ℓ_<sup>_′_(</sup><sup>**h**).</sup> The attentive reader may have noticed that we defined _ϕℓ_ only over the domain _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>_∪H_</sup> **_ψ_**<sup>❖</sup> _ℓ_<sup>instead</sup> over R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> . We note that it is simple to extend _ϕℓ_ to an _ϕ_<sup>_′_</sup> _ℓ_<sup>defined over R</sup><sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_.Let</sup><sup>`id`be the identity</sup> function and `extract` _ `unused` _ `rep` be defined by the algorithm given in Fig. 7. A bijective function _ϕ_<sup>_′_</sup> _ℓ_<sup>over R</sup><sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_mapping to R</sup><sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_can be defined as:</sup> 



which completes the proof. 

### **F.1 Discussion about Assumptions** 

**Assump. 1 (Countable input-space).** While this assumption cannot be made on all neural networks like MLPs, it holds for models working on language and images. The set of all images with a specific resolution is finite, as it considers a finite number of pixels where each pixel has a finite number of channels (e.g. values for red, green and blue) and each channel is a number between 0 and 255. The set of all sequences in a language model is also countably infinite, as each set of sequences of some length is finite given finite tokens; so we have a set made out of the countable union of finite sets, which is still countable. 

**Assump. 2 (Input-injectivity in all layers).** Neural network layers (e.g., MLP blocks) are not necessarily injective. The usage of learnable weights, activation functions like ReLU (Nair and Hinton, 2010) and information bottlenecks makes it possible to have a non-injective model. However, we prove in App. G that transformers, at least, are almost surely injective at initialisation on their inputs. Further, Nikolaou et al. (2025) recently published a proof—as well as empirical evidence—that transformers are almost surely injective in the hidden states of their last token, both at initialisation and after training. We also see in our empirical experiments in App. H that the MLPs we analyse are also, in practice, injective—or close enough to it that we observe no collisions in embedding space. 

**Assump. 3 (Strict output-surjectivity in all layers).** Surjectivity can be defined on the output distribution _f_<sup>_ℓ_</sup> `N`<sup>::</sup><sup>_H_</sup><sup>**_ψ_**</sup> _ℓ_<sup>_→_∆</sup><sup>_|Y|−_1, but that is a rather strong assumption.For our proofs, we will rely</sup> on strict surjectivity on the classification space instead ( _τ_ **_η_ y** _◦ f_ `N`<sup>_ℓ_::</sup><sup>_H_</sup><sup>**_ψ_**</sup> _ℓ_<sup>_→|Y|_), such that every class</sup> can be predicted. However, surjectivity on the classification space still does not necessarily hold for DNNs. LLMs have problems like the softmax bottleneck (Yang et al., 2018), which can lead to a model having insufficient capacity to predict all possible tokens. Grivas et al. (2022) also evaluate and find this problem, but show that surjectivity on the tokens is still likely in practice, making this a reasonable assumption in LLM settings. 

**Assump. 4 (Algorithm and DNN have matchable partial-orderings).** We assume this since, for a neural network `N` to be abstracted by the algorithm `A` , we need it to have this minimal width and depth. 

**Assump. 5 (DNN solves the task).** We assume this because, if a neural network does not solve the given task, it will also not be abstracted by an algorithm which implements it. 

### **F.2 Detailed Version of Definition 7** 

Definition 7 can also be written without referring to previous definitions as following: 

**Alternative Definition 1** ( **Equivalent to Definition 7** ) **.** _An algorithm_ `A` _is an_ **_input-restricted distributed abstraction_** _of a neural network_ `N` _iff there exists an τ ,_ **_I_** `A` _, and_ **_I_** `N` _such that_ 

25 

- _τ is a distributed abstraction map. I.e., there exists an alignment map ϕ, a latent-variable partition {_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_| η∈_</sup><sup>**_η_**</sup><sup>`int`</sup><sup>_} ∪{_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ} of_</sup><sup>**_ψ_**</sup> `int`<sup>_ϕ(with non-empty_</sup><sup>**_ψ_**</sup><sup>_ϕ_</sup> _η_<sup>_), and subabstraction maps_</sup> _{τη | η ∈_ **_η_** `int` _} such that τ is equivalent to computing the value of each node block-wise with τη_ ( **h** **_ψ_** _ϕη_<sup>)</sup><sup>_.Analignmentmapϕisabijectivefunctionthatmapstheinnerneurons_</sup> **_ψ_** `int` _of_ `N` _onto an equal-sized set of latent variables_ **_ψ_**<sup>_ϕ_</sup> `int`<sup>_, with ϕ respecting the network’s_</sup> _computational order by being the combination of layer-wise bijections ϕℓ_ : R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _→_ R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> _applied to the neurons of each of the DNN’s layers (ℓ);_ 

- **_I_** `A` _and_ **_I_** `N` _are a maximal input-restricted intervention set. A maximal input-restricted intervention set is composed of all interventions produced from other input-restricted interventions, i.e., it is a set with_ **h** **_ψ_** _ϕ ←_ **c** **_ψ_** _ϕ (where_ **_ψ_**<sup>_ϕ_</sup> _⊆_ **_ψ_** `int`<sup>_ϕ) or_</sup><sup>**v**</sup><sup>**_η_**</sup><sup>_←_</sup><sup>**c**</sup><sup>**_η_**</sup><sup>_(where_</sup><sup>**_η_**</sup><sup>_⊆_</sup> **_η_** `int` _) where_ **c** **_ψ_** _ϕ or_ **c** **_η_** _arise from valid input-restricted computations (e.g.,_ **c** **_ψ_** _ϕ_ = _f_ `N`<sup>**_ψ_**</sup><sup>_ϕ_(</sup><sup>**x**)</sup> _or_ **c** **_η_** = _f_ `A`<sup>:</sup><sup>**_η_**(</sup><sup>**x**</sup><sup>_,_</sup><sup>**c**</sup><sup>**_η_**</sup><sup>_←f_:</sup> `A`<sup>**_η_**(</sup><sup>**x**))</sup><sup>_)._</sup> 

- _τ is surjective;_ 

- **_I_** `A` = _ωτ_ ( **_I_** `N` ) _;_ 



### **F.3 Useful Definitions and Lemmas for Theorem 1** 

**Definition 10.** _We say the composition of a function f_ : R<sup>_d_</sup> _→_ ∆<sup>_|Y|−_1</sup> _with_ argmax _is_ **_strictly surjective_** _if, for any output_ **y**<sup>_⋆_</sup> _∈Y, there exists an input_ **h** _∈_ R<sup>_d_</sup> _for which f outputs_ **y**<sup>_⋆_</sup> _no matter how ties are broken in the_ argmax _. Formally:_ 



**Lemma 1.** _Let_ `N` _be a DNN and_ `A` _be an algorithm. Further, let τ be a distributed abstraction map with partition_ **Ψ** _and {τη}η∈_ **_η_** `int` _. If, for all η ∈_ **_η_** _ℓ, τη satisfies the conditions in_ **1** _(defined in Theorem 1’s proof) applied on layer_ **_ℓ_** _’s pre-intervention hidden variables, i.e., if:_ 



_where we note that f_ **_ψ_** `N`<sup>_ϕ_</sup> _ℓ_ = _ϕℓ ◦ f_<sup>:</sup> `N`<sup>_ℓwhen no intervention is applied to layer ℓ.Then τηalso satisfies_</sup> _this condition when applied to layer ℓ’s post-intervention hidden variables:_ 



_Proof._ Let _τη_ be the abstraction map of _η ∈_ **_η_** _ℓ_ . By assumption, condition **1** holds for all preintervention hidden variables, i.e., hidden variables of the form **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>= [</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`))]</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>.We can</sup> show the same function applies to post-intervention hidden variables, i.e., hidden variables of the form: 



Now let **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ_be any intervention and</sup><sup>**x**</sup><sup>_∈X_be any input.Further, let</sup><sup>**I**</sup><sup>`A`=</sup><sup>_ωτ_(</sup><sup>**I**</sup><sup>`N`).If</sup><sup>**I**</sup><sup>`N`</sup><sup>_∈_</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ−_1</sup> , then the post-intervention hidden variable is identical to a pre-intervention one, and the conditions in **1** still hold, i.e.,: **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>= [</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`))]</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>and</sup><sup>_τη_is such that</sup><sup>_τη_(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>) =</sup><sup>_f η_</sup> `A`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`).If</sup><sup>**I**</sup><sup>`N`</sup> _̸_<sup>_∈_</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ−_1</sup> , for each node’s hidden variables **_ψ_**<sup>_ϕ_</sup> _η_<sup>, we might or not intervene on it.If we do not intervene on node</sup> _η_ , then we still have the case **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>= [</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`))]</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>and thus</sup><sup>_τη_still gives us the correct solution,</sup> i.e., _τη_ ( **h**<sup>_′_</sup> `A`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`).If we intervene on</sup><sup>_η_, then we know there exists an intervention of form</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>) =</sup><sup>_f η_</sup> 

26 

**h** **_ψ_** _ϕη_<sup>_←f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_′,_</sup><sup>**I**</sup> `N`<sup>_′_) in</sup><sup>**I**</sup><sup>`N`, for which</sup><sup>**I**</sup> `N`<sup>_′∈_</sup><sup>**_I_**:</sup> `N`<sup>_ℓ−_1</sup> , as our interventions are input-restricted. We also know (by eq. (4)) that there exists an equivalent intervention _vη ← τη_ ( _f_ `N`<sup>:</sup><sup>_ℓ_(</sup><sup>**x**</sup><sup>_′,_</sup><sup>**I**</sup> `N`<sup>_′_)) in</sup><sup>**I**</sup><sup>`A`.We thus</sup> have that _τη_ ( **h**<sup>_′_</sup> `A`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`).</sup> **_ψ_**<sup>_ϕ_</sup> _η_<sup>) =</sup><sup>_f η_</sup> 

**Lemma 2.** _Let_ `N` _be a DNN and_ `A` _be an algorithm. Further, let τ be a distributed abstraction map with partition_ **Ψ** _. If, for all η ∈_ **_η_** : _ℓ−_ 1 _, there exists a function gη which satisfies the conditions in_ **3** _(defined in Theorem 1’s proof) applied on layer_ **_ℓ_** _’s pre-intervention hidden variables, i.e., if:_ 



_where we note that f_ **_ψ_** `N`<sup>_ϕ_</sup> _ℓ_ = _ϕℓ ◦ f_<sup>:</sup> `N`<sup>_ℓwhen no intervention is applied to layer ℓ.Then gηalso satisfied_</sup> _this condition when applied to layer ℓ’s post-intervention hidden variables:_ 



_Proof._ Let _η ∈_ **_η_** : _ℓ−_ 1 and _gη_ be a function that satisfies condition **3** for it. Note that condition **3** holds for all pre-intervention hidden variables, i.e., hidden variables of the form **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_=</sup> [ _ϕℓ_ ( _f_<sup>:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`))]</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_.We can show the same function</sup><sup>_gη_also applies to post-intervention hidden</sup> variables, i.e., hidden variables of the form: 



Now, let **I** `N` _∈_ **_I_**<sup>:</sup> `N`<sup>_ℓ_,</sup><sup>**x**</sup><sup>_∈X_.Further,let</sup><sup>**I**</sup><sup>`A`=</sup><sup>_ωτ_(</sup><sup>**I**</sup><sup>`N`).If</sup><sup>**I**</sup><sup>`N`</sup><sup>_∈_</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ−_1</sup> , then the post-intervention hidden variable is identical to a pre-intervention one, and the conditions in **3** still hold, i.e.,: **h**<sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_=[</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`))]</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_and</sup><sup>_gη_is such that</sup><sup>_gη_(</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_)=</sup><sup>_f η_</sup> `A`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`).If</sup><sup>**I**</sup><sup>`N`</sup> _̸_<sup>_∈_</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ−_1</sup> , it means that we intervene on at least one hidden variable in this layer **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>.However, we never intervene</sup> on neurons in **_ψ_** _⊥_<sup>_ϕ_, meaning that for those we still have the case</sup><sup>**h**</sup><sup>_′_</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_= [</sup><sup>_ϕℓ_(</sup><sup>_f_:</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`))]</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_</sup> and thus the same function _gη_ still satisfies our condition _gη_ ( **h**<sup>_′_</sup> `A`<sup>(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`A`).</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_∩_</sup><sup>**_ψ_**</sup> _⊥_<sup>_ϕ_) =</sup><sup>_f η_</sup> 

**Lemma 3.** _Under Assump. 1 and given a fixed ϕ, the set of input-restricted interventions_ **_I_** `N` _is countable._ 

_Proof._ This can be shown by induction. More specifically, we show that for any layer _ℓ_ , the set of input-restricted interventions **_I_**<sup>:</sup> `N`<sup>_ℓ_is countable for a specific</sup><sup>_ϕ_.</sup> 

**Base Case (** **_ℓ_ = 0).** The base case can be proved trivially, as **_I_**<sup>:0</sup> `N`<sup>=</sup><sup>_{∅}_.</sup> 

**Induction step (** **_ℓ_ given** **_ℓ −_ 1).** By the induction hypothesis, **_I_**<sup>:</sup> `N`<sup>_ℓ−_1</sup> is countable. Now, note that **_I_**<sup>:</sup> `N`<sup>_ℓ_can be decomposed as:</sup> 



As the Cartesian product of two countable sets is itself countable, and as **_I_**<sup>:</sup> `N`<sup>_ℓ−_1</sup> is countable by the inductive hypothesis, we only need to show that **_I_**<sup>_ℓ_</sup> `N`<sup>is countable to complete our proof.This set</sup><sup>**_I_**</sup><sup>_ℓ_</sup> `N` is defined as the set of all input-restricted interventions to layer _ℓ_ . Given a set of neurons or hidden variables in this layer **_ψ_**<sup>_′_</sup> , we are thus dealing with interventions of the form: **h** **_ψ_** _′ ← f_ `N`<sup>**_ψ_**</sup><sup>_′_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`),</sup> where: (i) **_ψ_**<sup>_′_</sup> _⊆_ **_ψ_** _ℓ_ or **_ψ_**<sup>_′_</sup> _⊆_ **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>;(ii)</sup><sup>**x**</sup><sup>_∈X_;and (iii)</sup><sup>**I**</sup><sup>`N`</sup><sup>_∈_</sup><sup>**_I_**:</sup> `N`<sup>_ℓ−_1</sup> . The set of all input-restricted interventions in this layer is thus bounded in size by the Cartesian product: _×_ **_ψ_** _∈_ **_ψ_** _ϕℓ_<sup>_X×_</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ−_1</sup> . These three sets are countable, and thus so is **_I_**<sup>_ℓ_</sup> `N`<sup>.This concludes our proof.</sup> 

**Lemma 4.** _Under Assump. 1 and given a fixed ϕ, the set of input-restricted pre-intervention hidden states in layer ℓ, i.e., H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>_, is countable._</sup> 

27 

_Proof._ The set of input-restricted hidden states _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>is formed by hidden states</sup><sup>**h**</sup><sup>**_ψ_**</sup> _ℓ_<sup>=</sup><sup>_f_</sup><sup>**_ψ_**</sup> `N`<sup>_ℓ_(</sup><sup>**x**</sup><sup>_,_</sup><sup>**I**</sup><sup>`N`),</sup> which we can write as: 



We thus have that the size of _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>is bounded by the size of the Cartesian product</sup><sup>_X×_</sup><sup>**_I_**</sup> `N`<sup>:</sup><sup>_ℓ_.As both</sup> of these sets are countable (by Assump. 1 and Lemma 3, respectively), _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>is also countable.This</sup> completes our proof. 

**Lemma 5.** _Under Assump. 1 and given a fixed ϕ, the set of input-restricted intervention-only hidden variables in layer ℓ, i.e., H_<sup>❖</sup> **_ψ_**<sup>_ϕ_</sup> _ℓ_<sup>_, is countable._</sup> 

_Proof._ A similar proof to Lemma 4 applies here. In short, we have three relevant sets for this proof. First, the set of input-restricted pre-intervention hidden variables: 



Second, we have the set of input-restricted post-intervention hidden variables: 



Both sets above are countable, since _H_ **_ψ_**<sup>◆</sup> _ℓ_<sup>is countable (by Lemma 4), and</sup><sup>**_η_**</sup><sup>_ℓ_is finite.Third, we have</sup> the set of input-restricted intervention-only hidden variables, defined as: 



Since _H_ **_ψ_** _ϕℓ_<sup>is countable,</sup><sup>_H_</sup> **_ψ_**<sup>❖</sup><sup>_ϕ_</sup> _ℓ_<sup>is clearly also countable.This completes the proof.</sup> 

**Lemma 6.** _Under Assump. 3 and given a target output_ **y**<sup>_⋆_</sup> _∈Y, we know that there is an uncountably infinite set H_ **_ψ_**<sup>(</sup><sup>**y**</sup> _ℓ_<sup>_⋆_)</sup> _which predicts it, i.e.,:_ 



_Proof._ Under Assump. 3, we know that—for any target output **y**<sup>_⋆_</sup> _∈Y_ —there is at least one hidden state **h**<sup>_⋆_</sup> _∈_ R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> which predicts it, i.e.: 



where we note that _f_<sup>_ℓ_</sup> `N`<sup>:(</sup><sup>**h**</sup><sup>_⋆_) outputs a probability distribution over</sup><sup>_Y_, i.e.,</sup><sup>_p_</sup><sup>`N`(</sup><sup>**y**</sup><sup>_′|_</sup><sup>**h**</sup><sup>_⋆_).</sup> To show that we have an uncountably infinite set, let us first notice that 



for _m_ 1 be the max value of _f_<sup>_ℓ_</sup> `N`<sup>:(</sup><sup>**h**) and</sup><sup>_m_2the second highest value of</sup><sup>_f ℓ_</sup> `N`<sup>:(</sup><sup>**h**).</sup><sup>_m_1</sup><sup>_> m_2follows by</sup> the strict subjectivity mentioned in _Assump._ 3. Eq. (50) follows by the definition of the euclidean norm ( _||.||_ 2), argmax and _f_<sup>_ℓ_</sup> `N`<sup>:(</sup><sup>**h**</sup><sup>_′_)</sup><sup>_∈_∆</sup><sup>_|_</sup><sup>**_ψ_**</sup><sup>_L|−_1as</sup><sup>_m_1has to be lowered at least</sup><sup>_<u>m</u>_</sup><sup><u>1</u></sup><sup>_−_</sup> 2<sup>_<u>m</u>_</sup><sup><u>2</u></sup> to increase _m_ 2 by<sup>_<u>m</u>_</sup><sup><u>1</u></sup><sup>_−_</sup> 2<sup>_<u>m</u>_</sup><sup><u>2</u></sup> for those two values to be the same. Increasing any other value in _f_<sup>_ℓ_</sup> `N`<sup>:(</sup><sup>**h**)would</sup> require _m_ 1 being lowered more than<sup>_<u>m</u>_</sup><sup><u>1</u></sup><sup>_−_</sup> 2<sup>_<u>m</u>_</sup><sup><u>2</u></sup> or any other value increased by more than<sup>_<u>m</u>_</sup><sup><u>1</u></sup><sup>_−_</sup> 2<sup>_<u>m</u>_</sup><sup><u>2</u></sup> . Now, given continuity of neural networks, we know that: 



28 

Therefore, we see that: 



We notice that _||_ **h** _−_ **h**<sup>_′_</sup> _||_ 2 _< δ_ for _δ >_ 0 denotes a continuous region in R<sup>_|_</sup><sup>**_ψ_**</sup><sup>_ℓ|_</sup> which therefore includes uncountably infinite points. 

## **G Transformers at Initialisation are Almost Surely Injective on each Layer** 

**Theorem 2.** _Transformers like DNN 2 with randomly independent initialised from a continuous distribution (riicd.) weights are almost surely injective at initialisation up to each layer_ 0 _≤ ℓ< L._ 

_Proof._ To show injectivity up to a layer _ℓ_<sup>_′_</sup> in a transformer, it suffices to show that _f_<sup>_ℓ_</sup> `N`<sup>is injective on</sup> any countable subset _H_ of its domain for all layers _ℓ_ (0 _≤ ℓ ≤ ℓ_<sup>_′_</sup> ). This suffices as we assume the set of inputs _X_ is countable, and the composition of injective functions is injective. Let Θ be the random variable representing the transformer’s weights. To show (almost sure) injectivity on layer _ℓ_ for any fixed input set _H_ , we need that (because of Lemma 10):<sup>14</sup> 



Since the transformer operates over sequences of tokens, any element **h** _∈H_ has its first dimension indexing the sequence length. Let _|_ **h** _|_ denote the sequence length and [ **h** ] _t_ refer to the _t_ -th element in **h** . Let _T_ be the set of token positions _T_ = _{_ 1 _, . . . ,_ `min` ( _|_ **h** 1 _|, |_ **h** 2 _|_ ) _}_ . For injectivity, it suffices to show that: 

_p_ Θ( _∀_ **h** 1 _,_ **h** 2 _∈H, t ∈ T,_ [ **h** 1] _t̸_ = [ **h** 2] _t_ : [ _f_ `N`<sup>_ℓ_(</sup><sup>**h**1)]</sup><sup>_t_</sup> _̸_<sup>= [</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**2)]</sup><sup>_t_) = 1</sup> (54) Note that eq. (54) only ensures injectivity when _|_ **h** 1 _|_ = _|_ **h** 2 _|_ . However, this is sufficient because when _|_ **h** 1 _|̸_ = _|_ **h** 2 _|_ , eq. (53) follows trivially: since _|f_ `N`<sup>_ℓ_(</sup><sup>**h**1)</sup><sup>_|̸_=</sup><sup>_|f ℓ_</sup> `N`<sup>(</sup><sup>**h**2)</sup><sup>_|_,weimmediatelyhave</sup> _f_<sup>_ℓ_</sup> `N`<sup>(</sup><sup>**h**1)</sup><sup>_̸_=</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**2).When</sup><sup>_|_</sup><sup>**h**1</sup><sup>_|_=</sup><sup>_|_</sup><sup>**h**2</sup><sup>_|_,wecanshowthateq.(54)implieseq.(53)asfollows:if</sup> **h** 1 _̸_ = **h** 2, then there exists at least one token position _t_<sup>_′_</sup> _∈ T_ where [ **h** 1] _t′̸_ = [ **h** 2] _t′_ . By eq. (54), this implies [ _f_<sup>_ℓ_</sup> `N`<sup>(</sup><sup>**h**1)]</sup><sup>_t′̸_= [</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**2)]</sup><sup>_t′_almost surely, and therefore</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**1)</sup><sup>_̸_=</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**2) almost surely.</sup> 

We observe that a transformer’s input set _X_ consists of all sequences formed from a finite token vocabulary, which is countably infinite. Since transformers are deterministic functions, the input set _H_ encountered at any sublayer is also countably infinite. Therefore, it suffices to prove eq. (54) for any fixed countably infinite input set _H_ . 

We show that Eq. (54) holds for any fixed countably infinite subset _H_ of the layer’s domain. This is established for the embedding layer ( _f_<sup>_ℓ_</sup> `N`<sup>(</sup><sup>**h**) =</sup><sup>**eh**), the MLP layer (</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**) =</sup><sup>**h**+</sup><sup>`mlp`(</sup><sup>`LN`(</sup><sup>**h**))), and the</sup> attention layer ( _f_<sup>_ℓ_</sup> `N`<sup>(</sup><sup>**h**) =</sup><sup>**h**+</sup><sup>`attn`(</sup><sup>`LN`(</sup><sup>**h**))) by Lemma 7, Lemma 8, and Lemma 9, respectively.</sup> 

### The 3 theorems facilitating the proof above are: 

**Lemma 7.** _Lets assume we have an embedding layer randomly independent initialized from a continuous distribution (riicd.) weights and any countably infinite input sets (in embeddings token indexes). We denote the set of random variables over the weights as_ Θ _. We then can show for any fixed countably infinite input set H that this Layer is injective almost surely._ 



_Proof._ See App. G.2. 

**Lemma 8.** _Lets assume we have a sub-block consisting of an MLP with a residual connection and layer norm (i.e.,_ **h** + ( `mlp` ( `LN` ( **h** )) _) with riicd. weights. We can show that, for any fixed countably_ 

> 14We note that _pZ_ ( _∀z ∈Z_ : _z_ ), where _Z_ is a set of events, is the same as _pZ_ ( _∩z∈Z {z}_ ) formally. 

29 

_infinite input set H, this layer is injective almost surely:_ 



_Proof._ See App. G.3. 

**Lemma 9.** _Lets assume we have a sub-block consisting of a self-attention with a residual connection and layer norm (i.e.,_ **h** + `attn` ( `LN` ( **h** )) _) with riicd. weights. We can show that, for any fixed countably infinite input set H, this layer is injective almost surely:_ 



_Proof._ See App. G.4 

### **G.1 Fundamental Lemmas** 

In this section we present some fundamental lemmas used to prove G.2 to G.4. 

**Lemma 10.** _For a layers function f_<sup>_ℓ_</sup> `N`<sup>_to be injective on its input set H, it has to hold that:_</sup> _p_ Θ( _∀_ **h** 1 _,_ **h** 2 _∈H_ : **h** 1 _̸_ = **h** 2 _⇒ f_ `N`<sup>_ℓ_(</sup><sup>**h**1)</sup><sup>_̸_=</sup><sup>_f ℓ_</sup> `N`<sup>(</sup><sup>**h**2)) = 1</sup> (58) 

_This can equivalently be written as:_ 



_Proof._ We can derive eq. (59) from eq. (58): 





**Lemma 11.** _If we have a countable set Z of almost sure events z, we know that their intersection is also almost surely. Formally:_ 



30 

_Proof._ First, observe that 



where _z_<sup>_c_</sup> is the complement of an event _z_ . Since _p_ ( _z_ ) = 1 for all _z ∈Z_ , it follows that _p_ ( _z_<sup>_c_</sup> ) = 0 for all _z ∈Z_ By the countable subadditivity of probability measures: 



Therefore, 



### **G.2 Proof of Lemma 7** 

In this section, we will prove Lemma 7 which states that the embedding layer is almost surely injective on countably infinite inputs. 

**Lemma 7.** _Lets assume we have an embedding layer randomly independent initialized from a continuous distribution (riicd.) weights and any countably infinite input sets (in embeddings token indexes). We denote the set of random variables over the weights as_ Θ _. We then can show for any fixed countably infinite input set H that this Layer is injective almost surely._ 



_Proof._ We can apply Lemma 11 three times (on **h** 1 _,_ **h** 2 and _t_ ) to show that eq. (55) is equivalent to, for any **h** 1 _,_ **h** 2 _∈H_ and _t ∈ T_ for which [ **h** 1] _t̸_ = [ **h** 2] _t_ , it holding that: 



We further note that, by the definition of an embedding block (Submodule 2): 

_p_ Θ([ **eh** 1] _t̸_ = [ **eh** 2] _t_ ) = 1 _⇔ p_ Θ( **e** [ **h** 1] _t̸_ = **e** [ **h** 2] _t_ ) = 1 (66) We can thus apply the law of total probability by defining Θ<sup>_′_</sup> as all the random variables Θ except the one for the first element of **e** [ **h** 1] _t_ , i.e., except [ **e** [ **h** 1] _t_ ]1,<sup>15</sup> and **e**<sup>_′_</sup> [ **h** 1] _t_<sup>as the embedding of [</sup><sup>**h**1]</sup><sup>_t_without</sup> the first element: 





This holds trivially when any embedding dimension other than the first of **e** [ **h** 1] _t_ and **e** [ **h** 2] _t_ differs. When all dimensions except the first are equal, we apply: 



The right-hand side [ **e** [ **h** 2] _t_ ]1 is a constant while the left-hand side [ **e** [ **h** 1] _t_ ]1 is a random variable over a continuous region; this event has measure 0, resulting in probability 0. 

### **G.3 Proof of Lemma 8** 

In this section, we will prove Lemma 8, which will show that the block consisting of an MLP, residual connection and layer norm is almost sure injective on its countably infinite inputs. 

**Lemma 8.** _Lets assume we have a sub-block consisting of an MLP with a residual connection and layer norm (i.e.,_ **h** + ( `mlp` ( `LN` ( **h** )) _) with riicd. weights. We can show that, for any fixed countably_ 

> 15By this we refer to the first element of the embedding vector of [ **h** 1] _t_ . 

31 

_infinite input set H, this layer is injective almost surely:_ 



_Proof._ For notational convenience, let **m** _i_ = `mlp` ( `LN` ( **h** _i_ )). Given Lemma 11, it suffices to prove that for any **h** 1 _,_ **h** 2 _∈H_ and _t ∈ T_ , where [ **h** 1] _t̸_ = [ **h** 2] _t_ , we have: 



Without loss of generality, fix one such **h** 1 _,_ **h** 2 _∈H_ and _t ∈ T_ . We can manipulate this probability distribution as: 



Therefore, it suffices to show that: 



where equality (1) holds since [ **m** 1] _t̸_ = [ **m** 2] _t_ implies there exists some index _i_ such that [ **m**<sup>_′_</sup> 1<sup>]</sup> [ _t,i_ ] _̸_<sup>=</sup> [ **m**<sup>_′_</sup> 2<sup>]</sup> [ _t,i_ ]<sup>..16(2) holds because if the inequality is satisfied for a single component of the vector, it</sup> must also be satisfied for the entire vector. In (3), we define Θ<sup>_′_</sup> as the random variable responsible for the value of [ **W** _L_ ][1 _,i_ ] and _θ_ as a realisation of the random variables Θ _\_ Θ<sup>_′_</sup> . Therefore, to prove 

> 16[ **m** _′_ 1<sup>]</sup> [ _t,i_ ]<sup>represents a two dimensional indexing, referring to the</sup><sup>_i_-th element of the representation of the</sup> _t_ -th token. 

32 

eq. (72), it suffices to show: 



For brevity, we omit repeating the conditions in the following probabilities as they remain unchanged to the previous equation: 



where the last step follows from the condition [ **m**<sup>_′_</sup> 1<sup>]</sup> [ _t,i_ ] _̸_<sup>= [</sup><sup>**m**</sup> 2<sup>_′_]</sup> [ _t,i_ ]<sup>which ensures the denominator is</sup> non-zero. Now, Eq. (75d) holds because the right-hand side is a constant (since its elements are fixed given the conditions of the probability) while the left-hand side is a random variable drawn from a continuous distribution (since the weights are riicd.). Therefore, the probability that this equality holds is zero, as the event has measure zero. 

### **G.4 Proof of Lemma 9** 

In this Section, we prove Lemma 9, which establishes that the self-attention sub-block (consisting of attention, residual connection, and layer normalisation) is almost surely injective on countably infinite inputs. The proof structure parallels that of Lemma 8, so we highlight the key differences and necessary adaptations without repeating the full derivation. 

**Lemma 9.** _Lets assume we have a sub-block consisting of a self-attention with a residual connection and layer norm (i.e.,_ **h** + `attn` ( `LN` ( **h** )) _) with riicd. weights. We can show that, for any fixed countably infinite input set H, this layer is injective almost surely:_ 



_Proof._ We follow a proof strategy analogous to that of Lemma 8 in App. G.3. Following the same steps up to eq. (72), it suffices to show for this lemma that for any **h** 1 _,_ **h** 2 _∈H_ and _t ∈ T_ , where [ **h** 1] _t̸_ = [ **h** 2] _t_ , we have: 



We can write this according to the definition of an attention block (Submodule 1): 



where **h**<sup>_′_</sup> 1<sup>and</sup><sup>**h**</sup><sup>_′_</sup> 2<sup>arethehiddenstatesafterconcatenationintheself-attentionmechanism(see</sup> eq. (12)). The remainder of the proof follows the same approach as the proof of Lemma 8 in App. G.3, starting from eq. (73a). 

33 

||All Pairs|Same Output|Not Same Output|Same Variables|Not Same Variables|
|---|---|---|---|---|---|
|Input|8_._5_e−_2_±_1_._1_e−_2|8_._5_e−_2_±_1_._1_e−_2|1_._7_e−_1_±_1_._9_e−_2|8_._5_e−_2_±_1_._1_e−_2|1_._6_e−_1_±_1_._9_e−_2|
|Layer 1|5_._7_e−_4_±_4_._5_e−_4|5_._7_e−_4_±_4_._5_e−_4|2_._4_e−_2_±_6_._2_e−_3|5_._7_e−_4_±_4_._5_e−_4|1_._1_e−_2_±_4_._3_e−_3|
|Layer 2|4_._5_e−_4_±_3_._8_e−_4|4_._5_e−_4_±_3_._8_e−_4|3_._2_e−_2_±_1_._2_e−_2|4_._5_e−_4_±_3_._8_e−_4|1_._2_e−_2_±_6_._7_e−_3|
|Layer 3|3_._3_e−_4_±_2_._8_e−_4|3_._3_e−_4_±_2_._8_e−_4|8_._9_e−_2_±_4_._3_e−_2|3_._3_e−_4_±_2_._8_e−_4|1_._5_e−_2_±_1_._0_e−_2|



Table 1: An approximation of the minimal Euclidean distance of the trained MLP model in the hierarchical equality task using 1 _,_ 280 _,_ 000 samples. The minimal Euclidean distance is computed between all samples to a randomly selected subset of 10 _,_ 000 samples. We compute it over 10 different random seeds and present the mean and standard deviation (the number after _±_ ). 

## **H MLP Injectivity in Hierarchical Equality Task** 

We see in Fig. 2 that the IIA remains low for the identity of first argument algorithm on a fully trained model even when using a _ϕ_<sup>`nonlin`</sup> alignment map (based on `revnet` ). A reasonable assumption for why would be that the fully trained model does not fulfil some assumption required by our proof of Theorem 1 (any algorithm is an input-restricted distributed abstraction for any model) given in §4. In this section, we present follow-up experiments investigating the reason for this disagreement between our empirical results on the identity of first argument algorithm and the theoretical result of Theorem 1. 

Let us first note that to prove Theorem 1 we rely on an existence proof: showing there exists a function _ϕ_ which satisfies the conditions for a DNN to be abstracted by an algorithm. It says nothing, however, about this function being learnable in practice. Our experiments, however, measure IIA on an unseen test set—which requires _ϕ_<sup>`nonlin`</sup> to not only fit a training set, but generalise to new data. Therefore, following our proof of Theorem 1 we explore the IIA on the train set. However, on the normal training set (with 1 _,_ 280 _,_ 000 samples), we still do not get an IIA over 0.55. On the other hand, if we repeat the experiment with only 1 _,_ 000 training samples, we see _ϕ_<sup>`nonlin`</sup> achieves an IIA of over 0 _._ 99 on the training set. Therefore, it is likely that the `revnet` used when defining _ϕ_<sup>`nonlin`</sup> does not have enough capacity to fit the overly complex function our proof describes. 

To further analyse why the capacity of the used `revnet` is not sufficient, we analyse the injectivity of the evaluated MLP by investigating its hidden representations. We first evaluate 1 _,_ 280 _,_ 000 randomly sampled inputs and their hidden states, checking if they are all unique. In these 1 _,_ 280 _,_ 000 samples (and repeating this experiment with 10 different random seeds), no collisions were found, implying the evaluated MLP is (at least close to) injective. 

We now examine the supposition that the model finds it more difficult to distinguish between hidden states that share the same values for the variables in both equality relations than between those that do not. To this end, we compute the minimal Euclidean distance between hidden states across the entire set of 1 _,_ 280 _,_ 000 samples to a randomly selected subset of 10 _,_ 000 samples. Specifically, we measure the minimal pairwise Euclidean distance among: (i) all sample pairs, (ii) sample pairs sharing the same output, (iii) sample pairs sharing the same values for both equality variables in both equality relations, (iv) sample pairs that do not share the same output and (v) sample pairs that do not share the same values for both equality variables. The results are presented in Table 1. We observe that the minimal Euclidean distances are smaller for pairs sharing the same output or the same equality-variable values compared to pairs that do not. This suggests that, although injectivity is preserved, a RevNet likely will find it more challenging to separate hidden states that share variable values. 

## **I Additional Experiment Details** 

In this section, we present additional details about our hierarchical equality task (in App. I.1) and indirect object identification (in App. I.2) experiments. We also present details and results on the distributive law task (in App. I.3). 

### **I.1 Hierarchical Equality Task** 

**Task 1** ( **from Geiger et al., 2024b** ) **.** _The_ **_hierarchical equality task_** _is defined as follows. Let_ **x** = **x** 1 _◦_ **x** 2 _◦_ **x** 3 _◦_ **x** 4 _be a 16-dimensional vector, where each_ **x** _i ∈_ R<sup>4</sup> _for i ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _}, and ◦ denotes vector concatenation. The input space is X_ = [ _−_ 0 _._ 5 _,_ 0 _._ 5]<sup>16</sup> _, and the output space is_ 

34 

_Y_ = _{_ `false` _,_ `true` _}. The task function is:_ 



_where the equality_ ( **x** _i_ == **x** _j_ ) _holds if and only if_ **x** _i and_ **x** _j are equal as vectors in_ R<sup>4</sup> _._ 

### **I.1.1 Algorithms** 

We define the following three candidate algorithms in detail. 

**Alg 1.** _The_ **_both equality relations alg._** _to solve Task 1 has_ **_η_** `inner` = _{η_ **x** 1== **x** 2 _,η_ **x** 3== **x** 4 _} and:_ 



**Alg 2.** _The_ **_left equality relation alg._** _to solve Task 1 has_ **_η_** `inner` = _{η_ **x** 1== **x** 2 _} and:_ 





### **I.1.2 Training Details** 

For the hierarchical equality task, we use a 3-layer MLP with _|_ **_ψ_** 1 _|_ = _|_ **_ψ_** 2 _|_ = _|_ **_ψ_** 3 _|_ = 16. The model is trained using the Adam optimiser with learning rate 0.001 and cross-entropy loss. We use a batch size of 1024 and train on 1,048,576 samples, with 10,000 samples each for evaluation and testing. Training runs for a maximum of 20 epochs with early stopping after 3 epochs of no improvement. 

For the training progression experiments, we use the same configuration but limit training to 2 epochs. 

When training the alignment maps _ϕ_ , we use a batch size of 6400 and train for up to 50 epochs with early stopping after 5 epochs of no improvement (using a threshold of 0.001 for the required change, compared to 0 for MLP training). We use the Adam optimiser with learning rate 0.001 and cross-entropy loss. To generate the datasets for DAS, for Alg. 1 we intervene with a probability of 1/3 on _η_ **x** 1== **x** 2, 1/3 on _η_ **x** 3== **x** 4, and 1/3 on both variables. The samples for the base and source inputs are generated such that ( **x** 1 == **x** 2) and ( **x** 3 == **x** 4) each hold 50% of the time. For Alg. 2 and Alg. 3 we intervene on _η_ **x** 1== **x** 2 and _η_ **x** 1 for all samples, respectively. For each algorithm, we sample 1 _,_ 280 _,_ 000 interventions for training, 10 _,_ 000 for evaluation, and 10 _,_ 000 for testing. 

### **I.1.3 Additional Results** 

We present results for the three candidate algorithms for the hierarchical equality task, analysing the effect of hidden size _d_ rn and intervention size _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_across all MLP layers.</sup> 

For the both equality relations algorithm, Fig. 8a, 9a and 9b demonstrate that the hidden size experiment aligns with previously reported trends, while also showing how alignment maps _ϕ_ of increasing complexity perform across training epochs, layers, and intervention sizes. 

For the left equality relation algorithm, as shown in Fig. 8b, 9c and 9d, we observe similar patterns: increasing hidden size and intervention size improves performance, and alignment is generally more successful in later layers during early training. 

For the identity of first argument algorithm, Fig. 8c, 9e and 9f reveal that, interestingly, some alignment is achieved—especially in layer 3—during the first half of training, but this effect diminishes in the second half. 

Overall, these results demonstrate that the hidden size experiment is consistent with the findings reported in the main paper. They also show that it is easier, in untrained models, to find an alignment map for later layers, and that transient alignment can occur in specific layers and algorithms during the initial stages of training. 

35 



<!-- Start of picture text -->
Layer 1 Layer 2 Layer 3<br>1 . 0<br>0 . 9<br>0 . 8<br>0 . 7<br>0 . 6 = 1<br>= 2<br>0 . 5 = 8<br>1 2 4 8 16 1 2 4 8 16 1 2 4 8 16<br>d rn d rn d rn<br>(a) both equality relations<br>Layer 1 Layer 2 Layer 3<br>1 . 0<br>0 . 9<br>0 . 8<br>0 . 7<br>0 . 6 == 2 1<br>0 . 5 = 8<br>1 2 4 8 16 1 2 4 8 16 1 2 4 8 16<br>d rn d rn d rn<br>(b) left equality relation<br>Layer 1 Layer 2 Layer 3<br>0 . 520<br>0 . 515<br>0 . 510<br>0 . 505<br>0 . 500<br>= 1<br>0 . 495 = 2<br>= 8<br>0 . 490<br>1 2 4 8 16 1 2 4 8 16 1 2 4 8 16<br>d rn d rn d rn<br>(c) identity of first argument<br>Accuracy<br>Accuracy<br>Accuracy<br><!-- End of picture text -->

Figure 8: Mean IIA over 5 seeds using _ϕ_<sup>`nonlin`</sup> ( _L_ rn = 1) on the trained DNN. Performance improves with larger hidden dimension _d_ rn and intervention size _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_.Each subplot corresponds to one of the</sup> three candidate algorithms for the hierarchical equality task, showing how the model’s representational capacity influences performance. 

### **I.2 Indirect Object Identification Task** 

**Task 2.** _The_ **_Indirect Object Identification_** _(IOI) task involves predicting the indirect object in sentences with a specific structure. Each input_ **x** _∈X consists of a text where a subject (_ `S` _) and an indirect object (_ `IO` _) are introduced, followed by the_ `S` _giving something to the_ `IO` _. For example:_ 

_"Friends Juana and Kristi found a mango at the bar. Kristi gave it to" ⇒ "Juana"_ 

_Here, "Juana" and "Kristi" are introduced, with "Kristi" (_ `S` _) appearing again before giving something to "Juana" (_ `IO` _). The output set Y consists of the first tokens of the two names:_ 



### **I.2.1 Algorithm** 

For this task, we evaluate the **ABAB-ABBA** algorithm. Denoting the two names in the story as A and B, this algorithm determines whether the sentence follows an ABAB pattern (e.g., _"Friends Juana and Kristi found a mango at the bar. Juana gave it to Kristi"_ ) or an ABBA pattern (e.g., _"Friends Juana and Kristi found a mango at the bar. Kristi gave it to Juana"_ ). If the pattern is ABAB (where B is the indirect object `IO` ), the algorithm predicts the first token of B. Conversely, for an ABBA pattern, it predicts the first token of A. In our experiments, we intervene on whether an input follows the ABAB pattern or not. 

36 



<!-- Start of picture text -->
= 1 = 2 = 12 = 1 = 2 = 12<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>DNN DNN<br>0 . 8 linear 0 . 8 linear<br>L rn = 1 , d rn = 2 4 L rn = 1 , d rn = 2 4<br>L rn = 5 , d rn = 2 4 L rn = 5 , d rn = 2 4<br>0 . 6 L rn = 10 , d rn = 2 4 0 . 6 L rn = 10 , d rn = 2 4<br>L rn = 10 , d rn = 2 7 L rn = 10 , d rn = 2 7<br>0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full<br>DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps<br>(a) Max IIA for both equality relations (b) Mean IIA for both equality relations<br>= 1 = 2 = 8 = 1 = 2 = 8<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>DNN DNN<br>0 . 8 linear 0 . 8 linear<br>L rn = 1 , d rn = 2 4 L rn = 1 , d rn = 2 4<br>L rn = 5 , d rn = 2 4 L rn = 5 , d rn = 2 4<br>0 . 6 L rn = 10 , d rn = 2 4 0 . 6 L rn = 10 , d rn = 2 4<br>L rn = 10 , d rn = 2 7 L rn = 10 , d rn = 2 7<br>0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full<br>DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps<br>(c) Max IIA for left equality relation (d) Mean IIA for left equality relation<br>= 1 = 2 = 8 = 1 = 2 = 8<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>DNN DNN<br>0 . 8 linear 0 . 8 linear<br>L rn = 1 , d rn = 2 4 L rn = 1 , d rn = 2 4<br>L rn = 5 , d rn = 2 4 L rn = 5 , d rn = 2 4<br>0 . 6 L rn = 10 , d rn = 2 4 0 . 6 L rn = 10 , d rn = 2 4<br>L rn = 10 , d rn = 2 7 L rn = 10 , d rn = 2 7<br>0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full<br>DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps<br>(e) Max IIA for identity of first argument (f) Mean IIA for identity of first argument<br>Layer3 Accuracy Layer3 Accuracy<br>Layer2 Accuracy Layer2 Accuracy<br>Layer1 Accuracy Layer1 Accuracy<br>Layer3 Accuracy Layer3 Accuracy<br>Layer2 Accuracy Layer2 Accuracy<br>Layer1 Accuracy Layer1 Accuracy<br>Layer3 Accuracy Layer3 Accuracy<br>Layer2 Accuracy Layer2 Accuracy<br>Layer1 Accuracy Layer1 Accuracy<br><!-- End of picture text -->

Figure 9: IIA over 5 seeds for each combination of MLP layer (rows) and intervention size (columns) during training progression for the tested algorithms. 

**Alg 4.** _The ABAB-ABBA algorithm for the IOI task has one inner node_ **_η_** `inner` = _{η_ 1 _} and is defined as follows:_ 



_Here,_ `get` ___ `name` ___ `a` ( **x** ) _extracts the first name (denoted A, e.g. Juana in our example) and_ `get` ___ `name` ___ `b` ( **x** ) _extracts the second name (denoted B, e.g. Kristi in our example) from the input sentence_ **x** _. The function_ `check` ___ `is` ___ `abab` ___ `pattern` ( **x** ) _returns_ `true` _if the sentence follows an “ABAB” structure (e.g., “A and B ... A gave to B”, meaning B is the_ `IO` _) and_ `false` _if it follows_ 

37 

_an “ABBA” structure (e.g., “A and B ... B gave to A”, meaning A is the_ `IO` _)._ `first` ___ `token` ( _Name_ ) _returns the first token of the specified name. The output_ **y** _is the first token of the indirect object._ 

### **I.2.2 Training Details** 



<!-- Start of picture text -->
\<br>wo} |<br>8!<br>i<br>¥<br>2 6<br>é<br>1<br>2<br>“>0000800010000‘Training Steps<br><!-- End of picture text -->

We use models from the Pythia suite (Biderman et al., 2023) to evaluate the IIA performance of the different _ϕ_ on the IOI task. Specifically, we employ the Pythia 31M, 70M, 160M, and 410M parameter models. We also examine different training checkpoints provided by these models to analyse how IIA evolves during training. To assess robustness, we replicate a subset of experiments using alternative Pythia model seeds from (van der Wal et al., 2025). 

We train all alignment maps on 2 epochs of 10<sup>6</sup> interventions based on data from Muhia (2022), Figure 10: Cross Entropy loss ( _y_ -axis) during trainwith a batch size of 256 and a learning rate of ing ( _x_ -axis) of the alignment map for the randomly 10<sup>_−_4</sup> . For all experiments, we set the interveninitialised _Pythia-31m_ . The loss plateaus between tion size _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_to half of the DNN’s hidden di-</sup> 4k and 6k steps, and suddenly drops after 6k steps. mension. For smaller models (31M, 70M), we train using float64 precision and a learning rate of 10<sup>_−_3</sup> , as these adjustments proved crucial for convergence. We also note that we observed quite severe grokking behaviour, where models had low IIA for a long time, which quickly jumped to high IIA values at a certain point of training (see Fig. 10; wandb run). 

Figure 10: Cross Entropy loss ( _y_ -axis) during training ( _x_ -axis) of the alignment map for the randomly initialised _Pythia-31m_ . The loss plateaus between 4k and 6k steps, and suddenly drops after 6k steps. 

### **I.2.3 Additional Results** 

**Robustness across random seeds.** In Fig. 11, we examine how our main results from §6 generalise across multiple training seeds of the Pythia model. The key trends hold consistently across all 5 seeds - we can find perfect alignments using _ϕ_<sup>`nonlin`</sup> in most cases. However, we observe two notable exceptions. For one seed, the DNN fails to learn the IOI task even after full training. For another seed, we cannot find an alignment using even complex alignment maps under our current setup.<sup>17</sup> All other seeds achieve perfect alignment under _ϕ_<sup>`nonlin`</sup> . We hypothesise that the alignment failure case is primarily due to suboptimal training of the alignment map. Due to computational constraints, we did not perform extensive hyperparameter tuning that might have achieved convergence. 



<!-- Start of picture text -->
1 . 0<br>0 . 8<br>0 . 6<br>0 . 4<br>DNN<br>0 . 2 non-linear<br>linear<br>0 . 0<br>oe Init. Full<br>DNN Training Step<br>Accuracy<br><!-- End of picture text -->

Figure 11: IIA of alignment between the ABAB-ABBA algorithm and the _Pythia-410m_ model across multiple seeds (seeds 1 to 5 from van der Wal et al., 2025), with interventions at layer 12. We evaluate the IIA of both _ϕ_<sup>`lin`</sup> and _ϕ_<sup>`nonlin`</sup> (with _d_ `h` = 64 _, K_ = 1) on randomly initialised (Init.) and fully trained (Full) DNNs. 

**Generalisation across distinct name sets.** In the main paper, we split the dataset from Muhia (2022) by ensuring that no two sentences appear in both the training and evaluation sets. However, this splitting strategy does not guarantee that the names themselves are distinct between training and evaluation sets. In Fig. 12, we examine the results when using completely different sets of names for training and evaluation. The results differ substantially: we cannot find an alignment using even complex alignment maps for the randomly initialised DNN. This suggests that IIA on the randomly initialised DNN may depend critically on overlap between the specific entities encountered during 

> 17These failures occur in different seeds: seed 3 shows poor IIA despite learning the task, while seed 4 fails to learn the IOI task. 

38 

training and evaluation. For the fully trained DNN, we observe perfect alignment using _ϕ_<sup>`nonlin`</sup> and reasonably high alignment using _ϕ_<sup>`lin`</sup> . 



<!-- Start of picture text -->
1 . 00<br>0 . 75<br>DNN<br>linear<br>L rn = 8<br>0 . 50 L rn = 16<br>0 Full<br>DNN Training Step<br>Accuracy<br><!-- End of picture text -->

Figure 12: IIA of alignment between the ABAB-ABBA algorithm and the _Pythia-410m_ model using a different set of names for training and evaluation, with interventions at layer 12. We evaluate the IIA of both _ϕ_<sup>`lin`</sup> and _ϕ_<sup>`nonlin`</sup> (with _d_ `h` = 64 _, K_ = 1) on randomly initialised and fully trained (Full) DNNs. 

### **I.3 Distributive Law Task** 

We now study a similar task to the hierarchical equality (in App. I.1), based on the distributive law of and ( _∧_ ) and or ( _∨_ ). 

**Task 3.** _The_ **_distributive law task_** _is defined as follows. Let_ **x** = **x** 1 _◦_ **x** 2 _◦_ **x** 3 _◦_ **x** 4 _◦_ **x** 5 _◦_ **x** 6 _be a 24-dimensional vector, where each_ **x** _i ∈_ [ _−_ 0 _._ 5 _,_ 0 _._ 5]<sup>4</sup> _for i ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 5 _,_ 6 _}, and ◦ denotes vector concatenation. The input space is X_ = [ _−_ 0 _._ 5 _,_ 0 _._ 5]<sup>24</sup> _, and the output space is Y_ = _{_ `false` _,_ `true` _}. The task function is_ 



_where the equality_ ( **x** _i_ == **x** _j_ ) _holds if and only if_ **x** _i and_ **x** _j are equal as vectors in_ R<sup>4</sup> _._ 

### **I.3.1 Algorithms** 

We define the following two candidate algorithms. 

**Alg 5.** _The_ **_And-Or-And alg._** _to solve Task 3 has_ 



_and it is defined as follows:_ 

**Alg 6.** _The_ **_And-Or alg._** _to solve Task 3 has_ 



_and it is defined as follows:_ 



### **I.3.2 Training Details** 

For the distributive law task, we use a 3-layer MLP (see App. E.1) with an input dimensionality of 24, hidden layers of dimensionality _|_ **_ψ_** 1 _|_ = _|_ **_ψ_** 2 _|_ = _|_ **_ψ_** 3 _|_ = 24, and an output dimensionality of 2. The model is trained using the Adam optimiser with a learning rate of 0.001 and cross-entropy loss. We use a batch size of 1024. The datasets are generated by randomly sampling input vectors **x** = **x** 1 _◦· · · ◦_ **x** 6 such that the target label _y_ ¯ = `T` ( **x** ) is true 50% of the time. We sample 1 _,_ 048 _,_ 576 samples for training, 10 _,_ 000 for evaluation, and 10 _,_ 000 for testing. Training runs for a maximum of 20 epochs with early stopping after 3 epochs of no improvement. 

For training _ϕ_ , we use a batch size of 6400 and train for up to 50 epochs with early stopping after 5 epochs of no improvement (using a threshold of 0.001 for the required change). We use the Adam 

39 

optimiser with learning rate 0.001 and cross-entropy loss. To generate the intervened datasets: For Alg. 5, we intervene with a probability of 1/3 on _η_ ( **x** 1== **x** 2) _∧_ ( **x** 3== **x** 4), 1/3 on _η_ ( **x** 3== **x** 4) _∧_ ( **x** 5== **x** 6), and 1/3 on both variables. For Alg. 6, we intervene with a probability of 1/3 on _η_ **x** 3== **x** 4, 1/3 on _η_ ( **x** 1== **x** 2) _∨_ ( **x** 5== **x** 6), and 1/3 on both variables. For both algorithms, the samples for the base and source inputs are generated such that the output of the intervention changes compared to the base input 50% of the time. We sample 1 _,_ 280 _,_ 000 interventions for training, 10 _,_ 000 for evaluation , and 10 _,_ 000 for testing for each algorithm. 

### **I.3.3 Results** 

In this section, we discuss the results on the distributed law task using the And-or-And and And-Or algorithms. Our findings corroborate the results presented in the main paper. As shown in Fig. 13, using linear and identity alignment maps reveals distinct dynamics. The And-Or algorithm achieves higher IIA using _ϕ_<sup>`lin`</sup> , particularly in later layers where the IIA of _ϕ_<sup>`lin`</sup> on the And-Or-And algorithm ~~approac~~ hes 0.5. However, these dynamics completely vanish when using a more complex alignment ~~<u>map lik</u>~~ e _ϕ_<sup>`nonlin`</sup> , where we achieve almost perfect IIA everywhere. 



<!-- Start of picture text -->
AndOr AndOrAnd<br>1.00<br>0.75<br>0.50<br>0.25 linear identity<br>non-linear<br>1 2 12 1 2 12 1 2 12 1 2 12 1 2 12 1 2 12<br>L1 L2 L3 L1 L2 L3<br>Accuracy<br><!-- End of picture text -->

Figure 13: IIA in the distributive law task for causal abstractions trained with different alignment maps _ϕ_ . The figure shows results for both analysed algorithms for this task. The bars represent the max IIA across 10 runs with different random seeds. The black lines represent mean IIA with 95% confidence intervals. The _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_denotes the intervention size per node.Without interventions,all</sup> DNNs reach 100% accuracy. The used _ϕ_<sup>`nonlin`</sup> uses _L_ rn = 10 and _d_ rn = 24. 

Fig. 14a and 14c present the evaluated IIA throughout model training. These training progression plots show that randomly initialised models often achieve IIA above 0.8 with non-linear alignment maps, supporting our insight that when the notion of causal abstraction is equipped with _ϕ_<sup>`nonlin`</sup> it may identify algorithms which are not necessarily implemented by the underlying model. In Fig. 14b and 14d, we plot the mean IIA over 5 seeds instead of the maximum IIA. 

The hidden size experiments (Fig. 15a and 15b) show that even RevNets with small _d_ `h` of 4 achieve near-perfect IIA for And-Or-And, while And-Or never reaches perfect IIA in the second layer, regardless of the _d_ `h` . The training progression plots suggest a possible explanation: IIA for And-OrAnd initially increases in the last two layers but then decreases, while RevNets maintain near-perfect IIA. This may indicate that And-Or-And is first implemented with simple encodings detectable by linear _ϕ_ s, before evolving into non-linear encodings that only RevNets can detect. The fact that And-Or never achieves high IIA in later layers further suggests it may not be a true abstraction of the model’s behaviour, though we note this remains a hypothesis requiring further investigation. 

**And-Or-And Training.** In this section, we analyse a DNN when this model is trained specifically to rely on the And-Or-And algorithm (and, consequently, to encode the values of its hidden nodes). We do so with the method from Geiger et al. (2022), training the DNN to encode And-Or-And’s hidden nodes’ values in its second layer, with an intervention size of 12. This method is similar to how we train _ϕ_ (see App. I.3.2), but _ϕ_ is fixed to the identity function, and the DNN itself is trained; further, the training dataset is composed of 1/4 non-intervened samples, 1/4 samples with interventions on _η_ ( **x** 1== **x** 2) _∧_ ( **x** 3== **x** 4), 1/4 on _η_ ( **x** 3== **x** 4) _∧_ ( **x** 5== **x** 6), and 1/4 on both variables. We then evaluate if this DNN abstracts both the And-Or-And and And-Or using different _ϕ_ (as before, after freezing the DNN). The IIA performance of these _ϕ_ is presented in Fig. 16. We can see here that, when using identity and linear alignment maps _ϕ_ , IIA scores suggest that the And-Or-And algorithm seems to be implemented perfectly given the second layer, where we have only around 0.75 IIA for 

40 



<!-- Start of picture text -->
= 1 = 2 = 12 = 1 = 2 = 12<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 DNNlinear 0 . 8 DNNlinear<br>L rn = 1 , d rn = 24 L rn = 1 , d rn = 24<br>L rn = 5 , d rn = 24 L rn = 5 , d rn = 24<br>0 . 6 L rn = 10 , d rn = 24 0 . 6 L rn = 10 , d rn = 24<br>L rn = 10 , d rn = 2 7 L rn = 10 , d rn = 2 7<br>0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full<br>DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps<br>(a) Max IIA for And-Or (b) Max IIA for And-Or<br>= 1 = 2 = 12 = 1 = 2 = 12<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 0 . 8<br>0 . 6 0 . 6<br>1 . 0 1 . 0<br>0 . 8 DNNlinear 0 . 8 DNNlinear<br>L rn = 1 , d rn = 24 L rn = 1 , d rn = 24<br>L rn = 5 , d rn = 24 L rn = 5 , d rn = 24<br>0 . 6 L rn = 10 , d rn = 24 0 . 6 L rn = 10 , d rn = 24<br>L rn = 10 , d rn = 2 7 L rn = 10 , d rn = 2 7<br>0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full 0 1M Full<br>DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps DNN Training Steps<br>(c) Max IIA for And-Or-And (d) Max IIA for And-Or-And<br>Layer3 Accuracy Layer3 Accuracy<br>Layer2 Accuracy Layer2 Accuracy<br>Layer1 Accuracy Layer1 Accuracy<br>Layer3 Accuracy Layer3 Accuracy<br>Layer2 Accuracy Layer2 Accuracy<br>Layer1 Accuracy Layer1 Accuracy<br><!-- End of picture text -->

Figure 14: IIA over 5 seeds for each combination of MLP layer (rows) and intervention size (columns) during training progression for the evaluated algorithms. 

the And-Or algorithm. However, these differences vanish almost completely using _ϕ_<sup>`nonlin`</sup> as our alignment map. 

**And-Or Training.** In this section, we report an experiment similar to the above, but we train our DNN to rely on the And-Or algorithm instead. These results are shown in Fig. 17. In this figure, we again see that, when using identity and linear as alignment map _ϕ_ , IIA performance suggests that the And-Or algorithm seems to be implemented perfectly given the second layer, where we have only around 0.65 IIA for the And-Or-And algorithm. These differences however vanish when using _ϕ_<sup>`nonlin`</sup> as alignment map, which leads to perfect IIA scores with either algorithm. 

## **J Computational Resources** 

The experiments on MLP were executed on CPU (10 computers with i7-4770 or newer) over 3 weeks, as we noticed that DAS on small MLPs are faster on CPU than on GPU. The experiments on the Pythia models were executed on a single A100 GPU with 80GB of memory using approximately 30 GPU hours, including the hyperparameter tuning. 

41 



<!-- Start of picture text -->
Layer 1 Layer 2 Layer 3<br>0 . 95<br>0 . 90<br>0 . 85<br>0 . 80<br>0 . 75<br>0 . 70<br>= 1<br>0 . 65 = 2<br>0 . 60 = 8<br>1 2 4 8 16 1 2 4 8 16 1 2 4 8 16<br>d rn d rn d rn<br>(a) And-Or<br>Layer 1 Layer 2 Layer 3<br>1 . 0<br>0 . 9<br>0 . 8<br>0 . 7<br>= 1<br>0 . 6 = 2<br>= 8<br>0 . 5<br>1 2 4 8 16 1 2 4 8 16 1 2 4 8 16<br>d rn d rn d rn<br>(b) And-Or-And<br>Accuracy<br>Accuracy<br><!-- End of picture text -->

Figure 15: Mean IIA over 5 seeds using _ϕ_<sup>`nonlin`</sup> ( _L_ rn = 1) on the trained DNN. Performance improves with larger hidden dimension _d_ rn and intervention size _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_.Each subplot corresponds to one of</sup> the t ~~wo can~~ didate algorithms for the distributed law task, showing how _ϕ_ ’s representational capacity influ ~~<u>ences p</u>~~ erformance. 



<!-- Start of picture text -->
AndOr AndOrAn d<br>1.00<br>0.75<br>0.50<br>0.25 linear identity<br>non-linear<br>1 2 12 1 2 12 1 2 12 1 2 12 1 2 12 1 2 12<br>L1 L2 L3 L1 L2 L3<br>Accuracy<br><!-- End of picture text -->

Figure 16: IIA in the distributive law task for causal abstractions trained with different alignment maps _ϕ_ and a DNN trained to use the And-Or-And algorithm. The figure shows results when evaluating if the DNN encodes either of the analysed algorithms for this task. The bars represent the max IIA across 10 runs with different random seeds. The black lines represent mean IIA with 95% confidence intervals. The _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_denotes the intervention size per node.All DNNs reach >99.9%</sup> accu ~~<u>racy af</u>~~ ter training. The used _ϕ_<sup>`nonlin`</sup> uses _L_ rn = 10 and _d_ rn = 16. 



<!-- Start of picture text -->
AndOr AndOrAnd<br>1.00<br>0.75<br>0.50<br>0.25 linear identity<br>non-linear<br>1 2 12 1 2 12 1 2 12 1 2 12 1 2 12 1 2 12<br>L1 L2 L3 L1 L2 L3<br>Accuracy<br><!-- End of picture text -->

Figure 17: IIA in the distributive law task for causal abstractions trained with different alignment maps _ϕ_ and a DNN trained to use the And-Or algorithm. The figure shows IIA results when evaluating if the DNN encodes either of the analysed algorithms for this task. The bars represent the max IIA across 10 runs with different random seeds. The black lines represent mean IIA with 95% confidence intervals. The _|_ **_ψ_**<sup>_ϕ_</sup> _η_<sup>_|_denotes the intervention size per node.Without interventions, all DNNs reach</sup> >99.9% accuracy. The used _ϕ_<sup>`nonlin`</sup> uses _L_ rn = 10 and _d_ rn = 16. 

42 

