# Monotonic Representation of Numeric Properties in Language Models

**arXiv:** 2403.10381 · **Source:** https://arxiv.org/abs/2403.10381
**Route:** markdown copied from `~/Papers/Symbolic_Embeddings/corpus/markdown/literature/2403.10381.md`. That library does not record which PDF converter produced it. Single-column source, so no column interleaving. Equations and numbers resolve against the PDF.
**Authors:** Benjamin Heinzerling, Kentaro Inui. Preprint.
**Corpus section:** Tyler Wheeler/interpretability
**Cited by:** NOT CITED BY WHEELER. Added for the audit phase of thread 01 (research/thread-01-amortized-inference/statement.md).
**Why it is here:** Numeric properties encoded monotonically in low-dimensional, editable subspaces. The simplest candidate form for how an embedding represents a particle energy.
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. Method literature, almost all on language models.
**Copied:** 2026-09-25

---

Preprint 

# **Monotonic Representation of Numeric Properties in Language Models** 

### **Benjamin Heinzerling** 

RIKEN / Tohoku University benjamin.heinzerling@riken.jp 

### **Kentaro Inui** 

MBZUAI / Tohoku University / RIKEN kentaro.inui@mbzuai.ac.ae 

## **Abstract** 

Language models (LMs) can express factual knowledge involving numeric properties such as _Karl Popper was born in 1902_ . However, how this information is encoded in the model’s internal representations is not understood well. Here, we introduce a simple method for finding and editing representations of numeric properties such as an entity’s birth year. Empirically, we find low-dimensional subspaces that encode numeric properties monotonically, in an interpretable and editable fashion. When editing representations along directions in these subspaces, LM output changes accordingly. For example, by patching activations along a ”birthyear” direction we can make the LM express an increasingly late birthyear: _Karl Popper was born in 1929_ , _Karl Popper was born in 1957_ , _Karl Popper was born in 1968_ . Propertyencoding directions exist across several numeric properties in all models under consideration, suggesting the possibility that monotonic representation of numeric properties consistently emerges during LM pretraining. Code: https://github.com/bheinzerling/numeric-property-repr 

## **1 Introduction: Do LMs represent numeric properties “appropriately”?** 

Language models (LMs) are capable of expressing factual knowledge (Petroni et al., 2019; Jiang et al., 2020; Roberts et al., 2020; Heinzerling & Inui, 2021; Kassner et al., 2021). For example, when queried _In which year was Karl Popper born?_ Llama 2 (Touvron et al., 2023) gives the correct answer _1902_ . While the question if and to what degree LMs can be said to “know” anything at all is subject of ongoing debate (Bender & Koller, 2020; Hase et al., 2023b; Mollo & Milliere` , 2023; Lederman & Mahowald, 2024), empirical work has progressed from behavioral analysis focused on the accuracy and robustness of knowledge expression (Shin et al., 2020; Jiang et al., 2021; Zhong et al., 2021; Youssef et al., 2023) to representational analysis aimed at understanding how factual knowledge is encoded<sup>1</sup> in model parameters (De Cao et al., 2021; Mitchell et al., 2021; Meng et al., 2022) and activations (Hernandez et al., 2023; Merullo et al., 2023; Geva et al., 2023; Gurnee & Tegmark, 2023). 

However, representational analysis has so far mainly targeted entity-entity relations such as _Warsaw is the capital of Poland_ (Merullo et al., 2023). If and how LM representations encode factual knowledge involving numeric properties, such as an entity’s birthyear, is less understood. Unlike entity-entity relational knowledge, numeric properties have natural ordering and monotonic structure, such as earlier/later and smaller/larger scales or geographic coordinate systems. While this kind of structure is natural and intuitive for humans, LMs encounter numeric properties only in form of largely unordered and unstructured textual mentions. This raises the question if LMs learn to represent numeric properties appropriately, i.e., according to their natural structure. 

> 1We say “X is encoded in Y” as shorthand for “X can be easily extracted from Y”. See caveats in §5. 

1 

Preprint 



<!-- Start of picture text -->
UM with “laterbictnyens” patch 1975t<br>“Later birthyear/ ion" (933 sop goo a0 683.660 BoM oon oon |<br>; 1202 on+ /7 > om | @xpenn eo eno ao EHD oDw EDO OOD |<br>(oa| ana esac ena aaaa) f na (SSRI |<br>H 650 00 5D GO SdH OO K00, 53 oe<br>H ph a hh8 eeWrepresentattion”Popper “Bicthyear direction" LM with “enclier bicthyenr” pateh 181<br>eee (Seana om aes apo eo mmm con oa)<br>1 ata yar Ha Paper tn A (cacao 0<br>ny ee | Suo'G50 ong.me0 mee eos Gan ouS GED |<br>“earlier bicthyear stirection”<br><!-- End of picture text -->

Figure 1: Sketch of our main finding. Patching entity representations along specific directions in activation space yields corresponding changes in model output. 

Here, we devise a simple method for identifying and manipulating representations of numeric properties in LMs. We find low-dimensional subspaces that strongly correlate with numeric properties across models and numeric properties, thereby confirming and extending prior observations of representations of numeric properties in LMs (Lietard et al.´ , 2021; Faisal & Anastasopoulos, 2023; Gurnee & Tegmark, 2023; Godey et al., 2024). Going beyond prior work (see §4), we show that by causally intervening along certain directions in these subspaces, LM output changes correspondingly. That is, we find a monotonic relationship between the intervention and the quantity expressed by the LM. For example, an entitity’s year of birth shifts according to the strength and sign of the intervention along a “birthyear” direction (Figure 1). Taken together, our findings suggest that LMs represent numeric properties in a way that reflects their natural structure and that such monotonic representations consistently emerge during LM pretraining. 

**Terminology.** Before moving to the main part, we briefly clarify important terms. A **quantity** consists of a scalar numeric **value** paired with a **unit** of measurement. A **numeric property** is a property that can naturally be described by a quantity, e.g., birthyear, population size, geographic latitude. A **numeric attribute** is an instance of a numeric property, associated with a particular entity. For example, Karl Popper has the numeric attribute birthyear:1902. By **linear representation** we denote the idea that a numeric attribute is encoded in a linear subspace of a LM’s activation space. Finally, a **monotonic representation** is a linear representation characterized by a monotonic relationship between directions in activation space and the value of the encoded numeric attribute. That is, as activations shift along a particular direction the value of the corresponding numeric attribute increases or decreases monotonically. 

## **2 Finding Property-Encoding Directions** 

### **2.1 Motivation: Representation learning, linear representation hypothesis** 

While numeric properties generally can be mapped naturally onto simple canonical structures, such as number lines or coordinate systems, it is not immediately obvious that pretraining on largely unstructured data enables LMs to appropriately represent such structures. Our main goal is to find out if and how numeric properties, such as an entity’s birthyear, are encoded in the geometry of LM representations. How could such an encoding look like? Based on two arguments, we hypothesize that numeric properties are encoded in low-dimensional linear subspaces of activation space. 

The first argument rests on a key principle in representation learning: a model generalizes if and only if its representations reflect the structure of the data (Conant & Ashby, 1970; Liu et al., 2022). To the degree that current LMs generalize, in the sense of achieving non-trivial 

2 

Preprint 

|Property|Prop. ID|Entity|EntityID|Prompt|Value|Unit|
|---|---|---|---|---|---|---|
|birthyear|P569|Nina Foch|Q235632|In what year was Nina Foch born?|1924|annum|
|death year|P570|Johannes R. Becher|Q58057|In what year did Johannes R. Becher die?|1958|annum|
|population|P1082|Akhisar|Q209905|What is the population of Akhisar?|173026|1|
|evelation|P2044|Sondrio|Q6274|How high is Sondrio?|360|metre|
|longitude|P625.long|Korean Empire|Q28233|What is the longitude of Korean Empire?|126.98|degree|
|latitude|P625.lat|K¨usnacht|Q69216|What is the latitude of K¨usnacht?|47.32|degree|



Table 1: Random sample of the entities used in our experiments, along with corresponding numeric attributes and prompts. See details and more samples in Appendix A. 

performance on benchmarks involving knowledge of numeric properties (Petroni et al., 2019), we can expect that their representations reflect the structure of numeric properties. And since the natural structures of many numeric properties are low-dimensional linear spaces such as number lines, we expect to find low-dimensional linear structure in the representations of a well-performing model. 

As second argument we adduce the linear representation hypothesis, which posits a correspondence between concepts and linear subspaces (Elhage et al., 2022; Park et al., 2023; Nanda et al., 2023). If the linear representation hypothesis is true,<sup>2</sup> this would imply that numeric properties are encoded in linear subspaces. For brevity, we will call a low-dimensional linear subspace of a LM’s activation space a _direction_ , regardless of whether it is one- or multi-dimensional. 

### **2.2 Method: Partial least squares regression on entity representations** 

Motivated by the hypothesis that numeric properties are encoded as directions in activation space, we now devise an experimental setup for finding out if such directions exist. An obvious choice for identifying linear structure would be principal component analysis (PCA; Pearson, 1901). However, PCA looks for directions of maximum variance and is unsupervised in the sense that we cannot cannot provide model outputs to steer the algorithm towards particular directions. What we would like to do instead is to provide supervision in order to find directions in activation space that maximally covary with model outputs. A standard method applicable to this kind of problem is partial least squares regression. 

Partial least squares regression (PLS; Wold et al., 2001) is a supervised method for finding relationships between two data matrices _X_ and _Y_ . Briefly summarizing the detailed exposition by Wold et al. (2001), the objective of PLS regression is to find a rank- _k_ decomposition of the data matrix _X_ into “ _X_ -scores” _T_ = _XW ∈_ **R**<sup>_n×k_</sup> and “ _X_ -loadings” _P ∈_ **R**<sup>_d×k_</sup> so that _X ≈ TP_<sup>T</sup> = _XWP_<sup>T</sup> , with weights _W ∈_ **R**<sup>_d×k_</sup> . The decomposition is subject to the constraint that the _X_ -scores (or conversely: the corresponding _k_ components) can be used to predict _Y_ , that is _Y ≈ XWC_<sup>T</sup> , with “ _Y_ -loadings” _C ∈_ **R**<sup>_k_</sup> . The _Y_ -loadings can be seen as coefficients that quantify how much each of the _k_ components contributes to the prediction of the regression model. There exist efficient methods for solving this optimization problem.<sup>3</sup> 

In our case _X_ corresponds to (a sample of) the activation space of a LM and _Y_ to corresponding LM outputs. Concretely, for a given numeric property, such as birthyear, we collect _n_ entities that have this property. For each entity _e_ we prepare a suitable prompt and encode this prompt with a LM to obtain an entity representation _xe_ of dimension _d_ . That is, _X_ = [ _x_ 1 _· · · en_ ]<sup>T</sup> _∈_ **R**<sup>_n×d_</sup> . Put simply, we encode prompts such as _When was Karl Popper born?_ and take the hidden state of a particular token at a particular layer as the LM’s entity representation. We also collect the quantity _ye_ expressed by the LM when queried for the numeric attribute of entity _e_ , that is, _Y_ = _{ye | e ∈_ [1 . . _n_ ] _} ⊆_ **R**<sup>_n_</sup> . In the case of our running example query for Karl Popper’s birthyear, Llama 2 expresses the quantity 

> 2For positive evidence, see Marks & Tegmark (2023); Merullo et al. (2023); Tigges et al. (2023); Jiang et al. (2024), i.a. 

> 3We use the Scikit-learn (Pedregosa et al., 2011) implementation of NIPALS (Wold, 1966). 

3 

Preprint 



<!-- Start of picture text -->
1.0 1.00 1.00<br>0.8 0.75 0.75<br>0.6 0.50 0.50<br>Method Method Method<br>0.4 PLS 0.25 PLS 0.25 PLS<br>0.2 PCA 0.00 PCA 0.00 PCA<br>0.0 PLS (shuffled labels) PLS (shuffled labels) PLS (shuffled labels)<br>PCA (shuffled labels) 0.25 PCA (shuffled labels) 0.25 PCA (shuffled labels)<br>0.2 PLS (random reprs.) PLS (random reprs.) PLS (random reprs.)<br>0.4 PCA (random reprs.) 0.50 PCA (random reprs.) 0.50 PCA (random reprs.)<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(a) Birthyear (b) Death year (c) Population<br>1.0 1.00 1.00<br>0.75 0.75<br>0.5 0.50 0.50<br>Method Method Method<br>0.0 PLS 0.25 PLS 0.25 PLS<br>PCA 0.00 PCA 0.00 PCA<br>PLS (shuffled labels) PLS (shuffled labels) PLS (shuffled labels)<br>0.5 PCA (shuffled labels) 0.25 PCA (shuffled labels) 0.25 PCA (shuffled labels)<br>PLS (random reprs.)PCA (random reprs.) 0.50 PLS (random reprs.)PCA (random reprs.) 0.50 PLS (random reprs.)PCA (random reprs.)<br>1.0 0.75<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(d) Elevation (e) Latitude (f) Longitude<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br><!-- End of picture text -->

Figure 2: Low-dimensional subspaces of Llama-2-13B’s 5120-dimensional activation space are predictive of the quantity expressed by the LM when queried for a numeric attribute of an entity, across six different numeric properties. Each subfigure shows the performance of a regression model fitted to predict the expressed quantities from LM-internal entity representations (in layer _l_ = 0.3), as a function of the number of PCA/PLS components used for prediction. Unlike regression on PCA components (dashed orange), partial least squares regression (PLS, solid blue) identifies a small set of predictive components. Controls with shuffled labels (dotted green, dash-dotted red) and random entity representations (long-dash-dot purple, dash-dot-dot brown) fail to find predictive subspaces. 

_ye_ = 1902. Having collected entity representations _X_ and associated LM outputs _Y_ , we fit a _k_ -component PLS regression model to predict _Y_ from a _k_ -dimensional subspace of _X_ . 

To measure how well numeric attributes can be predicted from low-dimensional subspaces of activation space, we vary the number of PLS regression components _k_ , i.e., the dimensionality of the subspace, and record goodness of fit using the coefficient of determination _R_<sup>2</sup> = 1 _−_ ∑<sup>_n_</sup> _e_ =1<sup>(</sup><sup>_ye −y_ˆ</sup><sup>_e_)2/∑</sup><sup>_n_</sup> _e_ =1<sup>(</sup><sup>_ye −y_¯)2, where</sup><sup>_y_ˆ</sup><sup>_e_is the prediction of the regression model</sup> and _y_ ¯ the sample-mean value of the numeric property in question.<sup>4</sup> 

### **2.3 Results: Low-dimensional subspaces predictive of numeric attribute expression** 

After selecting six of the most frequent numeric properties<sup>5</sup> in Wikidata (Vrandeciˇ c´ & Krotzsch¨ , 2014), for each property we randomly sample _n_ = 1000 popular<sup>6</sup> entities and prompt the LM (in English) for the corresponding attribute. Samples of entities and prompts are shown in Table 1 and Appendix A. To obtain entity representations we take the _d_ - dimensional hidden state of the entity mention’s last subword token at layer _l_ to obtain a sample of the activation space _X_ , choosing _l_ on a development set as described in Appendix E. We record the _n_ quantities parsed from the LM output to obtain _Y_ and fit PLS models with varying numbers of components to find property-encoding directions. 

> 4 _R_ 2 = 1 for a perfect model, _R_ 2 = 0 for the mean (constant) model, and _R_ 2 _<_ 0 for models that perform worse than the mean model. 

> 5These include longitude, which is non-monotonic due to the wrap-around at _±_ 180 _◦_ . In defense of this choice we note that longitude is locally monotonic almost everywhere. 

> 6We define popular entities as those in the top decile of the rank mean of Wikidata node degree and Wikipedia article length. 

4 

Preprint 



<!-- Start of picture text -->
2468 2375 4915489280<br>20<br>10 1962 2003 20 867305<br>10<br>0 1937 1974 161694<br>0 0<br>10 1905 10 1932 53931<br>20<br>20 1832 20 1826 19135<br>40<br>-3760 30 -2566 0<br>20 0 20 20 0 20 20 0 20<br>PLS component 1 PLS component 1 PLS component 1<br>(a) Birthyear (b) Death year (c) Population<br>8849 90.00 40 178.43<br>20 20<br>30<br>493 51.49 30.39<br>10 10<br>20<br>0 242 0 46.24 10 9.92<br>10 114 10 40.77 0 -0.35<br>20<br>20 30 34.58 10 -77.03<br>30 20<br>-213 -90.00 -178.00<br>“le 20 0 20 20 0 20 40 20 0 20<br>PLS component 1 PLS component 1 PLS component 1<br>(d) Elevation (e) Latitude (f) Longitude<br>PLS component 2 PLS component 2 PLS component 2<br>PLS component 2 PLS component 2 PLS component 2<br><!-- End of picture text -->

Figure 3: Projection onto the top two components of per-property partial least squares regressions reveals monotonic structure in LM representations. We first fit a PLS model on Llama 2 13B entity representations from our training split for each property, project entity representations from the test split, and then plot the resulting 2-d projections. Each dot represents one entity and color saturation represents the value of the corresponding entity attribute. See units for each property in Table 1. 

PLS regression results for Llama 2 13B representations are shown in Figure 2 and results for additional models in Appendix B. All numeric properties can be predicted well ( _R_<sup>2</sup> _≥_ 0.79), with the exception of the elevation property ( _R_<sup>2</sup> = 0.43). Across all six properties, PLS identifies small sets of predictive components. For example, a PLS model with _k_ = 7 components achieves a goodness of fit of _R_<sup>2</sup> = 0.91 when predicting birthyear attributes from entity representations. Generally, all LMs appear to encode almost the entirety (95% of maximum _R_<sup>2</sup> ) of their stored numeric attribute information in two- to six-dimensional subspaces and a large fraction (80% of maximum _R_<sup>2</sup> ) appears to be encoded in two to three dimensions (see Appendix C). 

To further illustrate the low dimensionality of numeric property representation, in Figure 3 we plot the projection of entity representations onto the top two components of PLS regressions. The plots for each property exhibit monotonic structure. Concomitant with poor regression fit, monotonic structure is least visible for the elevation property. All other plots show clearly visible directions along which attribute values increase, reflecting the good fit of low-dimensional PLS regression models for these properties. 

## **3 Causal Effect of Property-Encoding Directions** 

### **3.1 Motivation: Do property-encoding directions affect model output?** 

So far, we have found correlative evidence for the existence of directions in activation space that monotonically encode numeric properties. Concretely, partial least squares regression on entity representations found directions that are predictive of what quantity the LM will express when queried for a particular numeric attribute of a particular entity. However, representation is not a sufficient criterion for computation (Lasri et al., 2022). In our case this means that numeric properties might be encoded in representations without affecting model 

5 

Preprint 

output. In order to make the stronger claim that numeric properties are not only encoded monotonically, but that these representations have a monotonic effect on LM output, we now perform interventions to establish causality. 

Intuitively, we want to find out if making “small” interventions leads to small changes in model output, if “large” interventions lead to large changes, and if the sign of the intervention matches the sign of the change. We now formalize this intuition by adapting the definition of linear representation proposed by Park et al. (2023) and Jiang et al. (2024). **Definition 1** (Linear representation of numeric properties, adapted from Jiang et al. (2024)) **.** A numeric property is represented linearly if for all pairs of attribute instances _i_ , _j_ with quantities _qi̸_ = _qj_ and their representations _⃗ xi_ , _⃗ xj_ , there exists a _steering vector⃗ u_ so that _xi −⃗ xj ∈_ Cone( _⃗u_ ), where Cone( _⃗v_ ) = _{α⃗v_ : _α >_ 0 _}_ is the cone of vector _⃗ v_ , i.e., its positive span. 

Linearity of representations only requires that representations lie in a cone, but says nothing about their ordering. To model the natural structure of numeric properties, we introduce the constraint that the ordering of quantities is preserved in representation space. 

**Definition 2** (Monotonic representation of numeric properties) **.** A numeric property is represented monotonically if it is represented linearly in Cone( _⃗u_ ) and for all triples of attribute instances _h_ , _i_ , _j_ with quantities _qh > qi > qj_ and representations _⃗ xh_ , _⃗ xi_ , _⃗ xj_ the following holds: _⃗ xh −⃗ xj_ = _αhj⃗u_ and _⃗ xi −⃗ xj_ = _αij⃗u_ if and only if _αhj > αij_ . 

The coefficients _αij_ and _αhj_ exist since the definition stipulates linear representation. There are many ways to operationalize this definition. One way is to prepare a “synthetic” series of monotonic representations in Cone( _⃗u_ ) by varying _α_ and then testing if these representations result in monotonic output changes, which is what we will do now. 

### **3.2 Method: Activation patching** 

Viewing the LM computation graph as causal graph (Meng et al., 2022; McGrath et al., 2023), we intervene on model activations via activation patching (Vig et al., 2020; Wang et al., 2022; Zhang & Nanda, 2024) and observe the effect on model output. Unlike the common activation patching setup in which one replaces activations resulting from one input with activations from a different input, we create patches by editing activations along particular directions, similar to the activation manipulation method of Matsumoto et al. (2022). 

Specifically, for each of the top _K_ directions _⃗ uk ∈ R_<sup>_d_</sup> , _k ∈_ [1 . . _K_ ] found by PLS as described in §2, we prepare patches _⃗ ps_ , _k_ = _αs⃗uk_ with edit weights _αs_ and edit step index _s ∈_ [1 . . _S_ ].<sup>7</sup> In the view proposed by Park et al. (2023) and Jiang et al. (2024), we can interpret PLS components as potential steering vectors and edit weights as their coefficients. Lacking a principled method for choosing edit weights _αs_ , we set their range to the minimum and maximum PLS loadings on each property’s training split. This choice yields patches covering the full empirical range of activation projections onto direction _⃗ uk_ . After sampling _ntrain_ = 1000 popular entities for each of the six numeric properties we first fit PLS models for each property, then apply activation patches _⃗ ps_ , _k_ to the representations of _ntest_ = 100 held-out<sup>8</sup> entities and for each entitiy record the LM’s expressed quantity _ys_ , _k_ . To evaluate monotonicity, i.e., the notion that small (large) edit weights _αs_ , _k_ should have a small (large) effects and that negative (positive) weights should decrease (increase) the expressed quantity _ys_ , _k_ , we quantify the intervention effect via the ranked Spearman correlation _ρ_ ( _αs_ , _k_ ; _ys_ , _k_ ). 

A question left open so far is where activation patching should be performed. While automatic methods for localizing model components and subnetworks of interest have been proposed (Conmy et al., 2023; Kramar et al.´ , 2024), for simplicity we perform a coarse, non-exhaustive search across layers and token positions based on one numeric property and use the found setting for all experiments (results shown in Appendix E). In addition to this edit locus, we also search for an edit window, whose purpose is to counteract self-repair 

> 7We choose _S_ = 80 edits steps to balance step size and computational cost. 

> 8The held-out entities are entities which were not used to fit PLS models. 

6 

Preprint 

|_αs_|_ys_,1|_ys_,2|_ys_,3|_ys_,4|_ys_,5|_ys_,6|_αs_|_ys_,1|
|---|---|---|---|---|---|---|---|---|
|1.00|1941|1955|1980|1980|2012|1929|1.00|7.5 billion|
|0.90|1941|1955|1955|1984|2012|1929|0.90|7.5 billion|
|0.80|1941|1955|1955|1984|2012|1929|0.80|7.5 billion|
|0.70|1941|1955|1955|1980|1968|1929|0.70|7.5 billion|
|0.60|1932|1955|1935|1958|1968|1929|0.60|7.5 billion|
|0.50|1932|1940|1935|1958|1964|1929|0.50|7.5 billion|
|0.40|1932|1930|1917|1958|1957|1902|0.40|1.3 billion|
|0.30|1929|1930|1906|1958|1929|1902|0.30|1.3 billion|
|0.20|1902|1902|1902|1934|1929|1902|0.20|1.3 billion|
|0.10|1902|1902|1902|1902|1902|1902|0.10|10 million|
|0.00|1902|1902|1902|1902|1902|1902|0.00|40,000|
|-0.10|1887|1902|1902|1902|1882|1902|-0.10|40,000|
|-0.20|1882|1902|1902|1887|1882|1902|-0.20|25,000|
|-0.30|1883|1902|1902|1887|1882|1902|-0.30|25,000|
|-0.40|1619|1902|1906|1887|1882|1901|-0.40|20,000|
|-0.50|1619|1902|1906|1887|1882|1906|-0.50|20,000|
|-0.60|1619|1902|1906|1887|1882|1906|-0.60|20,000|
|-0.70|1619|1902|1906|1887|1880|1906|-0.70|12,000|
|-0.80|1888|1902|1902|1887|1880|1906|-0.80|12,000|
|-0.90|1815|1902|1902|1858|1880|1906|-0.90|12,000|
|-1.00|1815|1902|1902|1858|1880|1906|-1.00|12,000|
|_ρ_(_αs_,_ys_,_k_)|0.91|0.87|0.72|0.97|**0.98**|0.39|_ρ_(_αs_,_ys_,_k_)|**0.98**|



(a) Birthyear of Karl Popper (b) Population of Zittau 

Table 2: The quantity _ys_ , _k_ expressed by a LM changes as a result of directed activation patching along direction _k_ with (normalized) edit weight _αs_ , with _αs_ = 0.00 corresponding to unedited model activations. Warm colors indicate values larger than and cold colors values smaller than the true value, which, if output by the LM, is printed black. Table (a) shows how one-dimensional directed patches along each of the top six “birthyear” PLS components change the answer given by Llama 2 13B to the prompt: _In what year was Karl Popper born? One word answer only_ . It is apparent that the most-correlated component ( _k_ = 1) does not necessarily correspond to the direction in which model behavior exhibits highest monotonicity, which in this case is component _k_ = 5 with a Spearman correlation of 0.98. Table (b) shows the effect of patching along the top “population” component on Llama 2 13B when prompted: _What is the population of Zittau? One word answer only_ . 

(McGrath et al., 2023) and iterative inference effects (Rushing & Nanda, 2024). Layer-wise we find that a window of _±_ 2 layers around the edit locus is most effective, which is smaller than the _±_ 5 layers used in prior work (Meng et al., 2022; Hase et al., 2023a). We also implement a token-wise window (Monea et al., 2024), finding that in addition to the last entity mention token, patching up to two token representations to the left and one token representation to the right works best for the prompts in our experiments. Typically, this token window size covers the entity mention and the main verb or last token of the prompt, depending on the numeric property (see prompts in Appendix A). In summary, we patch activations in a 5-layer window centered on layer _l_ = 0.3 and an up-to 4-token window surrounding the last entity mention subword token. To improve output format adherence, we append the instruction _One word answer only_ to all prompts. 

### **3.3 Results: Property-encoding directions have effects and side-effects** 

We are interested in the effects and side effects on model output when patching activations along property-specific directions. Looking at effects first, Table 2 gives examples of how numeric attribute expression changes as a result of directed activation patching. Patching along “birthyear” directions results in the expression of different years, although the degree of monotonicity, as quantified by Spearman correlation _ρ_ , varies. Patching along the top “population” direction causes the model to generate a range of outputs that can be interpreted as population sizes, although the largest values are more suited to a planetary than a 

7 

Preprint 



<!-- Start of picture text -->
150 1e9<br>100<br>125 4<br>75<br>100<br>50 3<br>75<br>25<br>50 0 2<br>25 25 1<br>0 50<br>0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(a) Birthyear (b) Death year (c) Population<br>600 14000<br>6000 12000<br>400<br>10000<br>4000 200 8000<br>6000<br>2000 0 4000<br>2000<br>0 200<br>0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(d) Elevation (e) Latitude (f) Longitude<br>Model output change (years) Model output change (years) Model output change (people)<br>Model output change (meters) Model output change (degrees) Model output change (degrees)<br><!-- End of picture text -->

Figure 4: Effect of activation patching along property-specific directions across several numeric properties. Each subplot shows the change in the numeric attribute value expressed by Llama 2 13B, as a function of the edit weight _αs_ . Dark red lines indicate means across 100 entities sampled from held-out test sets and bands show standard deviations. 

municipal scale. The sequence of outputs has rather sudden jumps, e.g., from _40,000_ (unedited model, _αs_ = 0.00) to _10 million_ after taking the first step in the “larger population” direction ( _αs_ = 0.10). The pattern of jumps and plateaus is plausibly connected to several factors such as tokenization effects and the likely high frequency of certain numerals ( _1.3 billion_ : population of China at some point in time; _7.5 billion_ : population of Earth, etc.) in the training data, but we leave a detailed investigation to future work. The pattern also indicates that activation space, while apparently monotonic, is not linear in this direction. The intervention also induces a switch from positional notation ( _40,000_ ) to named numbers ( _million_ , _billion_ ), which showcases effects beyond single tokens. 

Moving to a more systematic analysis, we plot mean-aggregated effects of directed activation patching across six numeric properties in Figure 4. We see that there are properties for which directed activation patching has highly monotonic effects, e.g., birthyear ( _ρ_ = 0.84), elevation ( _ρ_ = 0.88), or work period start ( _ρ_ = 0.90), suggesting that these properties have highly monotonic representations. Other properties exhibit a much smaller degree of monotonic editability, e.g., longitude ( _ρ_ = 0.55) and population (0.65), suggesting that LM representations do not encode these properties as well. Figures for additional models lead to similar to conclusions and are shown in Appendix F. 

Having observed the effects of our interventions we now turn to analyzing their side effects on the expression of unrelated, non-targeted numeric properties, i.e., properties that were not the target of the intervention. For example, if we fitted a PLS regression to find and patch along “birthyear” directions, birthyear is our targeted property and all other properties, such as death year, population, or longitude are non-targeted properties. Using the same directions found in §2, we prompt LMs for non-targeted attributes, perform directed activation patching with weight _αs_ in a direction found for the targeted property and record expressed quantities _y_<sup>_′_</sup> _s_ , _k_<sup>.Toseeifnon-targetedpropertiesareaffectedina</sup> similar monotonic fashion as targeted ones, we quantify the side-effect of directed activation patching as the mean Spearman correlation _ρ_ ( _αs_ , _y_<sup>_′_</sup> _s_ , _k_<sup>),aggregatedover100entitiesper</sup> property. We perform this procedure for all combinations of targeted and non-targeted properties, including three additional properties, and show results in Figure 5. In this figure, entries on the diagonal show the mean effect size for targeted properties and off-diagonal 

8 

Preprint 



<!-- Start of picture text -->
1.0 1.0<br>area 0.49 0.67 0.64 0.65 0.58 0.52 0.52 0.52 0.56 area 0.54 0.73 0.69 0.70 0.68 0.68 0.60 0.34 0.67<br>birthyear 0.65 0.82 0.72 0.53 0.68 0.62 0.44 0.70 0.75 0.8 birthyear 0.57 0.84 0.66 0.42 0.64 0.56 0.43 0.36 0.84 0.8<br>death year 0.50 0.71 0.73 0.59 0.63 0.41 0.34 0.46 0.70 death year 0.62 0.68 0.82 0.44 0.66 0.52 0.54 0.63 0.52<br>elevation 0.52 0.61 0.49 0.80 0.56 0.43 0.39 0.46 0.52 0.6 elevation 0.45 0.62 0.52 0.88 0.61 0.43 0.40 0.51 0.59 0.6<br>inception 0.23 0.38 0.46 0.41 0.53 0.36 0.34 0.56 0.46 inception 0.41 0.67 0.59 0.50 0.70 0.64 0.56 0.44 0.60<br>latitude 0.51 0.65 0.63 0.63 0.56 0.63 0.54 0.49 0.58 0.4 latitude 0.57 0.55 0.61 0.58 0.51 0.76 0.35 0.50 0.49 0.4<br>longitude 0.49 0.47 0.61 0.50 0.47 0.40 0.49 0.49 0.52 longitude 0.71 0.52 0.54 0.57 0.56 0.57 0.55 0.58 0.58<br>population 0.56 0.71 0.61 0.59 0.61 0.45 0.49 0.65 0.57 0.2 population 0.54 0.76 0.71 0.67 0.70 0.69 0.51 0.65 0.68 0.2<br>work period start 0.46 0.60 0.60 0.64 0.50 0.44 0.42 0.42 0.81 work period start 0.41 0.77 0.74 0.49 0.65 0.54 0.48 0.52 0.90<br>0.0 0.0<br>Measured property Measured property<br>(a) Llama 2 7B (b) Llama 2 13B<br>Targeted property Targeted property<br>Edit effect (Spearman correlation) Edit effect (Spearman correlation)<br>area birthyear death year elevation inception latitude longitude population work period start area birthyear death year elevation inception latitude longitude population work period start<br><!-- End of picture text -->

Figure 5: Mean-aggregated effects and side effects when performing activation patching along property-specific directions in activation space. Diagonal entries (top-left to bottom right) show the effect on the targeted property in terms of mean Spearman correlation between edit weight _alphas_ , _k_ and expressed quantity _ys_ , _k_ . For example, patching an entity representation along a “birthyear” direction results in a corresponding change in the quantity expressed by Llama 2 13B with a correlation strength of 0.84. Off-diagonal entries show the side-effects of activation patching, e.g., “birthyear” patches affect LM output when queried for an entity’s death year with a correlation strength of 0.68. 

entries the size of side-effects. For Llama 2 7B, the mean effect size _ρ_ ¯ = 0.65 _±_ 0.12 (i.e., the mean of diagonal entries), is not much larger than the mean side-effect size _ρ_ ¯ = 0.53 _±_ 0.11 (mean of off-diagonal entries). In contrast, for Llama 2 13B the effect size of _ρ_ ¯ = 0.85 _±_ 0.07 is considerably larger than the size of side effects ( ¯ _ρ_ = 0.58 _±_ 0.18). A plausible explanation for these results is that in case of Llama 2 7B different properties share a subspace which encodes generic numeric ranges or generic small-large ranges that are translated into quantities depending on context, while the representational space of Llama 2 13B is more akin to a mixture of generic numeric and property-specific, “more orthogonal” subspaces. Clearly, more work is needed to verify this hypothesis. 

The analysis of side-effects is complicated by actual correlations between properties: Birthyear and death year distances are bounded by the human life span, latitude and population are correlated since the Earth’s northern hemisphere is more populous, locations with higher elevation tend to have smaller populations, etc. Consequently, one might argue that, say, editing an entity’s birthyear should also affect LM output when querying the entity’s death year. 

## **4 Related Work** 

Shaped by the locality of physical reality, the locality of human experience (Prystawski et al., 2023) gives rise to distributional patterns of language use. Such patterns include patterns of geographic and temporal coherence (Heinzerling et al., 2017), which reflect spatiotemporal proximity of real-world entities. These patterns can be picked up by statistical models and allow, e.g., to predict geographic information from co-occurrence statistics of cities mentioned in news articles (Louwerse & Zwaan, 2009). Probing static word vector representations for numeric attributes of geopolitical entities, Gupta et al. (2015) obtain good relative rankings, but do not evaluate absolute values nor analyze the geometry of representations. Continuing this line of research, Lietard´ et al. (2021) probe LM representations for GPS coordinates. Perhaps due to the—by current standards—small scale of the studied LMs, they find only limited success but report that larger models appeared to encode more geographic information. Faisal & Anastasopoulos (2023) measure how well the geographic proximity of countries can be recovered from LM representations but differ from our work in their focus on the impact of politico-cultural factors. 

9 

Preprint 

Closest to our work is the analysis of geo-temporal information encoded in Llama 2 representations by Gurnee & Tegmark (2023). Our work corroborates their finding of linear subspaces of activation space which are predictive of numeric attributes, but is distinct in three important aspects. First, as we show in §2, the subspaces found PCA, as used by Gurnee & Tegmark, are of considerably higher dimensionality (50 _−_ 100) than the subspaces found by partial least-square regression (2 _−_ 17). Our finding thus tightens the upper bound on the complexity of numeric property representation in recent LMs. Second, we make explicit and formalize the notion of monotonic representation. Third, our interventions via directed activation patching (§3) found one-dimensional directions with fine-grained effects on the expression of numeric attributes, across all numeric properties and models we analyzed, thereby establishing a causal relationship between monotonic representations and LM behavior. 

## **5 Limitations** 

### **5.1 General limitations of representational analysis** 

None of the language models studied in this work are embodied agents or otherwise capable of embodied cognition. Lacking direct sensorimotor grounding (Harnad, 1990; Mollo & Milliere` , 2023; Harnad, 2024), LMs cannot directly perceive, let alone precisely measure, the numerical attributes of which we claim to have found monotonic representations. It follows that any such representations are an artifact of distributional patterns in their training data, and that the best one can hope for is isomorphy between model representations and the properties of the real-world entities to which we tie those representations. 

Leaving the groundedness of representations aside, the idea that concepts, knowledge, or behavior are “encoded” in neural representations might seem intuitively appealing, but has been strongly critized, on theoretical grounds in the context of biological and artificial neural networks in general (Brette, 2019), and on empirical grounds in the context of pretrained language models in particular (Hase et al., 2023a; Niu et al., 2024). 

Analysis of LM representations also has well-known limitations. Under the mild assumption that there exists a bijection between inputs and their representations, all information extractable from the input, i.e., the natural language prompt, can also be extracted from the LM’s representation of that sequence (Pimentel et al., 2020b). Hence the question to be answered by representational analysis is not whether a feature of interest can be extracted or not, but how easy it is to extract. How to best quantify “ease of extraction” (Pimentel et al., 2020b) is an open question, although methods have been proposed (Pimentel et al., 2020a; Voita & Titov, 2020). 

### **5.2 Specific limitations of the representational analysis conducted in this work** 

The low-dimensional linear subspaces found in this work allow relatively “easy” extraction when compared to the nominally high dimensionalities of activation space, but are still higher-dimensional than necessary, since the represented structures (e.g., years, geographic coordinates) are canonically one- to two-dimensional. Furthermore, activation space is nominally high-dimensional but its intrinsic dimension is believed to be much lower (Li et al., 2018; Aghajanyan et al., 2021; Razzhigaev et al., 2024). For example Razzhigaev et al. (2024) provide estimates for the intrinsic dimension of various LMs, ranging from about 10 to 70 dimensions (the models used in our experiments are not covered). If we view a non-linear, non-monotonic representation of full intrinsic dimensionality as the most complex encoding with worst-case ease of extraction, and one- to two-dimensional linear monotonic encodings as the simplest representation with optimal ease of extraction, then the low-dimensional subspaces we found fall somewhere between these bounds. Whether they are low-dimensional relative to the models’ intrinsic dimension is currently unknown. Put differently, if the intrinsic dimension of Llama 2 7B turns out to be, say, 10, then finding, a 10-dimensional subspace that encodes all latitude information (see Appendix C) is not surprising, but necessary. 

10 

Preprint 

While we found evidence for monotonic representation of numeric properties, it is likely that our causal interventions via activation patching along one-dimensional directions are too simplistic, considering the fact that according to our PLS regression results, numeric properties are encoded in low- but not one-dimensional subspaces. Hence it is possible that a more refined editing method operating on higher-dimensional directions will allow more precise control over LM output. Furthermore, our analysis is limited to popular entities, frequent numeric properties, and English queries, i.e., the combination most likely to be well-represented in the LM training data. 

## **6 Conclusions and Open Questions** 

We used partial least-squares regression to identify low-dimensional subspaces of activation space that are predictive of the quantity an LM expresses when queried for numeric attributes such as an entity’s birthyear. We then performed activation patching along onedimensional directions in these subspaces and observed corresponding changes in model output. Our results suggest that LMs learn monotonic representations of numeric properties and that these monotonic representations exist in all of the language models we examined. 

While this work takes a step towards a better understanding of numeric property representation in LMs, it leaves many questions unanswered. Some of these are: 

- What exactly do the subspaces we found encode? Large side effects of activation patching suggest that there exist directions that encode generic numeric ranges. At the same time, the pronounced difference between large effect size and smaller sideeffects we observed for the largest model in our analysis constitutes evidence for the existence of property-specific subspaces. Can we find more specific directions, possibly by adding constraints to minimize side effects into the regression objective, similar to (Meng et al., 2022)? 

- If there are subspaces encoding generic numeric ranges, are changes along directions in these subspaces absolute or relative to typical value ranges? E.g., does a relative large change in birthyears, say 100 years, translate into a relatively large change in elevation, say 1000 meters, or into a change with the same value, i.e., 100 meters? 

- Is the quality of numeric attribute representations in any way connected to LM performance on tasks involving the attributes in question? For example, does the model answer queries like _Who was born earlier, Albert Einstein or Karl Popper?_ correctly if the corresponding entity representations are appropriately positioned along a “birthyear” direction, and incorrectly if they are not? 

**Acknowledgements.** This work was supported by JST CREST Grant Number JPMJCR20D2 and JSPS KAKENHI Grant Number 21K17814. 

## **References** 

- Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 7319–7328, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.568. URL https://aclanthology.org/2021.acl-long.568. 

- Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Merouane´ Debbah, Etienne<sup>´</sup> Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. The falcon series of open language models, 2023. 

- Emily M. Bender and Alexander Koller. Climbing towards NLU: On meaning, form, and understanding in the age of data. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel 

11 

Preprint 

Tetreault (eds.), _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 5185–5198, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.463. URL https://aclanthology.org/2020.acl-main. 463. 

- Romain Brette. Is coding a relevant metaphor for the brain? _Behavioral and Brain Sciences_ , 42: e215, 2019. 

- Roger C. Conant and W. Ross Ashby. Every good regulator of a system must be a model of that system. _International journal of systems science_ , 1(2):89–97, 1970. 

- Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria` Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability, 2023. 

- Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_ , pp. 6491–6506, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.522. URL https: //aclanthology.org/2021.emnlp-main.522. 

- Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. 

- Fahim Faisal and Antonios Anastasopoulos. Geographic and geopolitical biases of language models. In Duygu Ataman (ed.), _Proceedings of the 3rd Workshop on Multilingual Representation Learning (MRL)_ , pp. 139–163, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.mrl-1.12. URL https: //aclanthology.org/2023.mrl-1.12. 

- Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 12216–12235, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.751. URL https://aclanthology.org/ 2023.emnlp-main.751. 

- Nathan Godey, Eric de la Clergerie, and Beno<sup>´</sup> ˆıt Sagot. On the scaling laws of geographical representation in language models, 2024. 

- Abhijeet Gupta, Gemma Boleda, Marco Baroni, and Sebastian Pado.´ Distributional vectors encode referential attributes. In Llu´ıs Marquez, Chris Callison-Burch, and Jian Su (eds.),` _Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing_ , pp. 12–21, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1002. URL https://aclanthology.org/D15-1002. 

Wes Gurnee and Max Tegmark. Language models represent space and time, 2023. 

- Stevan Harnad. The symbol grounding problem. _Physica D: Nonlinear Phenomena_ , 42(1-3): 335–346, 1990. 

- Stevan Harnad. Language writ large: Llms, chatgpt, grounding, meaning and understanding, 2024. 

- Charles R. Harris, K. Jarrod Millman, Stefan´ J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernandez´ del R´ıo, Mark Wiebe, Pearu Peterson, Pierre Gerard-Marchant,´ Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, 

12 

Preprint 

Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. _Nature_ , 585(7825):357–362, September 2020. doi: 10.1038/s41586-020-2649-2. URL https: //doi.org/10.1038/s41586-020-2649-2. 

- Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models, 2023a. 

- Peter Hase, Mona Diab, Asli Celikyilmaz, Xian Li, Zornitsa Kozareva, Veselin Stoyanov, Mohit Bansal, and Srinivasan Iyer. Methods for measuring, updating, and visualizing factual beliefs in language models. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pp. 2714–2731, Dubrovnik, Croatia, May 2023b. Association for Computational Linguistics. URL https://aclanthology.org/ 2023.eacl-main.199. 

- Benjamin Heinzerling and Kentaro Inui. Language models as knowledge bases: On entity representations, storage capacity, and paraphrased queries. In Paola Merlo, Jorg Tiedemann, and Reut Tsarfaty (eds.), _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pp. 1772–1791, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.153. URL https://aclanthology.org/2021.eacl-main.153. 

- Benjamin Heinzerling, Michael Strube, and Chin-Yew Lin. Trust, but verify! better entity linking through automatic verification. In Mirella Lapata, Phil Blunsom, and Alexander Koller (eds.), _Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers_ , pp. 828–838, Valencia, Spain, April 2017. Association for Computational Linguistics. URL https://aclanthology.org/E17-1078. 

- Evan Hernandez, Belinda Z. Li, and Jacob Andreas. Inspecting and editing knowledge representations in language models. In _Arxiv_ , 2023. URL https://arxiv.org/abs/2304. 00740. 

- J. D. Hunter. Matplotlib: A 2d graphics environment. _Computing in Science & Engineering_ , 9 (3):90–95, 2007. doi: 10.1109/MCSE.2007.55. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut´ Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023. 

- Yibo Jiang, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, and Victor Veitch. On the origins of linear representations in large language models, 2024. 

- Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. How can we know what language models know? _Transactions of the Association for Computational Linguistics_ , 8:423– 438, 2020. doi: 10.1162/tacl ~~a 0~~ 0324. URL https://aclanthology.org/2020.tacl-1.28. 

- Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. How can we know when language models know? on the calibration of language models for question answering. _Transactions of the Association for Computational Linguistics_ , 9:962–977, 2021. doi: 10.1162/ tacl ~~a 0~~ 0407. URL https://aclanthology.org/2021.tacl-1.57. 

- Nora Kassner, Philipp Dufter, and Hinrich Schutze.¨ Multilingual LAMA: Investigating knowledge in multilingual pretrained language models. In Paola Merlo, Jorg Tiedemann, and Reut Tsarfaty (eds.), _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pp. 3250–3258, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.284. URL https://aclanthology.org/2021.eacl-main.284. 

- Janos Kram´ ar, Tom Lieberum, Rohin Shah, and Neel Nanda.´ Atp*: An efficient and scalable method for localizing llm behaviour to components, 2024. 

13 

Preprint 

- Karim Lasri, Tiago Pimentel, Alessandro Lenci, Thierry Poibeau, and Ryan Cotterell. Probing for the usage of grammatical number. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 8818–8831, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.603. URL https://aclanthology.org/2022.acl-long.603. 

- Harvey Lederman and Kyle Mahowald. Are language models more like libraries or like librarians? bibliotechnism, the novel reference problem, and the attitudes of llms, 2024. 

- Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. _arXiv preprint arXiv:1804.08838_ , 2018. 

- Bastien Lietard,´ Mostafa Abdou, and Anders Søgaard. Do language models know the way to Rome? In Jasmijn Bastings, Yonatan Belinkov, Emmanuel Dupoux, Mario Giulianelli, Dieuwke Hupkes, Yuval Pinter, and Hassan Sajjad (eds.), _Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP_ , pp. 510–517, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.blackboxnlp-1.40. URL https://aclanthology.org/2021. blackboxnlp-1.40. 

- Ziming Liu, Ouail Kitouni, Niklas S Nolte, Eric Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An effective theory of representation learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), _Advances in Neural Information Processing Systems_ , volume 35, pp. 34651–34663. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper ~~f~~ iles/paper/2022/ file/dfc310e81992d2e4cedc09ac47eff13e-Paper-Conference.pdf. 

- Max M. Louwerse and Rolf A. Zwaan. Language encodes geographical information. _Cognitive Science_ , 33(1):51–73, 2009. doi: https://doi.org/10.1111/j.1551-6709.2008.01003.x. 

- Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2023. 

- Yuta Matsumoto, Benjamin Heinzerling, Masashi Yoshikawa, and Kentaro Inui. Tracing and manipulating intermediate values in neural math problem solvers. In Deborah Ferreira, Marco Valentino, Andre Freitas, Sean Welleck, and Moritz Schubotz (eds.), _Proceedings of the 1st Workshop on Mathematical Natural Language Processing (MathNLP)_ , pp. 1–6, Abu Dhabi, United Arab Emirates (Hybrid), December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.mathnlp-1.1. URL https://aclanthology.org/2022. mathnlp-1.1. 

- Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. The hydra effect: Emergent self-repair in language model computations, 2023. 

- Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. _Advances in Neural Information Processing Systems_ , 36, 2022. 

- Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. A mechanism for solving relational tasks in transformer language models, 2023. 

- Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. Fast model editing at scale. _CoRR_ , 2021. URL https://arxiv.org/pdf/2110.11309.pdf. 

- Dimitri Coelho Mollo and Rapha¨el Milli`ere. The vector grounding problem, 2023. 

- Giovanni Monea, Maxime Peyrard, Martin Josifoski, Vishrav Chaudhary, Jason Eisner, Emre Kıcıman, Hamid Palangi, Barun Patra, and Robert West. A glitch in the matrix? locating and detecting language model grounding with fakepedia, 2024. 

14 

Preprint 

- Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi (eds.), _Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP_ , pp. 16–30, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.blackboxnlp-1.2. URL https://aclanthology.org/2023.blackboxnlp-1.2. 

- Jingcheng Niu, Andrew Liu, Zining Zhu, and Gerald Penn. What does the knowledge neuron thesis have to do with knowledge? In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum?id=2HJRwwbV3G. 

- The Pandas development team. pandas-dev/pandas: Pandas, February 2020. URL https: //doi.org/10.5281/zenodo.3509134. 

- Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models, 2023. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. _Advances in neural information processing systems_ , 32, 2019. 

- Karl Pearson. On lines and planes of closest fit to systems of points in space. _The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science_ , 2(11):559–572, 1901. doi: 10.1080/14786440109462720. URL https://doi.org/10.1080/14786440109462720. 

- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_ , 12:2825–2830, 2011. 

- Fabio Petroni, Tim Rocktaschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang¨ Wu, and Alexander Miller. Language models as knowledge bases? In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pp. 2463–2473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1250. URL https://aclanthology.org/D19-1250. 

- Tiago Pimentel, Naomi Saphra, Adina Williams, and Ryan Cotterell. Pareto probing: Trading off accuracy for complexity. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 3138–3153, Online, November 2020a. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.254. URL https://aclanthology.org/2020.emnlp-main.254. 

- Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, and Ryan Cotterell. Information-theoretic probing for linguistic structure. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 4609–4622, Online, July 2020b. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.420. URL https://aclanthology.org/2020.acl-main.420. 

- Ben Prystawski, Michael Y. Li, and Noah D. Goodman. Why think step by step? reasoning emerges from the locality of experience, 2023. 

- Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Goncharova, Ivan Oseledets, Denis Dimitrov, and Andrey Kuznetsov. The shape of learning: Anisotropy and intrinsic dimensions in transformer-based models, 2024. 

- Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? In Bonnie Webber, Trevor Cohn, Yulan He, 

15 

Preprint 

and Yang Liu (eds.), _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 5418–5426, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.437. URL https:// aclanthology.org/2020.emnlp-main.437. 

Cody Rushing and Neel Nanda. Explorations of self-repair in language models, 2024. 

- Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 4222–4235, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.346. URL https://aclanthology.org/2020.emnlp-main. 346. 

- Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda. Linear representations of sentiment in large language models, 2023. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. 

- Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. _Advances in neural information processing systems_ , 33:12388–12401, 2020. 

- Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stefan J.´ van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, Ilhan Polat, Yu Feng,<sup>˙</sup> Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro,ˆ Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. _Nature Methods_ , 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2. 

- Elena Voita and Ivan Titov. Information-theoretic probing with minimum description length. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pp. 183–196, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.emnlp-main.14. URL https://aclanthology.org/2020.emnlp-main.14. 

- Denny Vrandeciˇ c´ and Markus Krotzsch.¨ Wikidata: a free collaborative knowledgebase. _Communications of the ACM_ , 57(10):78–85, 2014. 

- Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small, 2022. 

- Michael L. Waskom. seaborn: statistical data visualization. _Journal of Open Source Software_ , 6 (60):3021, 2021. doi: 10.21105/joss.03021. URL https://doi.org/10.21105/joss.03021. 

16 

Preprint 

- Herman Wold. Estimation of principal components and related models by iterative least squares. _Multivariate analysis_ , pp. 391–420, 1966. 

- Svante Wold, Michael Sjostr¨ om,¨ and Lennart Eriksson. Pls-regression: a basic tool of chemometrics. _Chemometrics and intelligent laboratory systems_ , 58(2):109–130, 2001. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Qun Liu and David Schlangen (eds.), _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pp. 38–45, Online, October 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos.6. URL https://aclanthology.org/ 2020.emnlp-demos.6. 

- Paul Youssef, Osman Koras¸, Meijie Li, Jorg Schl¨ otterer, and Christin Seifert.¨ Give me the facts! a survey on factual knowledge probing in pre-trained language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), _Findings of the Association for Computational Linguistics: EMNLP 2023_ , pp. 15588–15605, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.1043. URL https: //aclanthology.org/2023.findings-emnlp.1043. 

- Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods, 2024. 

- Zexuan Zhong, Dan Friedman, and Danqi Chen. Factual probing is [MASK]: Learning vs. learning to recall. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pp. 5017–5033, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.398. URL https://aclanthology.org/2021.naacl-main.398. 

17 

Preprint 

## **A Data sample** 

|Property<br>|Prop. ID|Entity<br>|EntityID<br>|Prompt<br>|Value|Unit|
|---|---|---|---|---|---|---|
|birthyear<br>bith|P569<br>P569|Nina Foch<br>Gff Hld|Q235632<br>Q945691|In what year was Nina Foch born?<br>I ht   Gff Hld b?|1924<br>1930|annum<br>|
|ryear<br>birthyear|P569|eorey oer<br>Harriette L. Chandler|Q5664432|n wa year was eorey oer orn<br>In what year was Harriette L. Chandler born?|1937|annum<br>annum|
|birthyear|P569|Gabriel Garc´ıa M´arquez|Q5878|<br>In what year was Gabriel Garc´ıa M´arquez<br>born?|1927|annum|
|birthyear|P569|Norman Schwarzkopf Jr.|Q310188|In what year was Norman Schwarzkopf Jr.<br>born?|1934|annum|
|birthyear<br>|P569|Paul de Vos<br>|Q2610964<br>|In what year was Paul de Vos born?<br>|1590|annum|
|birthyear<br>birthear|P569<br>P569|Nicolas Carnot<br>Steve Harve|Q181685<br>Q2347009|In what year was Nicolas Carnot born?<br>In what ear was Steve Harve born?|1796<br>1957|annum<br>annum|
|y<br>birthyear|P569|y<br>Tommy Lawton|Q726272|y   y<br>In what year was Tommy Lawton born?|1919|annum|
|birthyear|P569|<br>Hans von B¨ulow|Q155540|<br>In whatyear was Hans von B¨ulow born?|1830|annum|
|death year|P570|Johannes R. Becher|Q58057|In what year did Johannes R. Becher die?|1958|annum|
|<br>death year|P570|Friedrich Georg Wilhelm<br>St|Q57164|<br>In what year did Friedrich Georg Wilhelm von<br>St di?|1864|annum|
|death year|P570|von ruve<br>Pierre Boulez|Q156193|ruve e<br>In what year did Pierre Boulez die?|2016|annum|
|<br>death year<br>death year|P570<br>P570|Giovanni da Palestrina<br>Abdurrauf Fitrat|Q179277<br>Q317907|<br>In what year did Giovanni da Palestrina die?<br>In what year did Abdurrauf Fitrat die?|1594<br>1938|annum<br>annum|
|<br>death year|P570|Lucian Freud|Q154594|<br>In what year did Lucian Freud die?|2011|annum|
|<br>death year<br>|P570<br>|<br>Akseli Gallen-Kallela<br>|Q170068<br>|<br>In what year did Akseli Gallen-Kallela die?<br>|1931<br>|annum|
|death year|P570|Spock|Q16341|In what year did Spock die?|2263|annum|
|<br>death year|P570|William Orpen|Q922483|<br>In what year did William Orpen die?|1931|annum|
|<br>deathyear|P570|<br>Carlos Santiago M´erida|Q1043100|<br>In whatyear did Carlos Santiago M´erida die?|1984|annum|
|population|P1082|Akhisar|Q209905|What is the population of Akhisar?|173026|1|
|population<br>oulation|P1082<br>P1082|Tripura<br>Albert|Q1363<br>Q30940|What is the population of Tripura?<br>What is the oulation of Albert?|3665958<br>9930|1<br>1|
|pp<br>population|P1082|High Wycombe|Q64116|pp<br>What is the population of High Wycombe?|120256|1|
|population<br>oulation|P1082<br>P1082|Pl¨on<br>Reublika Srska|Q497060<br>Q11196|What is the population of Pl¨on?<br>What is the oulation of Reublika Srska?|8914<br>1228423|1<br>1|
|pp<br>population|P1082|p p<br>Lebanese|Q2606511|pp  p p<br>What is the population of Lebanese?|8000000|1|
|population|P1082|Geraardsbergen|Q499532|<br>What is the population of Geraardsbergen?|33403|1|
|population|P1082|Gorz´ow Wielkopolski|Q104731|<br>What is the population of Gorz´ow Wielkopol-<br>|124295|1|
|population|P1082|Harran|Q199547|ski?<br>What is thepopulation of Harran?|47606|1|
|evelation|P2044|Sondrio|Q6274|How high is Sondrio?|360|metre|
|evelation<br>evelation|P2044<br>P2044|Rio Branco<br>Demmin|Q171612<br>Q50960|<br>How high is Rio Branco?<br>How high is Demmin?|158<br>8|metre<br>metre|
|evelation|P2044|Cetinje|Q173338|<br>How hih is Cetinje?|650|metre|
|evelation|P2044|Highland Park|Q576671|g<br>How high is Highland Park?|503|metre|
|evelation|P2044|Gozo|Q170488|How high is Gozo?|195|metre|
|evelation|P2044|Saint-Jean-de-<br>|Q208860|<br>How high is Saint-Jean-de-Maurienne?|566|metre|
|evelation|P2044|Maurienne<br>Butte|Q467664|How high is Butte?|1688|metre|
|evelation<br>|P2044<br>|Cottbus<br>|Q3214<br>|<br>How high is Cottbus?<br>|76<br>|metre<br>|
|evelation|P2044|Mahilio˘u Region|Q189822|How high is Mahilio˘u Region?|191|metre|
|longitude|P625.long|Korean Empire|Q28233|What is the longitude of Korean Empire?|126.98|degree|
|longitude<br>lid|P625.long<br>P625l|Pine Bluff<br>T|Q80012<br>260130|What is the longitude of Pine Bluff?<br>Wh i h lid f T?|-92.00<br>1176|degree<br>d|
|ongtue<br>longitude|.ong<br>P625.long|egernsee<br>Old C¨olln|Q<br>Q269622|at s te ongtue o egernsee<br>What is the longitude of Old C¨olln?|.<br>13.40|egree<br>degree|
|longitude<br>|P625.long<br>|Cambridge<br>|Q49111<br>|<br>What is the longitude of Cambridge?<br>|-71.11<br>|degree<br>|
|longitude<br>longitude|P625.long<br>P625.long|Stryn<br>Ciudad Real Province|Q5223<br>Q54932|What is the longitude of Stryn?<br>What is the longitude of Ciudad Real<br>Province?|6.86<br>-4.00|degree<br>degree|
|longitude|P625.long|Santa Catarina|Q41115|What is the longitude of Santa Catarina?|-50.49|degree|
|longitude|P625.long|Wake Forest University|Q392667|What is the longitude of Wake Forest Univer-<br>it?|-80.28|degree|
|longitude|P625.long|West Lothian|Q204940|sy<br>What is the longitude of West Lothian?|-3.50|degree|
|latitude|P625.lat|K¨usnacht|Q69216|What is the latitude of K¨usnacht?|47.32|degree|
|latitude|P625.lat|Mount Jerome Cemetery|Q917854|What is the latitude of Mount Jerome Ceme-<br>|53.32|degree|
|||||tery?<br>|||
|latitude|P625.lat|Dayton Children’s Hospi-<br>tal|Q5243510|What is the latitude of Dayton Children’s Hos-<br>ital?|39.77|degree|
|latitude<br>|P625.lat<br>|Le Flore County<br>|Q495944<br>|p<br>What is the latitude of Le Flore County?<br>|34.90|degree<br>|
|latitude|P625.lat|Czechoslovakia|Q33946|What is the latitude of Czechoslovakia?|50.08|degree|
|latitude<br>|P625.lat<br>|Pembroke College<br>|Q956501<br>|What is the latitude of Pembroke College?<br>|52.20<br>|degree<br>|
|latitude<br>|P625.lat<br>|Hayward<br>|Q491114|What is the latitude of Hayward?<br>|37.67|degree<br>|
|latitude<br>ltitd|P625.lat<br>P625lt|Banaskantha district<br>Crbil-Enn|Q806125<br>Q208812|What is the latitude of Banaskantha district?<br>Wht i th ltitd f Crbil-Enn?|24.17<br>4861|degree<br>dr|
|aue<br>latitude|.a<br>P625.lat|oessoes<br>Elbasan|Q114257|a s e aue o oessoes<br>What is the latitude of Elbasan?|.<br>41.11|egee<br>degree|



Table 3: Random sample of the entities used in our experiments, along with corresponding numeric attributes and prompts. Entities, their English labels, and numeric attributes for each property are extracted from an April 2022 dump of Wikidata (wikidata-20220421-all). In many cases Wikidata contains multiple values for a given numeric attribute, e.g., reflecting chronological change such as the population of a city, or owing to conflicting sources. In such cases we take the mode of the distribution as gold value. We also filter out quantities with non-standard units, such as elevations measured in feet. 

18 

Preprint 

## **B Regression on entity representations: Additional figures** 



<!-- Start of picture text -->
1.00 1.0 1.0<br>0.75<br>0.50 0.5 0.5<br>0.25 Method Method Method<br>0.00 PLSPCA 0.0 PLSPCA 0.0 PLSPCA<br>0.25 PLS (shuffled labels) 0.5 PLS (shuffled labels) PLS (shuffled labels)<br>PCA (shuffled labels) PCA (shuffled labels) 0.5 PCA (shuffled labels)<br>0.50 PLS (random reprs.) PLS (random reprs.) PLS (random reprs.)<br>0.75 PCA (random reprs.) 1.0 PCA (random reprs.) PCA (random reprs.)<br>1.0<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(c) Birthyear (d) Death year (e) Population<br>1.0 1.00 1.00<br>0.75 0.75<br>0.5 0.50 0.50<br>Method 0.25 Method Method<br>0.0 PLS PLS 0.25 PLS<br>PCA 0.00 PCA 0.00 PCA<br>PLS (shuffled labels) 0.25 PLS (shuffled labels) PLS (shuffled labels)<br>0.5 PCA (shuffled labels) 0.50 PCA (shuffled labels) 0.25 PCA (shuffled labels)<br>1.0 PLS (random reprs.)PCA (random reprs.) 0.75 PLS (random reprs.)PCA (random reprs.) 0.50 PLS (random reprs.)PCA (random reprs.)<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(f) Elevation (g) Latitude (h) Longitude<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br><!-- End of picture text -->

Figure 6: Regression curves for Llama 2 7B. See explanation in Figure 2. 



<!-- Start of picture text -->
1.0 1.00 1.00<br>0.8 0.75 0.75<br>0.6 0.50 0.50<br>0.4 PLSMethod 0.25 PLSMethod 0.25 PLSMethod<br>0.2 PCA 0.00 PCA 0.00 PCA<br>0.0 PLS (shuffled labels)PCA (shuffled labels) 0.25 PLS (shuffled labels)PCA (shuffled labels) 0.25 PLS (shuffled labels)PCA (shuffled labels)<br>0.2 PLS (random reprs.) 0.50 PLS (random reprs.) 0.50 PLS (random reprs.)<br>0.4 PCA (random reprs.) 0.75 PCA (random reprs.) PCA (random reprs.)<br>0.75<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(a) Birthyear (b) Death year (c) Population<br>1.00 1.00 1.00<br>0.75 0.75 0.75<br>0.50 0.50 0.50<br>0.25 PLSMethod 0.25 PLSMethod 0.25 PLSMethod<br>0.00 PCA 0.00 PCA 0.00 PCA<br>PLS (shuffled labels) 0.25 PLS (shuffled labels) PLS (shuffled labels)<br>0.25 PCA (shuffled labels) PCA (shuffled labels) 0.25 PCA (shuffled labels)<br>0.50 PLS (random reprs.) 0.50 PLS (random reprs.) 0.50 PLS (random reprs.)<br>PCA (random reprs.) 0.75 PCA (random reprs.) PCA (random reprs.)<br>0.75 0.75<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(d) Elevation (e) Latitude (f) Longitude<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br><!-- End of picture text -->

Figure 7: Regression curves for Falcon 7B. See explanation in Figure 2. 

19 

Preprint 



<!-- Start of picture text -->
1.00 1.00 1.0<br>0.75 0.75<br>0.50 0.50 0.5<br>Method 0.25 Method Method<br>0.25 PLSPCA 0.00 PLSPCA 0.0 PLSPCA<br>0.00<br>PLS (shuffled labels) 0.25 PLS (shuffled labels) PLS (shuffled labels)<br>0.25 PCA (shuffled labels)PLS (random reprs.) 0.50 PCA (shuffled labels)PLS (random reprs.) 0.5 PCA (shuffled labels)PLS (random reprs.)<br>0.50 PCA (random reprs.) 0.75 PCA (random reprs.) PCA (random reprs.)<br>1.0<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(a) Birthyear (b) Death year (c) Population<br>1.0 1.00 1.0<br>0.75<br>0.5 0.50 0.5<br>0.0 PLSPCAMethod 0.250.00 PLSPCAMethod 0.0 PLSPCAMethod<br>0.5 PLS (shuffled labels) 0.25 PLS (shuffled labels) PLS (shuffled labels)<br>PCA (shuffled labels) 0.50 PCA (shuffled labels) 0.5 PCA (shuffled labels)<br>1.0 PLS (random reprs.) PLS (random reprs.) PLS (random reprs.)<br>PCA (random reprs.) 0.75 PCA (random reprs.) 1.0 PCA (random reprs.)<br>0 10 20 30 40 50 0 10 20 30 40 50 0 10 20 30 40 50<br>#components #components #components<br>(d) Elevation (e) Latitude (f) Longitude<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br>2Goodness of fit ()R 2Goodness of fit ()R 2Goodness of fit ()R<br><!-- End of picture text -->

Figure 8: Regression curves for Mistral 7B. See explanation in Figure 2. 

## **C Regression on entity representations: Additional analysis** 



<!-- Start of picture text -->
Property Model R 2 C � maxR 2� C � ≥ 0.95 R 2� C � ≥ 0.90 R 2� C � ≥ 0.80 R 2� C � ≥ 0.70 R 2� C � ≥ 0.60 R 2� C � ≥ 0.50 R 2�<br>birthyear (P569) Falcon 7B 0.75 4 2 2 2 1 1 1<br>birthyear (P569) Llama 2 13B 0.91 7 4 3 2 2 2 1<br>birthyear (P569) Llama 2 7B 0.90 11 6 4 3 2 2 1<br>birthyear (P569) Mistral 7B 0.89 4 3 2 2 2 1 1<br>death year (P570) Falcon 7B 0.61 2 2 2 2 1 1 1<br>death year (P570) Llama 2 13B 0.84 12 4 3 2 2 1 1<br>death year (P570) Llama 2 7B 0.82 11 4 4 3 2 1 1<br>death year (P570) Mistral 7B 0.80 4 3 3 2 2 1 1<br>latitude (P625.lat) Falcon 7B 0.67 6 3 3 3 2 2 2<br>latitude (P625.lat) Llama 2 13B 0.82 10 5 4 3 3 2 2<br>latitude (P625.lat) Llama 2 7B 0.83 10 5 3 2 2 2 2<br>latitude (P625.lat) Mistral 7B 0.79 9 4 3 3 2 2 2<br>longitude (P625.long) Falcon 7B 0.74 7 5 3 3 2 2 2<br>longitude (P625.long) Llama 2 13B 0.79 17 6 5 3 3 2 2<br>longitude (P625.long) Llama 2 7B 0.83 9 5 3 3 2 2 2<br>longitude (P625.long) Mistral 7B 0.78 6 5 3 3 2 2 1<br>population (P1082) Falcon 7B 0.67 4 3 3 2 1 1 1<br>population (P1082) Llama 2 13B 0.79 5 4 4 2 2 1 1<br>population (P1082) Llama 2 7B 0.73 5 4 3 2 2 1 1<br>population (P1082) Mistral 7B 0.76 5 4 2 2 1 1 1<br>evelation (P2044) Falcon 7B 0.23 2 2 2 2 2 1 1<br>evelation (P2044) Llama 2 13B 0.43 3 2 2 2 2 2 1<br>evelation (P2044) Llama 2 7B 0.37 2 2 2 2 2 1 1<br>evelation (P2044) Mistral 7B 0.41 3 3 2 2 2 1 1<br><!-- End of picture text -->

Table 4: Number of partial least squares regression components _C_ [ _T_ ] required for a given goodness of fit _T_ , found using the experimental setup described in §2. For example, the _C_ � _≥_ 0.95 _R_ 2� column shows the number of components required to reach 95 percent of the maximum goodness of fit for the respective property and model. From this column we can read that, e.g., two components of Falcon 7B’s activation space are sufficient to reach 95 percent of the maximum goodness of fit when predicting the birthyear of entities, indicating that this property is almost entirely encoded in a two-dimensional subspace of this model’s activation space. 

20 

Preprint 

## **D PLS projections of entity representations: Additional figures** 



<!-- Start of picture text -->
2468 2375 4915489280<br>0 30<br>10<br>1962 25 2003 20 867305<br>0 1937 50 1974 10 161694<br>75 0<br>10 1905 100 1932 10 53931<br>20<br>20 1832 125 1826 19135<br>30<br>150<br>-3760 -2566 40 0<br>ca 20 0 20 20 0 20 20 0 20<br>PLS component 1 PLS component 1 PLS component 1<br>(a) Birthyear (b) Death year (c) Population<br>8849 20 90.00 178.43<br>0 20<br>10 493 51.49 30.39<br>20<br>0<br>0 242 40 46.24 9.92<br>114 60 40.77 20 -0.35<br>10<br>80 40<br>30 34.58 -77.03<br>100<br>20<br>60<br>-213 120 -90.00 -178.00<br>20 0 20 0 50 100 0 50 100<br>PLS component 1 PLS component 1 PLS component 1<br>(d) Elevation (e) Latitude (f) Longitude<br>PLS component 2 PLS component 2 PLS component 2<br>PLS component 2 PLS component 2 PLS component 2<br><!-- End of picture text -->

Figure 9: PLS projections of Llama 2 7B entity representations. See explanation in Figure 3. 



<!-- Start of picture text -->
2468 2375 4915489280<br>30<br>20<br>10 1962 2003 20 867305<br>10<br>10<br>0 1937 0 1974 0 161694<br>10 1905 10 1932 10 53931<br>20 1832 20 1826 20 19135<br>30<br>30<br>30 -3760 -2566 0<br>aa 40 20 0 20 20 0 20 20 0 20<br>PLS component 1 PLS component 1 PLS component 1<br>(a) Birthyear (b) Death year (c) Population<br>15 8849 20 90.00 178.43<br>10 30<br>493 10 51.49 30.39<br>5 20<br>242 0 46.24 9.92<br>0 10<br>5 114 10 40.77 0 -0.35<br>10 20<br>30 34.58 10 -77.03<br>15 30<br>-213 -90.00 20 -178.00<br>1] 10 0 10 20 20 0 20 La 20 0<br>PLS component 1 PLS component 1 PLS component 1<br>(d) Elevation (e) Latitude (f) Longitude<br>PLS component 2 PLS component 2 PLS component 2<br>PLS component 2 PLS component 2 PLS component 2<br><!-- End of picture text -->

Figure 10: PLS projections of Falcon 7B entity representations. See explanation in Figure 3. 

21 

Preprint 



<!-- Start of picture text -->
2468 2375 4915489280<br>20<br>10 1962 2003 20 867305<br>10<br>0 1937 1974 161694<br>0 0<br>1905 1932 53931<br>10 10<br>20<br>1832 20 1826 19135<br>20<br>40<br>-3760 -2566 0<br>20 0 20 20 0 20 20 0 20<br>PLS component 1 PLS component 1 PLS component 1<br>(a) Birthyear (b) Death year (c) Population<br>8849 90.00 178.43<br>20 40<br>20<br>493 51.49 30 30.39<br>10 10<br>20<br>0 242 0 46.24 9.92<br>10<br>114 10 40.77 -0.35<br>10 0<br>20<br>20 30 34.58 10 -77.03<br>30<br>20<br>-213 -90.00 -178.00<br>10 0 10 20 30 20 10 0 10 20 40 20 0 20<br>PLS component 1 PLS component 1 PLS component 1<br>(d) Elevation (e) Latitude (f) Longitude<br>PLS component 2 PLS component 2 PLS component 2<br>PLS component 2 PLS component 2 PLS component 2<br><!-- End of picture text -->

Figure 11: PLS projections of Mistral 7B entity representations. See explanation in Figure 3. 

## **E Choice of probing and edit locus** 



<!-- Start of picture text -->
1.0<br>1.00 .00 .00 .00<br>0.8<br>0.80 .89 .46 .00<br>0.60 .94 .96 .87 0.6<br>0.40 .99 .99 .89 0.4<br>0.20 .99 .97 .90<br>0.2<br>0.00 .93 .97 .97<br>0.0<br>Layer (relative)<br>In hatw eary asw [mention last] born ? Correlation of edit strength and model output change<br><!-- End of picture text -->

Figure 12: Results of a cursory search for the best probing and edit locus, using Llama 2 7B. Varying token position and layer, we edit the hidden state at this locus as described in §3 and record the Spearman correlation between edit strength and the change in the quantity (here: birthyear) expressed by the model. Correlation is highest (0.99) in the region between layers 0.2 and 0.4 and the last subword token of the entity mention and the following token. Based on this, we choose the last mention token and the middle point at layer _l_ = 0.3 as locus for the regression experiments in §2 and activation patching experiments in §3, across all numeric properties and LMs, but acknowledge that a more exhaustive search would likely find better probing and edit loci. 

22 

Preprint 

## **F Edit curves for additional language models** 



<!-- Start of picture text -->
140 100 1e9<br>120 0 1.2<br>100 100 1.0<br>80 0.8<br>200<br>60 0.6<br>300<br>40 0.4<br>20 400 0.2<br>0 500 0.0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(a) Birthyear (b) Death year (c) Population<br>15000<br>10000<br>6000<br>10000 8000<br>6000 4000<br>5000<br>4000<br>2000<br>0 2000<br>5000 0 0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(d) Elevation (e) Latitude (f) Longitude<br>Model output change (years) Model output change (years) Model output change (people)<br>Model output change (meters) Model output change (degrees) Model output change (degrees)<br><!-- End of picture text -->

Figure 13: Effect of activation patching along property-specific directions across several numeric properties with Llama 2 7B. See explanation in Figure 4. 



<!-- Start of picture text -->
1e7<br>100 0 3.0<br>50 100 2.5<br>200 2.0<br>0<br>300 1.5<br>50<br>400 1.0<br>100 500 0.5<br>150 600 0.0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(a) Birthyear (b) Death year (c) Population<br>80 10 40<br>60 0 30<br>40 20<br>10<br>20 10<br>20 0<br>0<br>10<br>30<br>20<br>20<br>40 40 30<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(d) Elevation (e) Latitude (f) Longitude<br>Model output change (years) Model output change (years) Model output change (people)<br>Model output change (meters) Model output change (degrees) Model output change (degrees)<br><!-- End of picture text -->

Figure 14: Effect of activation patching along property-specific directions across several numeric properties with Falcon 7B (Almazrouei et al., 2023). See explanation in Figure 4. 

23 

Preprint 



<!-- Start of picture text -->
125 1e9<br>200<br>100<br>0 6<br>75<br>200<br>50<br>4<br>25 400<br>0 600 2<br>25 800<br>50 1000 0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(a) Birthyear (b) Death year (c) Population<br>7500 4<br>5000<br>5000<br>2 4000<br>2500<br>0 0 3000<br>2500 2000<br>5000 2<br>1000<br>7500<br>4<br>10000 0<br>1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0 1.0 0.5 0.0 0.5 1.0<br>Edit weight (scaled) Edit weight (scaled) Edit weight (scaled)<br>(d) Elevation (e) Latitude (f) Longitude<br>Model output change (years) Model output change (years) Model output change (people)<br>Model output change (meters) Model output change (degrees) Model output change (degrees)<br><!-- End of picture text -->

Figure 15: Effect of activation patching along property-specific directions across several numeric properties with Mistral 7B (Jiang et al., 2023). See explanation in Figure 4. 

## **G Software** 

The following is a list of the main libraries used in this work: 

- Numpy (Harris et al., 2020) 

- Scikit-learn (Pedregosa et al., 2011) 

- Pytorch (Paszke et al., 2019) 

- Transformers (Wolf et al., 2020) 

- seaborn (Waskom, 2021) 

- Matplotlib (Hunter, 2007) 

- SciPy (Virtanen et al., 2020) 

- Pandas (Pandas development team, 2020) 

We thank all authors and the open source community in general for creating and maintaining publicly and freely available software. 

24 

