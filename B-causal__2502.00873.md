# Language Models Use Trigonometry to Do Addition

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** B-causal — Kantamneni & Tegmark. The exemplar: a geometric representation claim (helix) validated by causal intervention, not by probe accuracy. The standard W3 would need to meet.
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/2502.00873.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

**Language Models Use Trigonometry to Do Addition** 

**Subhash Kantamneni**<sup>1</sup> **Max Tegmark**<sup>1</sup> 

# **Abstract** 

Mathematical reasoning is an increasingly important indicator of large language model (LLM) capabilities, yet we lack understanding of how LLMs process even simple mathematical tasks. To address this, we reverse engineer how three mid-sized LLMs compute addition. We first discover that numbers are represented in these LLMs as a generalized helix, which is strongly causally implicated for the tasks of addition and subtraction, and is also causally relevant for integer division, multiplication, and modular arithmetic. We then propose that LLMs compute addition by manipulating this generalized helix using the “Clock” algorithm: to solve _a_ + _b_ , the helices for _a_ and _b_ are manipulated to produce the _a_ + _b_ answer helix which is then read out to model logits. We model influential MLP outputs, attention head outputs, and even individual neuron preactivations with these helices and verify our understanding with causal interventions. By demonstrating that LLMs represent numbers on a helix and manipulate this helix to perform addition, we present the first representation-level explanation of an LLM’s mathematical capability. 

# **1. Introduction** 

Large language models (LLMs) display surprising and significant aptitude for mathematical reasoning (Ahn et al., 2024; Satpute et al., 2024), which is increasingly seen as a benchmark for LLM capabilities (OpenAI; Glazer et al., 2024). Despite LLMs’ mathematical proficiency, we have limited understanding of how LLMs process even simple mathematical tasks like addition. Understanding mathematical reasoning is valuable for ensuring LLMs’ reliability, interpretability, and alignment in high-stakes applications. 

In this study, we reverse engineer how GPT-J, Pythia-6.9B, and Llama3.1-8B compute the addition problem _a_ + _b_ for _a, b ∈_ [0 _,_ 99]. Remarkably, we find that LLMs use a form 

> 1Massachusetts Institute of Technology. Correspondence to: Subhash Kantamneni _<_ subhashk@mit.edu _>_ . 



<!-- Start of picture text -->
a +  b  =<br>Step 1: Embed  and  as a helix using modular circles  a b<br>( T = [2,5,10,100]) and a line<br>Step 2: Create helix( a +  b )<br>Step 3: Translate answer helix to logits<br>Toa 3+63 T=5 T = 10 Helix<br>3+63 3<br>3 3+63<br>SL Beaty<br>T 63 63 Sty<br>100 10 Feae<br>SS<br>“SS a<br>*<br>3+63 63 *<br>3+63 3<br>63 « on Ces<br>Abe<br>3 Sy.<br>63<br><!-- End of picture text -->

_Figure 1._ **Illustrating the Clock algorithm.** We find that LLMs represent numbers on a helix. When computing the addition problem _a_ + _b_ , LLMs rotate the _a_ and _b_ helices, as if on a clock, to create the _a_ + _b_ helix and read out the final answer. 

of the “Clock” algorithm to compute addition, which was previously proposed by Nanda et al. (2023a) as a mechanistic explanation of how one layer transformers compute modular addition (and later named by Zhong et al. (2023)). 

To compute _a_ + _b_ , all three LLMs represent _a_ and _b_ as a helix on their tokens and construct helix( _a_ + _b_ ) on the last token, which we verify with causal interventions. We then focus on how GPT-J implements the Clock algorithm by investigating MLPs, attention heads, and even specific neurons. We find that these components can be understood as either constructing the _a_ + _b_ helix by manipulating the _a_ and _b_ helices, or using the _a_ + _b_ helix to produce the answer in the model’s logits. We visualize this procedure in Fig. 1 as rotating the dial of a clock. 

Our work is in the spirit of mechanistic interpretability (MI), which attempts to reverse engineer the functionality of machine learning models. However, most LLM MI research focuses either on identifying circuits, which are the minimal set of model components required for computations, 

1 

**Language Models Use Trigonometry to Do Addition** 

or understanding features, which are the representations of concepts in LLMs. A true mechanistic explanation requires understanding both how an LLM represents a feature and how downstream components manipulate that feature to complete a task. To our knowledge, we are the first work to present this type of description of an LLM’s mathematical capability, identifying that LLMs represent numbers as helices and compute addition by manipulating these helices with the interpretable Clock algorithm. 

Stolfo et al. (2023) identifies the circuit used by LLMs in addition problems, and Nikankin et al. (2024) claims that LLMs use heuristics implemented by specific neurons rather than a definite algorithm to compute arithmetic. Zhou et al. (2024) analyze a fine-tuned GPT-2 and found that Fourier components in numerical representations are critical for addition, while providing preliminary results that larger base LLMs might use similar features. 

# **3. Problem Setup** 

# **2. Related Work** 

**Circuits.** Within mechanistic interpretability, circuits research attempts to understand the key model components (MLPs and attention heads) that are required for specific functionalities (Olah et al., 2020; Elhage et al., 2021). For example, Olsson et al. (2022) found that in-context learning is primarily driven by induction attention heads, and Wang et al. (2023) identified a sparse circuit of attention heads that GPT-2 uses to complete the indirect object of a sentence. Understanding how multilayer perceptrons (MLPs) affect model computation has been more challenging, with Nanda et al. (2023b) attempting to understand how MLPs are used in factual recall, and Hanna et al. (2023) investigating MLP outputs while studying the greater-than operation in GPT-2. 

**Features.** Another branch of MI focuses on understanding how models represent human-interpretable concepts, known as features. Most notably, the Linear Representation Hypothesis posits that LLMs store features as linear directions (Park et al., 2023; Elhage et al., 2022), culminating in the introduction of sparse autoencoders (SAEs) that decompose model activations into sparse linear combinations of features (Huben et al., 2024; Bricken et al., 2023; Templeton et al., 2024; Gao et al., 2024; Rajamanoharan et al., 2024). However, recent work from Engels et al. (2024) found that some features are represented as non-linear manifolds, for example the days of the week lie on a circle. Levy & Geva (2024) and Zhu et al. (2025) model LLMs’ representations of numbers as a circle in base 10 and as a line respectively, although with limited causal results. Recent work has bridged features and circuits research, with Marks et al. (2024) constructing circuits from SAE features and Makelov et al. (2024) using attention-based SAEs to identify the features used in Wang et al. (2023)’s IOI task. 

**Reverse engineering addition.** Liu et al. (2022) first discovered that one layer transformers generalize on the task of modular addition when they learn circular representations of numbers. Following this, Nanda et al. (2023a) introduced the “Clock” algorithm as a description of the underlying angular addition mechanisms these transformers use to generalize. However, Zhong et al. (2023) found the “Pizza” algorithm as a rivaling explanation for some transformers, illustrating the complexity of decoding even small models. 

**Models** As in Nikankin et al. (2024), we analyze 3 LLMs: GPT-J (6B parameters) (Wang & Komatsuzaki, 2021), Pythia-6.9B (Biderman et al., 2023), and Llama3.1-8B (Grattafiori et al., 2024). All three models are autoregressive transformers which process tokens _x_ 0 _, ..., xn_ to produce probability distributions over the likely next token _xn_ +1 (Vaswani et al., 2017). The _i_ th token is embedded as _L_ hidden state vectors (also known as the residual stream), where _L_ is the number of layers in the transformer. Each hidden state is the sum of multilayer perceptron (MLP) and attention (attn) layers. 



GPT-J and Pythia-6.9B use simple MLP implementations, namely MLP( _x_ ) = _σ_ ( _xW_ up) _W_ down, where _σ_ ( _x_ ) is the sigmoid function. Llama3.1-8B uses a gated MLP, MLP( _x_ ) = _σ_ ( _xW_ gate) _◦_ ( _xW_ in) _W_ out, where _◦_ represents the Hadamard product (Liu et al., 2021). GPT-J tokenizes the numbers [0 _,_ 361] (with a space) as a single token, Pythia6.9B tokenizes [0 _,_ 557] as a single token, and Llama3.1-8B tokenizes [0 _,_ 999] as a single token. We focus on the singletoken regime for simplicity. 

**Data** To ensure that answers require only a single token for all models, we construct problems _a_ + _b_ for integers _a, b ∈_ [0 _,_ 99]. We evaluate all three models on these 10,000 addition problems, and find that all models can competently complete the task: GPT-J achieves 80.5% accuracy, Pythia6.9B achieves 77.2% accuracy, and Llama3.1-8B achieves 98.0% accuracy. For the prompts used and each model’s performance heatmap by _a_ and _b_ , see Appendix A. Despite Llama3.1-8B’s impressive performance, in the main paper we focus our analysis on GPT-J because its simple MLP allows for easier neuron interpretation. We report similar results for Pythia-6.9B and Llama3.1-8B in the Appendix. 

2 

**Language Models Use Trigonometry to Do Addition** 



_Figure 2._ **Number representations are both periodic and linear.** _Top_ The residual stream after layer 0 in GPT-J is sparse in the Fourier domain when batching the hidden states for _a ∈_ [0 _,_ 360] together. We average the magnitude of the Fourier transform of the batched matrix _h_<sup>0</sup> 360<sup>across the model dimension.</sup><sup>_Bottom_In</sup> addition to this periodicity, the first PCA component is roughly linear in _a_ for _a ∈_ [0 _,_ 99]. 

Additionally, when we compare the residual streams of all pairs of integers _a_ 1 and _a_ 2, we see that there is distinct periodicity in both their Euclidean distance and cosine similarity (Fig. 13, Appendix B). These Fourier features were also identified by Zhou et al. (2024), and although initially surprising, are sensible. The units digit of numbers in base 10 is periodic ( _T_ = 10), and it is reasonable that qualities like evenness ( _T_ = 2) are useful for tasks. 

## **4.2. Parameterizing the Structure as a Helix** 

To account for both the periodic and linear structure in numbers, we propose that numbers can be modeled helically. Namely, we posit that _h_<sup>_l_</sup> _a_<sup>, the residual stream immediately</sup> preceding layer _l_ for some number _a_ , can be modeled as 



# **4. LLMs Represent Numbers as a Helix** 

To generate a ground up understanding of how LLMs compute _a_ + _b_ , we first aim to understand how LLMs represent numbers. To identify representational trends, we run GPT-J on the single-token integers _a ∈_ [0 _,_ 360]. We do not use _a_ = 361 because 360 has more integer divisors, allowing for a simpler analysis of periodic structure. We conduct analysis on _h_<sup>0</sup> 360<sup>,whichistheresidualstreamfollowing</sup> layer 0 with shape [360 _,_ model ~~d~~ im]. We choose to use the output of layer 0 rather than directly analyzing the embeddings because prior work has shown that processing in layer 0 is influential for numerical tasks (Nikankin et al., 2024). 

_C_ is a matrix applied to the basis of functions _B_ ( _a_ ), where _B_ ( _a_ ) uses _k_ Fourier features with periods _T_ = [ _T_ 1 _, . . . Tk_ ]. The _k_ = 1 case represents a regular helix; for _k >_ 1, the independent Fourier features share a single linear direction. We refer to this structure as a generalized helix, or simply a helix for brevity. 

We identify four major Fourier features: _T_ = [2 _,_ 5 _,_ 10 _,_ 100]. We use the periods _T_ = [2 _,_ 5 _,_ 10] because they have significant high frequency components in Fig. 2. We are cautious of low frequency Fourier components, and use _T_ = 100 both because of its significant magnitude, and by applying the inductive bias that our number system is base 10. 

## **4.1. Investigating Numerical Structure** 

**Linear structure** . To investigate structure in numerical representations, we perform a PCA (F.R.S., 1901) on _h_<sup>0</sup> 360<sup>and</sup> find that the first principal component (PC1) for _a ∈_ [0 _,_ 360] has a sharp discontinuity at _a_ = 100 (Fig. 15, Appendix B), which implies that GPT-J uses a distinct representation for three-digit integers. Instead, in the bottom of Fig. 2, we plot PC1 for _h_<sup>0</sup> 99<sup>and find that it is well approximated by a line</sup> in _a_ . Additionally, when plotting the Euclidean distance between _a_ and _a_ + _δn_ for _a ∈_ [0 _,_ 9] (Fig. 14, Appendix B), we see that the distance is locally linear. The existence of linear structure is unsurprising - numbers are semantically linear, and LLMs often represent concepts linearly. 

**Periodic Structure.** We center and apply a Fourier transform to _h_<sup>0</sup> 360<sup>with respect to the number</sup><sup>_a_being represented</sup> and the model ~~d~~ im. In Fig. 2, we average the resulting spectra across model ~~d~~ im and observe a sparse Fourier domain with high-frequency components at _T_ = [2 _,_ 5 _,_ 10]. 

## **4.3. Fitting a Helix** 

We fit our helical form to the residual streams on top of the _a_ token for our _a_ + _b_ dataset. In practice, we first use PCA to project the residual stream at each layer to 100 dimensions. To ensure we do not overfit with Fourier features, we consider all combinations of _k_ Fourier features, with _k ∈_ [1 _,_ 4]. If we use _k_ Fourier features, the helical fit uses 2 _k_ + 1 basis functions (one linear component, 2 _k_ periodic components). We then use linear regression to find some coefficient matrix _C_ PCA of shape 100 _×_ 2 _k_ + 1 that best satisfies PCA( _h_<sup>_l_</sup> _a_<sup>) =</sup><sup>_C_PCA</sup><sup>_B_(</sup><sup>_a_)</sup><sup>_T_.Finally, we use the</sup> inverse PCA transformation to project _C_ PCA back into the model’s full residual stream dimensionality to find _C_ . 

We visualize the quality of our fit for layer 0 when using all _k_ = 4 Fourier features with _T_ = [2 _,_ 5 _,_ 10 _,_ 100] in Fig. 3. To do so, we calculate _C_<sup>_†_</sup> _h_ , where _C_<sup>_†_</sup> is the MoorePenrose pseudo-inverse of _C_ . Thus, _C_<sup>_†_</sup> _h_ represents the 

3 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
25<br>1 cos(a|T = 2)sin(a|T = 2) 1 3 2 35 30 20 15<br>10<br>4 1 40<br>2<br>5 0<br>cos(a|T = 5) 0 5 cos(a|T = 10) 0 50 cos(a|T = 100)<br>sin(a|T = 5) sin(a|T = 10) sin(a|T = 100)<br>55 95<br>0 3 90<br>4 6 7 8 9 60 6570 7580 85<br>0a 5 10 15 20 25 30 35 40 50 55 60 65 70 75 80 85 90<br>a<br><!-- End of picture text -->

_Figure 3._ **Helix subspace visualized.** For GPT-J’s layer 0 output, we project each of the numbers _a ∈_ [0 _,_ 99] onto our fitted _T_ = [2 _,_ 5 _,_ 10 _,_ 100] helix subspace, and visualize it. In the top row, we plot sin(<sup><u>2</u></sup> _T_<sup>_<u>π</u>_</sup> _i_<sup>_a_) vs cos(</sup><sup><u>2</u></sup> _T_<sup>_<u>π</u>_</sup> _i_<sup>_a_) for each</sup><sup>_Ti∈T_and plot all</sup><sup>_a_congruent</sup> under _a_ mod _T_ in the same color and annotate their mean. The bottom row contains the linear component subplot. 

projection of the residual stream into the helical subspace. When analyzing the columns of _C_ , we find that the Fourier features increase in magnitude with period and are mostly orthogonal (Appendix C.1). 



<!-- Start of picture text -->
8 Patch layer<br>9dim PCA patch<br>6 Helix: [2, 5, 10, 100]<br>Circle: [2, 5, 10, 100]<br>9th Deg Polynomial<br>4 7 Params<br>Helix: [5, 10, 100]<br>2 5 Params<br>Helix: [10, 100]<br>3 Params<br>0 Helix: [100]<br>0 5 10 15 20 25<br>Layer<br>Logit Difference<br><!-- End of picture text -->

_Figure 4._ **Helix causal intervention results.** We use activation patching to causally determine if our fits preserve the information the model uses to compute _a_ + _b_ . We find that our helical and circular fits are strongly causally implicated, often outperforming the PCA baseline. 

## **4.4. Evaluating the Quality of the Helical Fit** 

We want to causally demonstrate that the model actually uses the fitted helix. To do so, we employ activation patching. Activation patching isolates the contribution of specific model components towards answer tokens (Meng et al., 2022; Heimersheim & Nanda, 2024). Specifically, to evaluate the contribution of some residual stream _h_<sup>_l_</sup> _a_ on the _a_ token, we first store _h_<sup>_l_</sup> _a,_ clean<sup>whenthemodelis</sup> run on a “clean” prompt _a_ + _b_ . We then run the model on the corrupted prompt _a_<sup>_′_</sup> + _b_ and store the model logits for the clean answer of _a_ + _b_ . Finally, we patch in the clean _h_<sup>_l_</sup> _a,_ clean<sup>on the corrupted prompt</sup><sup>_a′_+</sup><sup>_b_and calculate</sup> _LDa_<sup>_l_= logitpatched(</sup><sup>_a_+</sup><sup>_b_)</sup><sup>_−_logitcorrupted(</sup><sup>_a_+</sup><sup>_b_), where</sup> 

_LDa_<sup>_l_is the logit difference for</sup><sup>_hl_</sup> _a_<sup>.By averaging over 100</sup> pairs of clean and corrupted prompts, we can evaluate _h_<sup>_l_</sup> _a_<sup>’s</sup> ability to restore model behavior to the clean answer _a_ + _b_ . To reduce noise, all patching experiments only use prompts the model can successfully complete. 

To leverage this technique, we follow Engels et al. (2024) and input our fit for _h_<sup>_l_</sup> _a,_ clean<sup>whenpatching.Thisallows</sup> us to causally determine if our fit preserves the information the model uses for the computation. We compare our _k_ Fourier feature helical fit with four baselines: using the actual _h_<sup>_l_</sup> _a,clean_<sup>(layerpatch),thefirst2</sup><sup>_k_+1PCAcom-</sup> ponents of _h_<sup>_l_</sup> _a,clean_<sup>(PCA),acircularfitwith</sup><sup>_k_Fourier</sup> components (circle), and a polynomial fit with basis terms _B_ ( _a_ ) = [ _a, a_<sup>2</sup> _, ...a_<sup>2</sup><sup>_k_+1</sup> ] (polynomial). For each value of _k_ , we choose the combination of Fourier features that maximizes _L_<sup><u>1</u></sup> � _l_<sup>_LD_</sup> _a_<sup>_l_as the best set of Fourier features.</sup> 

In Fig. 4, we see that the helical fit is most performant against baselines, closely followed by the circular fit. This implies that Fourier features are predominantly used to compute addition. Surprisingly, the _k_ = 4 full helical and circular fits dominate the strong PCA baseline and approach the effect of layer patching, which suggests that we have identified the correct “variables” of computation for addition. Additionally, we note a sharp jump between the fit for layer 0’s input (the output of the embedding) and layer 1’s input, aligning with evidence from Nikankin et al. (2024) that layer 0 is necessary for numerical processing. 

In Appendix C.2, we provide evidence that Llama3.1-8B and Pythia-6.9B also use helical numerical representations. Additionally, we provide evidence that our fits are not overfitting by using a train-test split with no meaningful effect on our results. The helix functional form is not overly expressive, as a helix trained on a randomized order of _a_ is not causally relevant. We also observe continuity when values of _a_ that the helix was not trained on are projected into the 

4 

**Language Models Use Trigonometry to Do Addition** 

_Table 1._ **Performance of fits across tasks.** We calculate max _l LDa_<sup>_l_for each fit across a variety of numerical tasks.While</sup> the helix fit is competitive, we find that it underperforms the PCA baseline on three tasks. 

|**Task**|**Full**|**PCA**|**Helix**|**Circle**|**Poly**|
|---|---|---|---|---|---|
|_a_+_b_=|**8.34**|6.13|**7.21**|6.83|3.09|
|_a −_23 =|**7.45**|6.16|**7.05**|6.52|2.93|
|_a//_5 =|**5.48**|**5.24**|4.55|4.25|4.55|
|_a ∗_1_._5 =|**7.86**|**6.65**|5.16|4.85|4.84|
|_a_ mod 2 =|**0.98**|1.00|1.11|**1.12**|0.88|
|_x −a_= 0**,**_x_=|**6.70**|**4.77**|4.56|4.34|2.98|



helical subspace. This satisfies the definition of a nonlinear feature manifold proposed by Olah & Jermyn (2024), and provides additional evidence for the argument of Engels et al. (2024) against the strongest form of the Linear Representation Hypothesis. 

## **4.5. Is the Helix the Full Picture?** 

To identify if the helix sufficiently explains the structure of numbers in LLMs, we test on five additional tasks. 



<!-- Start of picture text -->
Layer Patch<br>7.5 27dim PCA<br>9dim PCA<br>5.0 helix(a,b,a+b)<br>helix(a+b)<br>2.5 helix(a,b)<br>0.0<br>0 5 10 15 20 25<br>Layer<br>Logit Difference<br><!-- End of picture text -->

_Figure 5._ **Last token hidden states are well-modeled by** helix( _a_ + _b_ ) **.** We use activation patching to show that _h_ =<sup>_l_for</sup> GPT-J is well modeled by helices, in particular helix( _a_ + _b_ ). 

# **5. LLMs Use the Clock Algorithm to Compute Addition** 

## **5.1. Introducing the Clock Algorithm** 

Taking inspiration from Nanda et al. (2023a), we propose that LLMs manipulate helices to compute addition using the “Clock” algorithm. 

To compute _a_ + _b_ =, GPT-J 

   1. Embeds _a_ and _b_ as helices on their own tokens. 

1. _a −_ 23 for _a ∈_ [23 _,_ 99] 

2. _a//_ 5 (integer division) for _a ∈_ [0 _,_ 99] 

3. _a ∗_ 1 _._ 5 for even _a ∈_ [0 _,_ 98] 

4. _a_ mod 2 for _a ∈_ [0 _,_ 99] 

5. If _x − a_ = 0, what is _x_ = for _a ∈_ [0 _,_ 99] 

For each task, we fit full helices with _T_ = [2 _,_ 5 _,_ 10 _,_ 100] and compare against baselines. In Table 1, we describe our results on these tasks by listing max _l LDa_<sup>_l_,whichisthe</sup> maximal causal power of each fit (full plot and additional task details in Appendix C.2). Notably, while the helix is causally relevant for all tasks, we see that it underperforms the PCA baseline on tasks 2, 3, and 5. This implies that there is potentially additional structure in numerical representations that helical fits do not capture. However, we are confident that the helix is used for addition. When ablating the helix dimensions from the residual stream (i.e. ablating _C_<sup>_†_</sup> from _h_<sup>_l_</sup> _a_<sup>), performance is affected roughly as much as</sup> ablating _h_<sup>_l_</sup> _a_<sup>entirely (Fig.21, Appendix C.1).</sup> 

Thus, we conclude that LLMs use a helical representation of numbers to compute addition, although it is possible that additional structure is used for other tasks. 

2. A sparse set of attention heads, mostly in layers 9-14, move the _a_ and _b_ helices to the last token. 

3. MLPs 14-18 manipulate these helices to create the helix _a_ + _b_ . A small number of attention heads help with this operation. 

4. MLPs 19-27 and a few attention heads “read” from the _a_ + _b_ helix and output to model logits. 

Since we have already shown that models represent _a_ and _b_ as helices (Appendix C.2), we provide evidence for the last three steps in this section. In Fig. 5 we observe that last token hidden states are well modeled by _h_<sup>_l_</sup> =<sup>=helix(</sup><sup>_a, b, a_+</sup><sup>_b_),wherehelix(</sup><sup>_x, y_)isshorthand</sup> to denote helix( _x_ ) + helix( _y_ ). Remarkably, despite only using 9 parameters, at some layers helix( _a_ + _b_ ) fits last token hidden states better than a 27 dimensional PCA. The _a_ + _b_ helix having such causal power implies it is at the heart of the computation. 

In Appendix D, we show that other LLMs also use helix( _a_ + _b_ ). Since the crux of the Clock algorithm is computing the answer helix for _a_ + _b_ , we take this as compelling evidence that all three models use the Clock algorithm. However, we would like to understand how specific LLM components implement the algorithm. To do so, we focus on GPT-J. 

In Fig. 6, we use activation patching to determine which last token MLP and attention layers are most influential for 

5 

**Language Models Use Trigonometry to Do Addition** 

the final result. We also present path patching results, which isolates how much components directly contribute to logits. For example, MLP18’s total effect (TE, activation patching) includes both its indirect effect (IE), or how MLP18’s output is used by downstream components like MLP19, and its direct effect (DE, path patching), or how much MLP18 directly boosts the answer logit.<sup>1</sup> In Fig. 6, we see that MLPs dominate direct effect. 



<!-- Start of picture text -->
3 MLP TE<br>MLP DE. Total = 8.20<br>2 Attention TE<br>Attention DE. Total = 1.20<br>1<br>0<br>0 5 10 15 20 25<br>Layer<br>Logit Difference<br><!-- End of picture text -->

_Figure 6._ **MLPs drive computation of** _a_ + _b_ **.** By using activation and path patching, we find that MLPs are most implicated in constructing the final answer, along with early attention layers. 

We now investigate specific attention heads, MLPs, and individual neurons. 

## **5.2. Investigating Attention Heads** 

In GPT-J, every attention layer is the sum of 16 attention heads whose outputs are concatenated. We activation and path patch each attention head on the last token and rank them by total effect (TE). To determine the minimal set of attention heads required, we activation patch _k_ attention heads at once, and find the minimum _k_ such that their combined total effect approximates patching in all attention heads. In Appendix D.1, we see that patching _k_ = 17 heads achieves 80% of the effect of patching in all 448 attention heads, and we choose to round up to _k_ = 20 heads (83.9% of effect). 

Since attention heads are not as influential as MLPs in Fig. 6, we hypothesize that they primarily serve two roles: 1) moving the _a, b_ helices to the last token to be processed by downstream components ( _a, b_ heads) and 2) outputting the _a_ + _b_ helix directly to logits ( _a_ + _b_ heads). Some mixed heads output all three _a, b,_ and _a_ + _b_ helices. We aim to categorize as few attention heads as mixed as possible. 

To categorize attention heads, we turn to two metrics. _ca,b_ is the confidence that a certain head is an _a, b_ head, which we quantify with _ca,b_ = (1 _−_<sup><u>DE</u></sup> TE<sup>)</sup> helix(helix _a,b,a_ <u>(</u> _a,b_ +) _b_ )<sup>.</sup> The first term represents the fractional indirect effect of the attention head, and the second term represents the head’s total effect recoverable by just using the _a, b_ he- 

> 1For more on path patching, we refer readers to GoldowskyDill et al. (2023); Wang et al. (2023) 

lices instead of helix( _a, b, a_ + _b_ ). Similarly, we calculate _ca_ + _b_ as the confidence the head is an _a_ + _b_ head, using _ca_ + _b_ =<sup><u>DE</u></sup> TE helix(helix _a,b,a_ <u>(</u> _a_ <u>+</u> _b_ +) _b_ )<sup>.</sup> 

We sort the _k_ = 20 heads by _c_ = max( _ca,b, ca_ + _b_ ). If a head is an _a_ + _b_ head, we model its output using the _a_ + _b_ helix and allow it only to output to logits (no impact on downstream components). If a head is an _a, b_ head, we restrict it to outputting helix( _a, b_ ). For _m_ = [1 _,_ 20], we allow _m_ heads with the lowest _c_ to be mixed heads, and categorize the rest as _a, b_ or _a_ + _b_ heads. We find that categorizing _m_ = 4 heads as mixed is sufficient to achieve almost 80% of the effect of using the actual outputs of all _k_ = 20 heads. Thus, most important attention heads obey our categorization. We list some properties of each head type below. 

- _a, b_ heads (11/20): In layers 9-14 (but two heads in _l_ = 16 _,_ 18), attend to the _a, b_ tokens, and output _a, b_ helices which are used mostly by downstream MLPs. 

- _a_ + _b_ heads (5/20): In layers 24-26 (but one head in layer 19), attend to the last token, take their input from preceding MLPs, and output the _a_ + _b_ helix to logits. 

- Mixed heads (4/20): In layers 15-18, attend to the _a, b,_ and _a_ + _b_ tokens, receive input from _a, b_ attention heads and previous MLPs, and output the _a, b,_ and _a_ + _b_ helices to downstream MLPs. 

For evidence of these properties refer to Appendix D.1. Notably, only mixed heads are potentially involved in creating the _a_ + _b_ helix, which is the crux of the computation, justifying our conclusion from Fig. 6 that MLPs drive addition. 

## **5.3. Looking at MLPs** 

GPT-J seems to predominantly rely on last token MLPs to compute _a_ + _b_ . To identify which MLPs are most important, we first sort MLPs by total effect, and patch in _k_ = [1 _, L_ = 28] MLPs to find the smallest _k_ such that we achieve 95% of the effect of patching in all _L_ MLPs. We use a sharper 95% threshold because MLPs dominate computation and because there are so few of them. Thus, we use _k_ = 11 MLPs in our circuit, specifically MLPs 14-27, with the exception of MLPs 15, 24, and 25 (see Appendix D.2 for details). 

We hypothesize that MLPs serve two functions: 1) reading from the _a, b_ helices to create the _a_ + _b_ helix and 2) reading from the _a_ + _b_ helix to output the answer in model logits. We make this distinction using two metrics: helix( _a_ + _b_ )/TE, or the total effect of the MLP recoverable from modeling its output with helix( _a_ + _b_ ), and DE/TE ratio. In Fig. 7, we see that the outputs of MLPs 14-18 are progressively better modeled using helix( _a_ + _b_ ). Most of their effect is indirect and thus their output is predominantly used by downstream components. At layer 19, helix( _a_ + _b_ ) becomes a worse 

6 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
1.00<br>0.75<br>helix(a+b)<br>0.50 DE<br>0.25<br>Layer<br>14 16 18 20 22 24 26<br>Fraction of TE<br><!-- End of picture text -->

_Figure 7._ **Two stages of circuit MLPs.** MLPs 14-18’s outputs are well modeled by helix( _a_ + _b_ ), suggesting that they build this helical representation, while MLPs 19-27 higher DE/TE ratio implies they output the answer to model logits. 

fit and more MLP output affects answer logits directly. We interpret this as MLPs 14-18 “building” the _a_ + _b_ helix, which MLPs 19-27 translate to the answer token _a_ + _b_ . 

However, our MLP analysis has focused solely on MLP outputs. To demonstrate the Clock algorithm conclusively, we must look at MLP inputs. Recall that GPT-J uses a simple MLP: MLP( _x_ ) = _σ_ ( _xW_ up) _W_ down. _x_ is a vector of size (4096 _,_ ) representing the residual stream, and _W_ up is a (4096 _,_ 16384) projection matrix. The input to the MLP is thus the 16384 dimensional _xW_ up. We denote the _n_ th dimension of the MLP input as the _n_ th neuron preactivation, and move to analyze these preactivations. 

## **5.4. Zooming in on Neurons** 

Activation patching the 27 _∗_ 16384 neurons in GPT-J is prohibitively expensive, so we instead use the technique of attribution patching to approximate the total effect of each neuron using its gradient (see Kramar´ et al. (2024)). We find that using just 1% of the neurons in GPT-J and mean ablating the rest allows for the successful completion of 80% of prompts (see Appendix D.2). Thus, we focus our analysis on this sparse set of _k_ = 4587 neurons. 

## 5.4.1. MODELING NEURON PREACTIVATIONS 

For a prompt _a_ + _b_ , we denote the preactivation of the _n_ th neuron in layer _l_ as _Nn_<sup>_l_(</sup><sup>_a, b_).When we plot a heatmap of</sup> _Nn_<sup>_l_(</sup><sup>_a, b_) for top neurons in Fig.8, we see that their preac-</sup> tivations are periodic in _a, b_ , and _a_ + _b_ . When we Fourier decompose the preactivations as a function of _a_ + _b_ , we find that the most common periods are _T_ = [2 _,_ 5 _,_ 10 _,_ 100], matching those used in our helix parameterization (Appendix D.2). This is sensible, as the _n_ th neuron in a layer applies _Wup_<sup>_n_of shape (4096</sup><sup>_,_) to the residual stream, which</sup> we have effectively modeled as a helix( _a, b, a_ + _b_ ). Subsequently, we model the preactivation of each top neuron as 



For each neuron preactivation, we fit the parameters _c_ and _d_ in Eq. 3 using gradient descent (see Appendix D.2 for details). In Fig. 8, we show the highest magnitude fit component for a selection of top neurons. 

We evaluate our fit of the top _k_ neurons by patching them into the model, mean ablating all other neurons, and measuring the resulting accuracy of the model. In Fig. 9, we see that our neuron fits provide roughly 75% of the performance of using the actual neuron preactivations. Thus, these neurons are well modeled as reading from the helix. 

## 5.4.2. UNDERSTANDING MLP INPUTS 

We use our understanding of neuron preactivations to draw conclusions about MLP inputs. To do so, we first path patch each of the top _k_ neurons to find their direct effect and calculate their DE/TE ratio. For each neuron, we calculate the fraction of their fit that helix( _a_ + _b_ ) explains, which we approximate by dividing the magnitude of _cT,a_ + _b_ terms by the total magnitude of _cT t_ terms in Eq. 3. For each circuit MLP, we calculate the mean of both of these quantities across top neurons, and visualize them in Fig. 10. 

Once again, we see a split at layer 19, where earlier neurons’ preactivation fits rely on _a, b_ terms, while later neurons use _a_ + _b_ terms and write to logits. Since the neuron preactivations represent what each MLP is “reading” from, we combine this result with our evidence from Section 5.3 to summarize the role of MLPs in addition. 

- MLPs 14-18 primarily read from the _a, b_ helices to create the _a_ + _b_ helix for downstream processing. 

- MLPs 19-27 primarily read from the _a_ + _b_ helix to write the answer to model logits. 

Thus, we conclude our case that LLMs use the Clock algorithm to do addition, with a deep investigation into how GPT-J implements this algorithm. 

## **5.5. Limitations of Our Understanding** 

There are several aspects of LLM addition we still do not understand. Most notably, while we provide compelling evidence that key components create helix( _a_ + _b_ ) from helix( _a, b_ ), we do not know the exact mechanism they use to do so. We hypothesize that LLMs use trigonometric identities like cos( _a_ + _b_ ) = cos( _a_ ) cos( _b_ ) _−_ sin( _a_ ) sin( _b_ ) to create helix( _a_ + _b_ ). However, like the originator of the Clock algorithm Nanda et al. (2023a), we are unable to isolate this computation in the model. This is unsurprising, as 

7 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
#0.L20N7741. TE: 0.24 #7.L21N13478. TE: 0.11 #4.L22N12504. TE: 0.14 #10.L22N9339. TE: 0.10<br>-5.0cos(2 /2(a+b+83.0)) -4.4cos(2 /10(a+b-1.7))+ 1.7cos(2 /100(a+b+11.6)) 1.31b+1.18a<br>NRMSE: 0.06 -3.8cos(2 /5(a+b+0.9)) NRMSE: 0.11 NRMSE: 0.10<br>80 NRMSE: 0.08<br>60<br>40<br>20<br>0<br>0 20 40 60 80 0 20 40 60 80 0 20 40 60 80 0 20 40 60 80<br>b b b b<br>a<br><!-- End of picture text -->

_Figure 8._ **Neuron preactivations and fits.** We visualize the preactivations _Nn_<sup>_l_</sup> in its preactivations, which we model using a helix inspired functional form. 

_n_<sup>_l_(</sup><sup>_a, b_) for four top neurons.Each neuron has clear periodicity</sup> 



<!-- Start of picture text -->
1.00<br>Helix Fitting Neuron Preactivations<br>0.75<br>Actual Neuron Preactivations<br>0.50<br>0.25<br>0.00<br>10 0 10 1 10 2 10 3<br>Number of Neurons Kept (Ablating Rest)<br>Accuracy<br><!-- End of picture text -->

_Figure 9._ **Evaluating neuron fits.** Patching in our fitted preactivations for the top _k_ neurons is roughly as effective as patching in their actual preactivations and ablating all other neurons. 



<!-- Start of picture text -->
0.6<br>2322 26 27 25<br>21<br>0.5<br>20<br>20<br>0.4 19<br>Mean of MLP's Top Neurons<br>0.3 14 1617 18 R² = 0.788<br>15<br>0.0 0.2 0.4 0.6 0.8<br>DE/TE Ratio<br>Layer<br>|a+b| Fraction<br><!-- End of picture text -->

_Figure 10._ **Neuron trends.** Neurons in MLPs 14-18 primarily read from the _a, b_ helices, while MLPs 19-27 primarily read from the _a_ + _b_ helix and write to logits. 

# **6. Conclusion** 

there is a large solution space for how models choose to implement low-level details of algorithms. For example, Yip et al. (2024) finds that in Zhong et al. (2023)’s “Pizza” algorithm for modular addition, MLPs in one layer transformers implement numerical integration techniques to transform cos(<sup>_<u>k</u>_</sup> 2<sup>(</sup><sup>_a_+</sup><sup>_b_)) to cos(</sup><sup>_k_(</sup><sup>_a_+</sup><sup>_b_)).</sup> 

The mere existence of the Pizza algorithm demonstrates that even one layer transformers have a complex solution space. Thus, even if the Clock algorithm is used by LLMs, it could be one method of an ensemble. We see evidence of this in Appendix D, in that helix( _a_ + _b_ ) is less causally implicated for Llama3.1-8B than other models, which we hypothesize is due to its use of gated MLPs. Additionally, other models must necessarily use modified algorithms for addition because of different tokenization schemes. For example, Gemma-2-9B (Riviere et al., 2024) tokenizes each digit of a number separately and must use additional algorithms to collate digit tokens. Additionally, at different scales LLMs potentially learn different algorithms, providing another reason to be skeptical that the Clock algorithm is the one and only explanation for LLM addition. 

We find that three mid-sized LLMs represent numbers as generalized helices and manipulate them using the interpretable Clock algorithm to compute addition. While LLMs could do addition linearly, we conjecture that LLMs use the Clock algorithm to improve accuracy, analogous to humans using decimal digits (which are a generalized helix with _T_ = [10 _,_ 100 _, . . ._ ]) for addition rather than slide rules. In Appendix E, we present preliminary results that GPTJ would be considerably less accurate on “linear addition” due to noise in its linear representations. Future work could analyze if LLMs have internal error-correcting codes for addition like the grid cells presented in Zlokapa et al. (2024). 

The use of the Clock algorithm provides striking evidence that LLMs trained on general text naturally learn to implement complex mathematical algorithms. Understanding LLM algorithms is important for safe AI and can also provide valuable insight into model errors, as shown in Appendix F. We hope that this work inspires additional investigations into LLM mathematical capabilities, especially as addition is implicit to many reasoning problems. 

8 

**Language Models Use Trigonometry to Do Addition** 

# **Acknowledgments** 

We thank Josh Engels for participating in extensive conversations throughout the project. We also thank Vedang Lad, Neel Nanda, Ziming Liu, David Baek, and Eric Michaud for their helpful suggestions. This work is supported by the Rothberg Family Fund for Cognitive Science and IAIFI through NSF grant PHY-2019786. 

# **References** 

- Ahn, J., Verma, R., Lou, R., Liu, D., Zhang, R., and Yin, W. Large language models for mathematical reasoning: Progresses and challenges, 2024. URL https: //arxiv.org/abs/2402.00157. 

- Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In _International Conference on Machine Learning_ , pp. 2397–2430. PMLR, 2023. 

- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. _Transformer Circuits Thread_ , 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html. 

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., HatfieldDodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. https://transformercircuits.pub/2021/framework/index.html. 

- Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., and Olah, C. Toy models of superposition. _Transformer Circuits Thread_ , 2022. URL https://transformer-circuits. pub/2022/toy_model/index.html. 

- Engels, J., Michaud, E. J., Liao, I., Gurnee, W., and Tegmark, M. Not all language model features are linear, 2024. URL https://arxiv.org/abs/2405. 14860. 

- Fiotto-Kaufman, J., Loftus, A. R., Todd, E., Brinkmann, J., Juang, C., Pal, K., Rager, C., Mueller, A., Marks, S., Sharma, A. S., Lucchetti, F., Ripa, M., Belfki, A., Prakash, N., Multani, S., Brodley, C., Guha, A., Bell, J., Wallace, B., and Bau, D. Nnsight and ndif: Democratizing access to foundation model internals. 2024. URL https://arxiv.org/abs/2407.14561. 

- F.R.S., K. P. Liii. on lines and planes of closest fit to systems of points in space. _The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science_ , 2(11): 559–572, 1901. doi: 10.1080/14786440109462720. 

- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders, 2024. URL https: //arxiv.org/abs/2406.04093. 

- Glazer, E., Erdil, E., Besiroglu, T., Chicharro, D., Chen, E., Gunning, A., Olsson, C. F., Denain, J.-S., Ho, A., de Oliveira Santos, E., Jarviniemi, O., Barnett, M., San-¨ dler, R., Vrzala, M., Sevilla, J., Ren, Q., Pratt, E., Levine, L., Barkley, G., Stewart, N., Grechuk, B., Grechuk, T., Enugandla, S. V., and Wildon, M. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai, 2024. URL https://arxiv.org/abs/2411. 04872. 

- Goldowsky-Dill, N., MacLeod, C., Sato, L., and Arora, A. Localizing model behavior with path patching, 2023. URL https://arxiv.org/abs/2304.05969. 

- Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., et al. The llama 3 herd of models, 2024. URL https: //arxiv.org/abs/2407.21783. 

- Hanna, M., Liu, O., and Variengien, A. How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https://openreview.net/forum? id=p4PckNQR8k. 

- Heimersheim, S. and Nanda, N. How to use and interpret activation patching, 2024. URL https://arxiv.org/ abs/2404.15255. 

- Huben, R., Cunningham, H., Smith, L. R., Ewart, A., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. In _The Twelfth International Conference on Learning Representations_ , 2024. URL https://openreview.net/forum? id=F76bwRSLeK. 

- Kramar, J., Lieberum, T., Shah, R., and Nanda, N. Atp*: An´ efficient and scalable method for localizing llm behaviour to components, 2024. URL https://arxiv.org/ abs/2403.00745. 

9 

**Language Models Use Trigonometry to Do Addition** 

- Levy, A. A. and Geva, M. Language models encode numbers using digit representations in base 10, 2024. URL https://arxiv.org/abs/2410.11781. 

   - nostalgebraist. interpreting GPT: the logit lens — LessWrong — lesswrong.com. https://www.lesswrong. com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens, 2020. [Accessed 14-01-2025]. 

- Liu, H., Dai, Z., So, D., and Le, Q. V. Pay attention to MLPs. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), _Advances in Neural Information Processing Systems_ , 2021. URL https://openreview.net/ forum?id=KBnXrODoBW. 

   - Olah, C. and Jermyn, A. What is a linear representation? what is a multidimensional feature?, 2024. URL https://transformer-circuits. pub/2024/july-update/index.html# linear-representations. 

- Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., and Williams, M. Towards understanding grokking: An effective theory of representation learning. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum? id=6at6rB3IZm. 

   - Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. Zoom in: An introduction to circuits. _Distill_ , 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in. 

   - Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. In-context learning and induction heads. _Transformer Circuits Thread_ , 2022. https://transformer-circuits.pub/2022/in-contextlearning-and-induction-heads/index.html. 

- Makelov, A., Lange, G., and Nanda, N. Towards principled evaluations of sparse autoencoders for interpretability and control. In _ICLR 2024 Workshop on Secure and Trustworthy Large Language Models_ , 2024. URL https: //openreview.net/forum?id=MHIX9H8aYF. 

- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., and Mueller, A. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models, 2024. URL https://arxiv.org/abs/2403. 19647. 

   - OpenAI. URL https://openai.com/index/ learning-to-reason-with-llms. 

- Meng, K., Bau, D., Andonian, A. J., and Belinkov, Y. Locating and editing factual associations in GPT. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), _Advances in Neural Information Processing Systems_ , 2022. URL https://openreview.net/forum? id=-h6WAS6eE4. 

   - Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models. In _Causal Representation Learning Workshop at NeurIPS 2023_ , 2023. URL https://openreview. net/forum?id=T0PoOJg8cK. 

   - Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kramar, J., Shah, R., and Nanda, N. Improving´ dictionary learning with gated sparse autoencoders, 2024. URL https://arxiv.org/abs/2404.16014. 

- Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. Progress measures for grokking via mechanistic interpretability. In _The Eleventh International Conference on Learning Representations_ , 2023a. URL https: //openreview.net/forum?id=9XFSbDPmdW. 

   - Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., et al. Gemma 2: Improving open language models at a practical size, 2024. 

- Nanda, N., Rajamanoharan, S., Kramar, J., and Shah, R. Fact finding: Attempting to reverseengineer factual recall on the neuron level, Dec 2023b. URL https://www.alignmentforum. org/posts/iGuwZTHWb6DFY3sKB/ 

   - Satpute, A., Gießing, N., Greiner-Petter, A., Schubotz, M., Teschke, O., Aizawa, A., and Gipp, B. Can llms master math? investigating. large language models on math stack exchange. In _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval_ , SIGIR ’24, pp. 2316–2320, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704314. doi: 10.1145/3626772.3657945. URL https://doi. org/10.1145/3626772.3657945. 

- fact-finding-attempting-to-reverse-engineer-factual-recallllms master math? . 

- Nikankin, Y., Reusch, A., Mueller, A., and Belinkov, Y. Arithmetic without algorithms: Language models solve math with a bag of heuristics. In _Submitted to The Thirteenth International Conference on Learning Representations_ , 2024. URL https://openreview.net/ forum?id=O9YTt26r2P. under review. 

10 

**Language Models Use Trigonometry to Do Addition** 

- Stolfo, A., Belinkov, Y., and Sachan, M. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. pp. 7035–7052, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.435. URL https:// aclanthology.org/2023.emnlp-main.435. 

- Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. _Transformer Circuits Thread_ , 2024. URL https: //transformer-circuits.pub/2024/ scaling-monosemanticity/index.html. 

   - Zhu, F., Dai, D., and Sui, Z. Language models encode the value of numbers linearly. In Rambow, O., Wanner, L., Apidianaki, M., Al-Khalifa, H., Eugenio, B. D., and Schockaert, S. (eds.), _Proceedings of the 31st International Conference on Computational Linguistics_ , pp. 693–709, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025. coling-main.47/. 

   - Zlokapa, A., Tan, A. K., Martyn, J. M., Fiete, I. R., Tegmark, M., and Chuang, I. L. Fault-tolerant neural networks from biological error correction codes. _Phys. Rev. E_ , 110:054303, Nov 2024. doi: 10.1103/PhysRevE.110. 054303. URL https://link.aps.org/doi/10. 1103/PhysRevE.110.054303. 

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper. pdf. 

- Wang, B. and Komatsuzaki, A. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/ mesh-transformer-jax, May 2021. 

- Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview.net/forum? id=NpsVSN6o4ul. 

- Yip, C. H., Agrawal, R., Chan, L., and Gross, J. Modular addition without black-boxes: Compressing explanations of mlps that compute numerical integration, 2024. 

- Zhong, Z., Liu, Z., Tegmark, M., and Andreas, J. The clock and the pizza: Two stories in mechanistic explanation of neural networks. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https: //openreview.net/forum?id=S5wmbQc1We. 

- Zhou, T., Fu, D., Sharan, V., and Jia, R. Pre-trained large language models use fourier features to compute addition. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ , 2024. URL https: //openreview.net/forum?id=i4MutM2TZb. 

11 

**Language Models Use Trigonometry to Do Addition** 

# **A. Performance of all models on** _a_ + _b_ = 

We test three models, GPT-J, Pythia-6.9B, and Llama3.1-8B on the task _a_ + _b_ =. At first, we attempted to prompt each model with just _a_ + _b_ =, but we achieved significantly better results by including additional instructions in the prompt. After non-exhaustive testing, we used the prompts listed in Table 2 to test each model for all 10000 addition prompts (for _a, b ∈_ [0 _,_ 99]). We plot a heatmap of the accuracy of the model by _a_ and _b_ in Fig. 11. All three models are able to competently complete the task, with Llama3.1-8B achieving an impressive 98% accuracy. However, in the main paper we focus on analyzing GPT-J because it employs simple MLPs that are easier to interpret. We note that all three models struggle with problems with larger values of _a_ and _b_ . 

# **B. Additional Results on the Structure of Numbers** 

Our Fourier decomposition results in Section 4.1 are sensitive to the number of _a_ values analyzed. In particular, we find that the _T_ = 2 Fourier component is not identified when analyzing _h_<sup>0</sup> 361<sup>, but is identified when analyzing</sup><sup>_h_0</sup> 360 (Fig. 12). While we consider this sensitivity to sample size to be a limitation of our Fourier analysis, we note that the Fourier analysis is itself preliminary. We find that the _T_ = 2 Fourier feature is causally relevant when fitting the residual stream in Section 4.4. Additionally, in later sections we find that neurons often read from the helix using the _T_ = 2 Fourier feature, indicating its use downstream (Fig. 34). Thus, we identify _T_ = 2 as an important Fourier feature. 

We compare the residual stream of GPT-J after layer 0 on the inputted integers _a_ 1 _, a_ 2 _∈_ [0 _,_ 99] using Euclidean distance and cosine similarity in Fig. 13. We visually note periodicity in the representations, with a striking period of 10. To analyze the similarity between representations further, we calculate the Euclidean distance between _a_ and _a_ + _δn_ for all values of _δn_ . In Fig. 14, we see that representations continue to get more distant from each other for _a ∈_ [0 _,_ 99] as _δn_ grows, albeit sublinearly. This provides evidence that LLMs represent numbers with more than just periodic features. When _a_ is restricted to _a ∈_ [0 _,_ 9], we observe a linear relationship in _δn_ , implying some local linearity. The first principal component of the numbers _a ∈_ [0 _,_ 360] (shown in Fig. 15) is also linear with a discontinuity at _a_ = 100. Thus, our focus on two digit addition is justified, as three-digit integers seem to be represented in a different space. 

# **C. Additional Helix Fitting Results** 

## **C.1. Helix Properties** 

For the input of layer 0 of GPT-J, we plot the magnitude of the helical fit’s Fourier features. We do so by taking the magnitude of columns of _C_ in Eq. 2. We find that these features roughly increase in magnitude as period increases, which matches the ordering in the Fourier decomposition presented in Fig. 2. 

Additionally, we visualize the cosine similarity matrix between columns of _C_ , which represents the similarity between helix components. In Fig. 17, we observe that components are mostly orthogonal, as expected. A notable exception is the similarity between the _T_ = 100 sin component and the linear component. 

To ensure that the helix represents a true feature manifold, we design a continuity experiment inspired by Olah & Jermyn (2024). We first fit all _a_ that do not end with 3 with a _T_ = [100] helix. Then, we project _a_ = 3 _,_ 13 _, . . . ,_ 93 into that helical space in Fig. 18. We find that each point is projected roughly where we expect it to be. For example, 93 is projected between 89 and 95. We take this as evidence that our helices represent a true nonlinear manifold. 

## **C.2. Additional Causal Experiments for Helix Fits** 

We first replicate our helix fitting activation patching results on Pythia-6.9B and Llama3.1-8B in Fig. 19. 

To ensure the helix fits are not overfitting, we use a traintest split. We train the helix with 80% of _a_ values and patch using the other 20% of _a_ values (left of Fig. 20). We observe that the helix and circular fits still outperform the PCA baseline. We also randomize the order of _a_ and find that the randomized helix is not causally relevant (middle of Fig. 20), suggesting that the helix functional form is not naturally over expressive. Finally, we demonstrate that our results hold when fitting the _b_ token on _a_ + _b_ with helix( _b_ ) (right of Fig. 20). Note that when activation patching fits on the _b_ token, we use clean/corrupted prompt pairs of the form ( _a_ + _b_<sup>_′_</sup> , _a_ + _b_ ), in contrast to the ( _a_<sup>_′_</sup> + _b_ , _a_ + _b_ ) pairs we used for the _a_ token. 

We perform an ablation experiment by ablating the columns of _C_<sup>_†_</sup> from each _h_<sup>_l_</sup> _a_<sup>.In Fig.21,we see that ablating the</sup> helix dimensions from the residual stream like this affects performance about as much as ablating the entire layer, providing additional causal evidence that the helix is necessary for addition. However, when we attempt to fit _a_ with helix( _a_ ) for other tasks in Fig. 22, we find that while the fit is effective, it sometimes underperforms PCA baselines. This suggests that while the helix is sufficient for addition, additional structure is required to capture the entirety of numerical representations. For a description of the prompts 

12 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Model Performance on a+b= for a,b € [0,99]<br>Blue is correct<br>gpt-j-6B pythia-6.9b Llama-3.1-8B<br>80.5% Accuracy 77.2% Accuracy 98.0% Accuracy<br>20 ena 204 re 20 ; ad<br>40 EE sree 404 — hae eg 40 wnat ad<br>co ORE ria 60 aetnatin 60 ees<br>0 50 0 50 ie) 50<br>a<br><!-- End of picture text -->

_Figure 11._ All models are able to competently perform the task _a_ + _b_ , with Llama3.1-8B performing best. 

_Table 2._ The prompts used and accuracy of each model on the addition task _a_ + _b_ . 

|**Model Name**|**Prompt**|**Accuracy**|
|---|---|---|
|GPT-J|Output ONLY a number. _{a}_+_{b}_=|80.5%|
|Pythia-6.9B|Output ONLY a number. _{a}_+_{b}_=|77.2%|
|Llama3.1-8B|The following is a correct addition problem._\_n_{a}_+_{b}_=|98.0%|





<!-- Start of picture text -->
Range: 0-360<br>20 T=100 T=10 T=5 Fourier Decomposition<br>T=2<br>10<br>| WalerkeedtteP<br>0.0 0.1 0.2 0.3 0.4 0.5<br>Frequency<br>Range: 0-361<br>20 T=100 T=10 Fourier Decomposition<br>T=5<br>10 cee, T=2<br>0.0 0.1 ——————————— 0.2 0.3 0.4 0.5<br>Frequency<br>Magnitude<br>Magnitude<br><!-- End of picture text -->

_Figure 12._ The _T_ = 2 Fourier feature is prominent when analyzing _h_<sup>0</sup> 360<sup>, but not when analyzing</sup><sup>_h_</sup> 361<sup>0.</sup> 



<!-- Start of picture text -->
°° EuclideanLayer 0Distancein gpt-6BMatrix<br>20<br>10<br>0<br>3a<br>:<br>So 9% umber<br><!-- End of picture text -->



<!-- Start of picture text -->
CosineLayerSimilarity0 in gptj-68.Matrix<br>| ee<br>2010] HastPiet eeaeae r y |<br>co | |alae A<br>20) te A<br>ie ee<br>“oan£%)eumtelente<br><!-- End of picture text -->

_Figure 13._ We see clear periodicity in GPT-J’s layer 0 representations of the numbers from 0 _−_ 99. 



<!-- Start of picture text -->
ws<br>.<br>afe<br>s<br>PLEFPPOLLESESE<br><!-- End of picture text -->



<!-- Start of picture text -->
© fet taoa, oars<br>fio]nf dg, = EnaeRER9<br>zgeae “"ORpseete| osgae<br>ie<br>SAPPORO LES<br><!-- End of picture text -->

_Figure 14._ For GPT-J layer 0, the Euclidean distance between _a_ and _a_ + _δn_ is approximately linear for _a ∈_ [0 _,_ 9], and sublinear for _a ∈_ [0 _,_ 99]. This provides additional evidence that GPT-J uses more than periodic features to represent numbers. 

13 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
@ Pcl 300<br>“s<br>& 20 S == Best fit (R? = 0.780)<br>gs=0 e ° 200<br>=g e 100<br>-20 7<br>o<br>0 100 200 300<br>Number<br><!-- End of picture text -->

_Figure 15._ The first principal component of GPT-J layer 0 for numbers _a ∈_ [0 _,_ 360] shows a discontinuity for three-digit _a_ , implying that three-digit numbers are represented in a different space. 



<!-- Start of picture text -->
Mogntude of Hel Components in gp. Layer 0<br>for task a+b= (a,b € [0.99})<br>028<br>_ 020<br>Foxoo<br>i<br>a a a a a aS<br>tar eompows<br><!-- End of picture text -->

_Figure 16._ We plot the magnitude of each column of _C_ , the coefficient matrix for the helical basis, to calculate the importance of each Fourier feature. Feature magnitude roughly increases with period, with the notable omission of the _T_ = 2 sin component. We do not plot the linear component’s magnitude because its output is a different scale. 



<!-- Start of picture text -->
Cosine Similarity of Helix C¢ ents in gpt-j-6B Layer 0<br>cos(alT = 2) 1.00<br>075<br>sin(a|T= 2)<br>cos(alT = 5) 050<br>sin(alT = 5) ors 2<br>cos(alT = 10) 000 &<br>sin(a|T= 10) 0.25 8&<br>costalt = 100) _os0<br>sin(a[T= 100) -0.75<br>a<br>1.00<br>PPP ASSP SH?<br>SOE EES<br>FSS SS eee<br><!-- End of picture text -->



<!-- Start of picture text -->
Helix Superimposed in gpt-j-6B Layer 0<br>34 1 o0 ge n<br>e) 3<br>© é ®@e 80<br>3<br>© so 3 é 60<br>¢ cos(alT = 100) @o6<br>@ sinajt= 100) @<br>ad<br>& 3 20<br>oe 3 By<br>6<br>* 78 84<br>0<br>ee 63 3<br>Ee da oF@z<br>50<br>4 :<br><!-- End of picture text -->

_Figure 18._ We fit all _a ∈_ [0 _,_ 99] that do not end with 3 using a helix with _T_ = [100], and project the residual stream for _a_ = 3 _,_ 13 _, . . . ,_ 93 onto the space. We find that there is continuity in the manifold, which we take as evidence that numbers are represented as a nonlinear feature manifold. 

_Figure 17._ We plot the cosine similarity between columns of _C_ , the coefficient matrix for the helical basis. Features are roughly orthogonal, which we expect for a helix, with the exception of the _T_ = 100 sin component and the linear component _a_ . We ignore the _T_ = 2 sin component because of its negligible magnitude. 

14 

**Language Models Use Trigonometry to Do Addition** 

used and accuracy of GPT-J on these other tasks, see Table 3. 

# **D. Additional Clock algorithm evidence** 



<!-- Start of picture text -->
Patching Experiments on pythia-6.9b<br>Modeling: a for a+b=<br>— 9dim PA<br>84 ozs ==<br>=7<S). — Circle:Helix: (2,(2, 5,5, 10,10, 100)100]<br>. +++ th Deg Polynomial<br>& 64 => _ 7 Params<br>fal Pita Helix:[5, 10, 100)<br>2@ 47 yesASTI RES 7 5Helix:Params(10, 100)<br>a Ba 3 Params<br>2 7 Helix: (100)<br>0<br>0 5 10 15 20 25 30<br>layer<br><!-- End of picture text -->



<!-- Start of picture text -->
Patching Experiments on Llama-3.1-8B<br>Modeling: a for a+b=<br>6<br>— 9dim PA<br>£ 45 PeaSSSR SARSOS ==oHs++ Sites9thHelix: Deg(2,(2.5.10. Polynomial5, 10, 100)200]<br>a — ie3 eeHelix:[5, 10, 100]<br>we 3] Gissrartieeeeeetteed 5 Params<br>52> | Same at __= 3Helix:faiParams[100]fo, 200)<br>1<br>0<br>0 5 10 15 20 25 30<br>layer<br><!-- End of picture text -->

_Figure 19._ We replicate our patching results on Pythia-6.9B and Llama3.1-8B. The helical and circular fits outperform baselines at most numbers of parameters. 

We show that helix( _a_ + _b_ ) fits last token hidden states for Pythia-6.9B and Llama3.1 8B in Fig. 23. Notably, the results for Llama3.1-8B are less significant than those for GPT-J and Pythia-6.9B. This is surprising, since the helical fit on the _a_ token is causal for Llama3.1-8B in Fig. 19, and is a sign that Llama3.1-8B potentially uses additional algorithms to compute _a_ + _b_ . We hypothesize that this might be due to Llama3.1-8B using gated MLPs, which could lead to the emergence of algorithms not present in GPT-J and Pythia-6.9B, which use simple MLPs. Nikankin et al. (2024)’s analysis of Llama3-8B’s top neurons in addition problems identifies neurons with activation patterns unlike those we identified in GPT-J. Due to this evidence, along with the importance of MLPs in the addition circuit, we consider it likely that Llama3.1-8B implements modified algorithms, but we do not investigate further. 

## **D.1. Attention Heads** 

In Fig. 24, we use activation patching to show that a sparse set of attention heads are influential for addition. In Fig. 25, we find that patching in _k_ = 20 heads at once is sufficient to restore more than 80% of the total effect of patching all _k_ = 448 heads. In Fig. 26, we also find that all attention heads in GPT-J are well modeled using helix( _a, b, a_ + _b_ ). We judge this by the fraction of a head’s total effect recoverable when patching in a helical fit. 

We categorize heads as _a, b_ , _a_ + _b_ , and mixed heads using a confidence score (detailed in Section 5.2). To make our categorization useful, we aim to categorize as few heads as mixed as possible. We find that using _m_ = 4 mixed heads is sufficient to achieve almost 80% of the effect of patching the actual outputs of the _k_ = 20 heads (Fig. 27), although using _m_ = 0 mixed heads still achieves 70% of the effect. In Fig. 28, we analyze the properties of each head type. _a_ + _b_ heads tend to attend to the last token and occur in layers 19 onwards. _a, b_ heads primarily attend to the _a_ and _b_ tokens and occur prior to layer 18. Mixed heads attend to the _a, b_ , and last tokens, and occur in layers 15-18. 

To understand what each head type reads and writes to, we use a modification of the path patching technique we have discussed so far. Specifically, we view mixed and _a, b_ heads as “sender” nodes, and view the total effect of each downstream component if only the direct path between the sender node and the component is patched in (not mediated by any other attention heads or MLPs). In Fig. 29, we find that both _a, b_ and mixed heads generally impact downstream MLPs most. Similarly, we consider mixed and _a_ + _b_ heads 

15 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Patching Experiments on gpt-68 w/ Traintest Spit<br>Modeling: a for a+b=<br>8 Zo sare”<br>5 ° Sta===Sith ete+++=> Gre,9th Deg25.20.2001 Polynomial5, 10,1001<br>Ba S<br>BAY escesetcsee<br>°<br>oo) 50 6 layer(SOO<br><!-- End of picture text -->



<!-- Start of picture text -->
Randomized Helix Experiments on gpt-68<br>Modeling: a for a+b=<br>. Ta samren,<br>lk htc, 101<br>5 iJ a -~) Tr hanceHeli [2. 5.20, 100)<br>ga m5<br>Ey<br>ol? nia<br>oo) 50 6 layer(SOO<br><!-- End of picture text -->



<!-- Start of picture text -->
Patching Experiments on gpt-68<br>Modeling: b for atb=<br>8 Tr samc”<br>——— hee, 2.1001<br>Polynomial<br>5— of POss Te+ ee9th Deg a5. 0,100)<br>Sa \<br>ce a<br>° oA<br>ee Layeree<br><!-- End of picture text -->

_Figure 20. Left_ When training the helix with 80% of _a_ values and activation patching with the other 20% of _a_ values, we see that the helix and circular fits still outperform the PCA baseline. _Middle_ We randomize _a_ while fitting the helix and see that the randomized helix is not causally relevant. _Right_ We show that our results for fitting the _a_ token can be extended to fitting the _b_ token on the prompt _a_ + _b_ with helix( _b_ ). 

_Table 3._ GPT-J’s task performance with corresponding domains, <u>prompts, and accuracies.</u> 

|**Task**|**Domain**|**Prompt**|**Accuracy**|
|---|---|---|---|
|_a −_23|_a ∈_[23_,_99]|Output ONLY a number. a-23=|89.61%|
|_a//_5|_a ∈_[0_,_99]|Output ONLY a number. a//5=|97.78%|
|_a ∗_1_._5|_a ∈_[0_,_98]|Output ONLY a number. a*1.5=|80.00%|
|_a_ mod 2|_a ∈_[0_,_99]|Output ONLY a number. a modulo 2=|95.56%|
|_x −a_= 0_, x_=|_a ∈_[0_,_99]|Output ONLY a number. x-a=0,x=|95.56%|





<!-- Start of picture text -->
Ablation Results for a+b= for a,b € [0,99]<br>gpt-j-6B<br>0.0<br>eg 1 aay-4<br>£ -2544 er<br>5 TSG<br>2 FG<br>= 5.0 jot ie — 100dim Pca<br>2 Prey == 9dim PCA<br>> BY — Helix: [2, 5, 10, 100)<br>3-75 v == Circle: (2,5, 10, 100]<br>+++ 9th Deg Polynomial<br>0 5 10 15 20 25<br>Layer<br><!-- End of picture text -->

_Figure 21._ We find that ablating the helix dimensions from the residual stream is roughly as destructive as ablating the entire layer, providing additional evidence that GPT-J uses a numerical helix to do addition. 

as “receiver” nodes, and patch in the path between all upstream components and the receiver node to determine what components each head relies on to achieve its causal effect. We find that _a_ + _b_ heads rely predominantly on upstream MLPs, while mixed heads use both _a, b_ heads and upstream MLPs. This indicates that mixed heads may have some role in creating helix( _a_ + _b_ ). 

## **D.2. MLPs and Neurons** 

In Fig. 30, we see that patching _k_ = 11 MLPs achieves 95% of the effect of patching all MLPs. We consider these MLPs to be circuit MLPs. Zooming in at the neuron level, we find that roughly 1% of neurons are required to achieve an 80% success rate on prompts while mean ablating all other neurons (Fig. 31). Note the use of accuracy over logit difference as a metric in this case. Fig. 31 shows that ablating some neurons actually _helps_ performance as measured by logit difference, while hurting accuracy. To account for this seemingly contradictory result, we hypothesize that ablating some neurons asymmetrically boosts the answer token across prompts, such that some prompts are boosted significantly while other prompts are not affected. We do not investigate this further as it is not a major part of our argument and instead use an accuracy threshold. 

When plotting the distribution of top neurons across layers in Fig. 32, we find that almost 75% of top neurons are located in the _k_ = 11 circuit MLPs we have identified. We then path patch each of these neurons to calculate their direct effect. In Fig. 33, we see that roughly 700 neurons 

16 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
— Helix: [2, 5,10, 100] == Circle: [2, 5,10, 100] ++ 9th Deg Polynomial == 9dimPCA —— Patch layer<br>atb= a-23= a//5=<br>754 15 ===><br>5.04! seN= g&v 5047 SSees==> 47 eonsme<br>=f Gy ne<br>= s<br>0.0 0.0 0<br>0 10 20 0 10 20 0 10 20<br>a*1.5= amod 2= If x-a=0, x=<br>75 © 1.0<br>ypren 8 4 6<br>5.04 fseam ©$ jBP aang =<br>iss £054: ‘<br>bel a Lee eee steers)<br>8 0.0 oe<br>0.0 s 0<br>oO 10 20 0 10 20 0 10 20<br>Layer Layer<br><!-- End of picture text -->

_Figure 22._ When testing our helical fit on other numerical tasks, we find that it performs competently, but does not always outperform the PCA baseline. This implies that additional structure may be present in numerical representations. 



<!-- Start of picture text -->
Fitting Answer Hidden States of pythia-6.9b<br>a+b= for a,b E[0, 99]<br>10<br>© — Layer Patch oS<br>2 84 —= 2701 Pca 2a<br>5 s+ Sdim PCA ot °<br>2 & T— hetixia.b.atb) Sentra<br>GB 44 helixiatb)<br>2 — hetix(a,b)<br>Bz<br>°<br>o 5 wo 1520580<br>Layers<br><!-- End of picture text -->



<!-- Start of picture text -->
Fitting Answer Hidden States of Llama-3.1-8B<br>atb= for a, b€[0, 99]<br>6<br>S| —= Layer27dimeca Patch site +S aN<br>SB 44er+ ogim Pca = mS<br>S| — hetixia.v.a+o) :<br>fa] —— helix(a+b) a<br>= 2-7 — helixia.b) a<br>s<br>°<br>o 5 1S S30<br>Layers<br><!-- End of picture text -->

_Figure 23._ We present helical fits on the last token for Pythia-6.9B and Llama3.1-8B. The fits are weaker for Llama3.1-8B, potentially indicating the use of non-Clock algorithms. 

are required to achieve 80% of the direct effect of patching in all _k_ = 4587 top neurons. 84% of the top DE neurons occur after layer 18, which corresponds with our claim that MLPs 19-27 primarily write the correct answer to logits. 

When we Fourier decompose the _k_ = 4587 top neurons’ preactivations with respect to the value of _a_ + _b_ in Fig. 34, we see spikes at periods _T_ = [2 _,_ 5 _,_ 10 _,_ 100]. These are the exact periods of our helix parameterization. To leverage this intuition, we fit the neuron preactivation patterns using the helix inspired functional form detailed in Eq. 2. We use a stochastic gradient descent optimizer with lr = 1 _e−_ 1 _,_ epochs = 2500 and a cosine annealing learning rate scheduler to minimize the mean squared error of the fit. In Fig. 35, we see that more important neurons with larger total effect are fit better with this functional form, as measured by normalized root mean square error (NRMSE). 

# **E. Why Use the Clock Algorithm at All?** 

We conjecture that LLMs use the Clock algorithm as a form of robust, error correcting code. If LLMs used a linear representation of numbers to do addition, that representation would have to be extremely precise to be effective. 

To preliminarily test this conjecture, we take the first 50 PCA dimensions of the number representations for _a ∈_ [0 _,_ 99] in GPT-J after layer 0 and fit a line _ℓ_ to it. The resulting line 

17 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Attention Head Activation Patching<br>Final Position<br>0<br>5<br>10<br>.&<br>4s<br>15<br>20<br>25<br>o 2 4 6 8 10 12 14<br>Head<br><!-- End of picture text -->



<!-- Start of picture text -->
Attn Head Helix Intervention Results for gpt-j-6B<br>Activation Patching for helix(a,b,a+b)<br>y 15 —= If fit was perfect - =<br>= “7<br>o<br>5<br>= 10 :7 @<br>a2<br>2& 05 mo) -@<br>Bs<br>rz<br>x 0.0<br>a = |e<br>-0.25 0.00 0.25 050 0.75 1.00 125 150<br>Full Intervention Logit Difference<br><!-- End of picture text -->

_Figure 26._ Attention head outputs are well modeled with helix( _a, b, a_ + _b_ ). Total effect shown. 

_Figure 24._ A sparse set of attention heads have causal effects on the output when patched (total effect visualized). 



<!-- Start of picture text -->
Effect of Activation Patching Top-k Attention Heads on Addition Task<br>30% BERS<br>os 80% atk=17__¥,<br>2 70% at k=13<br>5<br>226 Sov—— aE KB<br>a4<br>22<br>Ss<br>°<br>10° 10 a0?<br>Number of Heads Patched (k)<br><!-- End of picture text -->



<!-- Start of picture text -->
= Helix Head Fitting as a fraction of Activation Patching<br>5 ons<br>3% ons<br>cg<br>BB 0750<br>7S ons<br>=<br>0700<br>2<br>0 Number2 4 of Mixed6 8  Categorizations10 1 a Allowed1 1 2%<br><!-- End of picture text -->

_Figure 27. m_ = 4 mixed heads are sufficient to achieve 80% of effect of patching in all _k_ = 20 heads normally. Even using _m_ = 0 mixed heads achieves 70% of the effect. 

_Figure 25._ Patching _k_ = 20 heads at once restores more than 80% of model behavior of patching all _k_ = 448 heads. 

18 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Atention Head Layer vs Confidence<br>10 Pury} e<br>oa t$ le<br>¥o6 ° oo<br>3 ° cee<br>é “|<br>02oo 4@S avbheedtnednenssbent yy °<br>Layer<br><!-- End of picture text -->



<!-- Start of picture text -->
tention Head Last Token attention vs Confidence<br>10 e e a<br>at> °<br>goof © oo<br>sie ° °<br>é-yr°<br>0200 4J @© svhedbentteins ° .<br>Last Token Attention<br><!-- End of picture text -->



<!-- Start of picture text -->
Attention Head AB Token tention vs Confidence<br>— e % Boe<br>ate °<br>goojie @ °<br>3,0 bd<br>8 .<br>ateoofig ©8tineBioheasbrent end<br>A.B Token Atention<br><!-- End of picture text -->

_Figure 28. Left a_ + _b_ heads generally occur late in the network, while _a, b_ heads occur earlier. Mixed heads occur in middle layers. _Middle a_ + _b_ heads attend more to the last token. _Right a, b_ heads attend mostly to tokens _a_ and _b_ . 



<!-- Start of picture text -->
Path Patching Results for sender #5. L14H13<br>TE = 0.331, DE = 0.093, DE/TE: 0.28, Type: a,b<br>27 0.08<br>26 0.06<br>25<br>24 0.04<br>5 2B<br>><br>4s5 2222 0.02<br>3<br>3€19 -002<br>@ ig<br>0 -o0e<br>16<br>15 0.06<br>4 0.08<br>SES PPELA<br>CEPOL CEPCE LE SOMSESE RSNES<br>Receiver Component<br><!-- End of picture text -->



<!-- Start of picture text -->
Path Patching Results for receiver #0. L24H10<br>TE = 1.454, DE = 0.927, DE/TE: 0.64, Type: a+b<br>23<br>z 02<br>29Ft8<br>16 on<br>&sie<br>Be<br>38<br>gn 0.0<br>3 lo<br>59<br>&8<br>$ oa<br>3<br>i -02<br>0<br>CEEEESHH LPLELE LH LE SOSEES Mr hes<br>Sender Component<br><!-- End of picture text -->



<!-- Start of picture text -->
Path Patching Results for sender #2. L18H10<br>TE = 0.530, DE = 0.072, DE/TE: 0.14, Type: Mixed<br>27<br>6 02<br>_g 25 O21<br>a<br>5 2 00<br>3 2<br>Ba -01<br>20<br>19 ~02<br>18<br>EEEECEEESS LF SLOPE EE  SPOLISEEE<br>Receiver Component<br><!-- End of picture text -->



<!-- Start of picture text -->
Path Patching Results for receiver #2. L18H10<br>TE = 0.530, DE = 0.072, DE/TE: 0.14, Type: Mixed<br>Bus<br>16 0.04<br>15<br>3<br>12 0.02<br>5pu<br>="<br>soo<br>5 8 0.00<br>3<br>57<br>- 3 655 -0.02<br>12 0.04<br>°<br>CPEEEEELESPSL PPS HPPeee So Mo Mdal<br>Sender Component<br><!-- End of picture text -->

_Figure 29._ We present path patching results for top attention heads. In the top row, we view one _a, b_ head (L14H13) and one mixed head (L18H10) as senders. We plot the total effect of each downstream component when only the path between the sender head and that component is patched into the model. We find that both _a, b_ and mixed heads write primarily to MLPs. Similarly, when viewing an _a_ + _b_ head (L24H10) and mixed head (L18H10) as receivers, we see that that the _a_ + _b_ head is primarily dependent on the output from preceding MLPs. While that is true for the mixed head as well, we see that the mixed head also takes input from other _a, b_ heads, implying that it could have some role in creating the _a_ + _b_ helix. 

19 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Faithfulness of k-MLPS in circuit<br>. [TT1T pevoopooooooooooooeossseecteees |<br>eel<br>5 it ro<br>ga Ht TH == 7o% of max at k3<br>2B2 mt 1 | 1 === som50% ormaxateasormax<br>- atios<br>youd 1 1 == 959% of max at keit<br>oto md 1 | Ne gos ormaxat k=15<br>0.2ay4 6 8 12 14 16 18 20 2 24 26<br>k MLPS in circuit<br><!-- End of picture text -->

_Figure 30. k_ = 11 MLPs are required to achieve 95% of the effect of patching in all MLPs. 



<!-- Start of picture text -->
Model Performance vs Fraction of Neurons Used<br>(Metric: Accuracy)<br>4-7.<br>ee<br>><br>3<br>% os<br>oe eeee)<br>02<br>ee esa<br>0.0 a<br>10sage10 10 ee 10 ee 10 dot 10'oP<br>Fraction of Neurons<br><!-- End of picture text -->



<!-- Start of picture text -->
Model Performance vs Fraction of Neurons Used<br>(Metric: Logit Diff)<br>1S50 ete en<br>Eas<br>ba2 of —No<br>3<br>-25<br>“50<br>wot 10-0000?<br>Fraction of Neurons<br><!-- End of picture text -->

_Figure 31. Top_ We see that using around 1% of top neurons and mean ablating the rest can restore the model to 80% accuracy. _Bottom_ When measuring logit difference, we find that mean ablating some neurons on average _increases_ the logit for the correct answer, but does not improve accuracy. We choose not to investigate this further. 



<!-- Start of picture text -->
Distribution of Top 1.0% Neurons Across Layers<br>Highlighting Circult vs Non-circult MLPs<br>409 TES neurons crcl LP Layers (74.39 of neurons)<br>350<br>2 300<br>;<br>3 m0<br>5 700<br>3 is<br>=<br>2 100<br>50°<br>0 2 4 6 8 1012 14 16 18 20 22 28 26 28<br>layer<br><!-- End of picture text -->

_Figure 32._ When we analyze approximately 1% of last token neurons most causally implicated in addition for GPT-J, we find that nearly 75% of them are in circuit MLPs, which is expected. 



<!-- Start of picture text -->
Cumulative Distribution of Path Patching Effects<br>8 tig 1g 18<br>g rif 13 be<br>57 181 oN<br>2 fe Ie 1S<br>z<br>s Isis Ig ty<br>g4 28 (8/8<br>ea] (8 98<br>2 ito<br>eey ftir<br>3 mre<br>a yontrue<br>°<br>° 1000 2000 3000 4000<br>Number of Neurons<br><!-- End of picture text -->



<!-- Start of picture text -->
Distribution of Top 1% Neurons Across Layers<br>Or sein ayers 19%<br>400 = openerPath Patch Neurons for 80.0% DE<br>300<br>3<br>200<br>100<br>° ° 5 1 18 20 5<br>layer<br><!-- End of picture text -->

_Figure 33. Top_ We find that roughly 700 neurons achieve 80% of the direct effect of patching in all _k_ = 4587 high impact neurons. _Bottom_ More than 80% of these high direct effect neurons are in layers 19 onwards, which we have identified as being responsible for translating the _a_ + _b_ helix to answer logits. 

20 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Distribution of a+b neuron periods<br>4581/4587 neurons plotted<br>% within =0<br>1500 2: 8.7%5 of period<br>™ 59.8%10: 13.4%<br>© 000 4100: 33.4%<br>3<br>8<br>500<br>°<br>2 5 100<br>a+b Period<br><!-- End of picture text -->

_Figure 34._ Top neurons’ preactivations are periodic in _a_ + _b_ with top periods of _T_ = [2 _,_ 5 _,_ 10 _,_ 100]. Percentages shown for Fourier periods within 5% of _T_ . 



<!-- Start of picture text -->
Addition Performance vs Answer Threshold<br>1.0<br>GPT-J<br>0.5 0.997 R^2 Linear Fit Addition<br>0 25 50 75 100<br>Answer Threshold<br>% Correct<br><!-- End of picture text -->

_Figure 36._ We find that using a line with _R_<sup>2</sup> = 0 _._ 997 leads to addition that performs considerably worse than GPT-J. We attribute this to the representational precision required to do addition along a line, indicating a possible reason LLMs choose to use helical representations. 

has an _R_<sup>2</sup> of 0 _._ 997, indicating a very good fit. We consider all problems _a_ 1 + _a_ 2. We do addition on this line by taking _ℓ_ ( _a_ 1) + _ℓ_ ( _a_ 2). If _ℓ_ ( _a_ 1) + _ℓ_ ( _a_ 2) is closest to _ℓ_ ( _a_ 1 + _a_ 2), we consider the addition problem successful. 

We then take the percentage of successful addition problems where the answer _a_ 1 + _a_ 2 is less than some threshold _α_ , and compare the accuracy as a function of _α_ for GPT-J and linear addition. Surprisingly, we find that for _α_ = 100, linear addition has an accuracy of less than 20%, while GPT-J has an accuracy of more than 80% (Fig. 36). 

Thus, even with very precise linear representations, doing linear addition leads to errors. We interpret LLMs use of modular circles for addition as a built-in redundancy to avoid errors from their imperfect representations. 



<!-- Start of picture text -->
Helix Fit Quality vs Neuron Logit Difference<br>‘Top 1% Most Important Addition Neurons, gpt-j-6B<br>_ 0.18020 @ ee &eoe @ e<br>x 016 Qo<br>3 ors eo.<br>38 oa e<br>222 010 e<br>z<br>0.08<br>vce & 6<br>107 10? 107<br>Logit Difference<br><!-- End of picture text -->

_Figure 35._ When calculating the NRMSE of the helix inspired fit for neuron preactivations, we find that more impactful neurons (with higher total effect) are typically fit better. 

# **F. Investigating Model Errors** 

Given that GPT-J implements an algorithm to compute addition rather than relying on memorization, why does it still make mistakes? For problems where GPT-J answers incorrectly with a number, we see that it is most often off by _−_ 10 (45.7%) and 10 (27.9%), cumulatively making up over 70% of incorrect numeric answers (Fig 37). We offer two hypotheses for the source of these errors: 1) GPT-J is failing to “carry” correctly when creating the _a_ + _b_ helix or 2) reading from the _a_ + _b_ helix to answer logits is flawed. 

We test the first hypothesis by analyzing the distribution of GPT-J errors. If carrying was the problem, we would expect that when the model is off by _−_ 10, the units digits of _a_ and _b_ add up to 10 or more. Using a Chi-squared test with a threshold of _α_ = 0 _._ 05, we see that the units digit of _a_ and _b_ summing to more than 10 is not more likely for when the model’s error is _−_ 10 than otherwise (Fig. 38). This falsifies our first hypothesis. Thus, we turn to understanding how the _a_ + _b_ helix is translated to model logits. 

Since MLPs most contribute to direct effect, we begin inves- 

21 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Distribution of Errors in Incorrect Addition Problems<br>1047/1945 non NaNs shown. 59.8% are negative<br>500<br>Top Error<br>400 Ga -10.0: 45.7%<br>10.0: 27.9%<br>< 300<br>3<br>8 200<br>100<br>°<br>s Ss PPoe SP S s<br>Difference Between Model Answer and True Answer<br><!-- End of picture text -->

_Figure 37._ When GPT-J incorrectly answers an addition prompt (19 _._ 5% of the time), it answers with a number more than half the time. That number is usually off by 10 or _−_ 10 from the correct answer. 



<!-- Start of picture text -->
Checking carying isthe Issue for Enor = 20<br>Brstrbuon of Ones big Sums<br>peo<br>2 006<br>a<br>oo 25 $0 75 10 ms bo 7s<br>Sum of Ones Dots<br><!-- End of picture text -->

tigating at the neuron level. We sort neurons by their direct effect, and take the _k_ = 693 highest DE neurons required to achieve 80% of the total direct effect (Fig 33). Then, we use the technique of LogitLens to understand how each neuron’s contribution boosts and suppresses certain answers (see nostalgebraist (2020) for additional details). For the tokens [0 _,_ 198] (the answer space to _a_ + _b_ ), we see that each top DE neuron typically boosts and suppresses tokens periodically (Fig. 39). Moreover, when we Fourier decompose the LogitLens of the max activating _a_ + _b_ example for each neuron, we find that a neuron whose preactivation fit’s largest term is _cTi,a_ + _b_ in Eq. 3 often has LogitLens with dominant period of _Ti_ as well (Fig. 40). We interpret this as neurons boosting and suppressing tokens with a similar periodicity that they read from the residual stream helix with. 

Despite being periodic, the neuron LogitLens are complex and not well modeled by simple trigonometric functions. Instead, we turn to more broadly looking at the model’s final logits for each problem _a_ + _b_ over the possible answer tokens [0 _,_ 198]. We note a similar distinct periodicity in Fig. 41. When we Fourier decompose the logits for all problems _a_ + _b_ , we find that the most common top period is 10 (Fig. 42). Thus, it is sensible that the most common error is _±_ 10, since _a_ + _b −_ 10, _a_ + _b_ + 10 are also strongly promoted by the model. To explain why _−_ 10 is a more common error than 10, we fit a line of best fit through the model logits for all _a_ + _b_ , and note that the best fit line almost always has negative slope (Fig. 42), indicating a preference for smaller answers. This bias towards smaller answers explains why GPT-J usually makes mistakes with larger _a_ and _b_ values (Fig. 11, Appendix A). 

# **G. Tooling and Compute** 

We used the Python library nnsight to perform intervention experiments on language models (Fiotto-Kaufman et al., 2024). All experiments were run on a single NVIDIA RTX A6000 GPU with 48GB of VRAM. With this configuration, all experiments can be reproduced in two days. 

_Figure 38._ If GPT-J was struggling to “carry” when creating the _a_ + _b_ helix, we would expect the ones digit of _a_ and _b_ to sum up to greater than 10 when the model is off by _−_ 10. However, we see that this is not significantly more likely for when the error is _−_ 10 than when it is not _−_ 10. 

22 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
#0. L20N7741 #7. L21N13478 #4, L22N12504 #10. L22N9339<br>Max Activating Example: 8+18 Max Activating Example: 17+79 Max Activating Example: 72+11 Max Activating Example: 96+54<br>4 == eter] ><br>2<br>3 1 ! 2 H<br>Zo 1 ° 1 1 1<br>S<br>3°1a !1 1<br>-2 - 1 11 -1 == 964541 Token<br>-2<br>° 100 200 0 100 200 0 100 200 ° 100 200<br>Token Token Token Token<br><!-- End of picture text -->

_Figure 39._ Using the LogitLens technique, we analyze the contributions of the top neurons presented in Fig. 8 for each neuron’s maximally activating _a_ + _b_ example. 



<!-- Start of picture text -->
Distribution of Logit Lens Fourier Periods for Top Path Patch Neurons<br>Grouped by Top Fitted Period<br>Top Input Period = 2 (40.3%) Top Input Period = 5 (18.4%)<br>40<br>100<br>5 5 20<br>g *° 8<br>°°<br>vo 8S 8 SES v o 8 & SES OS<br>‘Top Input Period = 10 (20.3%) ‘Top Input Period = 100 (20.9%)<br>60<br>€ 40 40<br>5 5<br>5284<br>°°<br>v o 8 6 SE 9 v o 8 6 Se 9<br>Top Logits Fourier Period Top Logits Fourier Period<br><!-- End of picture text -->

_Figure 40._ For all neurons with top _a_ + _b_ fit component with _T_ = [2 _,_ 5 _,_ 10 _,_ 100], we plot the distribution of the top Fourier period in their LogitLens taken over the tokens [0 _,_ 198]. We see that a neuron with top fitted period of _Ti_ often has a LogitLens with top Fourier period _Ti_ . Surprisingly, 200 is a common Fourier period, possibly used to differentiate numbers in [0 _,_ 99] from [100 _,_ 198]. 



<!-- Start of picture text -->
Linear SeSS2S sy<br>so See us<br>2" A<br>Bos<br>vue<br><!-- End of picture text -->



<!-- Start of picture text -->
Lnear Subtractedtop Fourier 8 9(0.26)<br>» i<br>.<br>En '<br>value<br><!-- End of picture text -->



<!-- Start of picture text -->
Linear LibiractTop Fouri r e d 458(0.19)<br>OE Be re ootnvea<br>-<br>Bw<br>value<br><!-- End of picture text -->

_Figure 41._ We plot the final model logits over the token space [0 _,_ 198] for some randomly selected examples. We see clear periodicity with a sharp period of 10, in addition to a general downward trend indicating a preference for smaller answers. 

23 

**Language Models Use Trigonometry to Do Addition** 



<!-- Start of picture text -->
Slope of Logits for Different a+b Prompts<br>Negative suggests a bias towards smaller answers<br>2 001 Individual arithmetic problems<br>a = = If no bias for smali/arge answers.<br>2 0.00 i<br>6<br>2 -0.01<br>5 -0.02<br>Fi£<br>= -0.03<br>270.04<br>&& -0.05<br>0 25 50 75 100 125 150 175 200<br>atb=<br><!-- End of picture text -->



<!-- Start of picture text -->
Distribution of Logits Top Period for all a+b<br>3000<br>& 2500<br>6<br>& 2000<br>2 1500<br>5<br>1000<br>=—<br>2<br>500<br>o<br>1 2 34567840 20 50 100 200<br>Period<br><!-- End of picture text -->

_Figure 42. Top_ When we plot the slope of the best fit line over all logits, we see that the slope is often negative, implying a bias towards smaller answers. _Bottom_ When applying a Fourier decomposition on the logits for all examples _a_ + _b_ over the token space [0 _,_ 198], we see that 10 is the most common period. Note that we subtract out the fitted linear component first before applying the Fourier transform. 

24 

