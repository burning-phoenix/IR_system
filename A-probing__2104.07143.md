# An Interpretability Illusion for BERT

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** A-probing — A probe can recover structure that is an artefact of the dataset rather than the model. The direct caution against W3's central instrument.
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/2104.07143.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

**An Interpretability Illusion for BERT** 

**Tolga Bolukbasi**<sup>* 1</sup> **Adam Pearce**<sup>* 1</sup> **Ann Yuan**<sup>* 1</sup> **Andy Coenen**<sup>1</sup> **Emily Reif**<sup>1</sup> **Fernanda Vi´egas**<sup>1</sup> **Martin Wattenberg**<sup>1</sup> 

# **Abstract** 

We describe an “interpretability illusion” that arises when analyzing the BERT model. Activations of individual neurons in the network may spuriously appear to encode a single, simple concept, when in fact they are encoding something far more complex. The same effect holds for linear combinations of activations. We trace the source of this illusion to geometric properties of BERT’s embedding space as well as the fact that common text corpora represent only narrow slices of possible English sentences. We provide a taxonomy of model-learned concepts and discuss methodological implications for interpretability research, especially the importance of testing hypotheses on multiple data sets. 

# **1. Introduction** 

An outstanding problem in the field of neural networks is understanding how they represent meaning. One simple hypothesis is that the activation level of an individual unit in a network encodes the presence or absence of a meaningful concept. A generalization of this idea suggests that concepts are encoded by linear combinations of neural activations. 

This point of view has proved fruitful in analyzing image networks (Bau et al., 2017; Olah et al., 2020; Kim et al., 2018). Could the same type of analysis uncover the key concepts that are learned by neural language networks? 

This paper describes a surprising phenomenon, a kind of “interpretability illusion,” that arose from such an experiment on BERT, a model for general language representations (Devlin et al., 2018). The basic set-up for our experiment (described in detail in Section 3.2) was designed to probe whether individual neurons in BERT might have humaninterpretable meaning. We gave BERT a large dataset of sentences as inputs, chose a target neuron from the final layer, and examined the sentences that maximally activated 

*Equal contribution 1Google Research, Cambridge, MA, USA. Correspondence to: Tolga Bolukbasi _<_ tolgab@google.com _>_ , Adam Pearce _<_ adampearce@google.com _>_ , Ann Yuan _<_ annyuan@google.com _>_ . 

it. Our intuition was that if these maximally-activating sentences shared a common pattern, it would be an indication that the target neuron had learned to detect this pattern. 

Indeed, many of the neurons we probed did show strong, consistent patterns of activation. For example, of the 164,246 sentences in the Quora Question Pairs dataset (Iyer et al., 2017), here are three of the sentences which activate neuron 221 in layer 12 most strongly: 

- _”What is the meaning behind the song ””Angel”” by Eric Clapton?”_ 

- _”What’s the meaning of Johnny Cash’s song ””King of the Hill””?”_ 

- _”What is the meaning behind the Tears for Fears song_ 

- _””Mad World””, such as the lyric, ””All around me are familiar faces””?”_ 

These strongly suggest that neuron 221 encodes a concept related to song titles, or perhaps the very specific syntactic structure of these sentences. Based on this evidence alone, it might be tempting to conclude that one can easily interpret the meanings of individual neurons in the final layer of BERT. 

The plot thickened, however, when we tried the same set of experiments with the same model (no fine-tuning or other modifications) but a different dataset. Using as input a question answering dataset drawn from Wikipedia (58,645 sentences), the top activating sentences for neuron 221 had nothing to do with the meanings of song titles. Instead, they included: 

- _On 16 June 2006, it was announced that Everton had entered into talks with Knowsley Council and Tesco over the possibility of building a new 55,000 seat stadium, ex-pandable to over 60,000, in Kirkby._ 

- _On 15 September 1940, known as the Battle of Britain Day, an RAF pilot, Ray Holmes of No. 504 Squadron RAF rammed a German bomber he believed was going to bomb the Palace._ 

- _On 20 August 2010, Queen’s manager Jim Beach put out a Newsletter stating that the band had signed a new contract with Universal Music._ 

Here we are forced to reconsider our initial interpretation. Based on this dataset, it would seem that neuron 221 encodes 

**An Interpretability Illusion for BERT** 

historical events, or perhaps sentences beginning with a date. 

In search of tie-breaking evidence, we repeated our analysis on a sample (198,085 sentences) from the Toronto BookCorpus dataset. The top activating sentences there included: 

- _Lara pulled out the document Reed had supplied from Gresham’s briefcase._ 

- _I take Kellan’s business card from my pocket and stretch it over to Realm._ 

- _Pilcher took a walkie-talkie out of his coat and spoke into the receiver._ 

The picture is now more complicated. The BookCorpus dataset suggests yet a third interpretation for neuron 221. Note that there’s nothing special about neuron 221: many other neurons show similar behavior. What we have seen is that for each data set, looking at maximally-activating sentences produces consistent, interpretable patterns for many neurons. These patterns, however, are _not_ consistent across datasets. We consider this to be an _interpretability illusion_ . The fact that a seemingly consistent pattern can turn out to be a mirage has clear implications for interpretability research. 

In this paper, we provide evidence that this illusion is a general, reproducible phenomenon of the BERT model. We also consider several possible explanations for the illusion, suggesting that it is useful to separate the notion of _datasetlevel_ concept directions from global concept directions. 

To summarize our main contributions: 

1. We identify an interpretability illusion that arises when analyzing a language model’s activation space. 

2. We provide the recommendation that interpretability researchers conduct their experiments on multiple datasets. 

3. We investigate possible causes of the illusion, using a taxonomy of the geometric properties of model-learned concepts: local, global, and dataset-level concept directions. 

# **2. Related work** 

There is a large body of prior work exploring the embedding spaces of language models. These embedding spaces exhibit both local and global structure. There is local structure in that the nearest neighbors of a datapoint’s embedding are similar to it (Bengio et al., 2003). Global directions in embedding spaces have also been found to encode specific concepts (Mikolov et al., 2013; Bolukbasi & Chang, 2016; Raghu et al., 2017; Olah et al., 2020; Vig et al., 2020; Li et al., 2015). In the case of language models, these directions combine to form representations that enable sophisticated language processing (Manning et al. (2020)). Indeed, following the probing method of Tenney et al. (2019), Durrani 

et al. (2020) finds that different elements of linguistic understanding can be localized to individual or small groups of neurons. 

There has been substantial work (both for images and text) on determining what a specific neuron encodes by considering which inputs maximally activate it. There are two main ways of doing this: first, by generating such inputs, which can provide clues regarding the encoded pattern (Nguyen et al., 2016; Poerner et al., 2018; Bauerle & Wexler¨ ), and second, by looking for patterns among real samples that maximally activate a neuron, where the samples are drawn from some preexisting dataset (Na et al. (2019)). 

Our paper focuses on the latter approach. To review a few more examples, Zhou et al. (2015) and Bau et al. (2017) use this technique to find neurons that respond to particular objects in natural scenes. Dalvi et al. (2019) look at language models, comparing the concepts for a neuron found by this technique to those found by probing. Szegedy et al. (2014) find that convolutional networks contain neurons that activate in response to semantically related inputs. Zeiler & Fergus (2014) obtain a similar result by visualizing the top activating images patches for a given feature. Olah et al. (2017) compare the maximally activating images from a dataset with images generated to maximize that neuron. 

There is also a body of work using concept directions for measuring and mitigating unwanted biases in models (Kaneko & Bollegala, 2019; Manzini et al., 2019; Bolukbasi & Chang, 2016). Our experiments suggest that using these techniques for sentence models without validating the directions on multiple datasets could have unintended effects. 

# **3. Establishing the illusion** 

Motivated by this body of work, we set out to find meaningful neurons and concept directions in BERT’s activation space. Our pilot experiments with a single test dataset suggested clear, consistent meanings for many neurons. However, as described in the introduction, many of those interpretations disappeared when we tried to confirm them with a different data set. 

## **3.1. Datasets** 

To test the extent of the illusion, we base our experiments on four different text corpora. 

1. Quora Question Pairs ( **QQP** ): QQP contains questions from the question-answering website Quora, with 164,246 datapoints. (Iyer et al., 2017). 

2. Question-answering Natural Language Inference ( **QNLI** ): QNLI contains passages from Wikipedia, with 58,645 datapoints. (Wang et al., 2019). 

**An Interpretability Illusion for BERT** 

3. Wikipedia ( **Wiki** ): Wiki contains a random subset of English Wikipedia as prepared in (Devlin et al., 2018), 203,736 datapoints. 

4. Toronto BookCorpus ( **Books** ): Books contains sentences from online novels. We sampled 198,085 sentences from the original data set. (Zhu et al., 2015). 

## **3.2. Experiments** 

We began by creating embeddings for the 624,712 sentences in our four datasets. To do this, we used the BERT-base uncased model from the HuggingFace Transformers library with no fine tuning or dataset specific modifications. We used the final layer hidden state of each sentence’s [ _CLS_ ] token as its embedding. 

Several methods for extracting aggregate sequence representations from BERT can be found in the literature (Reimers & Gurevych, 2019). However our method remains the default for HuggingFace pipelines, and is used in the original BERT paper. We also built an exploratory visualization (Figure 1) demonstrating that these embeddings give rise to highly coherent clusters across scales, giving us additional confidence in their representational validity. 

We then randomly picked a set of neurons and looked at their top activating sentences from each dataset. For convenience, we identify a neuron with a basis vector in BERT’s 768dimensional embedding space; that is, a one-hot vector _x_ ( _d_ ) _∈ R_<sup>_k_</sup> where: 



For each neuron we find its top activating sentences by sorting sentence embeddings according to their activation level, meaning the dot product with this vector. 

We also find top activating sentences for a set of random directions, rather than basis vectors. In this case we sort sentence embeddings according to their inner product with the random direction. Specifically, for a dataset _S_ and a vector _v_ : 



In this paper, we will refer to the dot product between a sentence embedding and a direction as their _projection score_ . 

Next we built an annotation interface that optionally shows: (1) the top ten activating sentences for a neuron, (2) the top ten activating sentences for a random direction, or (3) a random set of ten sentences. We annotated whether each set of ten sentences contained a pattern, and if so, which sentences demonstrated the pattern. During annotation we knew which dataset the sentences were drawn from, but _not_ 



<!-- Start of picture text -->
\<br>dW Yr<br>L .<br>R: Pay<br>@ QqP 7 ~ :<br>ont<br>© Wiki . . .<br>© Books fj ‘<br>.<br><!-- End of picture text -->

_Figure 1._ Visualization of sentence embeddings from the QQP, QNLI, Wiki, and Books datasets using UMAP showing that the four datasets form distinct clusters. Our methodology for extracting these embeddings is described in Section 3.2. 

whether the sentences were top activating (conditions (1), (2)) or randomly drawn (condition (3)). In total we annotated 25 neurons (randomly selected), 33 random directions, and 29 random sets of sentences (Table 1). 

To define our notion of pattern: a pattern is simply a property shared by a set of sentences. The property may be structural, i.e. the sentences are all the same length. It may also be lexical, i.e. the sentences all contain some variant of the phrase ’coat of arms’. We use these patterns as proxies for learned concepts by the model. 

# **4. The Illusion** 

Each set of sentences was annotated by two annotators. Table 1 shows how often both annotators found at least one pattern. _Conflicting_ indicates that one annotator found a pattern and the other did not. To establish a baseline, we also looked for patterns among randomly drawn sets of ten sentences. We found 14% of random sets of sentences to contain patterns, suggesting that our datasets contain intrinsic topic biases. However, these biases are not sufficient to explain our results. We found that more than 80% of top activating sentences contained patterns (Table 1). The patterns found among top activating sentences were also much stronger in that they contained more positive examples than those found in random sets of sentences (Figure 2). Finally, annotators were more in agreement about whether top activating sentences were meaningful: only 8% of annotators were split over whether a neuron was meaningful, and 18% were split over whether a random direction was meaningful, 

**An Interpretability Illusion for BERT** 

||**QQP**||**QNLI**||**Wiki**||**Books**||**All**|||
|---|---|---|---|---|---|---|---|---|---|---|---|
|**Contains patterns?**|**Yes**|**No**|**Yes**|**No**|**Yes**|**No**|**Yes**|**No**|**Yes**|**No**|**Conflicting**|
|**Neurons**|3 (60%)|1 (20%)|8 (100%)|0|6 (100%)|0|3 (50%|) 2 (33%)|20 (80%)|3 (12%)|2 (8%)|
|**Random direction**|10 (100%|) 0|10 (83%)|0|5 (100%)|0|2 (33%|) 0|27 (82%)|0|6 (18%)|
|**Random sentences**|0|5 (100%)|<br>2 (22%)|3 (33%)|<br>1 (11%)|3 (33%)|1 (17%|) 3 (50%)|4 (14%)|14 (48%)|11 (38%)|



_Table 1._ Annotation results for each dataset. For individual datasets, the remaining percentage is for conflicting annotations. 

|**QQP**|**QNLI**|**Wiki**|**Books**|
|---|---|---|---|
|Nested quotes, Colors, Mathe-<br>matics, Military conflict, Pop-<br>ulation statistics, Relationship<br>advice, School exam questions,|Biology, Geography, Technol-<br>ogy, Numbers and dates, Mili-<br>tary conflict, Population statis-<br>tics, War history, Windows 8,|Direct statement of fact, Mu-<br>sic, Sporting, Age distribution,<br>Television shows,<br>Olympic<br>facts, Legalese, Measurements,|Interpersonal relationships, Na-<br>ture, Quoted speech, Spanish,<br>Sentence fragments, Medieval<br>Europe, Very long sentences,|
|Questions of comparison, Pro-<br>gramming|Etymology|School districts|Flirtation|



_Table 2._ Sample annotated patterns for each dataset. 

compared to 38% in the case of random sets of sentences. 

Examples of our annotations are found in Table 2 (a full list can be found in the Appendix - Table 7). Many of the patterns we found are quite general, for example that a set of sentences contain quoted speech, or that they concern nature. Nevertheless, the annotations of top activating sentences often changed dramatically depending on which dataset the sentences were drawn from. In fact, for each neuron we measured on average 2.5 _distinct_ patterns across QQP, QNLI, Wiki, and Books (Figure 2). 

Our results suggest that the illusion of meaningfulness we observed in neuron 221 (Section 1) is a general, reproducible phenomenon of BERT. They also show that the illusion is equally pervasive among neurons and random directions. Based on an informal investigation of layers 2 and 7 we believe the illusion occurs for earlier layers as well, although a full analysis is beyond the scope of this paper. 

## _regions of BERT’s embedding space._ 

This would contribute to the illusion because if the four datasets occupy non-overlapping slices of BERT’s embedding space, then in any direction the top activating sentences from each dataset will come from distinct regions of the embedding space (Figure 3). 

## **Experiments** 

To test our hypothesis, we performed two experiments: 



<!-- Start of picture text -->
Number of patterns found for each neuron (= 2.5)<br>ge<br>8<br>2<br>Sa<br>“8 i ? 3 ‘<br>Number of patterns per neuron<br><!-- End of picture text -->

# **5. Explaining the illusion** 

What gives rise to this illusion? How could the same direction seemingly encode completely different concepts? 

We propose that the illusion can be traced to three sources: 

1. Dataset idiosyncrasy 

2. Local semantic coherence in BERT’s embedding space 

3. Annotator error 

Next, we discuss each source in detail. 

## **5.1. Datasets are idiosyncratic** 

First, we consider the hypothesis: **Hypothesis 1.** _QQP, QNLI, Wiki, and Books occupy distinct_ 

|**Condition**|**Mean**|**Stdev**|
|---|---|---|
|**Neurons**|6.80|2.37|
|**Random Directions**|6.89|2.13|
|**Random Sentences**|5.05|1.96|



_Figure 2._ Annotation statistics. ( **Top** ) The number of distinct patterns found for each annotated neuron across all datasets. We manually looked over the annotations and counted the number of unique patterns per neuron, grouping semantically equivalent annotations together. ( **Bottom** ) The number of sentences belonging to a pattern for the different experimental conditions for the sentence groups that are found to be meaningful. We required each pattern to have at least three positive examples. At most a pattern could have ten positive examples, because we only showed the top ten activating sentences for any given direction. 

**An Interpretability Illusion for BERT** 



<!-- Start of picture text -->
A B<br>° °<br>© oo eo?<br>@ °<br>gerbe Me<br>ee,<br>© cco<br>° ee<br>Pee<br><!-- End of picture text -->

_Figure 3._ Schematic illustration of how top activating sentences from geometrically distinct datasets could be semantically unrelated. The arrow represents a direction in embedding space. Red dots represent sentences from dataset A, and blue dots represent sentences from dataset B. The dots outlined in black represent the the top activating sentences. 

1. We built an exploratory visualization of the QQP, QNLI, Wiki, and Books dataset embeddings using the UMAP dimensionality reduction algorithm. 

2. We trained a linear SVM classifier to distinguish between the four datasets based on their sentence embeddings. 

## **Results** 

Our exploratory visualization shows that sentences cluster neatly by dataset (Figure 1). Our linear SVM is also able to distinguish between the datasets with high accuracy (Figure 4). These results suggest that QQP, QNLI, Wiki and Books represent relatively idiosyncratic slices of language, thus the top activating sentences for a given neuron from one dataset do not necessarily resemble those from another dataset despite having similar activation values—looking at all the pairs of datasets across all neurons, 38% of the ranges of the top 10 activations overlap. Our results align with previous work demonstrating that BERT representations can be used to disambiguate datasets (Aharoni & Goldberg, 2020). However, while this explains why we would find different patterns in different datasets, why should there be patterns at all? We address this question in the next section. 

## **5.2. Local semantic coherence** 

We hypothesize that another source of the illusion is local semantic coherence in BERT’s embedding space geometry. 

Before testing this, we observe: 

**Observation.** _Top activating sentences manifest patterns from both local semantic coherence and global directions._ 

This observation may seem obvious. However, suppose that a neuron encodes a certain concept: one might assume that its top activating sentences would clearly point to the en- 



<!-- Start of picture text -->
Predicted- Py<br>S222<br>oO Ff 8<br>— QQP MMH o 0 0<br>Z ON 0 BBs) 01<br>2 Wiki 0 25 (BE 03<br>Books 02 0 .02 QS<br><!-- End of picture text -->

_Figure 4._ Confusion matrix for a linear classifier trained to separate QQP, QNLI, Wiki, and Books sentence embeddings. Most datasets are easily separable in the embedding space. 



<!-- Start of picture text -->
<< 3)<br>ees<br>..<br>7<br>@<br><!-- End of picture text -->

_Figure 5._ Global versus local concepts. **Left: Global concept illustration.** Blue circles represent sentences containing a global concept. As one moves in the concept direction indicated by the arrow, the density of sentences containing the global concept increases. **Right: Local concepts illustration.** The red, yellow, and blue shapes represent sentences containing three different concepts. Sentences containing the same concept cluster together, such that there is local concept coherence throughout the space. However, there is no global direction along which any particular concept becomes more dominant. 

coded concept. Instead, we observe that the sentences may manifest patterns that do not match the encoded concept. 

Before further analysis, we describe three ways in which BERT may learn to represent concepts (Figure 5): 

_Global concepts:_ Global concepts become increasingly prevalent as one moves through the embedding space along a linear trajectory. For example, if _math_ is a global concept, then there is a direction in the embedding space such that, starting from any point and moving along that direction, one will tend to find more and more sentences relating to math. Or if the concept is _positivity_ , words like _happy, sunny, bliss, awesome_ might increase in frequency. In our analysis we use the presence of certain tokens as a proxy for concepts. By extension, if a direction in embedding space is correlated with the occurrence of certain tokens, it suggests the existence of an underlying concept direction. 

_Dataset-level concepts:_ Like global concepts, a _datasetlevel_ concept is associated with a direction in the embedding 

**An Interpretability Illusion for BERT** 

space. However, unlike global concepts, this direction is only meaningful within the region of the embedding space where samples from the dataset tend to be found. Thus, dataset-level concepts do not generalize to arbitrary inputs. 

_Local concepts:_ On the other hand, _local_ concepts emerge only as clusters in the embedding space, and lack a direction. For example, suppose that _math_ is a local concept. Then if one looks in the neighborhood of the sentence _e_ = _mc_<sup>2</sup> , one may find many other sentences containing numbers and symbols because the model groups math-related sentences together. However, there will not be any particular direction along which such sentences become increasingly prevalent. 



<!-- Start of picture text -->
®©<br>%%<br>= 40 = 40<br>Gao<br>soe om eo 100% by BO Oe. OO. EO 100%<br>©®<br>%%<br>§ 40 g 40<br>8 0 8 co<br>3 0 3 20<br>0<br>Qe vaneoe eo 100% by “20 40% Oe. EO 100%<br><!-- End of picture text -->

_Figure 6._ **(Top left)** The frequency of the word “quick” monotonically decreases as we look at sentences that increasingly activate neuron 275 in QQP. This suggests that neuron 275 encodes a global concept (Section 5.2) that relates to “quick”. **(Top right)** Most neurons, such as neuron 266, do not correlate with the frequency of “quick”. These are illustrative examples, Table 3 shows over a quarter of neuron/token pair are monotonic. 

We find evidence for both global and dataset-level concept directions in BERT’s embedding space. Figure 6 illustrates the change in certain token frequencies as one moves along various neuron directions. Some tokens, such as “weird”, monotonically change in frequency as one moves along neuron 266, while other token frequencies are unchanged. Table 3 illustrates that this is not an isolated phenomenon, rather there are many tokens that monotonically increase or decrease along different neurons. Table 3 further shows that these patterns can be unique to a particular dataset, suggesting the existence of dataset-level concept directions. We note that the baseline probability of a token (e.g. “weird”ness) being monotonic with respect to a neuron 

|**Datas**|**ets**|||**Monotonic**|**Increasing**|**Decreasing**|
|---|---|---|---|---|---|---|
|Books||||27.0%|13.4%|13.6%|
||QNLI|||22.7%|11.3%|11.4%|
|||Wiki||29.6%|14.7%|15.0%|
||||QQP|27.8%|13.9%|13.9%|
|Books|QNLI|||7.4%|3.0%|2.9%|
|Books||Wiki||9.9%|4.3%|4.3%|
||QNLI|Wiki||10.9%|5.3%|5.4%|
|Books|||QQP|9.2%|3.9%|3.9%|
||QNLI||QQP|7.7%|3.2%|3.2%|
|||Wiki|QQP|9.8%|4.0%|4.1%|
|Books|QNLI|Wiki||4.2%|1.8%|1.8%|
|Books|QNLI||QQP|3.0%|1.2%|1.2%|
|Books||Wiki|QQP|3.9%|1.7%|1.6%|
||QNLI|Wiki|QQP|4.2%|1.8%|1.9%|
|Books|QNLI|Wiki|QQP|1.9%|0.8%|0.8%|



_Table 3._ The table shows how often the token counts across quintiles of activations for each neuron/token pair are monotonically increasing or decreasing in each combination of datasets. Combinations with more datasets have darker backgrounds. There are some linear directions in the embedding space that are correlated with the same tokens across all datasets (for the 915 tokens appearing at least 100 times in each of the four datasets). 

**Tokens whose frequencies change monotonically:** ~~ee~~ **”** (125), **can** (120), **is** (99), **are** (98), **was** (97), **that** (91), **if** (88), **were** (86), **to** (85), **would** (84), **a** (82), **it** (80), **they** (78), **not** (77), **god** (76), **for** (75), **which** (73), **more** (73), **she** (70), **of** (68) 

_Table 4._ This table lists the tokens from all four datasets that are most often monotonically changing across the embedding space. For each token, we indicate in parenthesis the number of neurons for which the token changes monotonically in frequency as one moves along the neuron axis. 

is much lower than the measured rates<sup>1</sup> . Table 4 shows the most monotonic tokens across datasets, hinting that BERT may have learned to encode certain pervasive concepts as global directions, e.g. pronouns and common verbs. 

Despite the fact that global and dataset-level concept directions exist, we posit that they will be difficult to identify from top activating sentences. First, sentences typically engage with multiple concepts. Thus even if a neuron encodes a concept, its top activating sentences may have other concepts in common as well. Second, the directions we annotated are likely to themselves align with multiple concepts, and the top activating sentences for each concept may look unrelated when listed together. As an example, suppose 

> 1We use quintiles to measure monotonicity. The probability of a random set of quintiles being monotonic is 2/5! (1.7%). 

**An Interpretability Illusion for BERT** 

we have a dataset in which every sentence represents exactly one concept and we pick a direction that is a linear combination of ten global concept directions. When we look at the ten sentences that most highly activate this direction, we may not see any patterns at all. In the extreme case, we may only see one sentence representing each concept, making it impossible to identify any patterns across sentences. 

Formally we define a concept distribution as: _C_ = [ _c_ 1 _, c_ 2 _, ..., cN_ ] where<sup>�</sup><sup>_N_</sup> _i_ =1<sup>_ci_= 1</sup><sup>_._0.The concept purity of</sup> a sentence or a direction is defined by the skew of its concept distribution vector. A sentence that aligns with exactly one concept would have a one-hot concept distribution vector. 

## 5.2.1. MEASURING LOCAL CONCEPTS 

Having discussed the difficulty of identifying concept directions from top activating sentences, in this section we aim to show that: 

**Hypothesis 2.** _When annotating top activating sentences, people identify concepts emerging from local semantic coherence._ 

This would explain the illusion because local concepts are not necessarily associated with a direction in the activation space, but rather with a semantic cluster. 

We propose the following analysis to test our hypothesis. The main intuition is that if a concept we annotate emerges from local semantic coherence, then for sentences manifesting the concept, their neighborhood in the original embedding space should look very similar to their neighborhood when projected onto the concept direction<sup>2</sup> . Rather than measuring exact equality of sentences in the two neighborhoods (which is very sensitive), we compare their pairwise distance distributions. 

More formally, let _Nk_ ( _s_ ) be the _k_ nearest neighbors of a sentence _s_ : 



where _es_ denotes the embedding of a sentence in the original embedding space. Consistent with our annotation protocol we use _k_ = 10. 

Let _Sp,k_ define the top activating _k_ sentences for a direction _p_ . For each sentence _s_ in _Sp,k_ , we measure the dot product between _s_ and its nearest neighbors _s_<sup>_′_</sup> _∈ Nk_ ( _s_ ). We 

2Our analysis measures local coherence among top activating sentences. If both neighborhoods match, our analysis cannot determine whether the local coherence comes from a local or a global concept. However we know that for the directions whose meaning changes across datasets, the source of local coherence cannot be a global concept. On the other hand, a neighborhood mismatch would mean something besides local coherence is responsible for any pattern. 

call this set of distances _Dp,nearest_ . We then measure the dot product between all pairs of top activating sentences ( _Sp,k_ ), calling this set _Dp,top_ . Finally, we measure the dot product between each top activating sentence and _k_ random sentences in the dataset to establish a baseline, calling this set _Dp,random_ . 

The histograms in Figure 7 shows how _Dp,nearest_ , _Dp,top_ , and _Dp,random_ compare. On top is a neuron our annotators deemed meaningful. The pairwise distances between the top activating sentences ( _Dp,top_ ) overlap with those between the top activating sentences and their nearest neighbors in the original embedding space ( _Dp,nearest_ ). By contrast, on the bottom is a meaningless neuron, where the pairwise distances between the top activating sentences ( _Dp,top_ ) match those between the top activating and random sentences ( _Dp,random_ ). 

We extend this analysis across all the directions we annotated and propose a metric based on the Jaccard similarity: 



_L_ measures the intersection over union between any two histograms _h_ 1 and _h_ 2. We call _L_ the locality score for a direction when applied to the histograms of _Dp,nearest_ and _Dp,top_ . If Hypothesis 2 is true, _L_ should be high for meaningful directions and low for meaningless directions. 

Table 5 shows that the locality scores for meaningless neurons are indeed significantly lower than for meaningful neurons. This means that the patterns found among top activating sentences arise primarily from _local_ geometry. 

|Locality (Mean)|**All**|**QQP**|**QNLI**|**Wiki**|**Books**|
|---|---|---|---|---|---|
|Meaningful|0.026|0.012|0.014|0.037|0.042|
|Meaningless|0.010|0.003|0.008|0.016|0.012|
|p-value|0.0004|0.009|0.090|0.064|0.014|



_Table 5._ Per sentence locality scores (based on the histogram intersection over union measure). 

## 5.2.2. ANALYZING OUTLIER SENTENCES 

We present our preceding analysis of local semantic coherence as general, but our experiment targets the edges of the activation space. Perhaps the properties of local semantic coherence are unique to a small set of outlier sentences which maximally activate across the directions we annotated. 

We consider this possibility for the QQP dataset. Below are the three QQP sentences that maximally activate the most neurons: 

- _”what does snoop dogg mean by ””lolos”” in the line_ 

- _””hit the corners in them lolos girl ””in dr. dre’s 2001 hit ””still d.r.e””?”_ (56 top activations) 

**An Interpretability Illusion for BERT** 



<!-- Start of picture text -->
1 QQP Neuron 65 cose Nearest neighbors<br>3 —— Random<br>3 Vaan<br>5 7 \<br>Bef<br>5 */ \<br>2 SN<br>Pairwise distance between sentences —><br>3.286 19.139<br>tT s*0+=++=+ Nearest neighbors<br>—— Random<br>8 QaP Neuron 44 ——-—Top activating<br>on<br>° fs<br>ee<br>z => a —<br>Pairwise distance between sentences —<br>3.26 14.86<br><!-- End of picture text -->

_Figure 7._ The distribution of distances in the original embedding space for QQP’s top activating sentence for neurons 65, 44. Orange lines show the distances between the neurons’ top activating sentence and its nearest neighbors in the original embedding space. Blue lines show the distances between the top activating sentence and other top activating sentences. Green lines show the distances between the top activating sentence and randomly drawn sentences. **(Top)** A neuron deemed meaningful by annotators. Observe that the blue line significantly overlaps with the orange line, suggesting that the meaning comes from the top activating sentence’s local neighborhood. **(Bottom)** A neuron deemed meaningless by annotators. Observe the high degree of overlap between the distribution of distances between the top activating sentences and the distances between random sentences. 

- _”what are some songs like ””the dying of the light - noel gallaghar””?”_ (40 top activations) 

- _”what inspired the tv show ””skins””?”_ (32 top activations) 

Taking the pairwise Euclidean distances between every QQP sentence, these three sentences have the highest mean distances. This pattern holds across the entire distribution: the 20 most distant QQP sentences all have at least 11 top activations and the most distant 1% of sentences account for 48% of all top activations. 

These numbers suggest a lack of diversity among QQP’s outlier sentences. Indeed, looking at our experimental results, it appears that we annotated the same sentences across multiple directions: several neurons are maximally activated by nested quotation marks, several others by movie plots and several others by math equations. However most sentences are only top activating once or twice; across the 7,680 top 

ten activating sentences, there are 4,551 unique sentences. And the illusion _persists_ when the most distant 1% and 10% of sentences are omitted. 

## **5.3. Annotator error** 

A final source of the illusion is our ability to see patterns where they may not exist. As noted in Section 4, annotators often disagreed over whether a set of sentences constituted a pattern. Additionally, annotators differ significantly in their tendency to find patterns (Table 6). This suggests certain patterns may be less an objective property of the sentences than products of an individual annotator’s (overactive) imagination. 

|**Annotator**<br>|**Directions annotated**<br>|**Patterns found**<br>|
|---|---|---|
|0<br>|30<br>|27 (0.90)<br>|
|1|33|27 (0.82)|
|2|24|23 (0.92)|
|3|40|23 (0.56)|
|4|52|30 (0.58)|
|5|25|40 (1.6)|



_Table 6._ The total number of annotations and the number of patterns found per annotator. The number of patterns can exceed the number of annotations because a single direction may exhibit multiple patterns. 

# **6. Conclusion** 

In this work we investigated how BERT represents meaning by looking for patterns among top activating sentences along directions in the model’s activation space. In doing so, we uncovered a counterintuitive “illusion”: seemingly consistent patterns that turn out to be contingent on the test dataset. We then identified several possible sources of this behavior: the idiosyncratic nature of widely used natural language datasets, the geometry of BERT’s activation space, and annotator bias. Furthermore, despite these “illusory” findings, we also provide evidence that there do exist meaningful global directions in BERT’s activation space. 

Several avenues for future work suggest themselves. It would be interesting to replicate our analysis in other settings, such as with other sentence models, with token-level embeddings, and with earlier layers of BERT. It would also be natural to look for similar illusions in other types of input: images, graphs, and so forth. In any of these contexts, there is clearly much work to be done in characterizing the geometry of concept representation. 

Our work also draws needed attention to the role of datasets in interpretability research. It is a classic problem in machine learning that a model trained on one dataset may not perform well on another. The results we have described illustrate something slightly different: how an _interpreta-_ 

**An Interpretability Illusion for BERT** 

_tion_ that seems valid for one dataset may not generalize to other contexts. Examining how interpretations function across different data distributions may be a helpful step in establishing their validity. 

- Kaneko, M. and Bollegala, D. Gender-preserving debiasing for pre-trained word embeddings. _CoRR_ , abs/1906.00742, 2019. URL http://arxiv.org/ abs/1906.00742. 

**Acknowledgements** We would like to thank Ian Tenney, Jasmijn Bastings, Katherine Lee and Been Kim for their reviews and helpful discussions during the process of writing this paper. 

- Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In _International conference on machine learning_ , pp. 2668–2677, 2018. 

# **References** 

   - Li, J., Chen, X., Hovy, E. H., and Jurafsky, D. Visualizing and understanding neural models in NLP. _CoRR_ , abs/1506.01066, 2015. URL http://arxiv.org/ abs/1506.01066. 

- Aharoni, R. and Goldberg, Y. Unsupervised domain clusters in pretrained language models. In _Association for Computational Linguistics_ , pp. 7747–7763, 2020. 

   - Manning, C., Hewitt, J., Clark, K., Khandelwal, U., and Levy, O. Emergent linguistic structure in artificial neural networks trained by self-supervision. _Proceedings of the National Academy of Sciences of the United States of America_ , 2020. 

- Bau, D., Zhou, B., Khosla, A., Oliva, A., and Torralba, A. Network dissection: Quantifying interpretability of deep visual representations. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pp. 6541–6549, 2017. 

   - Manzini, T., Lim, Y. C., Tsvetkov, Y., and Black, A. W. Black is to criminal as caucasian is to police: Detecting and removing multiclass bias in word embeddings. _CoRR_ , abs/1904.04047, 2019. URL http://arxiv.org/ abs/1904.04047. 

- Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. A neural probabilistic language model. _The journal of machine learning research_ , 3:1137–1155, 2003. 

- Bolukbasi, T. and Chang, K.-W. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In _Proceedings of the 30th International Conference on Neural Information Processing Systems_ , pp. 4356–4364, 2016. 

   - Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space, 2013. 

   - Na, S., Choe, Y. J., Lee, D.-H., and Kim, G. Discovery of natural language concepts in individual units of cnns. 2019. In the Proceedings of ICLR. 

- Bauerle,¨ A. and Wexler, J. What does bert dream of? URL https://pair-code.github. io/interpretability/text-dream/ explainable/. 

   - Nguyen, A., Dosovitskiy, A., Yosinski, J., Brox, T., and Clune, J. Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. In _Advances in Neural Information Processing Systems_ , pp. 3387–3395, 2016. 

- Dalvi, F., Durrani, N., and Sajjad, H. What is one grain of sand in the desert? analyzing individual neurons in deep nlp models. In _Proceedings of the 33rd AAAI Conference on Artificial Intelligence_ , pp. 6309–6317, 2019. 

   - Olah, C., Mordvintsev, A., and Schubert, L. Feature visualization. _Distill_ , 2017. doi: 10.23915/distill.00007. https://distill.pub/2017/feature-visualization. 

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_ , 2018. 

   - Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. Zoom in: An introduction to circuits. _Distill_ , 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in. 

- Durrani, N., Dalvi, F., Sajjad, H., and Belinkov, Y. Analyzing individual neurons in pre-trained language models. 2020. URL https://arxiv.org/pdf/2010. 02695.pdf. 

- Poerner, N., Roth, B., and Schutze,¨ H. Interpretable textual neuron representations for NLP. In _Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_ , pp. 325–327, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/ v1/W18-5437. URL https://www.aclweb.org/ anthology/W18-5437. . 

Iyer, S., Dandekar, N., and Csernai, K. First quora dataset release: Question pairs, 2017. URL https://data.quora.com/ First-Quora-Dataset-Release-Question-Pairs. 

**An Interpretability Illusion for BERT** 

- Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability, 2017. 

- Reimers, N. and Gurevych, I. Sentence-bert: Sentence embeddings using siamese bert-networks. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing_ , pp. 3982––3992, 2019. 

- Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks. In _International Conference on Learning Representations_ , 2014. URL http: //arxiv.org/abs/1312.6199. 

- Tenney, I., Xia, P., Chen, B., Wang, A., Poliak, A., McCoy, R. T., Kim, N., Durme, B. V., Bowman, S., Das, D., and Pavlick, E. What do you learn from context? probing for sentence structure in contextualized word representations. In _International Conference on Learning Representations_ , 2019. 

- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Sakenis, S., Huang, J., Singer, Y., and Shieber, S. Causal mediation analysis for interpreting neural nlp: The case of gender bias, 2020. 

- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. GLUE: A multi-task benchmark and analysis platform for natural language understanding. 2019. In the Proceedings of ICLR. 

- Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. In _European conference on computer vision_ , pp. 818–833. Springer, 2014. 

- Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A. Object detectors emerge in deep scene cnn. In _International Conference on Learning Representations_ , 2015. 

- Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., and Fidler, S. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books, 2015. 

**An Interpretability Illusion for BERT** 

|**QQP**|**QNLI**|**Wiki**|**Books**|
|---|---|---|---|
|School exam questions, High<br>school math and physics, Com-<br>parison questions, Indian jobs,<br>Jobs for students, Large body<br>physics, Movies, Music, Math,<br>Physics, Indian cities, Race,<br>Science, Technology and con-<br>sulting, Containing multiple<br>quoted phrases, Hygiene, Tech<br>support, Drugs, India, Pro-<br>gramming, Mathemtical and<br>chemical formulas, Cosmol-<br>ogy, Containing quoted sub-<br>jects, Internet companies, Cor-<br>porate jargon, Quoted songs,<br>Indian paperwork|Military<br>conflict,<br>Ethnic<br>groups<br>(language,<br>culture,<br>history), Numbers and dates,<br>Standards,<br>Named<br>places<br>(state, university, etc.), DNA,<br>Text book material, Population<br>statistics / census, War history,<br>Military moves, Geographic<br>locations,<br>Africa<br>(mostly<br>Somalia), Engineering, Gov-<br>ernmental power, Geography,<br>Language / linguistics, Group<br>theory,<br>Politics,<br>Military,<br>Governments<br>and<br>leaders<br>and<br>revolutions,<br>Displaced<br>people and slavery, Animals,<br>Science,<br>Renovations<br>and<br>infrastructure<br>changes,<br>US<br>southern cities and especially<br>in NC, Countries, Bible quotes<br>and locations, Resettling and<br>boundaries, War, Histories of<br>ethnic groups, Math, Census<br>data,<br>Military<br>maneuvers,<br>Somalia / Eritrea, Law and<br>legislative bodies, Historical<br>events, Etymology, Windows<br>8,<br>Mathematical<br>groups,<br>Years, Political conflict and<br>revolution, Political history,<br>Animals, Family, Geography,<br>Ecology, Census, Chemistry,<br>Weather history,<br>Municipal<br>facts, Indian, Demographics,<br>Weather, Raleigh|Settlement history, Direct state-<br>ments of fact, Named locations,<br>Voting stats of cities, Birth<br>family occupations of individ-<br>uals, Locations, British his-<br>tory, Age distribution, Census<br>data, TV shows, Olympians,<br>Daughters, War, Law / con-<br>tracts, International sports, Me-<br>dia, School districts, National<br>borders, Music, Properties of<br>villages, Anatomical descrip-<br>tions, Indiginous communities,<br>Chemistry, Printing presses,<br>Math,<br>Statistics,<br>Television<br>shows, Legalese, French / Ger-<br>man municipalities, Measure-<br>ments, Rules, Minerals, Soccer,<br>Medicine, School rules, Bands|Spanish, Description then talk-<br>ing, Snippets describing peo-<br>ple, Fantasy travel, Quotes,<br>Body parts, Plans, Person do-<br>ing a small physical action<br>(laughing, rubbing hands, etc),<br>French or Spanish, Different<br>forms of ”what´s up? how´s it<br>going?”, Sentence fragments<br>(long noun phrases), Short<br>statements about a character<br>performing an action, About<br>medieval European islands /<br>cold places, Long blocks of<br>quoted text, Short fragments<br>separated by commas, Long<br>blocks of text, Flirtation, Mil-<br>itary, Non-English, Questions,<br>Violence|



_Table 7._ Annotated patterns for each dataset. Edited for brevity, duplicates removed. 

**An Interpretability Illusion for BERT** 

# **A. Normalization** 

We decided to conduct our analysis on raw embeddings, which are un-normalized. We found that the top activating sentences for directions in the embedding space were in general the same with and without normalization, therefore we decided to keep the vector space as close to how the neural network utilizes it as possible by not normalizing. 

In addition, we observed that most vector norms are concentrated around _∼_ 14 which is about _sqrt_ (768) _/_ 2. This is about the norm of a vector which has all elements equal to 0.5. We also observed that the outputs of the neurons typically fall between -1.0 and 1.0. 

