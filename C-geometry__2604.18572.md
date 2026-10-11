# Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale

**arXiv:** 2604.18572v2 · **Source:** https://arxiv.org/html/2604.18572v2
**Route:** arXiv LaTeXML HTML -> markdown, math preserved as $...$ from alttext. Equations still resolve against the PDF, not this file.
**Authors:** A. Sophia Koepke, Daniil Zverev, Shiry Ginosar, Alexei A. Efros
**Corpus section:** Tyler Wheeler/interpretability
**Cited by:** NOT CITED BY WHEELER. Added 2026-10-01 for research/SSL.
**Why it is here:** 2026: the convergence evidence weakens as the evaluation set grows.
**Fetched:** 2026-10-01 by utils/fetch_corpus.py

---
# Back into Plato’s Cave: Examining Cross-modal Representational Convergence at Scale

A. Sophia Koepke  Affiliation: Technical University Munich, MCML  Daniil Zverev  Affiliation: Technical University Munich, MCML  Shiry Ginosar  Affiliation: Toyota Technical Institute at Chicago  Alexei A. Efros  [1.8ex] UC Berkeley  [1ex] University of Tübingen  Tübingen AI Center 

###### Abstract

The Platonic Representation Hypothesis [42] suggests that neural networks trained on different modalities (e.g., text and images) align and eventually converge toward the same representation of reality. If true, this has significant implications for whether modality choice matters at all. We show that the experimental evidence for this hypothesis is fragile and depends critically on the evaluation regime. Alignment is measured using mutual nearest neighbors on small datasets ( $\approx$ 1K samples) and degrades substantially as the dataset is scaled to millions of samples. The same behavior is observed beyond text-image, for text-audio and text-video alignment. The alignment that remains between model representations reflects coarse semantic overlap rather than consistent fine-grained structure. Moreover, the evaluations in Huh et al. [42] are done in a one-to-one image-caption setting, a constraint that breaks down in realistic many-to-many settings and further reduces measured alignment. We also find that the reported trend of stronger language models increasingly aligning with vision does not appear to hold for newer models. Overall, our findings suggest that the current evidence for cross-modal representational convergence is considerably weaker than subsequent works have taken it to be. Models trained on different modalities may learn equally rich representations of the world, just not the same one.   
Project page: https://akoepke.github.io/cave_umwelten

## 1 Introduction

The success of Large Language Models (LLMs) is causing much hand-wringing in the computer vision community: do we even need pixels to build machines that understand our world, or is language “all you need”?

Several works have demonstrated that models trained only on text data successfully solve what were thought to be fundamentally visual problems, such as visual question answering (VQA) [32, 40], visual reasoning [11, 39, 98, 2], or embodied robotics applications [1, 57]. This resonates with the suggestion that text data may make other modalities redundant [88], on the premise that the part of the world that is relevant to humans is manifest in language. On the other hand, it is argued that linguistic data alone cannot yield genuine understanding [9] or allow actual embodiment. After all, there is a reason we visit art museums rather than just read descriptions of paintings in a catalogue. This raises a central question: how do models trained on different modalities represent reality?

The Platonic Representation Hypothesis [42] offers a compelling answer: as neural networks grow larger and consume more data, their learned representations will become more and more aligned, no matter which data modality (text, vision, audio, touch, etc.) was used for training. Proponents of language-only learning have interpreted this as validation of their approach: since the choice of modality does not matter as they all lead to the same shared representation, one might as well use language as the most convenient source of data.11 1 But analogously, the same argument could be made for vision-only learning [43]. However, the strength of a hypothesis depends on the strength of its evidence, and the experimental protocol underpinning the claim rests on specific methodological choices that have largely gone unexamined in subsequent work.

Figure 1: Illustration of the mutual nearest neighbor metric used by Huh et al. [42] to measure cross-modal alignment. (a) Sparse regime: given a query image and caption (blue), nearest neighbors (NN) are retrieved independently in image and text embedding spaces. Mutual NN alignment measures whether the NNs are consistent across modalities. (b) Dense regime: as dataset size increases, NNs within each modality get better. The vision model retrieves a car in the same pose, and the language model retrieves a caption of the same car model regardless of pose. At scale, improved within-modality organization does not translate into cross-modal agreement. 

In this paper, we take a closer look at the experimental evidence for the hypothesis and find it to be fragile and to depend critically on the evaluation regime. Huh et al. [42] conducted their analysis on small, sparse datasets with one-to-one correspondences between modalities. However, real-world multi-modal data is large, dense, and inherently many-to-many: one image has many valid descriptions, and a single caption can correspond to many plausible images. These differences fundamentally change what it means for two representations to “align”.

In a small dataset, weakly related samples may become nearest neighbors simply because no better alternatives exist (Fig. 1a). Here, two models can agree despite organizing their representations differently. As the dataset grows (i.e. the gallery used for retrieving nearest neighbors gets denser), both models find closer neighbors and cross-modal consistency requires more fine-grained structural alignment (Fig. 1b). A vision model may retrieve an image of a car taken from a similar angle as the query, while the language model retrieves a caption describing the same car model as the query but in a different pose. Both are valid, but inconsistent between modalities, producing a mismatch that gets penalized under the mutual nearest-neighbor metric. This illustrates how mutual nearest-neighbor agreement becomes an increasingly strict measure for alignment in many-to-many regimes. Using a mutual $k$ NN metric with $k>1$ is less strict, but does not fundamentally change the conclusion.

In this paper, we examine how cross-modal alignment changes in evaluation settings with large, dense, and non-bijective datasets, and observe the following:

Alignment degrades with scale: Increasing the gallery from $1024$ to millions of samples causes a sharp drop in cross-modal mutual nearest-neighbor agreement. Coarse agreement persists but fine-grained agreement does not: In controlled settings (e.g., ImageNet), vision and language models reliably retrieve correct-class neighbors but rarely agree on the same instance. Many-to-many correspondence reduces alignment: Allowing multiple valid correspondences per sample leads to drops in agreement, even when retrieved neighbors are semantically sensible. Previously reported trends may not hold for newer models: The claim that stronger language models align better with vision seems to weaken for more recent models.

These findings paint a more mixed picture than the small-gallery results from Huh et al. suggest. Models trained on different modalities can learn rich and semantically meaningful structure, yet still organize that structure differently. Low agreement does not imply poor representations, it reflects differences in how information is arranged. These patterns are not specific to text and images. We observe similar behavior at scale for text-audio and text-video alignment. Nearly a century ago, von Uexküll [95] argued that every organism inhabits its own perceptual world, or Umwelt, shaped by its senses rather than by an observer-independent reality. The same, we believe, might hold for our models: each constructs its own representational structure, determined by its modality and training data, rather than converging toward a shared model of reality. Though it is still early days, we suspect future evidence will favor von Uexküll over Plato.

## 2 Related Work

One Platonic Ideal vs many Umwelten. In his “Theory of Forms”, Plato argued that every physical object we perceive is a flawed imitation (a shadow) of some eternal, abstract “ideal” form [76], and only by escaping from the tyranny of our physical senses (leaving the cave of shadows), we can achieve true understanding. But in the 20th century, this argument for a single, unified Platonic Ideal representation has been repeatedly undercut by biologists, psychologists, and philosophers. Biologist von Uexküll argued that every organism inhabits its own perceptual environment, or Umwelt [95]: a tick lives in a world of thermal gradients, a bat in a world of echoes. The different Umwelten might have only little overlap with each other22 2 For a tour of von Uexküll’s ideas, see Koenderink’s delightful book [48].. Gibson’s ecological psychology [27] pushed this further, proposing that perception is shaped by what an organism can do in its environment, not by an observer-independent reality. Philosopher Wittgenstein, thinking about language, arrived at a strikingly similar conclusion. He famously argued: “If a lion could speak, we could not understand him” [102], meaning that the lion’s world (goals, instincts, perceived reality) is so utterly different from our own, that even if it spoke English, we would not comprehend the meaning33 3 For a great treatment of Wittgenstein’s argument in popular culture, see the episode Darmok of the American TV series Star Trek: The Next Generation.. Building on Wittgenstein, psychologist Rosch developed her Prototype Theory of Categorization [80], arguing against a single platonic ideal as a representation of object categories, proposing a data-driven clustering-based model instead.

Representational alignment. The question of representational similarity has been studied extensively in the neurosciences [21, 35, 50]. In machine learning, the parallel question of whether independently trained networks learn similar internal structure has received growing attention. Lenc and Vedaldi [54] investigated the equivalence of representations from different trained models and found that early convolutional layers are more interchangeable than later ones. This task, also referred to as “model stitching”, was later revisited by Bansal et al. [6]. Related to this, Li et al. [56] proposed methods to align neurons across independently trained networks. More recently, Dravid et al. [20] introduced “Rosetta Neurons,” showing that different vision models share common units corresponding to similar visual concepts across architectures, tasks, and training data. Tasker et al. [89] propose the Universal Normal Embedding hypothesis, where encoder and generative latents are interpreted as noisy views of an approximately Gaussian shared space.

Furthermore, alignment has been linked with shared model capabilities measured by task performances [5, 47, 42, 69, 6]. To directly quantify representational similarities, several metrics have been used to measure correlations between features [69, 38]. Kornblith et al. [49] introduced Central Kernel Alignment (CKA) as a robust measure invariant to orthogonal transformations and isotropic scaling. Huh et al. [42] found the CKA metric to reveal only a “very weak trend of alignment between models” and therefore proposed the use of the mutual $k$ NN metric that measures the overlap of two sets of neighborhoods of size $k$ .

Multi-modal alignment. Early efforts to connect images and text utilized human annotations [96]. The curation of large-scale paired image-caption datasets, such as MS-COCO [58] and Visual Genome [51], facilitated the systematic study of cross-modal correspondence and models. The CLIP model [78] by Radford et al. formed a turning point by demonstrating that contrastive learning on web-scale image-text pairs could produce shared embedding spaces.

Since, a growing body of work has investigated whether such alignment arises even without explicit joint training. Merullo et al. [67] showed that a simple learned linear transformation could map between frozen vision encoders and LLMs. Moschella et al. [70] use similarities to an anchor set. Maniparambil et al. [64] demonstrated that even unaligned unimodal encoders possess high semantic similarity. Li et al. [55] study alignment at the level of shared categories. Complementary to that, Lu et al. [63] define each sample by its angular distances to others, improving training-free alignment across vision, language, and audio.

Fully unsupervised approaches include blind vision-language matching [82] and unpaired embedding translation via cycle-consistency [44, 107]. Finally, Gupta et al. [33] show that an orthogonal map can map between independently trained multi-modal contrastive models. These results are often seen as evidence for representational convergence, also supported by Lu et al. [62] in their survey of multimodal alignment. However, those results are obtained in restricted settings (e.g. [82] experiments on CIFAR-100 and ImageNet-100) and do not scale to real-world multi-modal data. Our work examines whether alignment survives beyond these constraints, showing that it decreases at scale and reflects coarse categorical agreement rather than shared fine-grained structure.

Limits and measurement of emergent cross-modal structure. Several analyses show that alignment between independently trained unimodal encoders depends strongly on data, architecture, and evaluation protocol. Tjandrasuwita et al. [92] find that alignment varies with modality similarity and the balance of shared versus unique information, while Hadgi et al. [34] report weaker alignment for “pure” 3D encoders without careful subspace selection. Zhu et al. [108] further show that video–text alignment depends on temporal richness and text availability.

Gröger et al. [31] show that global similarity measures such as CKA are sensitive to network scale and can be altered via null calibration, largely removing evidence of global convergence while leaving local neighborhood similarity (e.g., mutual $k$ NN) more stable, though still evaluated under small-scale and bijective regimes. Beyond similarity metrics, Smith et al. [86] and Kumar et al. [52] show that functional agreement and output behavior can persist even when internal representations are misaligned or entangled, suggesting that behavioral compatibility does not imply shared structure. These caveats echo grounding arguments that text-only learning may be insufficient to recover perceptual structure [7, 53], and motivate multimodal foundation models that integrate perception and language at scale [41, 4, 68, 37].

## 3 Experimental setup

Mutual $k$ NN metric. To measure alignment between representations from different models, we use the mutual $k$ -nearest-neighbor metric (illustrated in Fig. 1), following Huh et al. [42]. Given a shared gallery set of $n$ datapoints (referred to as mini-batch sampled from the data distribution in [42]) encoded into feature vectors $\mathbf{a_{i}}\in\mathbb{R}^{d_{1}}$ and $\mathbf{b_{i}}\in\mathbb{R}^{d_{2}}$ by two models with $i\in\{1,\cdots,n\}$ , we first L2-normalize each representation. We then retrieve the $k$ nearest neighbors of every query point independently for each model (e.g. image and text query for vision and language encoders):

|  $$\mathcal{N}^{\mathbf{a}}_{k}(i)=\operatorname{argtopk}_{j\neq i}\mathbf{a}_{i}^{\top}\mathbf{a}_{j},\qquad\mathcal{N}^{\mathbf{b}}_{k}(i)=\operatorname{argtopk}_{j\neq i}\mathbf{b}_{i}^{\top}\mathbf{b}_{j}.$$  |   
---|---|---  
  
The per-sample score is the number of overlapping samples normalized by $k$ :

|  $$s_{i}=\frac{|\mathcal{N}^{\mathbf{a}}_{k}(i)\cap\mathcal{N}^{\mathbf{b}}_{k}(i)|}{k},$$  |   
---|---|---  
  
and the overall mutual- $k$ NN score is the mean over all samples. A score of 1 means that every point’s $k$ nearest neighbors are identical in both spaces, and a score of 0 means that the $k$ nearest neighbors do not overlap. In the sparse gallery in Fig. 1, the query (blue) retrieves the same neighbor in both image and text spaces, giving a mutual $k$ NN score of 1 for $k{=}1$ . A score of $\frac{k}{n}$ suggests chance-level expected overlap for independent random retrieval, which decreases for growing $n$ . Note that throughout we report raw mutual $k$ NN (as in [42]).

Figure 2:  Nearest-neighbor quality depends on data density. We show $10$ within-modality nearest neighbors for image (DINOv2) and text (LLM) embeddings on a sparse WIT-1024 gallery (top) and a denser WIT-1M gallery (bottom). For text queries, retrieved captions and their corresponding reference images are shown. At smaller scale, nearest neighbors are less semantically precise. Nearest-neighbor structure becomes more semantically refined as gallery density increases. 

Implementation details. For most experiments, we use DINOv2-base [74] as the vision encoder. We refer to this model as DINOv2 in the following. Our primary language model is OpenLlama3b [93, 26] (abbreviated as OpenLlama). Additional models are considered in the supplementary material (Section A.4). For each image and text sample, we extract the representations from all layers of their respective encoders and follow the experimental protocol from [42]. Details about additional models used in Section 4 are provided in Section E.3.2 in the supplementary material. We use Faiss [19] for nearest neighbor computation at scale. Specifically, we use their exact nearest neighbor implementation with IndexFlatL2, equivalent to using cosine similarity on normalized vectors.

(a) Effect of neighborhood sizes $k$ in mutual $k$ NN (both axes log-scaled). Trivially, mutual $k$ NN converges to 1.0 as $k$ approaches the full gallery size. [42] utilized mutual $k$ NN for $k=10$ .

(b) Alignment between DINOv2 and different LLMs, measured on WIT-1024 and WIT-1M. As observed in [42], alignment (mutual $k$ NN) increases with language performance (measured as $1-\text{bitsperbyte}$ from [42]) on the 1024-sample set, but this trend breaks with larger gallery size.

Figure 3: Mutual $k$ NN text-image feature alignment when scaling from WIT-1024 to WIT-1M. (a) shows the dependence on neighborhood size $k$ , while (b) examines alignment for different LLMs. The observation from [42], that more capable language models align better with vision at fixed small $k$ largely vanishes at WIT-1M scale.

## 4 How much do representations align?

In this section, we take a close look at the experimental evidence underpinning the Platonic Representation Hypothesis [42]. The experiments in [42] rest on two foundations that warrant scrutiny: the use of mutual $k$ NN alignment on a small evaluation set of only 1024 samples from the Wikipedia Image-Text (WIT) dataset [87] (WIT-1024), and the use of data with effectively bijective (one-to-one) image-text correspondences, an assumption implicit in the WIT-1024 setup rather than stated explicitly in [42]. Typically, these choices are not acknowledged when the hypothesis is cited [65, 81, 60, 10, 14]. The claim is usually invoked in its broad, appealing form rather than in the narrow terms under which experimental support was provided.

Here, we analyze how alignment behaves for a finer-grained metric ( $k{=}1$ instead of $k{=}10$ ), and a denser gallery (million(s of) instead of 1024 samples). We then decompose what mutual $k$ NN alignment actually measures in a controlled setup on ImageNet. This reveals that models individually retrieve correct-class neighbors but rarely agree on which one, suggesting that information is organized differently in each unimodal model. We then turn to the bijective assumption, and examine what happens when it is relaxed. We further test whether these patterns extend beyond text-image to the text-audio and text-video settings. Finally, we perform a trend check to ask whether the predictions from [42] have held up as models have improved.

Sensitivity to $k$ in mutual $k$ NN. Huh et al. [42] reported mutual $k$ NN alignment for $k{=}10$ . We additionally evaluate at $k{=}1$ , which requires the two representation spaces to agree on the single nearest neighbor. As shown in Fig. 3(a), the metric trivially converges to 1 as $k$ approaches the full gallery size $n$ , since both neighbor sets then contain all samples. Even moderate values of $k$ can inflate scores by capturing broadly similar rather than precisely matching neighbors. In our analyses at larger gallery scales, we perform deduplication to prevent near-duplicate samples from trivially inflating neighborhood overlap (see Section E.1 in the supplementary material for details).

### 4.1 Alignment across dataset scales

Nearest neighbors in sparse gallery. We now turn to the data, and ask whether the 1024-sample gallery used in [42] is too sparse to capture more than coarse structural agreement.

Table 1: Nearest-neighbor quality across gallery sizes. As the gallery grows, nearest neighbors get closer to the query set in both DINOv2 and OpenLlama embedding spaces, facilitating a more fine-grained analysis of cross-modal alignment.

Gallery | Model |  $k{=}1$  |  $k{=}10$   
---|---|---|---  
WIT-1024 | DINOv2 | 0.799 | 0.717  
WIT-1024 | OpenLlama | 0.502 | 0.400  
WIT-1M | DINOv2 | 0.906 | 0.888  
WIT-1M | OpenLlama | 0.757 | 0.701  
  
As shown in Table 1, the mean cosine similarity between queries and nearest neighbors in terms of both image (DINOv2) and text features (OpenLlama) is significantly lower for the WIT-1024 gallery compared to WIT-1M (e.g. 0.799 compared to 0.906 for DINOv2 at $k{=}1$ ). Note that in both cases, we use WIT-1024 as the query set.

We visualize nearest neighbors for $k{=}10$ for image and text features on WIT-1024 and WIT-1M in Fig. 2. At low density, semantically unrelated samples may end up as nearest neighbors as there is nothing closer available, meaning that measured mutual $k$ NN agreement mainly can reflect the shared lack of alternatives. To get more meaningful insights, we scale the density of the retrieval gallery in the following section.

Figure 4: Scaling the gallery size to 1M (WIT) and 15M (LAION) shows a large drop in mutual $k$ NN alignment for $k{=}1$ and $k{=}10$ for DINOv2 and OpenLlama features.

Densification by scaling the gallery size. Having established that the WIT-1024 gallery captures mainly coarse structure, we densify the gallery and test whether alignment persists. We evaluate on up to 1M and 15M gallery samples from the English-text WIT [87] and LAION400M [83] respectively. The best layer pair was determined on the 1024-sample subset of WIT (see ablation in Section A.2 in the supplementary material), following [42].

As shown in Fig. 4, alignment scores decrease as gallery size grows for fixed $k$ and query set (WIT-1024). The mutual $k$ NN alignment scores drop from 0.135 and 0.058 on the 1024-sample gallery to 0.008 and 0.001 on LAION-15M for $k=10$ and $k=1$ respectively. This confirms that the agreement observed at small scale declines with the transition to finer-grained evaluation at large scale. Nearest neighbors become closer and more semantically similar to the query, placing greater demand on the two representation spaces to agree on subtle distinctions.

Interestingly, alignment at $k{=}n/100$ remains relatively stable across scales, suggesting that models share some degree of coarse structural agreement. We hypothesize that this amounts to precisely the kind of broad categorical correspondence one would expect from models trained on overlapping internet data, and lacks signal about whether representations are organized in the same way.

We also analyze how mutual $k$ NN alignment for various LLMs and DINOv2 behaves at the WIT-1M scale. Reproducing the setting of [42], Fig. 3(b) shows a clear trend on WIT-1024: stronger language models exhibit higher alignment with visual features. This is a central finding of [42] and one of their most compelling pieces of evidence. However, when we scale the gallery to 1M samples, this trend largely vanishes for fixed small $k$ . The gap between LLMs narrows considerably, and the relationship between model capability and alignment weakens. This suggests that the observation in [42] may be a result of the sparse evaluation setting.

Re-running this analysis at $k{=}\frac{n}{100}$ on WIT-1M (Fig. 14 in the supplementary material), the alignment trend matches the $k{=}10$ trend on WIT-1024. This is consistent with the stable broad categorical correspondence across scales that one expects from models trained on overlapping web data. It appears that the original mutual- $k$ NN evidence establishes coarse semantic-category overlap, not the fine-grained representational convergence the Platonic Representation Hypothesis is taken to claim. If modalities truly converged to the same representation, they would have to agree on fine-grained structure, not merely on category.

The nearest-neighbor examples in Figs. 5 and 6 further illustrate this. Matches at small gallery sizes often break down as candidates are added, with each modality finding better but divergent neighbors. The matches at 1M and 15M scale are mostly near-duplicates our deduplication missed (e.g. a crop shifted by a few pixels). We show additional visualizations in Figs. 26, 27, 28 and 29 in the supplementary material.

Figure 5: Nearest-neighbor ( $k{=}1$ ) examples with DINOv2 and OpenLlama across gallery scales on WIT-1M. Captions are shown with corresponding images. Mutual $k$ NN matches across modalities are framed green. While the bottom example shows a match at 1M scale, at larger scales each model finds closer but different matches (top two). Figure 6: Nearest-neighbor ( $k{=}1$ ) examples with DINOv2 and OpenLlama across gallery scales on LAION-15M. As the gallery densifies, each model finds closer but different matches (top example). The match at 15M (bottom right) is a near-duplicate that survived our deduplication pipeline.

We additionally test whether the alignment drop with increasing gallery size is merely an artifact of the mutual $k$ NN metric being harder at scale. Specifically, we measure within-modality alignment for two pairs of models: two language models of different scale (OpenLlama-3b and OpenLlama-13b), and, separately, two vision models (DINOv2-base and DINOv2-giant). If mutual $k$ NN alignment collapses for dense galleries regardless of the models being compared, the cross-modal drop observed would be uninformative. If within-modality alignment remains stable, the cross-modal drop is meaningful.

As shown in the supplementary material (Fig. 13), unimodal alignment remains stable across gallery sizes. For the OpenLlama pair, mutual $k$ NN at $k{=}1$ stays between $[0.59,0.62]$ , and for the DINOv2 pair between $[0.35,0.45]$ , across all gallery scales. This confirms that mutual $k$ NN does not inherently collapse at scale.

Observation 1: Mutual $k$ NN alignment scores decrease for denser galleries for fixed small $k$ , suggesting that mutual $k$ NN is sensitive to gallery sparsity. Furthermore, the reported trend that stronger language models align better with vision weakens substantially for fixed small $k$ at WIT-1M scale, with all models scoring near zero.

(a) The query image (left) is matched with galleries of increasing density. As the gallery becomes more dense, DINOv2 and OpenLlama retrieve from the same class, but different instances, illustrating how within-class structure is organized differently across modalities.

(b) Per-modality retrieval accuracy and cross-modal mutual $k$ NN alignment ( $k{=}1$ ) as images / captions per class in gallery increase. Modalities individually improve with gallery density, but alignment does not.

Figure 7: Decomposing cross-modal alignment on ImageNet val. (a) shows a qualitative retrieval example where both models find plausible neighbors but disagree on the specific instance. (b) quantifies this: individual class-level retrieval accuracy improves with gallery density, yet strict alignment remains flat, illustrating that models organize within-class structure differently.

### 4.2 What is captured by cross-modal mutual $k$ NN alignment?

Low mutual $k$ NN alignment at fixed small $k$ could mean two things: the models individually retrieve poor neighbors, or they each retrieve good neighbors but different ones. The stability at $k=\frac{n}{100}$ hints at the models agreeing at a coarse level but diverging on fine-grained structure. To test this, we use the ImageNet [16] validation set, where class labels let us evaluate each model’s retrieval independently.

We decompose each query into: (i) whether each model individually retrieves a correct-class neighbor, (ii) whether both do, and (iii) whether they agree on the exact same gallery item (mutual $k$ NN with $k{=}1$ ). Our query set consists of one image per class (1000 images), and we vary the number of images per class (ipc) in the gallery from 1 to 49. We use detailed image captions (981 words on average) generated by gemini-3-flash-preview [91, 75], making this a favorable setting for alignment (details are provided in Section E.2 in the supplementary material).

Fig. 7b reveals that as the gallery densifies, in line with Cover and Hart [13], both models individually improve at retrieving correct-class neighbors. This indicates some degree of shared coarse structure. At larger scale, both models retrieve reasonable neighbors but different ones. We see similar trends when looking at coarser evaluation with $k{=}10$ (see Section B.3 in the supplementary material). At 49 images per class in the gallery, DINOv2 succeeds 46.1% of the time and OpenLlama 58.0%.

Yet strict alignment on the exact same gallery item remains flat around 11%, even with detailed captions. For reference, alignment with captions that only consist of the class dropping from 0.42 to near zero as ipc increases.

Figure 8: Shared mistake at ipc=1. The query image (bookstore) is matched by both DINOv2 and OpenLlama to a library image. The models agree, but on the wrong answer.

The models are individually capable but organize within-class structure differently (Fig. 7(a)). At ipc=1, strict alignment (23.1%) actually exceeds the rate at which both models retrieve a correct-class neighbor (11.7%), meaning the models often agree on semantically plausible but technically incorrect neighbors (Fig. 8).

This reveals what mutual $k$ NN actually captures. It does not measure unimodal representation quality, but agreement on fine-grained structure. Our experiments provide direct evidence that low cross-modal alignment in terms of mutual $k$ NN is not due to poor representations but rather due to fundamentally different representational organization within modalities. Both models learn structured, high-quality representations. They simply do not structure them the same way.

Observation 2: As the data gets denser, both models retrieve correct-class neighbors at increasing rates, yet strict cross-modal mutual $k$ NN alignment is flat at 11%. This is not a failure of unimodal representation quality, but of unaligned representational organization across modalities.

Figure 9: Effect of relaxing the bijective assumption on text-image alignment, using the CycleReward dataset [3]. We densify one modality by adding more images per caption (left) or more captions per image (right) while keeping the other fixed. Mutual $k$ NN alignment decreases consistently for both $k{=}1$ and $k{=}10$ .

### 4.3 What happens when the data is not bijective?

Figure 10: Illustration of non-bijective (many-to-many) correspondence between image and captions. The nearest neighbor of a text caption for one image (blue) is a caption for a different image (red). However, the nearest image neighbor for a given image may be another image with the same caption.

In practice, the relationship between modalities, such as image and text, is inherently many-to-many: a single image can be described by countless text descriptions, and a single text caption can correspond to a large set of visually distinct images. More fundamentally, modalities often differ in information content.

Specifically, images encode spatial, textural, and perceptual structure that text captures only to a limited extent. On the other hand, text encodes abstraction, negation, and compositional semantics that images do not. One could, in principle, bridge this gap trivially. For instance, one could encode pixel values as text or render captions as images and establish a bijection between those. Those preserve the information, but the inductive structure (the modality-specific properties that make each modality useful) of each modality is lost.

To test what happens when bijectivity is relaxed, we use the CycleReward dataset [3] which pairs each real sample with multiple synthetic candidates. The I2T subset contains 11 generated captions per real image, and T2I consists of 12 synthetic images for each text prompt. This directly breaks the bijection, i.e. one-to-one matching, that our earlier analysis assumes.

We evaluate mutual $k$ NN by densifying one modality at a time: for T2I we keep the text fixed and increase the number of generated images per prompt, and for I2T we keep the image fixed and add generated captions. For illustration, let us consider the T2I experiments where the closest neighbor in the densified modality is more likely to be a similar image that is associated with the same caption, while in the sparse text space the NN will be a caption for a different image. This creates a scenario where mutual $k$ NN fails (see Fig. 10).

We adapt the mutual $k$ NN metric so that a match is counted when the retrieved item corresponds to the same source sample, even if it is not the exact same caption or image. This is a generous relaxation within the mutual- $k$ NN framework. In Fig. 9, we see that the mutual $k$ NN scores still decrease as the one-to-one assumption is relaxed. This drop may reflect genuinely weaker alignment, or simply that mutual $k$ NN cannot capture alignment well once a query has multiple valid correspondences. Distinguishing these would require a metric designed for many-to-many settings. Nevertheless, the convergence evidence in [42] rests on a bijective pairing that real-world multi-modal data rarely satisfies, and the standard metric does not extend cleanly beyond it.

Observation 3: A single image can be described in countless ways, and a single caption can match many visually distinct images. When we progressively relax the one-to-one assumption, mutual $k$ NN alignment drops consistently. However, the mutual $k$ NN metric cannot distinguish between genuine misalignment and many-to-many correspondence.

### 4.4 Text-audio and text-video alignment

To test whether our findings extend beyond the text-image setting, we evaluate alignment between LLMs and a video encoder (VideoMAE-v2 [97]) on the PVD-100k dataset, a 100k subset of PVD [8, 100], and between LLMs and an audio encoder (Dasheng [17]) on a 100k subset of LAION-Audio630k [103].

Fig. 11 shows that the same patterns observed for text-image alignment hold across these modality pairs. Here, the LLM used is Gemma-2-9b [24]. Alignment at fixed small $k$ drops with gallery size, while alignment at $k{=}n/100$ remains stable across scales. The trend that stronger LLMs align better with the non-text modality (Fig. 11(c)) is weak: a slight positive slope for text-video and essentially flat for text-audio.

Our findings again suggest that models independently trained on different modalities organize information differently. Across text-image, text-video, and text-audio pairs, cross-modal alignment is limited (at fine granularity), consistent with the broader picture of modality-specific organizational structure rather than convergence to a shared representation.

(a) Text-video alignment.

(b) Text-audio alignment.

(c) Alignment vs. LLM performance.

Figure 11: Cross-modal alignment beyond text-image. (a, b) Mutual $k$ NN text-video (on PVD-100k [8, 100]) and text-audio alignment (on LAION-Audio-100k [103]) scales similarly to the text-image setting. (c) Alignment vs. LLM performance at 1k: weak trend for video, flat for audio.

### 4.5 Trend check: Are the predictions from [42] holding up so far?

Huh et al. [42] predict that as LLMs become stronger, their representations align more with vision representations. This claim is evaluated using three proxies for language performance: HellaSwag [106], GSM8K [12], and $(1-\text{bitsperbyte})$ .

In this section, we revisit this trend analysis for an extended set of models and benchmarks. We follow the original experimental setup from [42] to evaluate 55 LLMs, spanning from BLOOMZ [71] to recently released models. We use the ARC Challenge [11], MMLU [36], and LogiQA2 [61] benchmarks. These probe arithmetic reasoning, general knowledge, and logical reasoning. We present results for models that surpass Llama-3-70B [29] (the strongest model in [42]) on at least one benchmark, testing whether the alignment-performance trend continues. The complete results for all models and benchmarks are included in Appendix D in the supplementary material.

Figure 12:  Testing whether the alignment-LLM performance trend from [42], tested on WIT-1024, holds for recent LLMs on ARC, GSM8K, MMLU, and LogiQA2. Dashed lines show the trend for models from [42] (circles). Recent LLMs (diamonds) do not follow the trend: stronger language models do not seem to be more aligned with DINOv2. 

Our observations depend on the benchmark used to measure language-model performance. For benchmarks close to those used in [42], such as HellaSwag and Wikitext, the trend partially extends to newer models, with positive $R^{2}_{\text{new}}$ of $0.30$ and $0.49$ . Those benchmarks measure next-token prediction and commonsense language understanding, closely tied to the autoregressive pretraining objective. However, these benchmarks are increasingly saturated for recent LLMs. We therefore focus on broader reasoning and knowledge benchmarks, which provide a more stringent test. On these benchmarks (Fig. 12), we observe that recent models do not continue the scaling lines extrapolated from the model set from [42]. Instead, their points do not follow the predicted trend and hint at saturation with respect to DINOv2 features. The $R^{2}$ ranges from $-1.41$ to $-0.58$ , confirming that the extrapolated trend is not continued by recent models.

## 5 Discussion and future work

The Platonic Representation Hypothesis suggests that models trained on different modalities converge toward a shared representation of reality as they scale. Our results indicate a more conditional interpretation of this claim. Mutual $k$ NN agreement is highly sensitive to the evaluation regime: it drops sharply when moving from small galleries to million-scale datasets and degrades further under many-to-many cross-modal correspondences. Moreover, the previously reported trend that stronger language models yield higher alignment does not consistently hold for recent models.

These findings do not rule out shared structure across modalities. Overall, our results indicate that small-scale mutual nearest-neighbor evaluations may overstate the degree of convergence by relying on restrictive gallery sizes and one-to-one pairing. Low mutual $k$ NN agreement should not be conflated with weak representations. As our analysis on ImageNet shows, it may instead reflect differences in how fine-grained structure is organized. Rather than converging on a single Platonic representation of reality, modalities appear to inhabit their own Umwelten, a distinct but coherent representational “cave” where alignment between them is local and partial.

Future work: in search of bijection. Prior work on the Platonic Representation Hypothesis [42] evaluates alignment under a one-to-one correspondence assumption between modalities. We have shown that mutual $k$ NN agreement is not reliable once this assumption is relaxed. Likewise, interpretations of the Platonic Representation Hypothesis as evidence that “language is all you need” [99] often rely on evaluation regimes that effectively assume bijective structure between images and text.

However, real-world image–text data is fundamentally many-to-many, and the extent to which any approximate bijection exists at the level of representations remains unclear. A key direction for future work is to directly test this assumption, for example by studying whether language can serve as a lossless bottleneck for image reconstruction (i.e. an image-text-image autoencoder). If, as we suspect, this proves illusive for realistic settings (e.g. text bottlenecks of under a thousand words), it would be very interesting to identify and model the part of the joint text-image space forming a bijection (the intersection of the Venn diagram), and disentangle it from the parts that do not.

##### Acknowledgments.

This work was in part supported by the BMFTR (FKZ: 16IS24060), the DFG (SFB 1233, project number: 276693517), NSF IIS-2403305, and ONR MURI. This research utilized compute resources at the Tübingen Machine Learning Cloud. The authors thank all Efros group members for valuable discussions that shaped this work, and particularly Tyler Bonnen and Amil Dravid for proofreading the draft. Lastly, we thank Phillip Isola for the thought-provoking hypothesis, and for discussing and engaging openly with our disagreements – a rare kind of intellectual generosity.

## References

  * [1] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al.  Do as i can, not as i say: Grounding language in robotic affordances.  _arXiv preprint arXiv:2204.01691_ , 2022. 
  * [2] E. Akyürek, M. Damani, A. Zweiger, L. Qiu, H. Guo, J. Pari, Y. Kim, and J. Andreas.  The surprising effectiveness of test-time training for few-shot learning.  _arXiv preprint arXiv:2411.07279_ , 2024. 
  * [3] H. Bahng, C. Chan, F. Durand, and P. Isola.  Cycle consistency as reward: Learning image-text alignment without human preferences.  _arXiv preprint arXiv:2506.02095_ , 2025. 
  * [4] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al.  Qwen3-vl technical report.  _arXiv preprint arXiv:2511.21631_ , 2025. 
  * [5] R. Balestriero et al.  A spline theory of deep learning.  In _ICML_ , 2018. 
  * [6] Y. Bansal, P. Nakkiran, and B. Barak.  Revisiting model stitching to compare neural representations.  In _NeurIPS_ , 2021. 
  * [7] E. M. Bender and A. Koller.  Climbing towards NLU: On meaning, form, and understanding in the age of data.  In D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, editors, _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , 2020. 
  * [8] D. Bolya, P.-Y. Huang, P. Sun, J. H. Cho, A. Madotto, C. Wei, T. Ma, J. Zhi, J. Rajasegaran, H. Rasheed, et al.  Perception encoder: The best visual embeddings are not at the output of the network.  _arXiv preprint arXiv:2504.13181_ , 2025. 
  * [9] J. Browning and Y. LeCun.  Ai and the limits of language.  _Noema Magazine_ , 2022. 
  * [10] W. Chai, E. Song, Y. Du, C. Meng, V. Madhavan, O. Bar-Tal, J.-N. Hwang, S. Xie, and C. D. Manning.  Auroracap: Efficient, performant video detailed captioning and a new benchmark.  In _ICLR_ , 2025. 
  * [11] F. Chollet.  On the measure of intelligence.  _arXiv preprint arXiv:1911.01547_ , 2019. 
  * [12] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman.  Training verifiers to solve math word problems.  _arXiv preprint arXiv:2110.14168_ , 2021. 
  * [13] T. Cover and P. Hart.  Nearest neighbor pattern classification.  _IEEE transactions on information theory_ , 1967. 
  * [14] G. Dar.  mini-vec2vec: Scaling universal geometry alignment with linear transformations.  _arXiv preprint arXiv:2510.02348_ , 2025. 
  * [15] DeepSeek-AI, D. Guo, et al.  Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.  _arXiv preprint arXiv:2501.12948_ , 2025. 
  * [16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei.  Imagenet: A large-scale hierarchical image database.  In _CVPR_ , 2009. 
  * [17] H. Dinkel, Z. Yan, Y. Wang, J. Zhang, Y. Wang, and B. Wang.  Scaling up masked audio encoder learning for general audio classification.  In _Interspeech_ , 2024. 
  * [18] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby.  An image is worth 16x16 words: Transformers for image recognition at scale.  In _ICLR_ , 2021. 
  * [19] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou.  The faiss library.  _arXiv preprint arXiv:2401.08281_ , 2024. 
  * [20] A. Dravid, Y. Gandelsman, A. A. Efros, and A. Shocher.  Rosetta neurons: Mining the common units in a model zoo.  In _ICCV_ , 2023. 
  * [21] S. Edelman.  Representation is representation of similarities.  _Behavioral and brain sciences_ , 1998. 
  * [22] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou.  The language model evaluation harness.  _Zenodo_ , 07 2024.  doi: 10.5281/zenodo.12608602.  URL https://zenodo.org/records/12608602. 
  * [23] G. D. Gemma Team.  Gemma: Open models based on gemini research and technology.  _arXiv preprint arXiv:2403.08295_ , 2024a. 
  * [24] G. D. Gemma Team.  Gemma 2: Improving open language models at a practical size.  _arXiv preprint arXiv:2408.00118_ , 2024b. 
  * [25] G. D. Gemma Team.  Gemma 3 technical report.  _arXiv preprint arXiv:2503.19786_ , 2025. 
  * [26] X. Geng and H. Liu.  Openllama: An open reproduction of llama, 2023.  URL https://github.com/openlm-research/open_llama. 
  * [27] J. J. Gibson.  _The Ecological Approach to Visual Perception_.  Houghton Mifflin, Boston, 1979.  ISBN 978-0898593019. 
  * [28] A. Gokaslan and V. Cohen.  Openwebtext corpus.  http://Skylion007.github.io/OpenWebTextCorpus, 2019. 
  * [29] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al.  The llama 3 herd of models.  _arXiv preprint arXiv:2407.21783_ , 2024. 
  * [30] D. Groeneveld, I. Beltagy, P. Walsh, A. Bhagia, R. Kinney, O. Tafjord, A. H. Jha, H. Ivison, I. Magnusson, Y. Wang, S. Arora, D. Atkinson, R. Authur, K. R. Chandu, A. Cohan, J. Dumas, Y. Elazar, Y. Gu, J. Hessel, T. Khot, W. Merrill, J. Morrison, N. Muennighoff, A. Naik, C. Nam, M. E. Peters, V. Pyatkin, A. Ravichander, D. Schwenk, S. Shah, W. Smith, E. Strubell, N. Subramani, M. Wortsman, P. Dasigi, N. Lambert, K. Richardson, L. Zettlemoyer, J. Dodge, K. Lo, L. Soldaini, N. A. Smith, and H. Hajishirzi.  Olmo: Accelerating the science of language models.  In _ACL_ , 2024. 
  * [31] F. Gröger, S. Wen, and M. Brbić.  Revisiting the platonic representation hypothesis: An aristotelian view.  _arXiv preprint arXiv:2602.14486_ , 2026. 
  * [32] S. Gu, C. Clark, and A. Kembhavi.  I can’t believe there’s no images! learning visual tasks using only language supervision.  In _ICCV_ , 2023. 
  * [33] S. Gupta, S. Kansal, S. Jegelka, P. Isola, and V. Garg.  Canonicalizing multimodal contrastive representation learning.  In _ICLR_ , 2026. 
  * [34] S. Hadgi, L. Moschella, A. Santilli, D. Gomez, Q. Huang, E. Rodolà, S. Melzi, and M. Ovsjanikov.  Escaping plato’s cave: Towards the alignment of 3d and text latent spaces.  In _CVPR_ , 2025. 
  * [35] J. V. Haxby, M. I. Gobbini, M. L. Furey, A. Ishai, J. L. Schouten, and P. Pietrini.  Distributed and overlapping representations of faces and objects in ventral temporal cortex.  _Science_ , 2001. 
  * [36] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt.  Measuring massive multitask language understanding.  In _ICLR_ , 2021. 
  * [37] W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan, et al.  Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning.  _arXiv preprint arXiv:2507.01006_ , 2025. 
  * [38] H. Hotelling.  Relations between two sets of variates.  In _Breakthroughs in statistics: methodology and distribution_. 1992. 
  * [39] X. Hu, S. Storks, R. L. Lewis, and J. Chai.  In-context analogical reasoning with pre-trained language models.  In _ACL_ , 2023a. 
  * [40] Y. Hu, H. Hua, Z. Yang, W. Shi, N. A. Smith, and J. Luo.  Promptcap: Prompt-guided image captioning for vqa with gpt-3.  In _ICCV_ , 2023b. 
  * [41] S. Huang, L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K. Mohammed, B. Patra, Q. Liu, K. Aggarwal, Z. Chi, J. Bjorck, V. Chaudhary, S. Som, X. Song, and F. Wei.  Language is not all you need: Aligning perception with language models.  In _NeurIPS_ , 2023. 
  * [42] M. Huh, B. Cheung, T. Wang, and P. Isola.  The platonic representation hypothesis.  In _ICML_ , 2024. 
  * [43] P. Isola.  Personal communication, 2025. 
  * [44] R. Jha, C. Zhang, V. Shmatikov, and J. X. Morris.  Harnessing the universal geometry of embeddings.  In _NeurIPS_ , 2025. 
  * [45] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. Renard Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. El Sayed.  Mistral 7B.  _arXiv preprint arXiv:2310.06825_ , 2023. 
  * [46] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al.  Mixtral of experts.  _arXiv preprint arXiv:2401.04088_ , 2024a. 
  * [47] J. Jiang, J. Zhou, and Z. Zhu.  Tracing representation progression: Analyzing and enhancing layer-wise similarity.  _arXiv preprint arXiv:2406.14479_ , 2024b. 
  * [48] J. J. Koenderink.  _Sentience_.  De Clootcrans Press, Trajectum, Netherlands, 2019. 
  * [49] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton.  Similarity of neural network representations revisited.  In _ICML_ , 2019. 
  * [50] N. Kriegeskorte, M. Mur, and P. A. Bandettini.  Representational similarity analysis-connecting the branches of systems neuroscience.  _Frontiers in systems neuroscience_ , 2008. 
  * [51] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, et al.  Visual genome: Connecting language and vision using crowdsourced dense image annotations.  _IJCV_ , 2017. 
  * [52] A. Kumar, J. Clune, J. Lehman, and K. O. Stanley.  Questioning representational optimism in deep learning: The fractured entangled representation hypothesis.  _arXiv preprint arXiv:2505.11581_ , 2025. 
  * [53] Y. LeCun et al.  A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27.  _Openreview_ , 2022. 
  * [54] K. Lenc and A. Vedaldi.  Understanding image representations by measuring their equivariance and equivalence.  In _CVPR_ , 2015. 
  * [55] J. Li, Y. Kementchedjhieva, C. Fierro, and A. Søgaard.  Do vision and language models share concepts? a vector space alignment study.  _Transactions of the Association for Computational Linguistics_ , 2024. 
  * [56] Y. Li, J. Yosinski, J. Clune, H. Lipson, and J. Hopcroft.  Convergent learning: Do different neural networks learn the same representations?  In _ICLR_ , 2016. 
  * [57] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng.  Code as policies: Language model programs for embodied control.  In _ICRA_ , 2023. 
  * [58] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick.  Microsoft coco: Common objects in context.  In _ECCV_ , 2014. 
  * [59] A. H. Liu, S. Subramanian, V. Jouault, A. Sadé, et al.  Ministral 3.  _arXiv preprint arXiv:2601.08584_ , 2026. 
  * [60] D. Liu, S. Zhao, L. Zhuo, W. Lin, Y. Xin, X. Li, Q. Qin, Y. Qiao, H. Li, and P. Gao.  Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining.  _arXiv preprint arXiv:2408.02657_ , 2025. 
  * [61] H. Liu, J. Liu, L. Cui, Z. Teng, N. Duan, M. Zhou, and Y. Zhang.  Logiqa2.0: The logicqa dataset for logical reasoning.  _IEEE Transactions on Audio, Speech, and Language Processing_ , 2023. 
  * [62] J. Lu, H. Wang, Y. Xu, Y. Wang, K. Yang, and Y. Fu.  Representation potentials of foundation models for multimodal alignment: A survey.  _EMNLP_ , 2025a. 
  * [63] J. Lu, H. Wang, K. Yang, Y. Zhang, S. Jenni, and Y. Fu.  The indra representation hypothesis for multimodal alignment.  In _NeurIPS_ , 2025b. 
  * [64] M. Maniparambil, R. Akshulakov, Y. A. D. Djilali, S. Narayan, M. E. A. Seddik, K. Mangalam, and N. E. O’Connor.  Do vision and language encoders represent the world similarly?  In _CVPR_ , 2024. 
  * [65] P. Marcos-Manchón and L. Fuentemilla.  Shared representations in brains and models reveal a two-route cortical organization during scene perception.  _arXiv preprint arXiv:2507.13941_ , 2026. 
  * [66] S. Merity, C. Xiong, J. Bradbury, and R. Socher.  Pointer sentinel mixture models.  _arXiv preprint arXiv:1609.07843_ , 2016. 
  * [67] J. Merullo, L. Castricato, C. Eickhoff, and E. Pavlick.  Linearly mapping from image to text space.  In _ICLR_ , 2023. 
  * [68] Meta AI.  The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, 2025.  URL https://ai.meta.com/blog/llama-4-multimodal-intelligence/. 
  * [69] A. S. Morcos, M. Raghu, and S. Bengio.  Insights on representational similarity in neural networks with canonical correlation.  In _NeurIPS_ , 2018. 
  * [70] L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, and E. Rodolà.  Relative representations enable zero-shot latent space communication.  In _ICLR_ , 2023. 
  * [71] N. Muennighoff, T. Wang, L. Sutawika, A. Roberts, S. Biderman, T. L. Scao, M. S. Bari, S. Shen, Z.-X. Yong, H. Schoelkopf, X. Tang, D. Radev, A. F. Aji, K. Almubarak, S. Albanie, Z. Alyafeai, A. Webson, E. Raff, and C. Raffel.  Crosslingual generalization through multitask finetuning.  In _ACL_ , 2023. 
  * [72] T. OLMo, P. Walsh, L. Soldaini, D. Groeneveld, K. Lo, S. Arora, A. Bhagia, Y. Gu, S. Huang, M. Jordan, N. Lambert, D. Schwenk, O. Tafjord, T. Anderson, D. Atkinson, F. Brahman, C. Clark, P. Dasigi, N. Dziri, A. Ettinger, M. Guerquin, D. Heineman, H. Ivison, P. W. Koh, J. Liu, S. Malik, W. Merrill, L. J. V. Miranda, J. Morrison, T. Murray, C. Nam, J. Poznanski, V. Pyatkin, A. Rangapur, M. Schmitz, S. Skjonsberg, D. Wadden, C. Wilhelm, M. Wilson, L. Zettlemoyer, A. Farhadi, N. A. Smith, and H. Hajishirzi.  2 olmo 2 furious.  _arXiv preprint arXiv:2501.00656_ , 2025. 
  * [73] OpenAI.  Introducing gpt-oss, 2025.  URL https://openai.com/index/introducing-gpt-oss/. 
  * [74] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski.  Dinov2: Learning robust visual features without supervision.  _TMLR_ , 2024. 
  * [75] S. Pichai, D. Hassabis, and K. Kavukcuoglu.  A new era of intelligence with Gemini 3.  Google Blog (The Keyword), Nov. 2025.  URL https://blog.google/products-and-platforms/products/gemini/gemini-3/.  Accessed: 2026-01-01. 
  * [76] Plato.  Republic.  c. 375 BC. 
  * [77] A. C. Qwen Team.  Qwen3 technical report.  _arXiv preprint arXiv:2505.09388_ , 2025. 
  * [78] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever.  Learning transferable visual models from natural language supervision.  In _ICML_ , 2021. 
  * [79] I. Research.  Granite 3.3 8b base, 2025.  URL https://huggingface.co/ibm-granite/granite-3.3-8b-base. 
  * [80] E. Rosch.  Principles of categorization.  In E. Rosch and B. B. Lloyd, editors, _Cognition and Categorization_ , pages 27–48. Lawrence Elbaum Associates, 1978. 
  * [81] J. Ruan, A. Abudula, X. Liu, B. Li, Y. Li, C. Wang, Y. Fan, Y. Ge, T. Xiao, and J. Zhu.  Ndp: Next distribution prediction as a more broad target.  _arXiv preprint arXiv:2408.17377_ , 2024. 
  * [82] D. Schnaus, N. Araslanov, and D. Cremers.  It’s a (blind) match! towards vision-language correspondence without parallel data.  In _CVPR_ , 2025. 
  * [83] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki.  Laion-400m: Open dataset of clip-filtered 400 million image-text pairs.  _arXiv preprint arXiv:2111.02114_ , 2021. 
  * [84] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela.  Public multimodal dataset (PMD).  URL https://huggingface.co/datasets/facebook/pmd. 
  * [85] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela.  Flava: A foundational language and vision alignment model.  In _CVPR_ , 2022. 
  * [86] D. Smith, H. Mannering, and A. Marcu.  Functional alignment can mislead: Examining model stitching.  In _ICML_ , 2025. 
  * [87] K. Srinivasan, K. Raman, J. Chen, M. Bendersky, and M. Najork.  Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning.  In _ACM SIGIR conference on research and development in information retrieval_ , 2021. 
  * [88] I. Sutskever.  “the mastermind behind gpt-4 and the future of ai” — eye on a.i. (podcast, season 2 episode 118).  https://podcasts.apple.com/us/podcast/ilya-sutskever-the-mastermind-behind-gpt-4-and/id1438378439?i=1000604382855, mar 2023.  Accessed: 2026-02-28. 
  * [89] C. Tasker, R. Betser, E. Gofer, M. Y. Levi, and G. Gilboa.  The universal normal embedding.  In _CVPR_ , 2026. 
  * [90] F.-L. Team.  The falcon 3 family of open models, 2024.  URL https://huggingface.co/blog/falcon3. 
  * [91] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al.  Gemini: a family of highly capable multimodal models.  _arXiv preprint arXiv:2312.11805_ , 2023. 
  * [92] M. Tjandrasuwita, C. Ekbote, L. Ziyin, and P. P. Liang.  Understanding the emergence of multimodal representation alignment.  In _ICML_ , 2025. 
  * [93] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al.  Llama: Open and efficient foundation language models.  _arXiv preprint arXiv:2302.13971_ , 2023a. 
  * [94] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al.  Llama 2: Open foundation and fine-tuned chat models.  _arXiv preprint arXiv:2307.09288_ , 2023b. 
  * [95] J. B. Uexküll and G. Kriszat.  _Streifzuge durch die Umwelten von Tieren und Menschen Ein Bilderbuch unsichtbarer Welten_.  Springer, 1934. 
  * [96] L. Von Ahn, S. Ginosar, M. Kedia, and M. Blum.  Improving image search with phetch.  In _ICASSP_ , 2007. 
  * [97] L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao.  Videomae v2: Scaling video masked autoencoders with dual masking.  In _CVPR_ , 2023a. 
  * [98] R. Wang, E. Zelikman, G. Poesia, Y. Pu, N. Haber, and N. D. Goodman.  Hypothesis search: Inductive reasoning with language models.  _arXiv preprint arXiv:2309.05660_ , 2023b. 
  * [99] S. L. Wang, P. Isola, and B. Cheung.  Words that make language models perceive.  _arXiv preprint arXiv:2510.02425_ , 2025. 
  * [100] X. Wang, J. Wu, J. Chen, L. Li, Y.-F. Wang, and W. Y. Wang.  Vatex: A large-scale, high-quality multilingual dataset for video-and-language research.  In _ICCV_ , 2019. 
  * [101] R. Wightman.  Pytorch image models, 2019.  URL https://github.com/huggingface/pytorch-image-models. 
  * [102] L. Wittgenstein.  _Philosophical Investigations_.  Wiley-Blackwell, 1953. 
  * [103] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Shier.  Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation.  In _ICASSP_ , 2023. 
  * [104] A. Young, B. Chen, C. Li, C. Huang, G. Zhang, G. Zhang, G. Wang, H. Li, J. Zhu, J. Chen, et al.  Yi: Open foundation models by 01.ai.  _arXiv preprint arXiv:2403.04652_ , 2024. 
  * [105] C. Zauner.  Implementation and benchmarking of perceptual image hash functions.  2010\. 
  * [106] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi.  Hellaswag: Can a machine really finish your sentence?  In _ACL_ , 2019. 
  * [107] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros.  Unpaired image-to-image translation using cycle-consistent adversarial networks.  In _ICCV_ , 2017. 
  * [108] T. Zhu, T. Han, L. Guibas, V. Pătrăucean, and M. Ovsjanikov.  Dynamic reflections: Probing video representations with text alignment.  In _ICLR_ , 2026. 


Supplementary Material   
Back into Plato’s Cave: Examining Cross-modal Representational Convergence at Scale

## Appendix A Is the drop in mutual $k$ NN alignment at scale caused by the metric, layer selection, caption quality, or by specific model choices?

In this section, we provide additional experiments to verify that the main findings in the paper are not due to a confounding variable.

### A.1 Sanity check: does mutual $k$ NN inherently drop at scale even within modalities?

We test whether the alignment drop with increasing gallery size (Fig. 4) is merely an artifact of the mutual $k$ NN metric being harder at scale.

Specifically, we measure within-modality alignment for two pairs of models: two language models of different scale (OpenLlama-3b and OpenLlama-13b), and, separately, two vision models (DINOv2-base and DINOv2-giant). If mutual $k$ NN alignment collapses for dense galleries regardless of the models being compared, the cross-modal drop observed in the paper would be uninformative. If within-modality alignment remains stable, the cross-modal drop is meaningful.

(a) OpenLlama-3b and OpenLlama-13b

(b) DINOv2-base and DINOv2-giant

Figure 13: Unimodal mutual $k$ NN alignment as a function of gallery size on WIT-1M. In contrast to cross-modal alignment (Fig. 4), unimodal alignment remains significantly more stable across scales.

As shown in Fig. 13, unimodal alignment remains much more stable across gallery sizes than the cross-modal alignment reported in the main paper. For the OpenLlama pair, mutual $k$ NN at $k{=}1$ stays between $[0.59,0.62]$ , and for the DINOv2 pair between $[0.35,0.45]$ , across all gallery scales. This confirms that mutual $k$ NN does not inherently collapse at scale, and that the degradation observed for cross-modal pairs reflects an actual property of the representation spaces.

We additionally verify that the alignment trend occurs at the coarse $k{=}\frac{n}{100}$ level: Fig. 14 compares the WIT-1024 trend at $k{=}10$ with the WIT-1M trend at $k{=}\frac{n}{100}$ , and the two largely match. This indicates that the original mutual- $k$ NN evidence establishes coarse semantic-category overlap rather than fine-grained representational convergence.

Figure 14: Alignment vs. LLM performance: the WIT-1024 trend at $k{=}10$ compared to WIT-1M at $k{=}\frac{n}{100}$ . Unlike the fixed- $k$ case (Fig. 3(b) in the main paper), the capability trend persists for $k{=}n/100$ , consistent with stable coarse overlap rather than fine-grained convergence.

### A.2 Sanity check: is the alignment drop an artifact of the layer pair selected at WIT-1024?

For our experiments, we select the best layer pair on the 1024-sample subset of WIT and reuse this pair across larger-scale evaluations. Since our central claim concerns how alignment behaves as the gallery densifies, we test whether the observed patterns persist when the layer pair is re-selected at each gallery scale.

Specifically, we re-run the layer search at each gallery size in $\{1024,10\text{K},50\text{K},100\text{K},500\text{K},1\text{M}\}$ and report the resulting mutual $k$ NN alignment. Fig. 15 shows the comparison: (a) re-selecting layers per scale and per $k$ (i.e., the optimal layer pair for $k{=}1$ , $k{=}10$ , and $k{=}\frac{n}{100}$ separately at each gallery size), and (b) re-selecting layers per scale at fixed $k{=}10$ and applying the same layer pair to other $k$ values. Exhaustive layer-pair search inflates alignment scores by selection. We follow the null calibration proposed by Gröger et al. [31], subtracting the expected alignment under permuted image-text pairings to correct for this bias (shown as dotted lines).

In both cases, the alignment trend with increasing gallery size is preserved. This indicates that the alignment drop for fixed small $k$ reported in the main paper does not suffer from an artifact of the layer pair being fixed at WIT-1024 scale.

(a) Per-scale layer re-selection performed independently for each $k$ . The alignment trends are preserved.

(b) Per-scale layer re-selection performed at $k{=}10$ and applied to all $k$ values. The same trend is observed.

Figure 15: Cross-modal mutual $k$ NN alignment on WIT-1M with the layer pair re-selected at each gallery scale, rather than fixed at WIT-1024. Dotted lines show calibrated scores following [31]. The qualitative scaling trend is unchanged, indicating that the alignment drop is not driven by suboptimal layer choices.

### A.3 WIT-1M-recap: Is the alignment drop caused by poor captions?

One might hypothesize that the alignment drop at scale is driven by low quality of the WIT captions rather than by a fundamental cross-modal difference. To test this, we recaption WIT-1M using gemini-3-flash-preview as described in Section E.2. The resulting WIT-1M-recap dataset contains visually detailed descriptions of around 500 words per image. As shown in Fig. 16, mutual $k$ NN alignment still drops with gallery size. More detailed captions give overall higher mutual $k$ NN scores, but do not prevent the decline in scores. This suggests that caption quality is not the primary driver of the mutual $k$ NN alignment drop.

Figure 16: Cross-modal mutual $k$ NN alignment on images recaptioned using gemini-3-flash-preview (WIT-1M-recap) as the gallery grows to 1M samples. Detailed captions result in overall higher mutual $k$ NN scores, but do not prevent the drop in scores.

### A.4 Mutual cross-modal $k$ NN alignment drops at scale across model pairs

The cross-modal alignment drop reported in Fig. 4 in the paper uses DINOv2-base and OpenLlama-3b. Here, we examine whether similar patterns hold for stronger models. In Fig. 17, we repeat the scaling experiment for two additional model pairs: DINOv2-base with OpenLlama-13b, and DINOv2-giant with OpenLlama-13b.

Replacing DINOv2-base with the stronger DINOv2-giant and OpenLlama-3b with OpenLlama-13b does not change the pattern. We observe that mutual $k$ NN still drops at scale. This is consistent with Fig. 3b in the paper, which shows low alignment scores across different LLMs at WIT-1M scale.

(a) DINOv2-base and OpenLlama-13b

(b) DINOv2-giant and OpenLlama-13b

Figure 17: Cross-modal mutual $k$ NN alignment as gallery grows from WIT-1024 to WIT-1M for additional, stronger model pairs. Replacing DINOv2-base with the stronger DINOv2-giant and OpenLlama-3b (Fig. 4 in the paper) with OpenLlama-13b does not prevent the drop. This suggests that the degradation in mutual $k$ NN alignment was not a result of the limitation of any individual model.

(a) Pixio-ViT-B/16 $\times$ OpenLlama-3B

(b) CLIP-B/16 $\times$ OpenLlama-3B

Figure 18: Cross-modal mutual $k$ NN alignment as the WIT gallery grows from 1024 to 1M, for vision encoders trained with different objectives (Pixio-style SSL and CLIP contrastive image-text pretraining), paired with OpenLlama-3B. The same scaling pattern holds: alignment at fixed small $k$ drops with gallery size while $k{=}n/100$ stays flat.

(a) DINOv2-B/14 $\times$ Gemma-7B

(b) DINOv2-B/14 $\times$ Mistral-7B

Figure 19: Cross-modal mutual $k$ NN alignment as the WIT gallery grows from 1024 to 1M, with DINOv2-B/14 paired with LLMs from different families (Gemma and Mistral). The scaling pattern is consistent across LLM families: drop at fixed small $k$ , stability at $k{=}n/100$ .

To test whether the pattern is specific to the DINOv2/OpenLlama choice, we evaluate two further axes of variation. First, we vary the vision encoder while keeping OpenLlama-3B fixed (Fig. 18): we replace DINOv2 with Pixio-ViT-B/16 (a different self-supervised pretraining recipe) and with CLIP-B/16 (image-text contrastive pretraining). Second, we vary the LLM while keeping DINOv2-base fixed (Fig. 19): we pair it with Gemma-7B [23] and Mistral-7B [45] from different model families. Across all four combinations, the alignment drop at fixed small $k$ persists, while alignment at $k=\frac{n}{100}$ remains stable.

This confirms that the degradation reported in the paper is not specific to that particular choice of models.

## Appendix B Additional ImageNet experiments

The controlled experimental setting on the ImageNet validation set in Sec. 4.2 of the paper provides one of our key findings: models individually retrieve correct-class neighbors at increasing rates as the gallery densifies, yet cross-modal agreement remains flat. Here, we verify that the ImageNet validation set serves as a suitable test bed. Furthermore, we confirm that our observations are not limited to our choice of models or metric settings.

### B.1 The ImageNet validation set is denser than WIT-1024

A natural question is how the gallery density in our ImageNet experiments compares to the WIT data. As shown in Table 2, nearest-neighbor cosine similarities on the ImageNet validation set are substantially higher than on WIT-1024 and comparable to WIT-1M. Even for only one image per class in the gallery (ipc=1), the ImageNet validation set provides a denser retrieval setting than WIT-1024.

This confirms that the ImageNet experiments operate in a denser retrieval regime comparable to WIT-1M, making this a meaningful test bed.

Table 2: Nearest-neighbor distances across gallery sizes. ImageNet, even with only one image per class in the gallery (ipc=1), has neighbor distances comparable to WIT-1M, confirming that it operates in a similarly dense retrieval regime.

Gallery  | Model  | Dim  |  $k{=}1$  |  $k{=}10$   
---|---|---|---|---  
WIT-1024  | DINOv2-base  | 768  | 0.799  | 0.717  
WIT-1024  | OpenLlama3b  | 3200  | 0.502  | 0.400  
WIT-1M  | DINOv2-base  | 768  | 0.906  | 0.888  
WIT-1M  | OpenLlama3b  | 3200  | 0.757  | 0.701  
ImageNet ipc=1  | DINOv2-base  | 768  | 0.823  | 0.763  
ImageNet ipc=1  | DINOv2-giant  | 1536  | 0.609  | 0.496  
ImageNet ipc=1  | OpenLlama3b  | 3200  | 0.928  | 0.904  
ImageNet ipc=1  | LLaMA-65B  | 8192  | 0.890  | 0.858  
ImageNet ipc=49  | DINOv2-base  | 768  | 0.887  | 0.861  
ImageNet ipc=49  | DINOv2-giant  | 1536  | 0.749  | 0.690  
ImageNet ipc=49  | OpenLlama3b  | 3200  | 0.954  | 0.944  
ImageNet ipc=49  | LLaMA-65B  | 8192  | 0.926  | 0.912  
  
### B.2 Stronger models do not close the gap for ImageNet

The ImageNet decomposition experiments in Sec. 4.2 in the main paper use DINOv2-base and OpenLlama-3b as its vision and language model respectively. Here, we probe whether stronger models would show better cross-modal agreement that closes the gap to unimodal retrieval accuracy.

In Fig. 20, we repeat the experiment with DINOv2-base paired with OpenLlama-65b, and DINOv2-giant paired with OpenLlama-65b. The pattern is unchanged: both models individually improve at retrieving correct-class neighbors as the gallery densifies, but strict cross-modal alignment remains flat. Using substantially stronger models on both sides does not close the gap between individual retrieval accuracy and cross-modal agreement.

(a) DINOv2-base and Llama-65b

(b) DINOv2-giant and Llama-65b

Figure 20: ImageNet per-modality retrieval accuracy and cross-modal mutual $k$ NN alignment ( $k{=}1$ ) as images / captions per class in gallery increase for different model pairs. Even with substantially stronger models (OpenLlama-65b, DINOv2-giant), individual retrieval improves with gallery density while cross-modal alignment remains flat.

### B.3 ImageNet ablation shows a similar pattern for $k=10$ 

The main paper reports the ImageNet decomposition experiments with mutual $k$ NN scores for $k{=}1$ . Here, we verify that the finding is not an artifact of this strict setting. We additionally present how mutual $k$ NN with $k{=}10$ evolves when the gallery grows in Fig. 21 for two different model pairs.

We again observe that individual retrieval accuracy improves with gallery density while cross-modal alignment, here in terms of mutual $k$ NN with $k{=}10$ , does not.

(a) DINOv2-base and OpenLlama-3b

(b) DINOv2-giant and OpenLlama-65b

Figure 21: Per-modality retrieval accuracy and cross-modal mutual $k$ NN alignment ( $k{=}10$ ) as images / captions per class in gallery increase for two different model pairs. Again, modalities individually improve with gallery density, but mutual $k$ NN alignment, here with $k{=}10$ , does not.

## Appendix C What happens with non-synthetic data that is not bijective?

In the main paper (Sec. 4.3), we use the CycleReward dataset [3] to test alignment when the bijective (one-to-one) assumption is relaxed with _synthetic_ multi-modal correspondences. Here, we complement this analysis using non-synthetic many-to-many correspondences from the WIT dataset [87].

### C.1 Non-synthetic dataset with many-to-many correspondences

Natural duplicates in WIT. The WIT dataset naturally contains many-to-many correspondences between images and captions: the same caption can describe many visually distinct images, and the same image is reused across Wikipedia articles with different corresponding text. Specifically, 7.1% of the captions are associated with more than one image, and 24.6% of the images have more than one caption before deduplication (see Section E.1). These naturally occurring one-to-many and many-to-one correspondences provide a complementary test bed for relaxing the bijective (one-to-one) setting without relying on generated images or captions.

Table 3: Natural duplicates in the WIT dataset after within-group deduplication.

| T2I | I2T  
---|---|---  
Unique elements | 3.2 M | 2.5 M  
Appearing  $>$ 1 time | 7.1% | 24.6%  
Groups with $\geq$ 5 corresp.  
Before dedup | 7,844 | 38,254  
After dedup | 4,975 | 24,853  
  
Within-group deduplication. Grouping by caption text (for T2I) or by image (for I2T) can include _within-group_ duplicates, i.e. a caption group may contain duplicate images, and an image group may contain repeated captions. After within-group deduplication, the number of qualifying one-to-many samples decreases from 7,844 to 4,975 for T2I and from 38,254 to 24,853 for I2T (Table 3).

We construct two complementary one-to-many datasets to mirror the experimental setup in Sec. 4.3 of the paper:

T2I (text-to-images): We select all 4,975 captions that are associated with at least 5 unique images. For each caption, we take 5 images, yielding a flat dataset of 24,875 image-text pairs.

I2T (image-to-texts): We identify 24,853 unique images that are paired with at least 5 distinct captions (by exact string matching). To match the T2I dataset size, we randomly subsample 4,975 images. For each image, we take 5 captions, again yielding 24,875 samples.

### C.2 Mutual $k$ NN also decreases on non-synthetic data when the bijective assumption is relaxed

We evaluate alignment between DINOv2-base [74] and OpenLlama-3b [26] on the WIT-based T2I and I2T datasets. Fig. 22 shows mutual $k$ NN alignment for $k{=}1$ and $k{=}10$ as the number of images per caption and vice versa increases from 1 to 5.

In both directions, alignment decreases as bijectivity is relaxed. This is consistent with the results on the CycleReward dataset in the main paper (Fig. 10) and reinforces the conclusion that mutual $k$ NN alignment is sensitive to the bijective assumption. When multiple valid correspondences exist for a query, the two modalities are less likely to agree on the same nearest neighbor, even if each individually retrieves a good match. This confirms that the observed drop in alignment for non-bijective setting is not an artifact of synthetic data.

(a) T2I: increasing the number of unique images per caption from 1 (bijective) to 5.

(b) I2T: increasing the number of unique captions per image from 1 (bijective) to 5.

Figure 22: Effect of relaxing the bijective assumption on mutual $k$ NN alignment using many-to-many correspondences in the WIT data (I2T and T2I subset). Mutual $k$ NN alignment on non-synthetic drops consistently as we increase the number of images per caption and vice versa. This confirms that the pattern observed on CycleReward is not an artifact of synthetic data.

## Appendix D Does the alignment vs performance trend predicted by Huh  _et al_. [42] continue with recent LLMs?

To assess whether the alignment vs performance trend predicted by Huh  _et al_. [42] continues with recent language models, we evaluate 55 LLMs (see Section E.3.2 for full list) on six standard benchmarks using the LM Evaluation Harness framework [22]. [42] originally used three benchmarks to measure language capability: HellaSwag [106], GSM8K [12], and $(1-\texttt{bitsperbyte})$ on OpenWebText [28]. We replace OpenWebText with Wikitext [66] and extend this analysis to three additional benchmarks that probe different aspects of language understanding: ARC Challenge [11], MMLU [36], and LogiQA2 [61].

#### D.0.1 Benchmarks and metrics.

Table 4 summarizes the evaluation configuration for each benchmark used in Sec. 4.4 of the paper (we used the default configurations from [22]).

Table 4: Overview of language model benchmarks used in Sec. 4.4 of the paper.

Benchmark | Capability | Few-shot | Metric  
---|---|---|---  
HellaSwag [106] | Commonsense reasoning | 0 | Accuracy  
Wikitext [66] | Language modeling | 0 |  $1-\text{bits per byte}$   
ARC Challenge [11] | Science QA | 4 | Accuracy  
GSM8K [12] | Math reasoning | 5 | Exact match  
MMLU [36] | General knowledge | 5 | Accuracy  
LogiQA2 [61] | Logical reasoning | 5 | Accuracy  
  
#### D.0.2 Does the alignment vs performance trend hold?

For each benchmark and each DINOv2 variant, we fit a linear regression on the base models used in [42], predicting mutual $k$ NN alignment $a$ from benchmark performance $p$ . We then evaluate how well this trend describes two populations:   
  
$R^{2}$  (Huh  _et al_.): The standard coefficient of determination on the data is used to fit the regression, i.e. $R^{2}(\text{Huh~\emph{et al}.\hbox{} })=r^{2}$ , where $r$ is the Pearson correlation between mutual $k$ NN alignment and language modelling benchmark score across the 19 base models from Huh  _et al_. [42].   
  
$R^{2}$  (new models): We apply the line fitted on the base models to the 36 recent models and compute the generalized $R^{2}$ :

|  $$R^{2}(\text{new})=1-\frac{\sum_{i\in\mathcal{M}_{\text{new}}}(a_{i}-\hat{a}_{i})^{2}}{\sum_{i\in\mathcal{M}_{\text{new}}}(a_{i}-\bar{a}_{\text{new}})^{2}},$$  |   
---|---|---  
  
where $\hat{a}_{i}$ are the linear regression alignment predictions based on language performance $p_{i}$ , $a_{i}$ is the alignment score for the $i$ -th model and $\bar{a}_{\text{new}}$ is the mean alignment of the new models. When $R^{2}(\text{new})>0$ , the relation between alignment and language performance predicted in Huh  _et al_. extrapolates; when $R^{2}(\text{new})<0$ , the regression line is a worse predictor than simply predicting the average $\bar{a}_{\text{new}}$ . $R^{2}_{\text{avg}}$ values are reported in Table 5 and Figs. 23 and 24.

Table 5: Average $R^{2}$ of the linear regression (fitted on the 19 base models from Huh  _et al_. [42]) evaluated on the base models themselves and on the 36 recent models, across all four DINOv2 variants. Positive $R^{2}_{\text{avg}}(\text{new})$ indicates that the trend from the base models is a good predictor for the new models. Negative values indicate that the regression line is a worse predictor than the mean.

Benchmark |  $R^{2}_{\text{avg}}$ (Huh  _et al_. ) |  $R^{2}_{\text{avg}}$ (new)  
---|---|---  
HellaSwag | 0.752 | 0.297  
Wikitext | 0.729 | 0.489  
ARC | 0.702 |  $-$ 0.575  
GSM8K | 0.336 |  $-$ 1.753  
MMLU | 0.430 |  $-$ 0.662  
LogiQA2 | 0.431 |  $-$ 1.414  
  
The results reveal a split across language modelling benchmarks. For HellaSwag and Wikitext, the relation between alignment and language performance observed by Huh  _et al_. partially extends to recent models: the $R^{2}_{\text{avg}}$ on new models remains positive (0.297 and 0.489, respectively), indicating that stronger language models according to these benchmarks have higher mutual $k$ NN alignment with DINOv2. Both benchmarks primarily measure next-token prediction quality and commonsense language understanding, which are closely related to the pretraining objective of autoregressive LLMs.

In contrast, for the four benchmarks that probe more specialized reasoning abilities: ARC (science QA), GSM8K (arithmetic), MMLU (general knowledge), and LogiQA2 (logical reasoning), the relation between alignment and language performance predicted from the base models Huh  _et al_. does not appear to hold for this set of recent models.

Specifically, the $R^{2}_{\text{avg}}$ on new models is consistently negative, ranging from $-0.575$ (ARC) to $-1.753$ (GSM8K). This means that the linear fit from the base models from Huh  _et al_. [42] is a worse predictor of alignment for recent models than simply predicting the mean. In Figs. 23 and 24, we observe that recent models that are stronger than the best base model (Meta-Llama-3-70B) do not show higher mutual $k$ NN alignment with DINOv2 features. Instead, their alignment scores seem to saturate or decrease.

We note that the 36 added (new) models are heterogeneous. They include new models trained on next-token prediction (pre-training), instruction-tuned models, and reasoning-distilled models (e.g. DeepSeek-R1-Distill). We treat them as a single population to test whether the trend extrapolates to recent LLMs.

The above results support and extend the finding from Sec. 4.4 of the main paper. The relationship between alignment and language performance from [42] holds for core language modelling benchmarks, but does not seem to generalize to reasoning benchmarks.

Figure 23: Mutual $k$ NN alignment vs. language benchmark performance for 55 LLMs across four DINOv2 variants, on WikiText, HellaSwag, and GSM8K. Dashed lines show the linear trend fit to the 19 base models from [42]. For WikiText and HellaSwag (top two plots), recent models roughly follow the trend. For GSM8K (bottom plot), the trend is not followed.

Figure 24: Mutual $k$ NN alignment vs. language benchmark performance for 55 LLMs across four DINOv2 variants, on ARC, LogiQA2, and MMLU. As with GSM8K, the alignment-performance trend from [42] does not extrapolate to recent models on any of these reasoning benchmarks. Stronger models do not appear to show higher mutual $k$ NN alignment with DINOv2 features.

## Appendix E Experimental setup

In Section E.1, we provide additional details about the deduplication pipeline for the WIT-1M and LAION-15M datasets. We then describe the captioning pipeline used for WIT-1M-recap and for the ImageNet validation set in Section E.2.

### E.1 WIT-1M and LAION-15M datasets

#### E.1.1 Image deduplication.

We deduplicate the gallery pools from WIT [87] and LAION400M [83] at the image level using perceptual hashing [105]. For each image, a 64-bit hash $\mathbf{h}_{i}\in\{0,1\}^{64}$ is computed. For this, we first convert the image to grayscale, resize it to $32\times 32$ , apply a 2D Discrete Cosine Transform, and threshold the top-left $8\times 8$ low-frequency coefficients against their median. This produces a binary fingerprint that is robust to minor changes, e.g. due to recompression. We consider images $i$ and $j$ duplicates if their Hamming distance satisfies

|  $$d{\text{H}}(\mathbf{h}_{i},\mathbf{h}_{j})\leq 2,$$  |   
---|---|---  
  
measuring the number of bit positions at which two hashes differ:

|  $$d_{\text{H}}(\mathbf{h}_{i},\mathbf{h}_{j})=\sum_{b=1}^{64}\mathbf{1}[h_{i,b}\neq h_{j,b}].$$  |   
---|---|---  
  
We perform deduplication of the gallery against the WIT-1024 images, and within the gallery by keeping the first occurrence in the case of image duplicates.

#### E.1.2 Caption deduplication.

In addition to image deduplication, we do a text deduplication pass to remove gallery samples with captions identical to another gallery sample or to a WIT-1024 query caption. Duplicate captions are undesirable because they allow trivial text-based query-gallery matching, inflating retrieval scores regardless of visual representations.   


We use exact string matching and remove any gallery samples that match WIT-1024 captions. Among the gallery samples, we discard duplicates and keep only the first occurrence of a duplicate caption-sample.

Table 6: Deduplication statistics for the WIT-1M and LAION-15M gallery pools. Image duplicates are detected via perceptual hashing (pHash) with Hamming distance $\leq 2$ . Caption duplicates are detected by exact string matching. WIT-1M and LAION-15M are 1M and 15M image-caption pairs randomly sampled from the remaining final pool.

| WIT-1M | LAION-15M  
---|---|---  
Raw pool | 3,582,610 | 20,000,000  
Image deduplication  
Duplicates (with WIT-1024) | 2,847 | 53  
Duplicates (within gallery) | 2,164,343 | 3,371,128  
Pool after image deduplication | 2,486,852 | 17,941,016  
Caption deduplication  
Duplicates (with WIT-1024) | 52 | 14  
Duplicates (within gallery) | 97,654 | 642,895  
Final pool size | 2,389,146 | 17,298,107  
  
Table 7: Distribution of caption duplicates in the WIT and LAION galleries after image deduplication. Note that the “unique captions” include some captions that are removed as query matches for the final pool. 

| WIT | LAION  
---|---|---  
Copies per caption | Unique captions | Total samples | Unique captions | Total samples  
1 | 2,357,353 | 2,357,353 | 16,897,221 | 16,897,221  
2 | 22,799 | 45,598 | 316,109 | 632,218  
3 | 3,807 | 11,421 | 48,864 | 146,592  
4 | 1,529 | 6,116 | 15,422 | 61,688  
5 | 787 | 3,935 | 6,932 | 34,660  
6–10 | 1,520 | 11,431 | 9,596 | 69,614  
11–100 | 1,283 | 29,765 | 3,838 | 76,200  
101–1,000 | 118 | 27,164 | 103 | 12,376  
$>$ 1,000 | 2 | 5,077 | 4 | 14,588  
Total | 2,389,198 | 2,486,852 | 17,298,111 | 17,941,016  
  
#### E.1.3 WIT-1M.

We obtain a raw pool of 3,582,610 samples from the English-text WIT dataset [87]. To construct the English-only subset of the dataset, we used a subset of [85, 84]. Since [84] only contains the image URLs, we retrieved the corresponding images from [87]. The raw pool undergoes our deduplication pipeline, resulting in 2,389,146 samples. We randomly sampled 1 million samples for the WIT-1M dataset. Deduplication statistics are provided in Table 6 and Table 7. In WIT, 31,845 captions appear repeatedly, accounting for 129,499 samples (5.21%); the most frequent captions are coat of arms (2716 $\times$ ) and Town hall (2361 $\times$ ).

#### E.1.4 LAION-15M.

We randomly sample 20M samples from the LAION-400M dataset [83] which consists of English image-text pairs. We randomly sample 20M samples as our raw pool which undergoes our deduplication pipeline. Finally, we randomly sample 15M from the final pool after deduplication, resulting in our LAION-15M data pool. Deduplication statistics are provided in Table 6 and Table 7. In LAION, 400,890 unique captions appear repeatedly, accounting for 1,043,795 samples (5.82%); the most frequent captions are Patent Drawing (10027 $\times$ ) and Throw Pillow (3246 $\times$ ).

### E.2 Captioning pipeline for the ImageNet validation set and WIT-1M-recap

We used gemini-3-flash-preview [91, 75] for captioning the images in the ImageNet validation [16] set and in WIT-1M. Specifically, we used the following text prompt for each image.

⬇

You are a precise image description system. Describe the image in the following JSON format.

Return ONLY a valid JSON object with exactly these 7 keys. No text before or after the JSON.

{

"one_sentence": "<exactly one sentence, strictly fewer than 15 words>",

"short": "<2-3 sentences, 20-40 words total>",

"100w": "<a paragraph, approximately 100 words>",

"250w": "<several paragraphs, approximately 250 words>",

"500w": "<detailed description, approximately 500 words>",

"750w": "<thorough description covering all visual details, approximately 750 words>",

"extreme_long": "<maximally detailed description covering every visible element, texture, color, spatial relationship, lighting, and context. YOU MUST WRITE AT LEAST 1000 WORDS. If your draft is under 1000 words, keep adding more detail about textures, materials, lighting, spatial layout, colors, and any other visible elements until you reach at least 1000 words.

Target 1000-1500 words.>"

}

Be factual and visual. Describe what you actually see: objects, people, animals, colors, textures, spatial relationships, background, lighting, and mood. Do not invent information not visible in the image.

For the ImageNet validation set, we perform experiments with captions of the extreme_long type. As shown in Fig. 25(a), alignment scores increase with caption length. Captions of approximately 500 words achieve scores close to the maximum, while the longest captions yield the best mutual $k$ NN alignment scores between DINOv2-base and OpenLlama-3b on the ImageNet validation set. A distribution over the number of words for captions of the extreme_long caption type is shown in Fig. 25(b). Despite prompting the model to produce at least 1,000 words per caption, 63.1% of captions fall below this target. The average caption length is 981 words.

For WIT-1M-recap, we caption the 1 million images in the WIT-1M dataset using the 500w variant for computational efficiency. This resulted in captions for 999,971 (29 images did not get processed by gemini-3-flash-preview) images of on average 478 words. We provide those in https://huggingface.co/datasets/askoepke/wit_1m_recaptioned.

(a) Mutual $k$ NN alignment for DINOv2-base and OpenLlama-3b increases with longer captions on the ImageNet validation set.

(b) Distribution of word counts for Gemini-generated extreme_long captions across 49,984 ImageNet validation images.

Figure 25: Generated image captions for the ImageNet validation set. a) shows the mutual $k$ NN alignment using captions of different lengths between DINOv2-base and OpenLlama3b on the ImageNet validation set. As also shown in [42], longer detailed captions yield higher alignment scores. b) shows the distribution over caption length (word count). We use captions of on average 981 words for our ImageNet experiments.

### E.3 Models and feature extraction pipeline

Our feature extraction pipeline is based on the experimental protocol from Huh  _et al_. [42], which we extend to include additional LLMs.

#### E.3.1 Vision models.

On the vision side, we use four DINOv2 [74] variants: ViT-S/14 (384-d), ViT-B/14 (768-d), ViT-L/14 (1024-d), and ViT-G/14 (1536-d) [18], loaded via the timm [101] library. For each image, we extract the CLS token representation from every transformer layer, yielding a per-sample feature tensor of shape $L\times d$ , where $L$ is the number of layers and $d$ the feature dimensionality.

#### E.3.2 Language models.

We evaluate 55 large language models spanning 13 model families. The first group comprises the 19 base models used by Huh  _et al_. [42]: BLOOMZ (560M–7.1B) [71], OpenLlama (3B–13B) [26], LLaMA (7B–65B) [93], OLMo (1B, 7B) [30], Gemma (2B, 7B) [23], Mistral-7B [45], Mixtral-8 $\times$ 7B [46], and Meta-Llama-3-70B [29].   
For the trend analysis in Appendix D, we extend this set with 36 recent models: LLaMA-2 (7B–70B) [94], Llama-3/3.1 [29], OLMo-2/3 [72], Ministral-3 (3B–14B) [59], Gemma-2 (2B–27B)[24], Gemma-3 (270M–27B) [25], DeepSeek-R1-Distill (1.5B–70B) [15], Qwen3 (1.7B–32B) [77], Falcon3 (7B, 10B) [90], 01.AI Yi-1.5 (34B) [104], IBM Granite (8b) [79], and OpenAI GPT-OSS (20b) [73].   
For each model, we extract hidden-state representations from all layers. Following [42], we apply average pooling over non-padding tokens to obtain a single vector per layer.

## Appendix F Additional qualitative results

We present additional qualitative results for nearest-neighbor retrieval at different gallery scales on WIT-1M and LAION-15M in Figs. 26 and 27 and Figs. 28 and 29 respectively. In addition to the near-duplicate matches in Figs. 5 and 6 of the main paper, we here show further examples for cross-modal agreement (green-bordered matches) at scale when the modalities happen to select the same neighbor. Others show agreement at WIT-1024 that breaks down as the gallery densifies. In those cases, each modality individually finds a better match at scale, but they no longer agree on the same one.

Figure 26: Additional nearest-neighbor examples with DINOv2 and OpenLlama-3b for $k{=}1$ across gallery scales on the WIT-1M dataset. For OpenLlama-3b, we show the (partial) retrieved captions along with the corresponding reference image (LLM-ref) for visualisation. Green-bordered captions and images indicate a mutual $k$ NN match across modalities. Figure 27: Additional nearest-neighbor examples with DINOv2 and OpenLlama-3b for $k{=}1$ across gallery scales on the WIT-1M dataset. Green-bordered captions and images indicate a mutual $k$ NN match across modalities. Figure 28: Additional nearest-neighbor examples with DINOv2 and OpenLlama-3b for $k{=}1$ across gallery scales on the LAION-15M dataset. Green-bordered captions and images indicate a mutual $k$ NN match across modalities. Figure 29: Additional nearest-neighbor examples with DINOv2 and OpenLlama-3b for $k{=}1$ across gallery scales on the LAION-15M dataset. Green-bordered captions and images indicate a mutual $k$ NN match across modalities.
