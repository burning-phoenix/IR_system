# Towards More Interpretable AI With Sparse Autoencoders

**Corpus section:** Tyler Wheeler / Interpretability
**Selected as:** E-sae — Engels, MIT MS thesis 2025. The long-form synthesis covering the multi-dimensional-feature and dark-matter work. 376 KB — the single best entry point.
**Source file:** `Papers/Symbolic_Embeddings/corpus/markdown/literature/engels-thesis-2025.md`
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. This folder is method literature copied in to support an
  interpretability read of his papers. It is not evidence about him, his lab, or his research direction, and no
  audit over this corpus should treat it as a body of work by anyone in it.

---

## **Towards More Interpretable AI With Sparse Autoencoders** 

by 

Joshua Engels 

B.S., Rice University, 2021 B.A., Rice University, 2021 

Submitted to the Department of Electrical Engineering and Computer Science in partial fulfillment of the requirements for the degree of 

MASTER OF SCIENCE 

at the 

MASSACHUSETTS INSTITUTE OF TECHNOLOGY 

May 2025 

© 2025 Joshua Engels. All rights reserved. 

The author hereby grants to MIT a nonexclusive, worldwide, irrevocable, royalty-free license to exercise any and all rights under copyright, including to reproduce, preserve, distribute and publicly display copies of the thesis, or release the thesis under an open-access license. 

Authored by: Joshua Engels Department of Electrical Engineering and Computer Science May 16, 2025 

Certified by: Max Tegmark Professor of Physics Thesis Supervisor 

Accepted by: Leslie A. Kolodziejski Professor of Electrical Engineering and Computer Science Chair, Department Committee on Graduate Students 

2 

### **Towards More Interpretable AI With Sparse Autoencoders** 

by 

Joshua Engels 

Submitted to the Department of Electrical Engineering and Computer Science on May 16, 2025 in partial fulfillment of the requirements for the degree of 

MASTER OF SCIENCE 

#### **ABSTRACT** 

While large language models demonstrate remarkable capabilities across diverse domains, the specific representations and algorithms they learn remain largely unknown. The quest to understand these mechanisms holds dual significance: scientifically, it represents a fundamental inquiry into the principles underlying intelligence, while practically–and with growing urgency– it is vital for mitigating risks from these very same increasingly powerful systems. 

The initial section of this thesis tackles this challenge of interpreting internal language model representations (features) by employing _sparse autoencoders_ (SAEs). An SAE decomposes neural network hidden states into a potentially more interpretable basis. In Chapter 2, we introduce an unsupervised, SAE-based methodology that successfully identifies inherently multi-dimensional features. Notably, we establish that language models causally represent concepts such as days of the week and months of the year using circular structures. This work provided the first definitive evidence of causal, multi-dimensional features, thereby refuting the one-dimensional linear representation hypothesis. Chapter 3 further assesses whether SAEs identify “true” atomic language model features. We compare the generalization performance and data efficiency of linear probes trained on SAE latents against those trained on the original hidden state basis. The negative outcomes of these experiments suggest limitations in SAEs for capturing the true ontology of language models. 

Motivated by the aforementioned limitations, the second part of this thesis investigates sparse autoencoders themselves, exploring potential improvements and characterizing their failure modes. Chapter 4 examines the portion of activations not reconstructed by SAEs, which we term “Dark Matter.” We find that a significant fraction of this dark matter is linearly predictable, and furthermore, that specific tokens poorly reconstructed by SAEs remain largely consistent across SAE sizes and sparsities. This suggests that SAEs may systematically fail to capture certain input subspaces, which we hypothesize to contain inherently dense features. Subsequently, Chapter 5 investigates a method to enhance SAE utility: freezing the learned SAE parameters and finetuning the surrounding language model components to minimize KL divergence with the original model’s output distribution. This technique results in a 30% to 55% decrease in the cross-entropy loss gap incurred by inserting the SAE into the model. 

Thesis supervisor: Max Tegmark Title: Professor of Physics 

3 

4 

# **Acknowledgments** 

First and foremost, I would like to thank my advisor, Max Tegmark. I am immensely grateful that Max took a chance on me when I was looking for an AI safety research group, and since then Max has always been an incredible mentor and collaborator. Max has especially taught me to focus on the important problems and to never believe that something is intractable until you have tried it. I would not be anywhere close to the researcher I am today without the incredible research environment of the Tegmark group that I have been so lucky to be a part of. 

I have also been incredibly lucky to have had a wonderful set of collaborators. The works in this thesis were done in collaboration with Eric Michaud, Isaac Liao, Subhash Kantamneni, Wes Gurnee, Neel Nanda, Senthooran Rajamanoharan, Mathew Chen, and Logan Riggs. All of these researchers are incredible scientists whom I have learned so much from. I would especially like to thank Neel, who through the MATS program has graciously mentored me and from whom I have learned to frequently step back and evaluate the bigger picture of a research direction, and Eric, with whom I have had many long conversations that have shaped my outlook on research and life. I have also collaborated with other excellent researchers on works not included in this thesis, including Lily Sun, Alessandro Stolfo, Ben Wu, Anish Mudide, David Baek, and Julian Yocum. 

Outside of co-authors, my PhD experience so far would not have been the same without lengthy conversations with my friends Ben, Daniel, and Ted over long walks on the Charles and (very) long hikes in the Whites. Neel’s MATS stream was also an amazingly rich intellectual environment, and I will forever be friends with Clement, Julian, Helena, Caden, Constantin and Ivan. Finally, I am thankful for the supportive mechanistic interpretability community, and especially for conversations with Jake Mendel, Jacob Dunefsky, David Bau, David Chanin, Adam Karvonen, and doubtless many others that I am forgetting. I would like to especially thank Sam Marks, who was the first to introduce me to mechanistic interpretability and who opened me up to the possibility of switching research groups into AI safety. 

I have additionally had the great pleasure to have been supported by the NSF GRFP over the last year, which I have greatly appreciated for the increased freedom it has given me to pursue ideas and projects. Thank you also to MATS for organizing an excellent research program and hosting me in Berkeley, and to MAIA for organizing the reading group that got me up to speed on AI safety. 

Finally, I would like to especially thank my parents, grandmother, and sister for their love and support as I have embarked on this research journey. 

5 

6 

# **Contents** 

|_List of _|_Figures_||11|
|---|---|---|---|
|_List of _|_Tables_||19|
|**1**<br>**Intr**|**oducti**|**on**|**21**|
|1.1|Outlin|e. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>21|
||1.1.1|Not All Language Model Features Are One-Dimensionally Linear<br>.|.<br>21|
||1.1.2|Are Sparse Autoencoders Useful? A Case Study in Sparse Probing .|.<br>22|
||1.1.3|Decomposing The Dark Matter of Sparse Autoencoders . . . . . . .|.<br>22|
||1.1.4|Low Rank Adapting Models For Sparse Autoencoders . . . . . . . .|.<br>23|
|1.2|My C|ontributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>23|
|**2**<br>**Not **|**All L**|**anguage Models Are One-Dimensionally Linear**|**25**|
|2.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>25|
||2.1.1|Contributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>26|
|2.2|Relate|d Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>27|
|2.3|Defini|itions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>28|
||2.3.1|Multi-Dimensional Features<br>. . . . . . . . . . . . . . . . . . . . . .|.<br>28|
||2.3.2|Superposition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>30|
|2.4|Sparse|Autoencoders Find Multi-Dimensional Features<br>. . . . . . . . . . .|.<br>30|
|2.5|Circul|ar Representations in Large Language Models . . . . . . . . . . . . .|.<br>32|
||2.5.1|Intervening on Circular Day and Month Representations<br>. . . . . .|.<br>33|
||2.5.2|SAE Plane Interventions . . . . . . . . . . . . . . . . . . . . . . . .|.<br>35|
||2.5.3|Continuity of Circular Representations . . . . . . . . . . . . . . . .|.<br>36|
|2.6|Discus|sion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>37|
|**A App**|**endix**|**: Not All Language Models Are One-Dimensionally Linear**|**39**|
|A.1|Multi-|Dimensional Feature Capacity<br>. . . . . . . . . . . . . . . . . . . . .|.<br>39|
|A.2|More|on Reducibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>42|
||A.2.1|Additional Intuition for Definitions . . . . . . . . . . . . . . . . . .|.<br>42|
||A.2.2|Empirical Irreducible Feature Test Details . . . . . . . . . . . . . .|.<br>42|
|A.3|Altern|ative Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>43|
||A.3.1|Alternative Definition: Interventions and Representation Spaces . .|.<br>46|
|A.4|Toy C|ase of Training SAEs on Circles . . . . . . . . . . . . . . . . . . . . .|.<br>46|
|A.5|Traini|ng Mistral SAEs . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>48|
|A.6|GPT-|2 and Mistral 7B Dictionary Element Clustering . . . . . . . . . . . .|.<br>48|



7 

||A.6.1|GPT-2-small methods and results . . . . . . . . . . . . . . . . . . . .|49|
|---|---|---|---|
||A.6.2|Mistral 7B methods and results . . . . . . . . . . . . . . . . . . . . .|49|
|A.7|Other|discovered clusters . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|51|
|A.8|Furthe|r Experiment Details<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|52|
||A.8.1|Assets Information . . . . . . . . . . . . . . . . . . . . . . . . . . . .|52|
||A.8.2|Machine Information . . . . . . . . . . . . . . . . . . . . . . . . . . .|52|
||A.8.3|Error Bar Calculation<br>. . . . . . . . . . . . . . . . . . . . . . . . . .|53|
|A.9|More `W`|`eekdays` and `Months` Plots and Details . . . . . . . . . . . . . . . . .|54|
|A.10|Patchi|ng . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|55|
|A.11|Explan|ation via Regression (EVR)<br>. . . . . . . . . . . . . . . . . . . . . . .|55|
||A.11.1|Using EVR to uncover a circular representation for _γ_ . . . . . . . . .|58|
||A.11.2|More Experiments with EVR<br>. . . . . . . . . . . . . . . . . . . . . .|58|
|**3**<br>**Are **|**Sparse**|**Autoencoders Useful? A Case Study in Sparse Probing**|**65**|
|3.1|Introdu|ction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|65|
|3.2|Metho|dology<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|67|
||3.2.1|Classification Datasets . . . . . . . . . . . . . . . . . . . . . . . . . .|68|
||3.2.2|Probing Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|68|
||3.2.3|Probing Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|69|
||3.2.4|Experimental Setup: Quiver of Arrows . . . . . . . . . . . . . . . . .|69|
|3.3|Compa|ring Probing Techniques in Different Regimes<br>. . . . . . . . . . . . .|70|
||3.3.1|Standard Conditions . . . . . . . . . . . . . . . . . . . . . . . . . . .|70|
||3.3.2|Data Scarcity, Class Imbalance, and Label Noise . . . . . . . . . . . .|71|
||3.3.3|Covariate Shift<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|72|
|3.4|Interpr|etability<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|72|
||3.4.1|Probe Intepretability: Pruning and Latent Generalization . . . . . . .|73|
||3.4.2|Latent Interpretability . . . . . . . . . . . . . . . . . . . . . . . . . .|75|
||3.4.3|Detecting Dataset Quality Issues<br>. . . . . . . . . . . . . . . . . . . .|75|
|3.5|Why D|idn’t This Work: Illusions of SAE Probes . . . . . . . . . . . . . . . .|77|
|3.6|Assessi|ng Improvements in SAE Architectures . . . . . . . . . . . . . . . . .|78|
|3.7|Conclu|sion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|79|
|**B App**|**endix: **|**Are Sparse Autoencoders Useful? A Case Study in Sparse**||
|**Pro**|**bing**||**81**|
|B.1|Relate|d Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|81|
||B.1.1|Probing<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|81|
||B.1.2|Challenges in Neural Network Interpretability . . . . . . . . . . . . .|81|
||B.1.3|SAE and SAE Applications<br>. . . . . . . . . . . . . . . . . . . . . . .|82|
|B.2|Classifi|ication Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|82|
|B.3|i<br> Probin|i<br>g Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|85|
||B.3.1|AUC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|85|
||B.3.2|Probing Method Validation Details . . . . . . . . . . . . . . . . . . .|85|
||B.3.3|Probing Method Hyperparameter Details . . . . . . . . . . . . . . . .|86|
||B.3.4|Formal Discussion of Quiver of Arrows . . . . . . . . . . . . . . . . .|87|
|B.4|Norma|l Conditions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|88|



8 

||B.4.1|Evaluating Baselines . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>88|
|---|---|---|---|
||B.4.2|Choosing which SAEs to Test . . . . . . . . . . . . . . . . . . . .|. .<br>88|
||B.4.3|Plotting Dataset Performance vs. K. . . . . . . . . . . . . . . . .|. .<br>89|
|B.5|Additi|onal Results for Various Regimes . . . . . . . . . . . . . . . . . . .|. .<br>89|
||B.5.1|Data Scarcity . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>89|
||B.5.2|Class Imbalance . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>89|
||B.5.3|Label Corruption . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>89|
|B.6|Repro|ducing Core Results on Llama-3.1-8b . . . . . . . . . . . . . . . . .|. .<br>89|
|B.7|Interp|retability<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>92|
||B.7.1|Pruning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>92|
||B.7.2|Latent Investigation<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>93|
||B.7.3|Glue CoLA<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>96|
||B.7.4|AI Made . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>96|
|B.8|Multip|le tokens results . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>97|
|B.9|SAE A|rchitectural Improvements . . . . . . . . . . . . . . . . . . . . . .|. .<br>98|
|B.10|Assess|ing SAE Probe Binarization . . . . . . . . . . . . . . . . . . . . . .|. .<br>100|
|**4**<br>**Dec**|**ompos**|**ing the Dark Matter of Sparse Autoencoders**|**103**|
|4.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>103|
|4.2|Relate|d Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>104|
|4.3|Notati|on . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>105|
|4.4|Predic|ting SAE Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>106|
||4.4.1|Predicting SAE Error Norm . . . . . . . . . . . . . . . . . . . . .|. .<br>107|
||4.4.2|Predicting SAE Error Vectors . . . . . . . . . . . . . . . . . . . .|. .<br>108|
||4.4.3|Nonlinear FVU and Breaking Down SAE Dark Matter<br>. . . . . .|. .<br>108|
||4.4.4|Predicting SAE Per-Token Error Norms<br>. . . . . . . . . . . . . .|. .<br>109|
|4.5|Analy|zing Components of SAE Error . . . . . . . . . . . . . . . . . . . .|. .<br>110|
||4.5.1|Applying the Norm Prediction Test: . . . . . . . . . . . . . . . . .|. .<br>111|
||4.5.2|Training SAEs on `SaeError`(**x**) Components: . . . . . . . . . . . .|. .<br>112|
||4.5.3|Downstream Cross Entropy Loss of `SaeError`(**x**) Components: . .|. .<br>113|
|4.6|Reduc|ing `NonlinearError`(**x**) . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>113|
||4.6.1|Using a More Powerful Encoder . . . . . . . . . . . . . . . . . . .|. .<br>114|
||4.6.2|Predicting SAE Errors Between SAEs: . . . . . . . . . . . . . . .|. .<br>115|
|4.7|Concl|usion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>115|
|**C App**|**endix**|**: Decomposing the Dark Matter of Sparse Autoencoders**|**117**|
|C.1|Extra|Error Prediction Experiments . . . . . . . . . . . . . . . . . . . . .|. .<br>117|
||C.1.1|Note on Feature Shrinkage . . . . . . . . . . . . . . . . . . . . . .|. .<br>117|
||C.1.2|Additional Error Prediction Plots . . . . . . . . . . . . . . . . . .|. .<br>117|
||C.1.3|Norm Prediction Baselines . . . . . . . . . . . . . . . . . . . . . .|. .<br>117|
||C.1.4|Breaking Apart Error Per Token<br>. . . . . . . . . . . . . . . . . .|. .<br>118|
|C.2|Model|ing Activations . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>118|
||C.2.1|Analyzing Error Norm Prediction . . . . . . . . . . . . . . . . . .|. .<br>120|
||C.2.2|Analyzing Error Vector Prediction. . . . . . . . . . . . . . . . . .|. .<br>120|
||C.2.3|Analyzing Per-Token Scaling Predictions . . . . . . . . . . . . . .|. .<br>121|



9 

|C.3|More I|nfo on Modeling Activations<br>. . . . . . . . . . . . . . . . . . . . .|. .<br>122|
|---|---|---|---|
||C.3.1|Proof of Claim from Section C.2.1 . . . . . . . . . . . . . . . . . .|. .<br>122|
||C.3.2|Empirical Norm Prediction. . . . . . . . . . . . . . . . . . . . . .|. .<br>124|
||C.3.3|Synthetic SAE Error Vector Experiments . . . . . . . . . . . . . .|. .<br>124|
|C.4|Synthe|tic Experiments with Random Data . . . . . . . . . . . . . . . . .|. .<br>124|
|C.5|Using|_∥_`NonlinearError`(**x**)_∥_to Predict Scaling: . . . . . . . . . . . . . .|. .<br>125|
|**5**<br>**Low**|**-Rank **|**Adapting Models for Sparse Autoencoders**|**129**|
|5.1|Introdu|ction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>129|
|5.2|Relate|d Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>131|
|5.3|Optimi|zing Models for Sparse Autoencoders . . . . . . . . . . . . . . . .|. .<br>132|
||5.3.1|TopK Sparse Autoencoders. . . . . . . . . . . . . . . . . . . . . .|. .<br>132|
||5.3.2|End-to-End Sparse Autoencoders . . . . . . . . . . . . . . . . . .|. .<br>133|
||5.3.3|JumpReLU Sparse Autoencoders<br>. . . . . . . . . . . . . . . . . .|. .<br>134|
||5.3.4|Method for Low-Rank Adapting Models to SAEs<br>. . . . . . . . .|. .<br>134|
|5.4|Results|. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>134|
||5.4.1|Scaling Laws for Downstream Loss<br>. . . . . . . . . . . . . . . . .|. .<br>134|
||5.4.2|Downstream Loss vs. Computational Cost . . . . . . . . . . . . .|. .<br>135|
||5.4.3|Adapting Multiple SAEs . . . . . . . . . . . . . . . . . . . . . . .|. .<br>136|
|5.5|Downs|tream Improvements. . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>136|
||5.5.1|SAEBench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>137|
||5.5.2|Feature Steering. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>138|
||5.5.3|General Language Model Capabilities . . . . . . . . . . . . . . . .|. .<br>140|
|5.6|Analyz|ing Why Our Method Works . . . . . . . . . . . . . . . . . . . . .|. .<br>141|
||5.6.1|Per Token Improvement Breakdown . . . . . . . . . . . . . . . . .|. .<br>141|
||5.6.2|Activation Distances<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>142|
||5.6.3|Single Layer Adapters<br>. . . . . . . . . . . . . . . . . . . . . . . .|. .<br>142|
|5.7|Conclu|sion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>143|
||5.7.1|Limitations<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>143|
|**D App**|**endix: **|**Low-Rank Adapting Models for Sparse Autoencoders**|**145**|
||D.0.1|Steering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>145|
||D.0.2|SAEBench Metrics . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>146|
||D.0.3|Activation Distances<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>147|
|_Referenc_|_es_||149|



10 

# **List of Figures** 

- 2.1 Circular representations of days of the week, months of the year, and years of the 20th century in layer 7 of GPT-2-small colored by the token they fire on. These representations were discovered via clustering SAE dictionary elements, described in Section 2.4. Points are colored according to the token which created the representation. See Fig. A.5 for other axes and Fig. A.6 for similar plots for Mistral 7B. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26 

- 2.2 Empirical _ϵ_ -mixture index and separability index for the “days of the week” cluster along PCA components 2 and 3. **Left:** The _ϵ_ band parameterized by **v** and _c_ that the optimization procedure found contained the highest fraction of points. **Mid:** Dot products of points in the feature distribution with the _ϵ_ band; _Mϵ_ ( **_f_** ) is the percent of dot products within _ϵ_ = 0 _._ 1 of 0. **Right:** Estimated mutual information for different rotations of the space; _S_ ( **_f_** ) is the minimum over all rotations. This point cloud has a lower _ϵ_ -mixture index and higher separability index than PCA projections within typical clusters (see Fig. 2.3), indicating that it is more likely to be an irreducible multi-dimensional feature. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31 

- 2.3 Mixture index and separability index of GPT-2 features. Features from Fig. 2.1, which we had manually identified, score highly as candidate multidimensional features with these metrics. . . . . . . . . . . . . . . . . . . . . . . . . . . . 32 

- 2.4 Top two PCA components on the _α_ token. Colors show _α_ . **Left:** Layer 30 of Mistral on `Weekdays` . **Right:** Layer 5 of Llama on `Months` . . . . . . . . . . . 33 

- 2.5 Visual representation of the circular intervention process. **Top:** We learn a circular probe on the PCA projection of a training set. **Bot:** To intervene, we change the circular representation to _αj_<sup>_′_andaverageablateotherdimensions.</sup> 34 

- 2.6 Mean and 96% error bars for intervening on the _α_ token across layers using different intervention methods. The circular intervention technique outperforms patching only the top 5 PCA components and leaving the rest unchanged, and almost reaches the upper bound performance of patching the entire layer. . . 34 

- 2.7 Off distribution interventions on Mistral layer 5 on the `Weekdays` task. The color corresponds to the highest logit _γ_ after performing the circular subspace intervention on that point. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35 

- 2.8 Interventions on the Mistral 7B `Weekdays` task with different methods of determining the probe. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36 

11 

- 2.9 Layer 30 Mistral 7B activations for [morning/evening] on [Monday/Tuesday/.../Sunday], plotted projected into the PCA plane for [Monday/Tuesday/.../Sunday]. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36 

- A.1 Testing irreducibility of synthetic features. **Left in each subfigure:** Distributions of **x** . For feature **a** , 63.96% lies within the narrow dotted lines, indicating the feature is likely a mixture. For feature **b** , 17.84% lies within the wide lines, indicating the feature is unlikely to be a mixture. The green cross indicates the angle _θ_ that minimizes mutual information. **Middle in each subfigure:** Histograms of the distribution of **v** _·_ **x** with red lines indicating a 2 _ϵ_ -wide region. **Right in each subfigure:** Mutual information between **a** and **b** as a function of the rotation angle _θ_ of matrix **R** . Feature **b** has a large minimum mutual information so is unlikely to be separable; feature **a** has a medium value of minimum mutual information of about 0 _._ 37 bits. . . . . . . 44 

- A.2 Testing irreducibility of synthetic features. **Left in each subfigure:** Distributions of **x** . For feature **c** , 7.94% lies within the narrow dotted lines, indicating the feature is unlikely to be a mixture. For feature **d** , 25.90% lies within the wide lines, indicating the feature is likely a mixture. The green cross indicates the angle _θ_ that minimizes mutual information. **Middle in each subfigure:** Histograms of the distribution of **v** _·_ **x** with red lines indicating a 2 _ϵ_ -wide region. **Right in each subfigure:** Mutual information between **a** and **b** as a function of the rotation angle _θ_ of matrix **R** . Both features have a small ( _<_ 0 _._ 5 bits) minimum mutual information and so are likely separable. . . . . . . . . . . . 45 

- A.3 SAEs trained to reconstruct a single 2d circle with _m_ = 2 (left) and _m_ = 10 (middle and right) dictionary elements. When there are several SAE features, there is no natural/canonical choice of feature directions, and the dictionary elements spread out across the circle. . . . . . . . . . . . . . . . . . . . . . . 47 

- A.4 Automatic discovery of synthetic circular features by clustering SAE dictionary elements. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48 

- A.5 Projections of days of week, months of year, and years of the 20th century representations onto top four principal components, showing additional dimensions of the representations than Fig. 2.1. . . . . . . . . . . . . . . . . . . . . . . . 50 

- A.6 Circular representations of days of the week and months of the year which we discover with our unsupervised SAE clustering method in Mistral 7B. Unlike similar features in GPT-2, we also find an additional “weekend” representation in between Saturday and Sunday representations (left) and additional representations of seasons among the months (right). For instance, “winter” tokens activate a region of the circle in between the representation of January and December. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51 

- A.7 Hyperparameter regimes where the days of the week cluster exists. The cluster exists in the regime between all features clumping together and all features being in their own cluster; this regime seems reasonably stable. . . . . . . . . 52 

- A.10 Projections of Mistral 7B `Weekdays` task activations at layer 8 into the plane discovered by clustering layer 8 SAE features. . . . . . . . . . . . . . . . . . 54 

12 

|A.11|Layer 30 Mistral 7B activations for [very early/very late] on [Monday/Tues-<br>day/.../Sunday], plotted projected into the PCA plane for [Monday/Tues-<br>day/.../Sunday]. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|54|
|---|---|---|
|A.14|Top two PCA components of residual errors after EVR with one-hot in _α_ and<br>_β_. Mistral 7B `Weekdays`, layer 25, final token. Colored by _γ_. . . . . . . . . .|58|
|A.8|Top 10 GPT-2 clusters by Mixture and Separability Index. . . . . . . . . . .|61|
|A.9|Projections onto the top two PCA dimensions of model hidden states on the<br>_α_ token show that circular representations of _α_ are present in various layers.|62|
|A.12|Attention and MLP patching results on `Weekdays`. Results are averaged over<br>20 different runs with fixed _α_ and varying _β_ and 20 different runs with fixed _β_<br>and varying _α_.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|63|
|A.13<br>A.15|Attention and MLP patching results on `Months`. Results are averaged over 20<br>different runs with fixed _α_ and varying _β_ and 20 different runs with fixed _β_<br>and varying _α_.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>EVR residual RGB plots on Mistral hidden states on the`Weekdays`final token,<br>layers 17 to 29. From top to bottom, we show each residual RGB plot after<br>adding the function(s) **g**_i_ labelled just underneath, as well as the resulting _r_<sup>2</sup><br>value. We write “tmr” meaning “tomorrow” for _β_ = 1. We also write “circle for<br>_x_” meaning the inclusion of two functions **g**_i_(_x_) =_{_cos_,_sin_}_(2_πx/_7). . . . . .|63<br>64|
|A.16|Iterative deconstruction of hidden state representations on the final token on<br>Llama 3 8B, `Months`. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|64|
|3.1|SAE probes underperform the baseline of logistic regression in each regime<br>when taking the mean across datasets. Additionally, we find that baseline<br>methods can provide many of the interpretability insights of SAE probes. . .|66|
|3.2|**Left:** An illustration of our SAE probing method. We pass in training acti-<br>vation vectors from each class and train an _L_1 regularized logistic regression<br>probe on the latents that differ the most between classes. **Right:** We ensure ro-<br>bustness of our results with the "quiver of arrows" approach (see Section 3.2.4):<br>we add SAE regression into a set of methods, and see if the test accuracy of<br>the best method (chosen by validation accuracy) increases. . . . . . . . . . .|67|
|3.3|For a given width, using higher L0 and constructing probes with a larger basis<br>of latents (_k_) is more performant. . . . . . . . . . . . . . . . . . . . . . . . .|70|
|3.4|In standard conditions, when SAE probes are added to the quiver, we find a<br>slight decrease in performance. . . . . . . . . . . . . . . . . . . . . . . . . . .|70|
|3.5|For three of the datasets in Table 3.1, we visualize the performance when SAE<br>probes are in the quiver (dashed) versus when they are not (solid) for the<br>regimes of data scarcity (left), class imbalance (middle), and label noise (right).<br>In all three regimes, we see that on average (bottom row), SAEs do not help.|71|
|3.6|For the regimes of data scarcity, class imbalance, and label noise, we see no<br>average improvement across datasets when adding SAEs to the quiver. Shading<br>represents 95% confidence intervals. . . . . . . . . . . . . . . . . . . . . . . .|71|
|3.7|SAE probes often generalize worse than baseline logistic regression with co-<br>variate shift. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|73|



13 

|3.8|Latent 122774 is most active on the English translation of “living room,” and<br>does not fire on French.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|74|
|---|---|---|
|3.9|Latent 369585 outperforms a dense SAE probe and logistic regression when<br>testing on mislabeled CoLA examples.<br>. . . . . . . . . . . . . . . . . . . . .|76|
|3.10|**Left:** Top 3 latents (w/ descriptions) by mean activation difference between<br>AI generated and human SAE latent activations. **Right:** Histograms of the<br>identity of the 4 most common last tokens in AI generated and human text.|77|
|3.11 <br>3.12|Illustration of a potentially misleading result we find: when comparing activa-<br>tion probes to pooled SAE probes, SAE probe win rate increases considerably,<br>but adding in a “pooled” attention-inspired probe brings down the SAE win-rate<br> We see that there is a slight uptick in probing performance compared to<br>baselines in standard conditions with more recent architectures. However, the<br>spread of the data is considerably larger than this improvement. . . . . . . .|. 77<br>79|
|B.1|Analysis of layer performance and methods. (a) Shows the optimal layer for<br>baseline performance. (b) Demonstrates the effectiveness of different methods<br>across layers.<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|87|
|B.2|Analysis of SAE width and L0 regularization effects on probe performance.<br>(a) Shows the Pareto frontier of L0 vs width trade-offs. (b) Illustrates the<br>relationship between SAE width and probe performance. . . . . . . . . . . .|88|
|B.3|Test AUC on all datasets vs. _k_ =_|I|_ using the Gemma Scope layer 20 SAE<br>with width 131k and _L_0 = 193.<br>. . . . . . . . . . . . . . . . . . . . . . . . .|90|
|B.4|Improvement from using SAE methods in the quiver across all datasets for<br>various corruption ratios, ratios of positive classes, and number of training<br>examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|91|
|B.5|Method selection analysis under different data conditions. (a) Selection under<br>data scarcity. (b) Selection under class imbalance. (c) Selection under data<br>corruption.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|92|
|B.6|Analysis of Llama-3.1-8b probe performance. (a) Layer-wise performance of<br>baseline methods. (b) Comparison between SAE and baseline probes under<br>standard conditions.<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|92|
|B.7|We find that Llama-3.1-8b SAE probes do not improve on baselines in the<br>settings of data scarcity, class imbalance, and label noise. . . . . . . . . . . .|93|
|B.8|Two datasets improve with pruning, but one does not. Indicates that it needs<br>more latents to construct answer. . . . . . . . . . . . . . . . . . . . . . . . .|93|
|B.9|Across regimes, we see a slight positive trend in probing performance with<br>more recent SAE architectures.<br>. . . . . . . . . . . . . . . . . . . . . . . . .|99|
|B.10|We run the multi-token experiments from Section 3.5 with and without bina-<br>rizing aggregated latents. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|100|
|4.1|A breakdown of SAE dark matter for layer 20 Gemma 9B SAEs, with dotted<br>lines assuming that observed trends continue for larger SAEs. See Section 4.4<br>for how we break down the overall fraction of unexplained variance into<br>absent features, linear error, and nonlinear error. See Section 4.6.1 for further<br>separating encoder error from nonlinear error. . . . . . . . . . . . . . . . . .|104|



14 

|4.2|**Left**: _R_<sup>2 </sup>of predicting SAE error norms (see Eq. (4.10),**Mid**: _R_<sup>2 </sup>of predicting<br>SAE error vectors (see Eq. (4.11),**Right:** 1_−R_<sup>2 </sup>of predicting model activations||
|---|---|---|
||given the SAE reconstruction and the SAE error vector prediction. We do a<br>linear interpolation between SAEs (black dots) . . . . . . . . . . . . . . . . .|107|
|4.3|_R_<sup>2 </sup>of SAE error norm predictions (see Eq. (4.10)) for Gemma Scope SAEs of<br>width _L_0 _≈_50 and Llama Scope SAEs of width _L_0 =_k_ = 50. . . . . . . . . .|109|
|4.4|_R_<sup>2 </sup>for linear probes of per token SAE errors of larger SAEs from smaller<br>SAEs. Prediction accuracy decreases as the SAEs get farther apart in scale,<br>but overall remains high. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|109|
|4.5|Per token scaling with average nonlinear error, layer20 Gemma 9B SAEs from<br>Gemma Scope closest to _L_0 = 60.<br>. . . . . . . . . . . . . . . . . . . . . . .|110|
|4.6|Violin plot of norm prediction tests for all SAEs (layer 12 Gemma 2B SAEs,<br>layer20Gemma 9B SAEs, all Gemma Scope SAEs with_L_0 closest to60across<br>layers, and all Llama Scope SAEs). We plot the _R_<sup>2 </sup>of a linear regression from<br>**x** to to each random vector’s norm squared.<br>. . . . . . . . . . . . . . . . . .|112|
|4.7|Auto-interpretability results on SAEs trained on the linear and nonlinear<br>components of `SaeError`(**x**) on a width 16k, _L_0 _≈_60, layer 20 Gemma Scope<br>SAE. “Not” represents contexts that the SAE latent did not activate on, while<br>each _Qi_ represents activating examples from decile _i_.<br>. . . . . . . . . . . . .|112|
|4.8|Results of intervening in the forward pass and replacing **x** with _SAE_(**x**) +<br>_NonlinearError_(**x**) and _SAE_(**x**) +_LinearError_(**x**) during the forward pass<br>on all layer 20 Gemma 9B SAEs with _L_0 _≈_60 or width of 16k.<br>. . . . . . .|113|
|4.9|Investigations towards reducing nonlinear SAE error.<br>. . . . . . . . . . . . .|114|
|C.1|Per-token breakdown of linearly predictable and non-linearly predictable SAE<br>error across SAE scale. We show the same tokens as in Fig. 4.5. The norm of<br>linear error decreases with SAE width, whereas the norm of nonlinear error<br>stays mostly constant.<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|118|
|C.2|_R_<sup>2 </sup>for linear regressions of SAE error norms with different regressors. We run<br>on Gemma Scope 9B SAEs of size 131_k_ with _L_0 _≈_60. Activations perform<br>the best except on the first few layers.<br>. . . . . . . . . . . . . . . . . . . . .|119|
|C.3|Average SAE latent activation and dot product of the latent with the learned<br>norm prediction vector**_a_**<sup>_∗_</sup>for the Gemma Scope layer20, width 131_k_, _L_0 = 62<br>SAE. We also plot a smoothed version of this dot product with a smoothed<br>window of 10. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|119|
|C.4|**Top:** When controlled amounts of noise are added to synthetic data `Sae`(**x**<sup>_′_</sup>)<br>and **x**<sup>_′_</sup>, the result is a plot similar to Fig. 4.1. **Bottom:** The nonlinear and<br>linear error estimates (as shown at top) accurately correlate with the amount<br>of noise added. The exact correlation between synthetic added noise and<br>resulting estimated error components across these noise levels are shown in<br>Table C.1<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|125|
|C.5|Percent decrease in FVU when additionally using the squared norms of nonlin-<br>ear error to predict SAE error norm.<br>. . . . . . . . . . . . . . . . . . . . . .|126|



15 

|C.6|Results of linearly predicting SAE error norm and SAE error from model<br>activations on Gemma 2 2B layer 5 (**top**) and Gemma 2 2B layer 19 (**bottom**).<br>The right plots show the _R_<sup>2 </sup>of predicting SAE error norms (see Eq. (4.10),<br>the middle plots show the _R_<sup>2 </sup>of predicting SAE error vectors (see Eq. (4.11),<br>and the right plots show 1_−R_<sup>2 </sup>of predicting model activations given the SAE<br>reconstruction and the SAE error vector prediction. Unlike for the middle<br>layers shown in the main body, `FVUnonlinear` decreases some as width is scaled,<br>although the sparsity of the SAEs in the space makes this harder to verify. .|127|
|---|---|---|
|C.7|Results of linearly predicting SAE error norm and SAE error from model<br>activations on Gemma 2 9B layer 9 (**top**) and Gemma 2 9B layer 31 (**bottom**).<br>The right plots show the _R_<sup>2 </sup>of predicting SAE error norms (see Eq. (4.10),<br>the middle plots show the _R_<sup>2 </sup>of predicting SAE error vectors (see Eq. (4.11),<br>and the right plots show 1_−R_<sup>2 </sup>of predicting model activations given the SAE<br>reconstruction and the SAE error vector prediction. Unlike for the middle<br>layers shown in the main body, `FVUnonlinear` decreases some as width is scaled,<br>although the sparsity of the SAEs in the space makes this harder to verify. .|128|
|5.1<br>5.2<br>5.3|Cross entropy loss vs. training time over 2B tokens for Gemma-2-2B TopK<br>SAEs withwidth= 18_,_432_,_L0 = 64. We find that our method (TopK + LoRA<br>in the plot) outperforms an e2e SAE and vanilla TopK SAE. . . . . . . . . .<br>Visual representation of our method, with a local SAE trained on layer 12 and<br>low-rank adapters trained on MLP and attention components on all layers.<br>Cross entropy loss improvement (**Top**: absolute, **Bottom:** percentage of CE<br>loss gap closed) using our method for Gemma Scope SAEs on Gemma-2-2B.<br>**Left:** Scaling across sparsity with fixed width=16k and layer=12, we see<br>the largest effect by percentage of our method at lower sparsities, but still<br>substantial effect at higher sparsities as well. **Middle:** Scaling across width<br>with fixed _L_0 = 68 and layer=12, the highest effect by percentage is at low<br>width but again this is not a large effect. **Right:**<br>Scaling across layer with<br>fixed _L_0 = 68 and width=16k, the highest effect of our method by percentage<br>is at layer 9 but it is mostly unaffected by layer.<br>. . . . . . . . . . . . . . .|129<br>132<br>133|
|5.4|Cross entropy loss improvement (**Top**: absolute, **Bottom:** percentage) for<br>Gemma Scope SAEs of width 16_k_ and _L_0 closest to 70 on Gemma-2-2B, 7B,<br>and 27B. We find that our method works increasingly well on larger models.|135|
|5.5|Cross entropy loss vs. training time for Llama-3.2-1B with TopK SAEs of<br>_L_0 = 64 and width 16384. Our method (TopK + LoRA) achieves lower CE<br>loss sooner than e2e SAE or vanilla TopK SAEs . . . . . . . . . . . . . . . .|138|
|5.6|Downstream cross entropy loss when multiple Llama Scope SAEs are inserted<br>into Llama-3.1-8B at once. “Base” is the original loss without any fine-tuning,<br>while “LoRA” is the loss after 15M tokens of LoRA training. . . . . . . . . .|139|
|5.7|The distribution of normalized log-likelihood change post-steering is more<br>pronounced on positive examples for the LoRA model. Results shown for the<br>SAE latent responsible for “Donald Trump”.<br>. . . . . . . . . . . . . . . . . .|140|



16 

|5.8|Distribution of loss improvements and loss degredations across validation tokens.<br>We see that more tokens have a loss improvement than degredation (although<br>a substantial number have a degradation) and most loss improvements and<br>degredations happen in a range of about 0.01 to 1 ∆_L_SAE nats. . . . . . . .|141|
|---|---|---|
|5.9|Plot of _L_SAE when running LoRA on just a single layer of Gemma Scope 2B.<br>We find that LoRA on layers closer to the SAE layer do better, and that also<br>LoRA on just the next layer achieves much of the loss reduction of training on<br>all layers.<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|141|
|D.1|Distribution plots of the change in normalized log-likelihood after steering for<br>various SAE latents. **Top left** is for machine learning (neuron 8421). **Top**<br>**right** is for San Francisco (neuron 2195). **Bottom left** is for Donald Trump<br>(neuron 13677). **Bottom right** is for COVID-19 (neuron 17811).<br>. . . . . .|145|
|D.2|Change in average distance to original activations before and after applying<br>LoRA; increases in cosine similarity (**Left**) and decreases in Euclidean distance<br>(**Right**) are good. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|148|



17 

18 

# **List of Tables** 

|2.1|Aggregate model accuracy on days of the week and months of the year modular<br>arithmetic tasks. Performance broken down by problem instance in Section A.9.|33|
|---|---|---|
|A.1 <br>A.2 <br>A.3|`Weekdays` finegrained results. Row ommited if both models get it correct. . .<br> `Months` finegrained results. Row ommited if both models get it correct. . . .<br> Highest intervention effect attention heads from fine-grained attention head<br>patching, as well as EVR results with one hot _α, β_ and one hot _α, β, γ_.<br>. . .|56<br>57<br>60|
|3.1<br>3.2|Example probing tasks. Only tasks with short prompts shown for conciseness.<br>Timeline of SAE Architecture Improvements. . . . . . . . . . . . . . . . . . .|68<br>78|
|B.1 <br>B.2|Binary classification tasks used. . . . . . . . . . . . . . . . . . . . . . . . . .<br> Selection methods to choose hyperparameters _hp_ when training on different<br>dataset sizes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|82<br>86|
|B.4 <br>B.7<br>B.3 <br>B.5|Latent categorization, characteristics, and performance.<br>. . . . . . . . . . .<br>Comparison of different logistic regression (“base”) and SAE probing methods<br>on a selection of 60 random datasets. Last is last token probing as in the rest<br>of our paper, concat is concatenating the top 20 PCA dimensions of all tokens,<br>mean is taking the mean across activation dimensions across the context, max<br>is taking the max across activation dimensions across the context, and Attn-like<br>probe is described in Section 3.5.<br>. . . . . . . . . . . . . . . . . . . . . . . .<br> Description of Latent Variables for Out-of-Distribution Detection. We find<br>that this procedure ranks latents roughly according to their OOD test AUC,<br>even though o1 did not have access to this information. However, the system is<br>not perfect. For example, latent 51330’s natural language description is ranked<br>last and seems unreleated to the task at hand. However, its OOD test AUC is<br>the highest of all latents. We do not investigate this further, but hypothesize<br>that either the latent descriptions require additional tuning, or the test set<br>truly has some spurious quality that this latent identifies. . . . . . . . . . . .<br> **Left** We show the top 5 examples where latent 369585 is active while the<br>CoLA prompt is labeled as grammatical. Most of these sentences are clearly<br>ungrammatical, which illustrates that the CoLA data is corrupted. **Right**<br>We then look at sentences a baseline classifier trained on CoLA marks as<br>ungrammatical when the label is grammatical. We reach the same conclusion -<br>the CoLA data is mislabeled.<br>. . . . . . . . . . . . . . . . . . . . . . . . . .|93<br>97<br>101<br>102|



19 

|B.6|Logistic regression classifier for 110_aimade_humangpt3’s top activating to-<br>kens when run on more than 2.5 million tokens from the Pile. The classifier<br>predominantly activates on punctuation. Only tokens with at least 10 oc-<br>curences shown. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|102|
|---|---|---|
|C.1|Correlation matrix between synthetic noise and estimated errors. . . . . . . .|121|
|5.1|Gemma-2-2B timing results and speedups, nearest hour . . . . . . . . . . . .|137|
|5.2|Llama-3.2-1B timing results and speedups, nearest hour . . . . . . . . . . . .|137|
|5.3|Using the same TopK SAE trained onGemma-2-2B, we compare the SAEBench<br>metrics when the underlying model is low-rank adapted with rank 64. We see<br>across most applicable metrics, the LoRA model shows meaningful improve-<br>ment. Full results over various thresholds are displayed in Table D.1.<br>. . . .|140|
|5.4|For each SAE latent, ∆POSITIVE and ∆NEGATIVE denote the 90% CI improve-<br>ment in normalized log-likelihood increase when using the LoRA model for steer-<br>ing on positive and negative examples, respectively. Because ∆POSITIVE _>_0<br>and ∆NEGATIVE _≤_0, we see the LoRA model is better at steering for a given<br>SAE latent while not affecting other features.<br>. . . . . . . . . . . . . . . . .|141|
|5.5|Comparisons of SAE, SAE + LoRA, and original model performance on<br>standard NLP benchmarks. Error ranges represent one standard error; largest<br>value between SAE and SAE + LoRA is bolded. . . . . . . . . . . . . . . . .|142|
|D.1|Using the same TopK SAE trained onGemma-2-2B, we compare the SAEBench<br>metrics when the underlying model is low-rank adapted with rank 64. The<br>threshold hyperparameter for SCR and TPP denotes how many of the top _n_<br>latents are used in the modified classifier. . . . . . . . . . . . . . . . . . . . .|148|



20 

# **Chapter 1** 

# **Introduction** 

Language models are some of the most sophisticated and impressive systems that humans have ever created, so it is particularly surprising that we fundamentally do not understand how they work. Although language model weights give a complete explanation of behavior on any particular set of tokens, this layer of abstraction–billions of floating point parameters–is not useful for many fundamental questions like “is the model deceiving me” or “does the model have a backdoor”. Thus, this thesis aims to build and analyze a set of tools that 1) raise the level of abstraction at which we understand language models while 2) remaining a faithful description of language model behavior. Specifically, we study _sparse autoencoders_ (SAEs), a tool for decomposing language model representations into a sparse over-complete basis of interpretable one-dimensional latents. SAEs thus promise this desired increased level of abstraction, and we indeed find that they are useful for discovering higher level structure in language models. However, we also show that SAEs have many flaws: they fail to find atomic concepts and consistently miss subspaces of the input. We leave off with a promising direction for how to improve SAEs: finetuning the language model itself to better work with an existing frozen SAE. 

## **1.1 Outline** 

The first two chapters of this thesis investigate the nature of language model representations using SAEs. We additionally ask to what extent SAEs capture these true model representations. 

### **1.1.1 Not All Language Model Features Are One-Dimensionally Linear** 

In Chapter 2, we challenge the prevailing hypothesis that language models use primarily one-dimensional features. We develop a rigorous definition for irreducible multi-dimensional features and present a scalable, unsupervised method that uses SAEs to automatically discover such features within GPT-2 and Mistral 7B. Using this method, we identify irreducible and interpretable multi-dimensional structures, most notably _circular_ features representing days of the week and months of the year. We provide evidence through intervention experiments 

21 

on Mistral 7B and Llama 3 8B that these circular features are causally involved in modular arithmetic tasks, thereby establishing them as fundamental units of computation. Ultimately, this chapter argues that understanding multi-dimensional features is necessary for decomposing certain model behaviors and provides the first conclusive evidence against the one-dimensional linear representation hypothesis. 

### **1.1.2 Are Sparse Autoencoders Useful? A Case Study in Sparse Probing** 

The prior chapter implies both positive updates about SAEs–they can discover interpretable multi-dimensional langauge model features–and negative ones–because SAE latents are inherently one-dimensional, they may not align with true (possibly multi-dimensional) language model concepts. Thus, in Chapter 3, we aim to asses whether the features identified by SAEs constitute a ’true’–or at least useful–basis for understanding model internals. We test this by evaluating whether SAE-latents improve performance on the downstream task of linear probing, particularly under challenging real-world conditions where linear probes on raw hidden states suffer. We specifically evaluate SAEs versus baselines on more than 100 datasets in data scarce, class imbalanced, label noised, and covariate shifted regimes. Contrary to the hypothesis that the interpretable, concept-level latents from SAEs would provide a strong inductive bias that would help in these regimes, our results show that SAEs do not consistently outperform strong baselines that use the original activation basis. We additionally investigate whether SAEs show potential in specific interpretabiltiy scenarios such as finding spurious correlations and incorrect labels. Although SAEs do well at these tasks, we find that comparable results are often achievable with simpler non-SAE based methods. These findings highlight the limitations of current SAEs and emphasize the need for rigorous evaluation of interpretability methods against strong baselines on downstream tasks. 

Motivated by the limitations identified above, the second part of this thesis shifts focus to analyzing and improving SAEs themselves. 

### **1.1.3 Decomposing The Dark Matter of Sparse Autoencoders** 

In Chapter 4, we investigate the portion of activations _not_ captured by SAE reconstructions, which we term “Dark Matter.” Rather than treating this SAE error as noise, we analyze it as an object of study in its own right. Surprisingly, we find that a substantial fraction of this dark matter (both the error vector itself and its norm) is linearly predictable from the original activation vector. Furthermore, we observe that the specific input tokens poorly reconstructed by SAEs remain remarkably consistent across different SAE sizes and sparsity levels. These results suggest that SAEs may systematically fail to capture certain input subspaces, potentially those containing inherently dense or non-linear features. We explore models of activation structure consistent with these findings and empirically analyze the properties of the linearly unpredictable component of the error, finding that it differs quantitatively and qualitatively from the predictable component. We end with some preliminary methods for for reducing nonlinear SAE error. 

22 

### **1.1.4 Low Rank Adapting Models For Sparse Autoencoders** 

Finally, Chapter 5 explores a novel direction for enhancing the practical utility of SAEs. Instead of focusing solely on improving SAE training, we propose adapting the _language model itself_ to better accommodate a pre-trained SAE. Specifically, we freeze the SAE parameters and use low-rank adaptation (LoRA) to finetune the surrounding language model components while minimizing the KL divergence with the original model’s output distribution. Analyzing this approach across various SAE configurations, model sizes, and layers (using the Gemma Scope family of SAEs), we demonstrate that this method reduces the downstream crossentropy loss gap incurred by SAE insertion by 30% to 55%. Compared to computationally expensive end-to-end SAE training methods, our approach achieves comparable downstream performance significantly faster (3 _×_ to 20 _×_ on Gemma-2-2B and 2 _×_ to 10 _×_ on Llama-3.2-1B). This chapter shows that optimizing the model itself for interpretability-tool compatibility is a viable path towards Pareto improvements in the trade-off between model performance and interpretability. 

## **1.2 My Contributions** 

I have been incredibly lucky to work with amazing coauthors on each of these projects. Below, I detail my specific contributions to each project: 

- Not All Language Model Features Are One-Dimensionally Linear: As the first author, I led this project, came up with the original idea of searching for multi-dimensional features in language models, found the first such features, performed all causal intervention experiments, and wrote most of the final paper. Eric Michaud and I worked together to develop the unsupervised SAE clustering algorithm to find multi-dimensional features, and Issac Liao, Max Tegmark, and I together developed the theoretical definitions and tests for identifying irreducible multi-dimensional features. 

- Are Sparse Autoencoders Useful? A Case Study in Sparse Probing: Subhash Kantamneni and I jointly led this project and wrote the paper. I came up with the main Quiver of Arrows approach we used for benchmarking SAEs against baseline methods, was responsible for running the sparse autoencoder baselines and multi-token experiments, and discovered the main case study results (GlUE CoLA, AI vs. Human text, and SAE latent reliance on language). 

- Decomposing The Dark Matter of Sparse Autoencoders: I was the first author and main contributor on this project; I found the initial surprising result that SAE errors could be linearly predicted from activations, ran all experiments in the main body except the SAE intervention experiment, and wrote most of the final paper. 

- Low Rank Adapting Models For Sparse Autoencoders: Mathew Chen and I jointly led this project. I came up with the original idea, met with Mathew weekly to suggest new experiments and directions, ran the multi-layer SAE experiments and the experiments examining the distribution of loss improvements across tokens, and and wrote much of the final paper. 

23 

24 

# **Chapter 2 Not All Language Models Are One-Dimensionally Linear** 

## **2.1 Introduction** 

Language models trained for next-token prediction on large text corpora have demonstrated remarkable capabilities, including coding, reasoning, and in-context learning (Bubeck et al. 2023; Achiam et al. 2023; Anthropic 2024b; Gemini Team et al. 2023). However, the specific algorithms models learn to achieve these capabilities remain largely a mystery to researchers; we do not understand how language models write poetry. Mechanistic interpretability is a field that seeks to address this gap by reverse-engineering trained models from the ground up into variables (features) and the programs (circuits) that process these variables (Olah et al. 2020). 

One mechanistic interpretability research direction has focused on understanding toy models in detail. This work has found multi-dimensional representations of inputs such as lattices (Eric J Michaud et al. 2024) and circles (Z. Liu, Kitouni, et al. 2022; Nanda, Chan, et al. 2023), and has successfully reverse-engineered the algorithms that models use to manipulate these representations. A separate direction has identified one-dimensional representations of high level concepts and quantities in large language models (Gurnee and Tegmark 2023; Marks and Tegmark 2023; Heinzerling and Inui 2024; Bricken, Templeton, et al. 2023). These findings have led to the _linear representation hypothesis_ (LRH): a hypothesis which has historically claimed both that 1. all representations in pretrained large language models lie along one-dimensional lines, and 2. model states are a simple sparse sum of these representations (Park, Choe, and Veitch 2023a; Bricken, Templeton, et al. 2023). In this work, we specifically call into question the first part of the LRH.<sup>1</sup> 

For the most part, these two directions have been disconnected: Yedidia 2023a and Gould 

> 1An earlier version of this manuscript sparked discussion in the mechanistic interpretability community on the distinction between non-linear features and multi-dimensional features, and in fact this discussion directly led to a consensus around the two part LRH we describe above (see (Olah 2024; Csordás et al. 2024; Mendel 2024; Kantamneni and Tegmark 2025)). To clarify, we agree with these discussions, and believe the multi-dimensional features we find are "linear" in the sense that they are contained in a low-dimensional linear subspace, but "non-linear" in the sense that this low-dimensional subspace is not one-dimensional (and this is the sense we mean in the title). 

25 



<!-- Start of picture text -->
Days of the Week Months of the Year Years of the 20th Century<br>Monday Friday January October<br>TuesdayWednesday SaturdaySunday FebruaryMarch NovemberDecember 1900 1950 1999<br>Thursday Other April Other<br>May<br>June<br>July<br>August<br>September<br>PCA axis 2 PCA axis 2 PCA axis 3<br>PCA axis 3 PCA axis 3 PCA axis 4<br><!-- End of picture text -->

Figure 2.1: Circular representations of days of the week, months of the year, and years of the 20th century in layer 7 of GPT-2-small colored by the token they fire on. These representations were discovered via clustering SAE dictionary elements, described in Section 2.4. Points are colored according to the token which created the representation. See Fig. A.5 for other axes and Fig. A.6 for similar plots for Mistral 7B. 

et al. 2023 find intriguing hints of circular language model representations, and Bricken, Templeton, et al. 2023 speculate about the existence of feature manifolds, but these brief results only serve to further emphasize the lack of a unifying and satisfying perspective on the nature of language model features. In this work, we seek to bridge this gap by formalizing, investigating, and systematically searching for multi-dimensional language model features. 

### **2.1.1 Contributions** 

1. In Section 2.3, we generalize the one-dimensional definition of a language model feature to multi-dimensional features and provide an updated multi-dimensional superposition hypothesis to account for these new features. 

2. In Section 2.4, we build on the definitions proposed in Section 2.3 to develop a theoretically grounded and empirically practical test that uses sparse autoencoders to find irreducible features. Using this test, we identify multi-dimensional representations automatically in GPT-2 and Mistral 7B, including circular representations for the day of the week and month of the year. 

3. In Section 2.5, we show that Mistral 7B and Llama 3 8B use these circular representations when performing modular addition in days of the week and in months of the year. To the best of our knowledge, we are the first to find causal circular representations of concepts in a language model. We additionally find that the model’s circular representations respect a continuous notion of time. 

26 

## **2.2 Related Work** 

**Linear Representations:** Early word embedding methods such as GloVe and Word2vec, although only trained using co-occurrence data, contained directions in their vector spaces corresponding to semantic concepts (Tomáš Mikolov, Yih, and Zweig 2013; Pennington, Socher, and Manning 2014; Tomas Mikolov et al. 2013). Recent research has found similar evidence of one-dimensional linear representations in sequence models trained only on next token prediction, including Othello board positions (Nanda, A. Lee, and Wattenberg 2023; K. Li et al. 2022), the truth value of assertions (Marks and Tegmark 2023), and numeric quantities such as longitude, latitude, birth year, and death year (Gurnee and Tegmark 2023; Heinzerling and Inui 2024). These results have inspired the linear representation hypothesis (Park, Choe, and Veitch 2023a; Elhage, Hume, et al. 2022) defined above. Y. Jiang et al. 2024 provide theoretical evidence for this hypothesis, assuming a latent (binary) variable-based model of language. Empirically, Bricken, Templeton, et al. 2023 and Cunningham et al. 2023b successfully use sparse autoencoders to break down a model’s feature space into an overcomplete basis of linear features. These works assume that the number of linear features stored in superposition exceeds the model dimensionality (Elhage, Hume, et al. 2022). 

**Multi-Dimensional Representations:** There has been comparatively little research on multi-dimensional features in language models. Shai et al. 2024 predict and verify that a transformer trained on a hidden Markov model uses a fractal structure to represent the probability of each next token, a clear example of a necessary multi-dimensional feature, but the analysis is restricted to a toy setting. Yedidia 2023a; Yedidia 2023b finds that GPT-2 learned position vectors form a helix, which implies a circle when “viewed” from below. Thus, we are not the first to find a circular feature in a language model. However, our work finds circular features that represent latent concepts from text, while the GPT-2 learned position vectors are specific to tokenization, separate from the rest of the model parameters, and causally implicated only due to positional attention masking. Another suggestive result, due to Hanna, O. Liu, and Variengien 2024, is the presence of a U-shape in the representation of numbers between 0 and 100; however, Hanna, O. Liu, and Variengien 2024 find that this representation is not causal, and they only show it exists within a specific prompt distribution. Recent work on dictionary learning (Bricken, Templeton, et al. 2023) has speculated about multi-dimensional _feature manifolds_ ; our work is similar to this direction and develops the idea of feature manifolds theoretically and empirically. Finally, in a separate direction, Black et al. 2022 argue for interpreting neural networks through the polytopes they split the input space into, and identifies regions of low polytope density as “valid” regions for a potential linear representation. 

**Circuits:** Circuits research seeks to identify and understand _circuits_ , subsets of a model (usually represented as a directed acyclic graph) that explain specific behaviors (Olah et al. 2020). The base units that form a circuit can be layers, neurons (Olah et al. 2020), or sparse autoencoder features (Marks, Rager, et al. 2024). In the first circuits-style work, Olah et al. 2020 found line features that were combined into curve detection features in the InceptionV1 image model. More recent work has examined language models, for example the indirect object identification circuit in GPT-2 (K. Wang et al. 2022). Given the difficulty of designing bespoke experiments, there has been increased research in automated circuit 

27 

discovery methods (Marks, Rager, et al. 2024; Conmy et al. 2023; Syed, Rager, and Conmy 2023). 

**Interpretability for Arithmetic Problems:** Z. Liu, Kitouni, et al. 2022 study models trained on modular arithmetic problems _a_ + _b_ = _c_ (mod _m_ ) and find that models that generalize well have circular representations for _a_ and _b_ . Further work by Nanda, Chan, et al. 2023 and Zhong et al. 2024 shows that models use these circular representations to compute _c_ via a “clock” algorithm and a separate “pizza” algorithm. These papers are limited to the case of a small model trained only on modular arithmetic. Another direction has studied how large language models perform basic arithmetic, including a circuits level description of the greater-than operation in GPT-2 (Hanna, O. Liu, and Variengien 2024) and addition in GPT-J (Stolfo, Belinkov, and Sachan 2023). These works find that to perform a computation, models copy pertinent information to the token before the computed result and perform the computation in the subsequent MLP layers. Finally, recent work by Gould et al. 2023 investigates language models’ ability to increment numbers and finds linear features that fire on tokens equivalent modulo 10. 

## **2.3 Definitions** 

This section focuses on hypotheses for how hidden states of language models can be decomposed into sums of functions of the input (features). We focus on _L_ layer transformer models _M_ that take in token input **t** = ( _t_ 1 _, . . . , tn_ ) from input token distribution _T_ , have hidden states **x** 1 _,l, . . . ,_ **x** _n,l_ for layers _l_ , and output logit vectors **y** 1 _, . . . ,_ **y** _n_ . Given a set of inputs _T_ , we let _Xi,l_ be the set of all corresponding **x** _i,l_ . We write matrices in capital bold, vectors and vector valued functions in lowercase bold, and sets in capital non-bold. 

### **2.3.1 Multi-Dimensional Features** 

**Definition 1** (Feature) **.** We define a _df -dimensional feature_ as a function **f** that maps a subset of the input space into R<sup>_df_</sup> . We say that a feature is _active_ on the aforementioned subset. 

The input token distribution _T_ induces a _df_ -dimensional probability distribution over feature vectors **f** ( _t_ ). As an example, let _n_ = 1 (so inputs are single tokens) and consider a feature **f** that maps integer tokens to their integer values in R<sup>1</sup> . Then **f** is a 1-dimensional feature that is active on integer tokens, and **f** ( _t_ ) is the marginal integer occurrence distribution from the token distribution. 

How can we differentiate "true" multi-dimensional features from sums of lower dimensional features? We make this distinction by examining the _reducibility_ of a potential multidimensional feature. That is, **f** is a "true" multi-dimensional feature if it cannot be written as the sum of two statistically independent features **and** it cannot be written as the sum of two non-co-occurring features. Formally, we have the following definition: 

**Definition 2.** A feature **f** is _reducible_ into features **a** and **b** if there exists an affine transformation 



28 

for some orthonormal _df × df_ matrix **R** and additive constant **c** , such that the transformed feature probability distribution _p_ ( **a** _,_ **b** ) satisfies at least one of these conditions: 

1. _p_ is _separable_ , i.e., factorizable as a product of its marginal distributions: _p_ ( **a** _,_ **b** ) = _p_ ( **a** ) _p_ ( **b** ). 

2. _p_ is a _mixture_ , i.e., a sum of disjoint distributions, one of which is lower dimensional: _p_ ( **a** _,_ **b** ) = _wp_ ( **a** ) _δ_ ( **_b_** ) + (1 _− w_ ) _p_ ( **a** _,_ **b** ) 

Here, _p_ is a probability density function that is conditional on the subset of _T_ that **_f_** is active on, _δ_ is the Dirac delta function, and 0 _< w <_ 1. By two probability distributions being disjoint, we mean that they have disjoint support (there is no set where both have positive probability measure, or equivalently the two features **a** and **b** cannot be active at the same time). In Eq. (2.1), **a** is the first _k_ components of the vector **Rf** + **c** and **b** is the remaining _df − k_ components. When _p_ is separable or a mixture, we also say that **f** is separable or a mixture. We term a feature _irreducible_ if it is not reducible, i.e., if no rotation and translation makes it separable or a mixture. 

An example of a feature that is a mixture is a one hot encoding along a simplex; an example of a feature that is separable is a normal distribution<sup>2</sup> . In natural language, a mixture might be a one hot encoding of “breed of dog”, while a separable distribution might be the “latitude” and “longitude” of location tokens. 

In practice, the mixture and separability definitions may not be precisely satisfied. Thus, we soften our definitions to permit degrees of reducibility: 

**Definition 3** (Separability Index and _ϵ_ -Mixture Index) **.** Consider a feature **f** . The **separability index** _S_ ( **f** ) measures the minimal mutual information between all possible **a** and **b** defined in Eq. (2.1): 



where _I_ denotes the mutual information. Smaller values of _S_ ( **f** ) mean that **f** is more separable. The _ϵ_ **-mixture index** _Mϵ_ ( **f** ) tests how often **f** can be projected near zero while it is active: 



Larger values of _Mϵ_ ( **f** ) mean that **f** is more of a mixture. 

In Section A.2, we expand on the intuition behind why the separability and _ϵ_ -mixture indices as defined here correspond to weakened versions of Definition 2. 

We develop optimization procedures to empirically solve for the separability and _ϵ_ -mixture indices of two dimensional feature distributions. At a high level, the separability procedure iterates over a sweep of rotations and estimates the mutual information between the axes for each angle, while the _ϵ_ -mixture index procedure performs gradient descent to find the _ϵ_ band that contains the largest possible fraction of the feature distribution. For more details on the implementation of the tests, see Section A.2.2. In Section 2.4, we apply these empirical tests to real language model feature distributions to find irreducible multi-dimensional features; we show the detailed test results on the “days of the week” cluster in Fig. 2.2 

> 2since any multidimensional Gaussian can be rotated to have a diagonal covariance matrix 

29 

### **2.3.2 Superposition** 

In this section, we propose an updated _superposition hypothesis_ (Elhage, Hume, et al. 2022) that takes into account multi-dimensional features. First, we restate the original superposition hypothesis: 

**Definition 4** ( _δ_ -orthogonal matrices) **.** Two matrices **A** 1 _∈_ R<sup>_d×d_1</sup> and **A** 2 _∈_ R<sup>_d×d_2</sup> are _δ_ - _orthogonal_ if _|_ **x** 1 _·_ **x** 2 _| ≤ δ_ for all unit vectors **x** 1 _∈_ colspace( **A** 1) and **x** 2 _∈_ colspace( **A** 2). 

**Hypothesis 1** (One-Dimensional Superposition Hypothesis, paraphrased from (Elhage, Hume, et al. 2022)) **.** Hidden states **x** _i,l_ are the sum of many ( _≫ d_ ) sparse one-dimensional features _fi_ and pairwise _δ_ -orthogonal vectors **v** _i_ such that **x** _i,l_ ( _t_ ) =<sup>�</sup> _i_<sup>**v**</sup><sup>_ifi_(</sup><sup>_t_).Weset</sup><sup>_fi_(</sup><sup>_t_)</sup> to zero when _t_ is outside the domain of _fi_ . 

In contrast, our new superposition hypothesis posits independence between irreducible multi-dimensional features instead of unknown levels of independence between one-dimensional features: **Hypothesis 2** (Multi-Dimensional Superposition Hypothesis, changes underlined) **.** Hidden states **x** _i,l_ are the sum of many ( _≫ d_ ) sparse low-dimensional irreducible features **f** _i_ and pairwise _δ_ -orthogonal matrices **V** _<u>i</u> ∈_ R<sup>_d×d_</sup><sup>**f**</sup><sup>_i_</sup> such that **x** _i,l_ ( _t_ ) =<sup><u>�</u></sup> _i_<sup>**V**</sup><sup>_<u>i</u>_</sup> **f** _i_ ( _t_ ). We set **f** _i_ ( _t_ ) to zero when _t_ is outside the domain of **f** _i_ . 

Note that since multi-dimensional features can be written as the sums of projections of lower-dimensional features, our new superposition hypothesis is a stricter version of Hypothesis 1. In the next section, we will explore empirical evidence for our hypothesis, while in Section A.1, we prove upper and lower bounds on the number of _δ_ -almost orthogonal matrices **V** _i_ that can be packed into _d_ dimensional space. 

## **2.4 Sparse Autoencoders Find Multi-Dimensional Features** 

In this section, we describe a method to identify multi-dimensional features in language model hidden states using sparse autoencoders (SAEs). Sparse autoencoders (SAEs) deconstruct model hidden states into sparse vector sums from an over-complete basis (Bricken, Templeton, et al. 2023; Cunningham et al. 2023b). For hidden states _Xi,l_ , a one-layer SAE of size _m_ with sparsity penalty _λ_ minimizes the following dictionary learning loss (Bricken, Templeton, et al. 2023; Cunningham et al. 2023b): 



In practice, the _L_ 0 loss on the last term is relaxed to _Lp_ for 0 _< p ≤_ 1 to make the loss differentiable. We call the _m_ columns of **D** (vectors in R<sup>_d_</sup> ) **dictionary elements** . 

We now argue that SAEs can discover irreducible multi-dimensional features by _clustering_ **D** . We will consider a simple form of clustering: build a complete graph on **D** with edge 

30 



<!-- Start of picture text -->
M (f) = 0.4750 S(f) = 0.9506<br>600<br>1.0<br>400<br>0.5<br>200<br>0 0.0<br>PCA 2 2 0 2 0 3 2<br>2 2<br>Norm. v f + c<br>Angle<br>PCA 3 Count<br>Mutual Info (bits)<br><!-- End of picture text -->

Figure 2.2: Empirical _ϵ_ -mixture index and separability index for the “days of the week” cluster along PCA components 2 and 3. **Left:** The _ϵ_ band parameterized by **v** and _c_ that the optimization procedure found contained the highest fraction of points. **Mid:** Dot products of points in the feature distribution with the _ϵ_ band; _Mϵ_ ( **_f_** ) is the percent of dot products within _ϵ_ = 0 _._ 1 of 0. **Right:** Estimated mutual information for different rotations of the space; _S_ ( **_f_** ) is the minimum over all rotations. This point cloud has a lower _ϵ_ -mixture index and higher separability index than PCA projections within typical clusters (see Fig. 2.3), indicating that it is more likely to be an irreducible multi-dimensional feature. 

weights equal to the cosine similarity between dictionary elements, prune all edges below a threshold _T_ , and then set the clusters equal to the connected components of the graph. If we now consider the spaces spanned by each cluster, they will be approximately _T_ -orthogonal by construction, since their basis vectors are all _T_ -orthogonal. Now, consider some irreducible two-dimensional feature **f** ; we claim that if the SAE is large enough and **f** is active enough such that the SAE can reconstruct **f** when **f** is active, one of the clusters is likely to be exactly equal to **f** . If **D** includes just two dictionary elements spanning **f** , then these elements both must have nonzero activations post-ReLU to reconstruct **f** (otherwise **f** is a mixture). Because of the sparsity penalty in Eq. (2.4), this two-vector solution to reconstruct **f** is disincentivized, so instead the dictionary is likely to learn many elements that span **f** . These dictionary elements will then have a high cosine similarity, and so the edges between them will not be pruned away during the clustering process; hence, they will be in a cluster. 

Thus, we have a way to operationalize Hypothesis 2: clustering **D** finds _T_ -orthogonal subspaces, and if irreducible multi-dimensional features exist, they are likely to be equal to some of these subspaces. This suggests a natural approach to using sparse autoencoders to search for irreducible multi-dimensional features: 

1. Cluster dictionary elements by their pairwise cosine similarity. We use both the simple similarity-based pruning technique described above, as well as spectral clustering; see Section A.6 for details, including comments on scalability. 

2. For each cluster, run the SAEs on all **x** _i,l ∈ Xi,l_ and ablate all dictionary elements not in the cluster. This will give the reconstruction of each **x** _i,l_ restricted to the cluster found in step 1 (if no cluster dictionary elements are non-zero for a given point, we ignore the point). 

3. Examine the resulting reconstructed activation vectors for irreducible multi-dimensional features. This step can be done manually by visually inspecting the PCA projections for 

31 

known irreducible multi-dimensional structures (e.g. circles, see Fig. A.1) or automatically by passing the PCA projections to the tests for Definition 3. 

Pseudocode for this method is in the appendix in Line 1. This method succeeds on toy datasets of synthetic irreducible multi-dimensional features; see Section A.4.<sup>3</sup> We apply this method to language models using GPT-2 (Radford et al. 2019) SAEs trained by Bloom 2024 for every layer and Mistral 7B (A. Q. Jiang et al. 2023) SAEs that we train on layers 8, 16, and 24 (training details in Section A.5). 

Strikingly, we reconstruct irreducible multidimensional features that are _interpretable_ circles: in GPT-2, days, months, and years are arranged circularly in order (see Fig. 2.1); in Mistral 7B, days and months are arranged circularly in order (see Fig. A.6). These plots contain the PCA dimensions that most clearly show circular structure; these best dimensions are usually the second and third because the first PCA dim is an “intensity” direction that manifests as the radius of the circle in Fig. 2.1 (thus the overall structure for these multi-d features is perhaps best thought of as a cone). See Fig. A.5 for all PCA dimensions visualized). 



<!-- Start of picture text -->
1.0<br>Other cluster<br>Cluster shown in Figure 1<br>0.8<br>days<br>0.6<br>months<br>years<br>0.4<br>0.2<br>0.0<br>1.0 0.8 0.6 0.4 0.2 0.0<br>M (f): more irreducible<br>f): more irreducible (S<br><!-- End of picture text -->

Figure 2.3: Mixture index and separability index of GPT-2 features. Features from Fig. 2.1,, which we had manually identified, score highly as candidate multidimensional features with these metrics. 

Figure 2.3: Mixture index and sepaFor each cluster of GPT-2 SAE features, we take rability index of GPT-2 features. Feathe reconstructed activations and project them onto tures from Fig. 2.1,, which we had PCA components 1-2, 2-3, 3-4, and 4-5 (or fewer if manually identified, score highly as there are fewer features in the cluster) and measure candidate multidimensional features the separability index and _ϵ_ -mixture index of each 2D with these metrics. point cloud as described in Section A.2.2. The mean scores across these planes are a computationally tractable approximation of Definition 3. We plot these mean scores in Fig. 2.3, and find that the features which we had manually identified in Fig. 2.1 are among the top scoring features along both measures of irreducibility. Thus, our theoretical tests can indeed be used to find interpretable irreducible features. We show the top 20 feature clusters, measured by the product of (1 _− ϵ_ -mixture index ) and separability index, in Section A.7. Out of all 1000 clusters, the Fig. 2.1 clusters rank 9 _,_ 28 _,_ and 15 by this metric, respectively.<sup>4</sup> 

## **2.5 Circular Representations in Large Language Models** 

In this section, we examine tasks in which models _use_ the multi-dimensional features we discovered in Section 2.4, thereby providing evidence that these representations are indeed the fundamental unit of computation for some problems. Inspired by prior work studying circular representations in modular arithmetic (Z. Liu, Kitouni, et al. 2022), we define two 

> 3Code: https://github.com/JoshEngels/MultiDimensionalFeatures 

> 4We also tried an alternative ranking scheme: we sorted the clusters by separability and irreducibility and set the cluster score equal to the minimum sorted position between the two sorted lists. The Fig. 2.1 clusters rank 8 _,_ 105 _,_ and 12 by this metric. 

32 

Table 2.1: Aggregate model accuracy on days of the week and months of the year modular arithmetic tasks. Performance broken down by problem instance in Section A.9. 

|Model|`Weekdays`<br>`Mon`|`ths`|
|---|---|---|
|Llama 3 8B|29 / 49<br>143|/ 144|
|Mistral 7B|31 / 49<br>125|/ 144|
|GPT-2|8 / 49<br>10 /|144|





<!-- Start of picture text -->
Tue Jan Feb<br>Mon Mar<br>Wed Dec<br>Apr<br>Sun Nov May<br>Oct<br>Sat Thu Jun<br>Fri Sep Jul<br>Aug<br><!-- End of picture text -->

Figure 2.4: Top two PCA components on the _α_ token. Colors show _α_ . **Left:** Layer 30 of Mistral on `Weekdays` . **Right:** Layer 5 of Llama on `Months` . 

prompts that represent “natural” modular arithmetic tasks: 

- `Weekdays` task: _“Let’s do some day of the week math. Two days from Monday is”_ `Months` task: _“Let’s do some calendar math. Four months from January is”_ 

For `Weekdays` , we range over the 7 days of the week and durations between 1 and 7 days to get 49 prompts. For `Months` , we range over the 12 months of the year and durations between 1 and 12 months to get 144 prompts. Mistral 7B and Llama 3 8B (AI@Meta 2024) achieve reasonable performance on the `Weekdays` task and excellent performance on the `Months` task (measured by comparing the highest logit valid token against the ground truth answer), as summarized in Table 2.1. Interestingly, although these problems are equivalent to modular arithmetic problems _α_ + _β ≡_ ? (mod _m_ ) for _m_ = 7 _,_ 12, both models get trivial accuracy on plain modular addition prompts, e.g. “5 + 3 (mod 7) _≡_ ”. Finally, although GPT-2 has circular representations, it gets trivial accuracy on `Weekdays` and `Months` . 

To simplify discussion, let _α_ be the day of the week or month of the year token (e.g. “Monday” or “April”), _β_ be the duration token (e.g. “four” or “eleven”), and _γ_ be the target ground truth token the model should predict, such that (abusing notation) we have _α_ + _β_ = _γ_ . Let the prompts of the task be parameterized by _j_ , such that the _j_ th prompt asks about _αj_ , _βj_ , and _γj_ . 

We confirm that Llama 3 8B and Mistral 7B have circular representations of _α_ on this task by examining the PCA projections of hidden states across prompts at various layers on the _α_ token. We plot two of these in Fig. 2.4 and show all layers in Fig. A.9. These plots show circular representations as the highest varying two components in the model’s representation of _α_ at many layers. 

### **2.5.1 Intervening on Circular Day and Month Representations** 

We now experiment with _intervening_ on these circular representations. We base our experiments on the common interpretability technique of activation patching, which replaces activations from a “dirty” run of the model with the corresponding activations from a “clean” run (Zhang and Nanda 2023). Activation patching empirically tests whether a specific model component, position, and/or representation has a causal influence on the model’s output. We 

33 



<!-- Start of picture text -->
Step 1: Learn Circular Probe<br>Training Find +Apply  Circular Learn 1 2 3<br>Points  MatrixPCA Probe   0 6 5 4<br>Step 2: Intervene with<br>Initial<br>point Apply MatrixPCA  5 6 5 0 4 132 Circular Probe Apply  00 1166 2255 3344<br>Apply intervention<br>New Apply<br>point  transpose of PCA  00 11 Apply  1 2 3<br>Matrix 5665 44 3322 pinv(P)  0 6 5 4<br><!-- End of picture text -->

Figure 2.5: Visual representation of the circular intervention process. **Top:** We learn a circular probe on the PCA projection of a training set. **Bot:** To intervene, we change the circular representation to _αj_<sup>_′_andav-</sup> erage ablate other dimensions. 



<!-- Start of picture text -->
Mistral Weekdays Mistral Months<br>2.5 5<br>0.0 0<br>2.5 5<br>Llama Weekdays Llama Months<br>2<br>5<br>0 0<br>5<br>2<br>0 10 20 0 10 20<br>Layer<br>No-op Patch circle Average ablate<br>Patch layer Patch PCA<br>Average logit diff<br><!-- End of picture text -->

Figure 2.6: Mean and 96% error bars for intervening on the _α_ token across layers using different intervention methods. The circular intervention technique outperforms patching only the top 5 PCA components and leaving the rest unchanged, and almost reaches the upper bound performance of patching the entire layer. 

employ a custom subspace patching method to allow testing for whether a specific _circular subspace_ of a hidden state is sufficient to causally explain model output. Specifically, our patching technique relies on the following steps (visualized in Fig. 2.5): 

1. **Find a subspace with a circle to intervene on** : Using a PCA reduced activation subspace to avoid overfitting, we train a “circular probe” to identify representations which exhibit strong circular patterns. More formally, let **x**<sup>_j_</sup> _i,l_<sup>bethehiddenstateatlayer</sup><sup>_l_token</sup> position _i_ for prompt _j_ . Let **W** _i,l ∈_ R<sup>_k×d_</sup> be the matrix consisting of the top _k_ principal component directions of **x**<sup>_j_</sup> _i,l_<sup>.Inourexperiments,weset</sup><sup>_k_= 5.Welearnalinearprobe</sup><sup>**P**</sup><sup>_∈_R2</sup><sup>_,k_</sup> from **W** _i,l · Xi,l_ to a unit circle in _α_ . In other words, if `circle` ( _α_ ) = [cos(2 _πα/_ 7) _,_ sin(2 _πα/_ 7)] for `Weekdays` and `circle` ( _α_ ) = [cos(2 _πα/_ 12) _,_ sin(2 _πα/_ 12)] for `Months` , **P** is defined as follows: 



2. **Intervene on the subspace** : Say our initial prompt had _α_ = _αj_ and we are intervening with _α_ = _αj′_ . In this step, we replace the model’s projection on the subspace **P** _·_ **W** _i,l_ , which will be close to `circle` ( _αj_ ), with the “clean” point `circle` ( _αj′_ ). Note that we do not use the hidden state **x**<sup>_j′_</sup> _i,l_<sup>fromthe“clean”run,onlythe“clean”label</sup><sup>_αj′_.Inpractice,othersubspaces</sup> of **x**<sup>_j_</sup> _i,l_<sup>maybeusedconcurrentlybythemodelin“backup”circuits(seee.g.K.Wangetal.</sup> 2022) to compute the answer, so if we just intervene on the circular subspace the remaining components of the activation may interfere in downstream computations. Thus, to isolate the effect of our intervention, we set the average ablate the portion of the activation not in 

34 

the intervened subspace. Letting **~~x~~** _i,l_ be the average of **x**<sup>_j_</sup> _i,l_<sup>acrossallpromptsindexedby</sup><sup>_j_</sup> and **P**<sup>+</sup> be the pseudoinverse of **P** , we intervene via the formula 



We run our patching on all 49 `Weekday` problems and 144 `Month` problems and use as “clean” runs the 6 or 11 other possible values for _β_ , resulting in a total of 49 _∗_ 6 patching experiments for `Weekdays` and 144 _∗_ 11 patching experiments for `Months` . We also run baselines where we (1) replace the entire subspace corresponding to the first 5 PCA dimensions with the corresponding subspace from the clean run, (2) replace the entire layer with the corresponding layer from the clean run, and (3) replace the entire layer with the average across the task. The metric we use is _average logit difference_ across all patching experiments between the original correct token ( _αj_ ) and the target token ( _αj′_ ). See Fig. 2.6 for these interventions on all layers of Mistral 7B and Llama 3 8B on `Weekdays` and `Months` . 

The main takeaway from Fig. 2.6 is that circular subspaces are causally implicated in computing _γ_ , especially for `Weekdays` . Across all models and tasks, early layer interventions on the circular subspace have almost the same intervention effect as patching the entire layer, and are usually better than patching the top PCA dimensions from the clean problem. 

Patching experiments in Section A.10 show _α_ is copied to the final token on layers 15 to 17, which is why interventions drop off there. Additionally, while in this section we train a probe on a dataset of prompts, in Section 2.5.2, we show that intervening on the circle discovered via SAE clustering in Section 2.4 also works. 

To investigate exactly how models use the circular subspace, we perform _off distribution_ interventions. We modify Eq. (2.6) so that instead of intervening on the circumference `circle` ( _α_ ), we sweep over a grid of positions ( _r, θ_ ) within the circle: 



We intervene with _r ∈_ [0 _,_ 0 _._ 1 _, . . . ,_ 2] _, θ ∈_ [0 _,_ 2 _π/_ 100 _, . . . ,_ 198 _π/_ 100] and record the highest logit _γ_ after the forward pass. Fig. 2.7 displays these results on Mistral layer 5 for _β ∈_ [2 _,_ 3 _,_ 45]. They imply that Mistral treats the circle as a multi-dimensional representation with _α_ encoded in the angle. 

### **2.5.2 SAE Plane Interventions** 

In the last section, we train probes manually on the PCA of the activations to fit a circle. A perhaps more natural approach is intervening with the precise circle we found in Section 2.4. To determine if this approach is feasible, we first project layer 8 Mistral 7B `Weekdays` activations into the weekdays plane that 



<!-- Start of picture text -->
Task Duration = 2 Days Task Duration = 3 Days<br>2 2<br>1 1<br>0 0<br>1 1<br>2 2<br>2 0 2 2 0 2<br>Task Duration = 4 Days Task Duration = 5 Days<br>2 2<br>1 1<br>0 0<br>1 1<br>2 2<br>2 0 2 2 0 2<br>Monday Wednesday Friday Sunday<br>Tuesday Thursday Saturday<br><!-- End of picture text -->

Figure 2.7: Off distribution interventions on Mistral layer 5 on the `Weekdays` task. The color corresponds to the highest logit _γ_ after performing the circular subspace intervention on that point. 

was discovered by clustering (see Fig. A.6; the plane is defined by PCA dimensions 2 and 3 

35 

of the cluster). In the rest of this section, we call this plane the SAE plane. In Fig. A.10, we find that indeed, the `Weekdays` representations projected into the SAE plane form a circle (see Fig. A.10). We thus can fit a circular probe to this 2D plane as in Eq. (2.5). Similarly, we call this probe the SAE probe. 

Because we only have layer 8 clustering results for Mistral, we train an SAE probe only on layer 8. We then evaluate interventions using this SAE probe on layer 8 of Mistral, but also at neighboring layers, since nearby layers should have similar representations (see e.g. Belrose et al. 2023). We compare to two baselines: 1) training a normal circular probe on the PCA projections of each layer as described in Section 2.5.1, and 2) training a circular probe only on layer 8 and then evaluating on adjacent layers (in the same way as for the layer 8 SAE probe). We show the results of these methods in Fig. 2.8. 



<!-- Start of picture text -->
Intervene w/ Layer 8 Probe<br>0.0 Intervene w/ SAE Probe<br>Intervene w/ Per-Layer Probe<br>0.5<br>1.0<br>1.5<br>2.0<br>2.5<br>3.0<br>6 7 8 9 10<br>Layer<br>Average logit difference after intervention<br><!-- End of picture text -->

Figure 2.8: Interventions on the Mistral 7B `Weekdays` task with different methods of determining the probe. 

We find that on all layers, using the SAE `Weekdays` task with different methods of deterprobe only slightly decreases intervention permining the probe. formance as compared to training a circular probe (from -2.58 to -2.01 average logit difference on layer 8). Even more interestingly, the layer 8 SAE probe is much more robust to layer shifts than the layer 8 circular probe; for example, using the layer 8 circular probe on layer 6 results in an average logit difference of 0.029, whereas using the layer 8 SAE probe results in an average logit difference of -2.32. This is intriguing evidence that the SAE is perhaps finding more “true” (or at least more robust) features than our circular probing technique. 

### **2.5.3 Continuity of Circular Representations** 

In past sections, the representations of the interpretable numeric quantities we have discovered have been mostly _discontinuous_ ; that is, the days of the week and months of the year in Fig. 2.1 and Fig. A.6 are clustered at the vertices of a heptagon and dodecagon, and there is nothing "between" adjacent weekdays or months along the circle. In this section, we will examine the "continuity" of the circular features we have discovered. Although continuity of the representation is not a requirement of Definition 3, it would further decrease the _ϵ_ -mixture index, and would also increase our subjective perception of the circular feature as an intrinsic model feature representing a continuous quantity (time). Thus, we create a synthetic dataset containing the text "[very early/very 



<!-- Start of picture text -->
Fri<br>Thu Evening ThuMorning FriEvening Fri Morning SatSat<br>Morning Thu Evening Sat<br>Evening Wed<br>Wed Morning Wed Morning Sun Sun<br>Evening Sun<br>Evening Tue<br>Tue Morning TueEvening MonMorning Mon<br>Mon<br><!-- End of picture text -->

Figure 2.9: Layer 30 Mistral 7B activations for [morning/evening] on [Monday/Tuesday/.../Sunday], plotted projected into the PCA plane for [Monday/Tuesday/.../Sunday]. 

36 

late] on [Monday/Tuesday/.../Sunday]" and simply plot the projections of the layer 30 activations into the top two PCA components of the activations of [Monday/Tuesday/.../Sunday]. The results, shown in Fig. 2.9, show that Mistral 7B indeed can map intermediate quantities to their expected place in the circle: the very early and very late version of each weekday are more towards the last and the next weekday along the circle, respectively. We show similar results for "[morning/evening] on [Monday/Tuesday/.../Sunday]" in Appendix Fig. A.11. 

## **2.6 Discussion** 

Our work proposes a significant refinement to the simple one-dimensional linear representation hypothesis. While previous work has convincingly shown the existence of one-dimensional features, we find evidence for irreducible multi-dimensional representations, requiring us to generalize the notion of a feature to higher dimensions. Fortunately, we find that existing unsupervised feature extraction methodologies like sparse autoencoders can readily be applied to discover multi-dimensional representations. However, we think our work raises interesting questions about whether individual SAE features are appropriate “mediators” (Mueller et al. 2024) for understanding model computation, if some features are in fact multi-dimensional. Although taking a multi-dimensional representation perspective may be more complicated, we believe that uncovering the true (perhaps multi-dimensional) nature of model representations is necessary for discovering the underlying algorithms that use these representations. Ultimately, our field aims to turn complex circuits in future more-capable models into formally verifiable programs (Tegmark and Omohundro 2023; Dalrymple et al. 2024), which requires the ground truth “variables” of language models; we believe this work takes an important step towards discovering these variables. 

**Limitations:** It is unclear why we did not find more interpretable multi-dimensional features. We are unsure if we are failing to interpret some of the high-scoring multi-dimensional features, if most multi-dimensional features lie in dimensions higher than two, if our clustering technique is not powerful enough to find some features, or if there are truly not that many. Additionally, our definitions for irreducible features (Definition 2) are purely statistical and not intervention based, and also had to be relaxed to hold in practice, resulting in measures that return a possibly subjective “degree” of reducibility (Definition 3). Thus, although this work provides preliminary evidence for the multi-dimensional superposition hypothesis (Hypothesis 2), it is still unclear if this theory provides the best description for the representations models use. Future work might make progress on this question by investigating new techniques for decomposing model representations, exploring higher dimensional representations, or determining conclusively whether models use representations in ways that necessitate the representations are non-linear. 

37 

38 

# **Appendix A** 

# **Appendix: Not All Language Models Are One-Dimensionally Linear** 

## **A.1 Multi-Dimensional Feature Capacity** 

The Johnson-Lindenstrauss (JL) Lemma (Johnson and Lindenstrauss 1984) implies that we can choose _e_<sup>_Cdδ_2</sup> pairwise one-dimensional _δ_ -orthogonal vectors to satisfy Hypothesis 1 for some constant _C_ , thus allowing us to build the model’s feature space with a number of one-dimensional _δ_ -orthogonal features exponential in _d_ . We now prove a similar result for low-dimensional projections (the main idea of the proof is to combine _δ_ -orthogonal vectors as guaranteed from the JL lemma): 

**Theorem 1.** _For any d_<sup>_′_</sup> _and δ, it is possible to choose d_ max1<sup>_eC_1(</sup><sup>_d/d′_2)</sup><sup>_δ_2</sup><sup>_pairwiseδ-orthogonal_</sup> _matrices_ **A** _i ∈_ R<sup>_ni×d′_</sup> _for some constant C_ 1 _. Furthermore, it is not possible to choose more than e_<sup>_C_2(</sup><sup>_d−d_max</sup><sup>_δ_log(</sup><sup><u>1</u></sup> _δ_<sup>))</sup> _for some constant C_ 2 _._ 

We will first prove a lemma that will help us prove Theorem 1. 

**Lemma 1.** _Pick n pairwise δ-orthogonal unit vectors in_ **v** 1 _, . . . ,_ **v** _n ∈_ R<sup>_d_</sup> _. Let_ **y** _∈_ R<sup>_d_</sup> _be a unit norm vector that is a linear combination of unit norm vectors_ **v** 1 _, . . . ,_ **v** _n with coefficients z_ 1 _. . . , zn ∈_ R _. We can write_ **A** = [ **v** 1 _, . . . ,_ **v** _n_ ] _and_ **z** = [ _z_ 1 _, . . . , zn_ ]<sup>_T_</sup> _, so that we have_ **y** =<sup>�</sup><sup>_n_</sup> _k_ =1<sup>_zk_</sup><sup>**v**</sup><sup>_k_=</sup><sup>**Az**</sup><sup>_Twith∥_</sup><sup>**y**</sup><sup>_∥_2= 1</sup><sup>_.Then,_</sup> 



_Proof._ We will first bound the _L_ 2 norm of **z** . If _σn_ is the minimum singular value of **A** , then we have via standard singular value inequalities (Higham 2021) 



Thus we now lower bound _σn_ . The singular values are the square roots of the eigenvalues of the matrix **A**<sup>_T_</sup> **A** , so we now examine **A**<sup>_T_</sup> **A** . Since all elements of **A** are unit vectors, 

39 

the diagonal of **A**<sup>_T_</sup> **A** is all ones. The off diagonal elements are dot products of pairs of _δ_ -orthogonal vectors, and so are within the range [ _−δ, δ_ ]. Then by the Gershgorin circle theorem (Gershgorin 1931), all eigenvalues _λi_ of **A**<sup>_T_</sup> **A** are in the range 



In particular, _σn_<sup>2=</sup><sup>_λn≥_1</sup><sup>_−δ_(</sup><sup>_n −_1),andthus</sup><sup>_σn≥_</sup> ~~<u>�</u>~~ 1 _− δ_ ( _n −_ 1). Plugging into our upper bound for _∥_ **z** _∥_ 2, we have that _∥_ **z** _∥_ 2 _≤_ 1 _/_ ~~�~~ 1 _− δ_ ( _n −_ 1). Finally, the largest _L_ 1 for a point on an _n_ -hypersphere of radius _r_ is when all dimensions are equal and such a point has magnitude<sup>_√_</sup> _<u>nr</u>_ , so 



**Theorem 1.** _For any d_<sup>_′_</sup> _and δ, it is possible to choose d_ max1<sup>_eC_1(</sup><sup>_d/d′_2)</sup><sup>_δ_2</sup><sup>_pairwiseδ-orthogonal_</sup> _matrices_ **A** _i ∈_ R<sup>_ni×d′_</sup> _for some constant C_ 1 _. Furthermore, it is not possible to choose more than e_<sup>_C_2(</sup><sup>_d−d_max</sup><sup>_δ_log(</sup><sup><u>1</u></sup> _δ_<sup>))</sup> _for some constant C_ 2 _._ 

_Proof._ By the JL lemma (Johnson and Lindenstrauss 1984), for any _d_ and _δ_ , we can choose _e_<sup>_Cdδ_2</sup> _δ_ -orthogonal unit vectors in R<sup>_d_</sup> indexed as **v** _i_ , for some constant _C_ . Let **A** _i_ = [ **v** _d_ max _∗i, . . . ,_ **v** _d_ max _∗i_ + _ni−_ 1] where each element in the brackets is a column. Then by construction all **A** _i_ are matrices composed of unique _δ_ -orthogonal vectors and there are _d_ max1<sup>_eCdδ_2matrices</sup><sup>**A**</sup><sup>_i_.</sup> 

Now, consider two of these matrices **A** _i_ = [ **v** 1 _, . . . ,_ **v** _ni_ ] and **A** _j_ = [ **u** 1 _, . . . ,_ **u** _nj_ ], _i̸_ = _j_ ; we will prove that they are _f_ ( _δ_ )-orthogonal for some function _f_ . Let **y** _i_ =<sup>�</sup><sup>_n_</sup> _k_ =1<sup>_izi,k_</sup><sup>**v**</sup><sup>_k_bea</sup> vector in the colspace of **A** _i_ and **y** _j_ =<sup>�</sup><sup>_n_</sup> _k_ =1<sup>_jzj,k_</sup><sup>**u**</sup><sup>_k_beavectorinthecolspaceof</sup><sup>**A**</sup><sup>_j_,such</sup> that **y** _i_ and **y** _j_ are unit vectors. To prove _f_ ( _δ_ )-orthogonality, we must bound the absolute dot product between **y** _i_ and **y** _j_ : 

40 



Thus **A** _i_ and **A** _j_ are _f_ ( _δ_ )-orthogonal for _f_ ( _δ_ ) = _δd_ max _/_ (1 _− δd_ max), and so it is possible to choose _d_ max1<sup>_eCdδ_2pairwise</sup><sup>_f_(</sup><sup>_δ_)-orthogonalprojectionmatrices.Remappingthevariable</sup><sup>_δ_</sup> with _δ �→ f_<sup>_−_1</sup> ( _δ_ ) = _δ/_ ( _d_ max(1 + _δ_ )), we find that it is possible to choose _d_ max1<sup>_eCdδ_2</sup><sup>_/_((1+</sup><sup>_δ_)2</sup><sup>_d_</sup> max<sup>2)</sup> pairwise _δ_ -orthogonal projection matrices. Because 1 + _δ_ is at most 2 with _δ ∈_ (0 _,_ 1), we can further simplify the exponent and find that it is possible to choose _d_ max1<sup>_eC_(</sup><sup>_d/d_</sup> max<sup>2)</sup><sup>_δ_2</sup><sup>_/_4</sup> pairwise _δ_ -orthogonal projection matrices. Absorbing the 4 into the constant _C_ finishes the proof of the lower bound. 

For the upper bound, we can proceed much more simply. Consider _k_ pairwise _δ_ -orthogonal matrices _Ai ∈_ R<sup>_d′_</sup> . Since these matrices are full rank, their column spaces each parameterize a subspace of dimension _d_<sup>_′_</sup> , and so by a result from (Alon 2003) it is possible to choose _e_<sup>_Cd′δ_2 log(</sup><sup><u>1</u></sup> _δ_<sup>)</sup> almost orthogonal vectors in this subspace. Furthermore, by our definition of _δ_ -orthogonal matrices, all pairs of these vectors between subspaces will be _δ_ -orthogonal. Finally, again by (Alon 2003) we cannot have more than _e_<sup>_Cdδ_2 log(</sup><sup><u>1</u></sup> _δ_<sup>)</sup> _δ_ -orthogonal vectors overall, so we have that 



and simplfying, 



41 

These results imply that models can still represent an exponential number of higher dimensional features. However, there is a large exponential gap between the lower and upper bound we have shown. If the lower bound is reasonably tight, then this would mean that models would be highly incentivized to fit features within the smallest dimensional space possible, suggesting a reason for recent work showing interesting compressed encodings of multi-dimensional features in toy problems (Morwani et al. 2023). 

Note that the proof assumes the “worst case” scenario that all of the features are dimension _d_ max, while in practice many of the features may be 1 or low dimensional, so the effect on the capacity of a real model that represents multi-dimensional features is unlikely to be this extreme. 

Finally, we note that the dictionary learning literature may have discovered similar results in the past (which we were unaware of), see Foucart and Rauhut 2013. 

## **A.2 More on Reducibility** 

### **A.2.1 Additional Intuition for Definitions** 

Here, we present some extra intuition and high level ideas for understanding our definitions and the motivation behind them. Roughly, we intend for our definitions in the main text to identify representations in the model that describe an object or concept in a way that fundamentally takes multiple dimensions. We operationalize this as finding a subspace of representations that 1. has basis vectors that “always co-occur” no matter the orientation 2. is not made up of combinations of independent lower-dimensional features. 

1. The first condition is met by the mixture part of our definition. The feature in question should be part of an irreducible manifold, and so should “fill” a plane or hyperplane. There shouldn’t be any part of the plane where the probability distribution of the feature is concentrated, because this region is then likely part of a lower dimensional feature. The idea of this part of the definition is to capture multi-dimensional objects; if the entire multi-dimensional space is truly being used to represent a high-dimensional object, then the representations for the object should be “spread out” entirely through the space. 

2. The second condition is met by the separability part of our definition. This part of the definition is intended to rule out features that co-occur frequently but are fundamentally not describing the same object or concept. For example, latitude and longitude are not a mixture in that they frequently co-occur, but we do not think it is necessarily correct to say they are part of the same multi-dimensional feature because they are independent. 

### **A.2.2 Empirical Irreducible Feature Test Details** 

Our tests for reducibility require the computation of two quantities _S_ ( **f** ) for the separability index and _Mϵ_ ( **f** ) for the _ϵ_ -mixture index. We describe how we compute each index in the following two subsections. 

42 

#### **Separability Index** 

We define the separability index in Equation 2.2 as 

#### _S_ ( **f** ) = min _I_ ( **a** ; **b** ) 

where the min is over rotations **R** used to split **f**<sup>_′_</sup> = **Rf** + **c** into **a** and **b** . In two dimensions, the rotation is defined by a single angle, so we can iterate over a grid of 1000 angles and estimate the mutual information between **a** and **b** for each angle. We first normalize **f** by subtracting off the mean and then dividing by the root mean squared norm of **f** (and multiplying by _√_ 2 since the toy datasets are in two dimensions). To estimate the mutual information, we first clip the data **f** to a 6 by 6 square centered on the origin. We then bin the points into a 40 by 40 grid, to produce a discrete distribution _p_ ( _a, b_ ). After computing the marginals _p_ ( _a_ ) and _p_ ( _b_ ) by summing the distribution over each axis, we obtain the mutual information via the formula 



#### **_ϵ_ -Mixture Index** 

We define the _ϵ_ -mixture index in Equation 2.3 as 



The challenge with computing _Mϵ_ ( **f** ) is to compute the maximum. We opted to maximize via gradient descent; and we guaranteed differentiability by softening the inequality _<_ with a sigmoid, 



where _T_ is a temperature, which we linearly decay from 1 to 0 throughout training. We optimize for **v** and _c_ using this loss _Mϵ,T_ ( **f** _,_ **v** _, c_ ) using full batch gradient descent over 10000 steps with learning rate 0 _._ 1. With the solution ( **v**<sup>_∗_</sup> _, c_<sup>_∗_</sup> ), the final value of _Mϵ,T_ =0( **f** _,_ **v**<sup>_∗_</sup> _, c_<sup>_∗_</sup> ) is then our estimate of _Mϵ_ ( **f** ). 

We also run the irreducibility tests on additional synthetic feature distributions in Fig. A.2a and Fig. A.2b. 

## **A.3 Alternative Definitions** 

In this section, we present an alternative definition of a reducible feature that we considered during our work. This chiefly deals with multi-dimensional features from the angle of _computational_ reducibility as opposed to _statistical_ reducibility. In other words, this definition considers whether representations of features on a specific set of tasks can be split up without 

43 



<!-- Start of picture text -->
2<br>2000<br>1<br>1000<br>Mϵ ( f ) = 0 . 6396<br>S ( f ) = 0 . 3691<br>0 − 5 0 5 00 π π 3 π 2 π<br>representation dim 1 normalized v  ·  f +  c 2 angle θ 2<br>(a) Testing S ( a ) and Mϵ ( a ) on a reducible feature a .<br>S ( f ) = 2 . 683<br>200 2<br>Mϵ ( f ) = 0 . 1784<br>100 1<br>0 − 1 . 5 − 1 . 0 − 0 . 5 0 . 0 00 π π 3 π 2 π<br>representation dim 1 normalized v  ·  f +  c 2 angle θ 2<br>(b) Testing S ( b ) and Mϵ ( b ) on an irreducible feature b<br>count<br>(bits)infoMutual<br>dim2representation<br>count<br>(bits)infoMutual<br>dim2representation<br><!-- End of picture text -->

Figure A.1: Testing irreducibility of synthetic features. **Left in each subfigure:** Distributions of **x** . For feature **a** , 63.96% lies within the narrow dotted lines, indicating the feature is likely a mixture. For feature **b** , 17.84% lies within the wide lines, indicating the feature is unlikely to be a mixture. The green cross indicates the angle _θ_ that minimizes mutual information. **Middle in each subfigure:** Histograms of the distribution of **v** _·_ **x** with red lines indicating a 2 _ϵ_ -wide region. **Right in each subfigure:** Mutual information between **a** and **b** as a function of the rotation angle _θ_ of matrix **R** . Feature **b** has a large minimum mutual information so is unlikely to be separable; feature **a** has a medium value of minimum mutual information of about 0 _._ 37 bits. 

44 



<!-- Start of picture text -->
100 0 . 50<br>Mϵ ( f ) = 0 . 0794<br>50 0 . 25<br>S ( f ) = 0 . 1156<br>0 − 2 0 2 0 . 000 π π 3 π 2 π<br>representation dim 1 normalized v  ·  f +  c 2 angle θ 2<br>(a) Testing S ( c ) and Mϵ ( c ) on a reducible feature c .<br>1000 1 . 0<br>Mϵ ( f ) = 0 . 259<br>500 0 . 5<br>S ( f ) = 0 . 2541<br>0 − 2 . 5 0 . 0 2 . 5 0 . 00 π π 3 π 2 π<br>representation dim 1 normalized v  ·  f +  c 2 angle θ 2<br>(b) Testing S ( d ) and Mϵ ( d ) on an irreducible feature d<br>count<br>(bits)infoMutual<br>dim2representation<br>count<br>(bits)infoMutual<br>dim2representation<br><!-- End of picture text -->

Figure A.2: Testing irreducibility of synthetic features. **Left in each subfigure:** Distributions of **x** . For feature **c** , 7.94% lies within the narrow dotted lines, indicating the feature is unlikely to be a mixture. For feature **d** , 25.90% lies within the wide lines, indicating the feature is likely a mixture. The green cross indicates the angle _θ_ that minimizes mutual information. **Middle in each subfigure:** Histograms of the distribution of **v** _·_ **x** with red lines indicating a 2 _ϵ_ -wide region. **Right in each subfigure:** Mutual information between **a** and **b** as a function of the rotation angle _θ_ of matrix **R** . Both features have a small ( _<_ 0 _._ 5 bits) minimum mutual information and so are likely separable. 

45 

changing the accuracy of the task. This captures an interesting (and important) aspect of feature reducibility, but because it requires a specific set of prompts (as opposed to allowing unsupervised discovery) we chose not to use it as our main definition. 

Our alternative definitions consider _representation spaces_ that are possibly multi-dimensional, and defines these spaces through whether they can completely explain a function _h_ on the output logits. We consider a _group theoretic_ approach to irreducible representations, via whether computation involving multiple group elements can be decomposed. 

### **A.3.1 Alternative Definition: Interventions and Representation Spaces** 

Assume that we restrict the input set of prompts _T_ = _{_ **t**<sup>_j_</sup> _}_ to some subset of prompts and that we have some evaluation function _h_ that maps from the output logit distribution of _M_ to a real number. For example, for the `Weekdays` problems, _T_ is the set of 49 prompts and _h_ could be the arg max over the days of week logits. Abusing notation, we let _M_ also be the function from the layer we are intervening on; this is always clear from context. Then we can define a representation space of **x**<sup>_j_</sup> _i,l_<sup>asasubspaceinwhichinterventionsalwayswork:</sup> 

**Definition 5** (Representation Space) **.** Given a prompt set _T_ = _{_ **t**<sup>_j_</sup> _}_ , a rank- _r_ dimensional _representation space_ of intermediate value **x**<sup>_j_</sup> _i,l_<sup>isarank</sup><sup>_r_projectionmatrix</sup><sup>_P_suchthatfor</sup> all _j, j_<sup>_′_</sup> , _h_ ( _M_ (( _I − P_ ) **x**<sup>_j_</sup> _i,l_<sup>+</sup><sup>_P_</sup><sup>**x**</sup><sup>_j_</sup> _i,l_<sup>_′_)) =</sup><sup>_h_(</sup><sup>_M_(</sup><sup>**x**</sup><sup>_j_</sup> _i,l_<sup>_′_)).</sup> 

Note that it immediately follows that the rank _d_ dimensional matrix _Id_ is trivially a rank _d_ representation space for all prompt sets _T_ . 

**Definition 6** (Minimality) **.** A representation space _P_ of rank _r_ is _minimal_ if there does not exist a lower rank representation space. 

A minimal representation with rank > 1 is a _multi-dimensional representation_ . 

**Definition 7** (Alternative Reducibility) **.** A representation space _P_ of rank _r_ is _reducible_ if there are orthonormal representation spaces _P_ 1 and _P_ 2 (such that _P_ 1 + _P_ 2 = _P_ , _P_ 1 _P_ 2 = 0) where 



for all _j, j_<sup>_′_</sup> . 

Suppose _T_ , _h_ and _M_ define the multiplication of two elements in a finite group _G_ of order _n_ . Then if we interpret the embedding vectors as the group representations, our definition of reducibility implies to the standard group-theoretical definition of irreducibility –– specifically, reducibility into a tensor product representation. 

## **A.4 Toy Case of Training SAEs on Circles** 

To explore how SAEs behave when reconstructing irreducible features of dimension _df >_ 1, we perform experiments with the following toy setup. Inspired by the circular representations 

46 

of integers that networks learn when trained on modular addition (Nanda, Chan, et al. 2023; Z. Liu, Kitouni, et al. 2022), we create synthetic datasets of activations containing multiple features which are each 2d irreducible circles. 

First however, consider activations for a single circle – points uniformly distributed on the unit circle in R<sup>2</sup> . We train SAEs on this data with encoder `Enc` ( **x** ) = `ReLU` ( **We** ( **x** _−_ **bd** ) + **be** ) and decoder `Dec` ( **f** ) = **Wdf** + **bd** . We train SAEs with _m_ = 2 and _m_ = 10 with the Adam optimizer and a learning rate of 10<sup>_−_3</sup> , sparsity penalty _λ_ = 0 _._ 1, for 20,000 steps, and a warmup of 1000 steps. In Fig. A.3 we show the dictionary elements of these SAEs. When _m_ = 2, the SAE must use both SAE features on each input point, and uses **db** to shift the reconstructed circle so it is centered at the origin. When _m_ = 10, the SAE learns **db** _≈_ 0 and the features spread out across the circle, arranged close together, and only a subset are active on each input. 



<!-- Start of picture text -->
sparsity loss: 1.93 sparsity loss: 1.27, seed=2 sparsity loss: 1.28, seed=4<br>2 SAE hidden activations<br>db<br>Reconstruction 1 1<br>1 Dictionary element<br>0 0 0<br>1<br>db db<br>1 Reconstruction 1 Reconstruction<br>2 many, | KA Dictionary element | Ap Dictionary element<br>2 1 0 1 2 1 0 1 1 0 1<br><!-- End of picture text -->

Figure A.3: SAEs trained to reconstruct a single 2d circle with _m_ = 2 (left) and _m_ = 10 (middle and right) dictionary elements. When there are several SAE features, there is no natural/canonical choice of feature directions, and the dictionary elements spread out across the circle. 

We now consider synthetic activations with multiple circular features. Our data consists of points in R<sup>10</sup> , where we choose two orthogonal planes spanned by ( **e1** _,_ **e2** ) and ( **e3** _,_ **e4** ), respectively. With probability one half a points is sampled uniformly on the unit circle in the **e1** - **e2** plane, otherwise the point will be sampled uniformly on the unit circle in the **e3** - **e4** plane. We train SAEs with _m_ = 64 on this data with the same hyperparameters as the single-circle case. 

We now apply the procedure described in Section 2.4 to see if we can automatically rediscover these circles. Encouragingly, we first find that the alive SAE features align almost exactly with either the **e1** - **e2** or the **e3** - **e4** plane. When we apply spectral clustering with `n_clusters` = 2 to the features with the pairwise angular similarities between dictionary elements as the similarity matrix (Fig. A.4, left), the two clusters correspond exactly to the features which span each plane. As described in Section 2.4, given a cluster of dictionary elements _S ⊂{_ 1 _, . . . , m}_ , we run a large set of activations through the SAE, then filter out samples which don’t activate any element in _S_ . For samples which do activate an element of _S_ , reconstruct the activation while setting all SAE features not in _S_ to have a hidden activation of zero. If some collection of SAE features together represent some irreducible 

47 

feature, we want to remove all other features from the activation vector, and so we only allow SAE features in the collection to participate in reconstructing the input activation. We find that this procedure almost exactly recovers the original two circles, which encouraged us to apply this method for discovering the features shown in Fig. 2.1 and Fig. A.6. 



<!-- Start of picture text -->
Cosine sim between alive features Cluster 1 reconstruction Cluster 2 reconstruction<br>0 1<br>1 1<br>15<br>0 0 0<br>30<br>1 1<br>1<br>0 15 30 1 0 1 1 0 1<br>alive SAE feature PCA axis 1 PCA axis 1<br>PCA axis 2 PCA axis 2<br>alive SAE feature<br><!-- End of picture text -->

Figure A.4: Automatic discovery of synthetic circular features by clustering SAE dictionary elements. 

## **A.5 Training Mistral SAEs** 

Our Mistral 7B (A. Q. Jiang et al. 2023) sparse autoencoders (SAEs) are trained on over one billion tokens from a subset of the Pile (Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, H. He, Thite, Nabeshima, et al. 2020a) and Alpaca (Peng et al. 2023) datasets. We train our SAEs on layers 8, 16, and 24 out of 32 total layers to maximize coverage of the model’s representations. We use a 16 _×_ expansion factor, yielding a total of 65536 dictionary elements for each SAE. 

To train our SAEs, we use an _Lp_ sparsity penalty for _p_ = 1 _/_ 2 with sparsity coefficient _λ_ = 0 _._ 012. Before an SAE forward pass, we normalize our activation vectors to have norm _√dmodel_ = 64 in the case of Mistral. We do not apply a pre-encoder bias. We use an AdamW optimizer with weight decay 10<sup>_−_3</sup> and learning rate 0.0002 with a linear warm up. We apply dead feature resampling (Bricken, Templeton, et al. 2023) five times over the course of training to converge on SAEs with around 1000 dead features. 

## **A.6 GPT-2 and Mistral 7B Dictionary Element Clustering** 

In this section, we first present pseudocode in Line 1 for the overall high level technique that finds multi-dimensional features and that uses clustering as a subroutine. We then provide the specific clustering algorithm implementations we use for GPT-2 and Mistral. 

48 

**Input:** Dictionary elements _D_ , activation vectors _Xi,l_ , `SAE` **Output:** Irreducible multi-dimensional features _Si,j ←_ `CosineSim` ( _Di, Dj_ ); _clusters ←_ `Cluster(` _S_ `)` ; _reconstructions ←{}_ ; **for** _cluster in clusters_ **do** _Rcluster ←_ ids of dictionary elements in _cluster_ ; **for** **_x_** _i,l in Xi,l_ **do** _encoding ←_ `ReLU` ( _E ·_ **_x_** _i,l_ ); **if** max( _encoding_ [ _Rcluster_ ]) _>_ 0 **then** _r ← D_ [: _, Rcluster_ ] _· encoding_ ; _reconstructions ← reconstructions ∪{r}_ ; **end end end** _features ←{}_ ; **for** _R in reconstructions_ **do** _proj ←_ `PCA(` _R_ `)` ; **if** `TestIrreducible(` _proj_ `)` **then** Add _proj_ to _features_ ; **end end return** _features_ **Algorithm 1:** High Level Clustering Approach For Finding Multi-D Features 

### **A.6.1 GPT-2-small methods and results** 

For GPT-2-small, we perform spectral clustering on the roughly 25k layer 7 SAE features from (Bloom 2024), using pairwise angular similarities between dictionary elements as the similarity matrix. We use `n_clusters` = 1000 and manually looked at roughly 500 of these clusters. For each cluster, we looked at projections onto principal components 1-4 of the reconstructed activations for these clusters. In Fig. A.5, we show projections for the most interesting clusters we identified, which appear to be circular representations of days of the week, months of the year, and years of the 20th century. 

### **A.6.2 Mistral 7B methods and results** 

For Mistral 7B, our SAEs have 65536 dictionary elements and we found it difficult to run spectral clustering on all of these at once. We therefore develop a simple graph based clustering algorithm that we run on Mistral 7B SAEs: 

1. Create a graph _G_ out of the dictionary elements by adding directed edges from each dictionary element to its _k_ closest dictionary elements by cosine similarity. We use _k_ = 2. 

2. Make the graph undirected by turning every directed edge into an undirected edge. 

49 



<!-- Start of picture text -->
Days of the Week<br>a<br>Dt . cetiie pnse rym 5 on nea + £3:<br>ila ae Pa<br>i a cae<br>PCA axis 1 PCA axis 2 PCA axis 3<br>Months of the Year<br>AE pone ey z y<br>3 eG er . Ss... er<br>5 Gime 5 arti © eae a Fey<br>5 SSeS § a ge re<br>2 Re 2 a a aD a 2<br>a nn aN A*<br>PCAaxis1<br>PCA axis2 "PCA axis 3<br>Years of the 20th Century<br>% “ 3a |<br>at \ “ 0.1900 50 1999<br>2 ie ancl iy re ie<br>S Sr r.  § ae EE, Se. —<br>PCA axis 1 PCA axis 2 PCA axis 3<br><!-- End of picture text -->

Figure A.5: Projections of days of week, months of year, and years of the 20th century representations onto top four principal components, showing additional dimensions of the representations than Fig. 2.1. 

3. Prune edges with cosine similarity less than a threshold value _τ_ . We use _τ_ = 0 _._ 5. 

4. Return the connected components as clusters. 

We run this algorithm on the Mistral 7B layer 8 SAE (2<sup>16</sup> dictionary elements) and find roughly 2700 clusters containing between 2 and 1000 elements. We manually inspected roughly 2000 of these. From these, we re-discover circular representations of days of the week and months of the year, shown in Fig. A.6. However, we did not find other obviously interesting and clearly irreducible features. 

We also investigate the sensitivity of this method to _τ_ and _k_ by varying _τ_ and _k_ and showing the max Jaccard similarity between any of the resulting clusters and the days of the week cluster we show in Fig. A.6. We show the results in Fig. A.7, where we find that varying _k_ has minimal effect, while varying _τ_ shows 3 regimes: small _τ_ causes all features to 

50 

group in one cluster, so the days of the week cluster is not found; medium _τ_ causes the days of the week cluster to become identifiable; large _τ_ causes all features to be divided into their own clusters. 



<!-- Start of picture text -->
Days of the Week (Mistral 7B) Months of the Year (Mistral 7B)<br>© Monday * © Friday © January + Winter<br>© Tuesday © Saturday @ February © Spring<br>. Y ase . © Other S May Other<br>eS hi .<br>. August B $e.<br>” . September Y I,<br>2 og : % ee. DSom § DeenaOctober eeso Searclee:8 She2<br>S oo* eee S ey eae<br>a 6 . . Taean-4 wily z\ “%<br>“eee . 4 7<br>Be § . 4<br>J a8<br>PCA axis 2 PCA axis 2<br><!-- End of picture text -->

Figure A.6: Circular representations of days of the week and months of the year which we discover with our unsupervised SAE clustering method in Mistral 7B. Unlike similar features in GPT-2, we also find an additional “weekend” representation in between Saturday and Sunday representations (left) and additional representations of seasons among the months (right). For instance, “winter” tokens activate a region of the circle in between the representation of January and December. 

As future work, we think it would be exciting to develop better clustering techniques for SAE features. Our graph based clustering technique could likely be improved by more recent efficient and high-quality graph based clustering techniques, e.g. hierarchical agglomerate clustering with single-linkage (Lattanzi et al. 2020). Additionally, we believe we would see a large improvement by setting edge weights to be a combination of both the cosine and Jaccard similarity of the dictionary elements, e.g. max(cosine, Jaccard). 

## **A.7 Other discovered clusters** 

In Fig. A.8, we plot the top 11 ranked clusters by the product of a) the measured separability index and b) one minus the measured _ϵ_ -mixture index with _ϵ_ = 0 _._ 1 (this is just one of many possible ways to get an ordered ranking from a two-parameter score). We color by both the current token (which results in clear patterns for all tokens) and the next token (to see if we find belief states as found by Shai et al. 2024 in toy transformers). We note that weekdays are ranked 9 and so are shown in the plot. Additionally, the next token patterns of the ‘such’ cluster and the ‘B’ cluster do seem to display some clustering independently of the the current token pattern, which might lend the belief state hypothesis some support. 

51 



<!-- Start of picture text -->
Cluster Stability Across Parameters Mean Cluster Size Across Parameters<br>0.10 1.0<br>0.11<br>0.13<br>0.140.16 Cluster part of large super cluster 0.8 10 3<br>0.18<br>0.21<br>0.23<br>0.26 0.6<br>0.30 10 2<br>0.34<br>0.38<br>0.43 0.4<br>0.48 Cluster formed<br>0.50<br>0.55 10 1<br>0.62 0.2<br>0.70<br>0.78<br>0.89 Cluster broken apart<br>1.00 10 0<br>k=2 k=3 k=4 k=2 k=3 k=4<br>Top-k for Graph (k=2,3,4) Top-k for Graph (k=2,3,4)<br>Similarity Cutoff<br>Mean Cluster Size (log scale)<br>Max Jaccard Similarity With GT Cluster<br><!-- End of picture text -->

Figure A.7: Hyperparameter regimes where the days of the week cluster exists. The cluster exists in the regime between all features clumping together and all features being in their own cluster; this regime seems reasonably stable. 

## **A.8 Further Experiment Details** 

### **A.8.1 Assets Information** 

We use the following open source models for our experiments: Llama 3 8B (AI@Meta 2024) (custom Llama 3 license https://llama.meta.com/llama3/license/), Mistral 7B (A. Q. Jiang et al. 2023) (released under the Apache 2 License), and GPT-2 (Radford et al. 2019) (modified MIT license, see https://github.com/openai/gpt-2/blob/master/LICENSE). 

### **A.8.2 Machine Information** 

Intervention experiments were run on two V100 GPUs using less than 64 GB of CPU RAM; all experiments can be reproduced from our open source repository in less than a day with this configuration. We use the TransformerLens library (Nanda and Bloom 2022) for intervention experiments. _ϵ_ -mixture index measurements on toy datasets took about one minute each, on 8GB of CPU RAM. EVR experiments take seconds on 8GB of CPU RAM and are dominated by time taken to human-interpret the RGB plots. 

GPT-2 SAE clustering and plotting was run on a cluster of heterogeneous hardware. Spectral clustering and computing reconstructions + plotting was done on CPUs only. We made reconstruction plots for 500 clusters, with each taking less than 10 minutes. Mistral 7B SAE reconstruction plots were made on the same cluster. We made roughly 2000 reconstruction plots for Mistral 7B (and manually inspected each), with each taking less than 20 minutes to generate. Jobs were allocated 64GB of memory each. 

Mistral SAE training was run on a single V100 GPU. Initially caching activations from Mistral 7B on one billion tokens took approximately 60 hours. Training the SAEs on the saved activations took another 36 hours. 

52 

### **A.8.3 Error Bar Calculation** 

In Fig. 2.6 we report 96% error bars for all intervention methods. To compute these error bars, we loop over all intervention methods and all layers and compute a confidence interval for each (method, layer) pair across all prompts. Assuming normally distributed errors, we compute error bars with the following standard formula: 



where _µ_ is the sample mean, _z_ is the z score (slightly larger than 2 for 96% error bars), and _SE_ is the standard error (the standard deviation divided by the square root of the number of samples). We use standard Python functions to compute this value. 

53 

The reason that the `Months` error bars are smaller than the `Weekdays` error bars is because there are more `Months` prompts: there are 12 _∗_ 12 _∗_ 11 = 1584 intervention effect values, rather than 7 _∗_ 7 _∗_ 6 = 294 intervention effect values. 

## **A.9 More** `Weekdays` **and** `Months` **Plots and Details** 

#### **Basic Plots** 

We show the results of Mistral 7B and Llama 3 8B on all individual instances of `Weekdays` that at least one of the models get wrong in Table A.1 and present a similar table for `Months` in Table A.2. 

We show projections onto the top two PCA directions for both Mistral 7B and Llama 3 8B in Fig. A.9 on the hidden layers on top of the _α_ token, colored by _α_ . These are similar plots to Fig. 2.4, except they are on all layers. The circular structure in _α_ is visible on many—but not all—layers. Much of the linear structure visible is due to _β_ . 

#### **Intervening with the SAE Probe** 

We show the results of projecting Mistral `Weekdays` representations into the plane discovered by clustering SAE features in Fig. A.10. The result is clearly circular. 

#### **Basic Patching** 

In Fig. A.12 and Fig. A.13, we report MLP and attention head patching results for `Weekdays` and `Months` . We experiment on 20 pairs of problems with the same _α_ and different _β_ and 20 pairs of problems with the same _β_ and different _α_ , for a total of 40 pairs of problems. For each pair of problems, we patch the MLP/attention outputs from the "clean" to the "dirty" problem for each layer and token, and then complete the forward pass. Defining the logit difference as the logit of the clean _γ_ minus the logit of the dirty _γ_ , we record what percent of the difference between the original logit difference of the dirty problem and the logit difference of the clean problem is recovered upon intervening, and average across these 40 percentages for each layer and token. This gives us a score we call the _Average Intervention Effect_ . 



<!-- Start of picture text -->
Monday Wednesday Friday Sunday<br>Tuesday Thursday Saturday<br>0.4<br>0.2<br>0.0<br>0.2<br>0.4<br>0.4 0.2 0.0 0.2 0.4 0.6<br>Projection onto second SAE PCA component<br>Projection onto third SAE PCA component<br><!-- End of picture text -->

Figure A.10: Projections of Mistral 7B `Weekdays` task activations at layer 8 into the plane discovered by clustering layer 8 SAE features. 



<!-- Start of picture text -->
Fri<br>Early FriLate Fri Sat<br>Thu<br>Early ThuLate Thu Late SatEarly Sat<br>Late Wed<br>Wed Early Wed Late SunEarly Sun Sun<br>Late Tue<br>Tue Early Tue Late MonEarly Mon<br>Mon<br><!-- End of picture text -->

Figure A.11: Layer 30 Mistral 7B activations for [very early/very late] on [Monday/Tuesday/.../Sunday], plotted projected into the PCA plane for [Monday/Tuesday/.../Sunday]. 

54 

For simplicity of presentation, we clip all of the 

(few) negative intervention averages to 0 (prior work (Zhang and Nanda 2023) has also found negative-effect attention heads during patching experiments). 

#### **Circle Continuity** 

Finally, in Fig. A.11, we show another example of the continuity of the circular days of the week representation in Mistral 7B. 

## **A.10 Patching** 

In this section, we present results to support a claim that MLPs (and not attention blocks) are responsible for computing _γ_ . In Fig. A.16, we deconstruct states on top of the final token (before predicting _γ_ ) on Llama 3 8B `Months` (we show a similar plot for the states on the final token of Mistral 7B on `Weekdays` in the main text in Fig. A.15. These plots show that the value of _γ_ is computed on the final token around layers 20 to 25. To show that this computation of occurs in the MLPs, we must show that no attention head is copying _γ_ from a prior token or directly computing _γ_ . 

We first perform a patching experiment with the same setup Fig. A.13 and Fig. A.12 on individual attention heads on the final token. From the patching results we identify the top 10 attention heads by average intervention effect. For each attention head, we compute one EVR run with explanatory functions equal to one-hot functions of _α_ and _β_ (resulting in 14 functions **g** _i_ for `Weekdays` and 24 for `Months` ) and one with explanatory functions equal to one-hot functions of _α_ , _β_ , and _γ_ . We find that for all layers before 25, adding _γ_ to the explanatory functions adds almost no explanatory power. Since we established above that the model has already computed _γ_ at this point, we know that attention heads do not participate in computing _γ_ . 

To isolate the rough circuit for `Weekdays` and `Months` , we perform layer-wise activation patching on 40 random pairs of prompts. The results, displayed in Fig. A.13 show that the circuit to compute _γ_ consists of MLPs on top of the _α_ and _β_ tokens, a copy to the token before _γ_ , and further MLPs there (roughly similar to prior work studying arithmetic circuits (Stolfo, Belinkov, and Sachan 2023)). Moreover, fine-grained patching in Section A.11 shows that there are just a few responsible attention heads for the writes to the token before _γ_ . However, patching alone cannot tell use _how_ or _where γ_ is represented. For that, we need a new technique, which we expand on in the next section. 

## **A.11 Explanation via Regression (EVR)** 

So far, we have focused on examining and intervening on the representation for _α_ , which we present as a circle in the top PCA components on top of the _α_ token. In this section, we examine how the generated output, _γ_ , is represented. 

First, to isolate the rough circuit for `Weekdays` and `Months` , we perform layer-wise activation patching on 40 random pairs of prompts. The results, displayed in Fig. A.13 and Fig. A.12, show that the circuit to compute _γ_ consists of MLPs on top of the _α_ and _β_ 

55 

Table A.1: `Weekdays` finegrained results. Row ommited if both models get it correct. 

|_α_|_β_|Ground truth _γ_|Mistral top _γ_|Mistral correct?|Llama top _γ_|Llama correct?|
|---|---|---|---|---|---|---|
|1|1|Wednesday|Wednesday|Yes|Thursday|No|
|3|1|Friday|Friday|Yes|Tuesday|No|
|4|1|Saturday|Saturday|Yes|Thursday|No|
|3|2|Saturday|Saturday|Yes|Tuesday|No|
|4|2|Sunday|Sunday|Yes|Wednesday|No|
|5|2|Monday|Monday|Yes|Tuesday|No|
|2|3|Saturday|Friday|No|Saturday|Yes|
|3|3|Sunday|Sunday|Yes|Tuesday|No|
|4|3|Monday|Monday|Yes|Tuesday|No|
|0|4|Friday|Thursday|No|Friday|Yes|
|3|4|Monday|Monday|Yes|Tuesday|No|
|0|5|Saturday|Friday|No|Saturday|Yes|
|1|5|Sunday|Saturday|No|Wednesday|No|
|2|5|Monday|Sunday|No|Monday|Yes|
|4|5|Wednesday|Tuesday|No|Tuesday|No|
|6|5|Friday|Thursday|No|Thursday|No|
|1|6|Monday|Sunday|No|Thursday|No|
|2|6|Tuesday|Monday|No|Tuesday|Yes|
|3|6|Wednesday|Tuesday|No|Tuesday|No|
|4|6|Thursday|Thursday|Yes|Tuesday|No|
|5|6|Friday|Friday|Yes|Thursday|No|
|6|6|Saturday|Thursday|No|Thursday|No|
|0|7|Monday|Sunday|No|Tuesday|No|
|1|7|Tuesday|Sunday|No|Tuesday|Yes|
|2|7|Wednesday|Sunday|No|Wednesday|Yes|
|3|7|Thursday|Sunday|No|Thursday|Yes|
|4|7|Friday|Thursday|No|Tuesday|No|
|5|7|Saturday|Friday|No|Saturday|Yes|
|6|7|Sunday|Friday|No|Thursday|No|



56 

Table A.2: `Months` finegrained results. Row ommited if both models get it correct. 

|_α_|_β_|Ground truth _γ_|Mistral top _γ_|Mistral correct?|Llama top _γ_|Llama correct?|
|---|---|---|---|---|---|---|
|0|4|May|April<br>|No|May|Yes|
|6|4|November|October|No|November|Yes|
|0|6|July|June|No|July|Yes|
|0|7|August|July|No|August|Yes|
|1|7|September|October|No|September|Yes|
|3|7|November|October|No|November|Yes|
|5|7|January|December|No|January|Yes|
|6|7|February|January|No|February|Yes|
|7|7|March|February|No|March|Yes|
|9|7|May|April|No|May|Yes|
|4|9|February|February|Yes|January|No|
|2|10|January|December|No|January|Yes|
|8|10|July|June|No|July|Yes|
|1|11|January|December|No|January|Yes|
|2|11|February|December|No|February|Yes|
|3|11|March|February|No|March|Yes|
|7|11|July|June|No|July|Yes|
|8|11|August|July|No|August|Yes|
|9|11|September|August|No|September|Yes|
|0|12|January|December|No|January|Yes|



57 

tokens, a copy to the token before _γ_ , and further MLPs there (roughly similar to what Stolfo, Belinkov, and Sachan 2023 find in prior work studying arithmetic circuits). Thus, we know _where_ to look for a representation of _γ_ : in the second half of the layers on the token before _γ_ . However, patching alone cannot tell use _how γ_ is represented. 

Unlike _α_ , _γ_ has no obvious circular (or linear) pattern in the top PCA components on these layers. To determine the representation for _γ_ , we introduce a more powerful technique we call Explanation via Regression (EVR): given a set of token sequences with a corresponding set of hidden states _Xi,l_ , we choose a set of interpretable explanation functions of the input tokens _{_ **g** _j_ ( _t_ ) _}_ . The _r_<sup>2</sup> value of a linear regression from _{_ **g** _j_ ( _t_ ) _}_ to _Xi,l_ tells us how much of the variance in the activations the _{_ **g** _j_ ( _t_ ) _}_ explain, and conversely the residuals show the exact components of the representation we have yet to explain. 

### **A.11.1 Using EVR to uncover a circular representation for** **_γ_** 

We first use EVR to determine the representation for _γ_ by plotting the top two PCA components of the layer 25 Mistral 7B activations after subtracting the components that can be 



<!-- Start of picture text -->
3<br>Tue Wed<br>2<br>Mon 1 Thu<br>0<br>4 2 1 0 2 Fri 4<br>2<br>Sun Sat<br>3<br><!-- End of picture text -->

Figure A.14: Top two PCA components of residual errors after EVR with one-hot in _α_ and _β_ . Mistral 7B `Weekdays` , layer 25, final token. Colored by _γ_ . 

explained using a regression with one hot functions in _α_ and _β_ (i.e. **g** 1 = [ _α_ = 0] _,_ **g** 2 = [ _β_ = 1] _,_ **g** 3 = [ _α_ = 1] _, . . ._ ). The result, shown in Fig. A.14, is an incredibly clear circle in _γ_ , which suggests that the model’s generated representation of _γ_ lies along a circle. A simple PCA projection was not enough to find this result because the representation for _γ_ has interference from _α_ and _β_ , which the EVR removes. This suggests that the models may be generating _γ_ by using a trigonometry based algorithm like the “clock” Nanda, Chan, et al. 2023 or “pizza” Zhong et al. 2024 algorithm in late MLP layers. 

### **A.11.2 More Experiments with EVR** 

We now apply EVR to `Months` and `Weekdays` to break down _Xi,l_ completely into interpretable functions. We build a list of **g** _i iteratively_ and _greedily_ . At each iteration, we perform a linear regression with the current list **g** 1 _. . ._ **g** _k_ , visualize and interpret the residual prediction errors, and build a new function **g** _k_ +1 representing these errors to add to the list. Once most variance is explained, we can conclude that **g** 1 _, . . . ,_ **g** _k_ constitutes the entirety of what is represented in the hidden states. This information tells us what can and cannot be extracted via a linear probe, without having to train any probes. Furthermore, if we treat each **g** _i_ as a _feature_ (see Definition 1), then the linear regression coefficients tell us which directions in _Xi,l_ these features are represented in, connecting back to Hypothesis 2. 

Since _Xi,l_ consists of modular addition problems with two inputs _α_ and _β_ , we can visualize the errors as we iteratively construct **g** 1 _, . . . ,_ **g** _k_ by making a heatmap with _α_ and _β_ on the two axes, where the color shows what kind of error is made. More specifically, we take the top 3 PCA components of the error distribution and assign them to the colors red, green, and 

58 

blue. We call the resulting heatmap a **residual RGB plot** . Errors that depend primarily on _α_ , _β_ , or _γ_ show up as horizontal, vertical, or diagonal stripes on the residual RGB plot. 

In Fig. A.15, we perform EVR on the layer 17-29 hidden states of Mistral 7B on the `Weekdays` task; additional deconstructions are in Section A.11. We find that a circle in _γ_ develops and grows in explanatory power; we plot the layer 25 residuals after explaining with one hot functions in _α_ and _β_ (i.e. **g** 1 = [ _α_ = 0] _,_ **g** 2 = [ _β_ = 1] _,_ **g** 3 = [ _α_ = 1] _, . . ._ ) in Fig. A.14 to show this incredibly clear circle in _γ_ . This suggests that the models may be generating _γ_ by using a trigonometry based algorithm like the “clock” (Nanda, Chan, et al. 2023) or “pizza” (Zhong et al. 2024) algorithm in late MLP layers. 

59 

Table A.3: Highest intervention effect attention heads from fine-grained attention head patching, as well as EVR results with one hot _α, β_ and one hot _α, β, γ_ . 

||(a)|Mistral 7B|, `Weekday`|`s`.||(b)|Llama 3 8B|, `Weekda`|`ys`.|
|---|---|---|---|---|---|---|---|---|---|
|L|H|Average<br>Inter-<br>ven-<br>tion<br>Effect|EVR<br>_R_<sup>2</sup><br>One<br>Hot _α_,<br>_β_|EVR<br>_R_<sup>2</sup><br>One<br>Hot _α_,<br>_β_, _γ_|L|H|Average<br>Inter-<br>ven-<br>tion<br>Effect|EVR<br>_R_<sup>2</sup><br>One<br>Hot _α_,<br>_β_|EVR<br>_R_<sup>2</sup><br>One<br>Hot _α_,<br>_β_, _γ_|
|28|18|0.22|0.39|0.73|17|0|0.18|0.98|0.99|
|18|30|0.17|0.95|0.96|17|1|0.08|0.98|0.98|
|15|13|0.17|0.94|0.95|19|10|0.08|0.95|0.96|
|22|15|0.11|0.77|0.82|30|17|0.07|0.85|0.90|
|16|21|0.09|0.92|0.93|17|3|0.07|0.93|0.95|
|28|16|0.08|0.42|0.69|17|27|0.06|1.00|1.00|
|15|14|0.06|0.98|0.99|31|22|0.05|0.37|0.78|
|30|24|0.05|0.43|0.79|21|9|0.04|0.73|0.78|
|21|26|0.04|0.53|0.63|20|28|0.04|1.00|1.00|
|14|2|0.04|0.93|0.95|30|16|0.04|0.73|0.85|
||(c)|Mistral 7|B, `Months`|.||(d)|Llama 3 8|B, `Month`|`s`.|
|L|H|Average<br>Inter-|EVR<br>_R_<sup>2</sup>|EVR<br>_R_<sup>2</sup>|L|H|Average<br>Inter-|EVR<br>_R_<sup>2</sup>|EVR<br>_R_<sup>2</sup>|
|||ven-|One|One|||ven-|One|One|
|||tion|Hot _α_,|Hot _α_,|||tion|Hot _α_,|Hot _α_,|
|||Effect|_β_|_β_, _γ_|||Effect|_β_|_β_, _γ_|
|20|28|0.15|0.76|0.76|15|13|0.26|0.62|0.62|
|17|0|0.10|0.77|0.77|16|21|0.17|0.76|0.76|
|25|14|0.08|0.19|0.61|18|30|0.13|0.77|0.77|
|17|1|0.07|0.80|0.82|28|18|0.11|0.13|0.52|
|17|3|0.06|0.71|0.71|28|16|0.07|0.13|0.52|
|31|22|0.06|0.12|0.67|21|25|0.05|0.65|0.70|
|17|27|0.05|0.58|0.58|15|14|0.03|0.72|0.72|
|19|4|0.05|0.40|0.66|17|26|0.02|0.77|0.77|
|19|10|0.04|0.62|0.62|31|1|0.02|0.11|0.57|
|30|26|0.04|0.51|0.62|21|24|0.02|0.30|0.45|



60 

61 

Layer 0 Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8 Layer 9 Layer 10 Layer 11 Layer 12 Layer 13 Layer 14 Layer 15 Layer 16 Layer 17 Layer 18 Layer 19 Layer 20 Layer 21 Layer 22 Layer 23 Layer 24 Layer 25 Layer 26 Layer 27 Layer 28 Layer 29 Layer 30 Layer 31 Monday Tuesday Wednesday Thursday Friday Saturday Sunday 

Layer 0 Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8 Layer 9 Layer 10 Layer 11 Layer 12 Layer 13 Layer 14 Layer 15 Layer 16 Layer 17 Layer 18 Layer 19 Layer 20 Layer 21 Layer 22 Layer 23 Layer 24 Layer 25 Layer 26 Layer 27 Layer 28 Layer 29 Layer 30 Layer 31 Monday Tuesday Wednesday Thursday Friday Saturday Sunday 

##### (a) Mistral 7B, `Weekdays` 

Layer 0 Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8 Layer 9 Layer 10 Layer 11 Layer 12 Layer 13 Layer 14 Layer 15 Layer 16 Layer 17 Layer 18 Layer 19 Layer 20 Layer 21 Layer 22 Layer 23 Layer 24 Layer 25 Layer 26 Layer 27 Layer 28 Layer 29 Layer 30 Layer 31 January March May July September November February April June August October December 

##### (b) Llama 3 8B, `Weekdays` 

Layer 0 Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6 Layer 7 Layer 8 Layer 9 Layer 10 Layer 11 Layer 12 Layer 13 Layer 14 Layer 15 Layer 16 Layer 17 Layer 18 Layer 19 Layer 20 Layer 21 Layer 22 Layer 23 Layer 24 Layer 25 Layer 26 Layer 27 Layer 28 Layer 29 Layer 30 Layer 31 January March May July September November February April June August October December 

##### (c) Mistral 7B, `Months` 

##### (d) Llama 3 8B, `Months` 

Figure A.9: Projections onto the top two PCA dimensions of model hidden states on the _α_ token show that circular representations of _α_ are present in various layers. 

62 



<!-- Start of picture text -->
*Monday 0.3 *Monday 0.4<br>is is 0.3<br>0.2<br>two two<br>0.2<br>days 0.1 days<br>0.1<br>from from<br>ee 0.0 0.0<br>0 10 20 30 0 10 20 30<br>Layer Layer<br>(a) Mistral 7B MLP Patching (b) Mistral 7B attention patching<br>*Monday *Monday<br>0.4<br>0.2<br>is is<br>two two<br>0.2 0.1<br>days days<br>from from<br>ee 0.0 ee 0.0<br>0 10 20 30 0 10 20 30<br>Layer Layer<br>(c) Llama 3 8B MLP patching (d) Llama 3 8B attention patching<br><!-- End of picture text -->

Figure A.12: Attention and MLP patching results on `Weekdays` . Results are averaged over 20 different runs with fixed _α_ and varying _β_ and 20 different runs with fixed _β_ and varying _α_ . 



<!-- Start of picture text -->
*Two 0.3 *Two 0.4<br>months months<br>0.3<br>0.2<br>from from<br>0.2<br>*January 0.1 *January<br>0.1<br>is is<br>a 0.0 0.0<br>0 10 20 30 0 10 20 30<br>Layer Layer<br>(a) Mistral 7B MLP Patching (b) Mistral 7B attention patching<br>*Two 0.4 *Two<br>0.3<br>months 0.3 months<br>0.2<br>from from<br>0.2<br>*January *January 0.1<br>0.1<br>is is<br>eee! 0.0 0.0<br>0 10 20 30 0 10 20 30<br>Layer Layer<br>(c) Llama 3 8B MLP patching (d) Llama 3 8B attention patching<br><!-- End of picture text -->

Figure A.13: Attention and MLP patching results on `Months` . Results are averaged over 20 different runs with fixed _α_ and varying _β_ and 20 different runs with fixed _β_ and varying _α_ . 

63 



<!-- Start of picture text -->
layer 17 layer 18 layer 19 layer 20 layer 21 layer 22 layer 23 layer 24 layer 25 layer 26 layer 27 layer 28 layer 29<br>original original original original original original original original original original original original original<br>r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0%<br>one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β one hot α , β<br>r2: 94.0% r2: 95.6% r2: 96.3% r2: 94.5% r2: 91.4% r2: 89.0% r2: 87.3% r2: 85.4% r2: 81.9% r2: 78.4% r2: 74.8% r2: 72.3% r2: 70.4%<br>α for tmr α for tmr α for tmr circle γ circle γ circle γ circle γ circle γ circle γ circle γ circle γ circle γ<br>r2: 98.1% r2: 98.2% r2: 96.8% r2: 94.4% r2: 93.6% r2: 93.4% r2: 92.2% r2: 91.2% r2: 88.7% r2: 87.7% r2: 86.4% r2: 84.6%<br>circle α − β circle γ circle γ circle α − β circle α − β circle α − β circle α − β circle α − β circle α − β circle α − β<br>r2: 98.6% r2: 98.7% r2: 98.1% r2: 96.3% r2: 95.6% r2: 95.5% r2: 94.4% r2: 93.7% r2: 91.0% r2: 89.9%<br>circle α − β circle α − β α for tmr α for tmr α for tmr<br>r2: 99.2% r2: 99.0% r2: 98.0% r2: 98.1% r2: 97.6%<br><!-- End of picture text -->

Figure A.15: EVR residual RGB plots on Mistral hidden states on the `Weekdays` final token, layers 17 to 29. From top to bottom, we show each residual RGB plot after adding the function(s) **g** _i_ labelled just underneath, as well as the resulting _r_<sup>2</sup> value. We write “tmr” meaning “tomorrow” for _β_ = 1. We also write “circle for _x_ ” meaning the inclusion of two functions **g** _i_ ( _x_ ) = _{_ cos _,_ sin _}_ (2 _πx/_ 7). 



<!-- Start of picture text -->
layer 13 layer 14 layer 15 layer 16 layer 17 layer 18 layer 19 layer 20 layer 21 layer 22 layer 23 layer 24 layer 25 layer 26 layer 27 layer 28 layer 29<br>original original original original original original original original original original original original original original original original original<br>r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0% r2: 0.0%<br>Seer<br>one hot β one hot β one hot β one hot β one hot β one hot β one hot β one hot β one hot β circle γ circle γ circle γ circle γ circle γ γ parity γ parity circle γ<br>r2: 69.7% r2: 72.8% r2: 69.9% r2: 89.0% r2: 76.6% r2: 81.8% r2: 54.1% r2: 47.8% On r2: 36.8% r2: 23.6% r2: 24.3% r2: 24.7% aan r2: 25.5% r2: 25.5% r2: 12.0% r2: 12.2% r2: 23.7%<br>one hot α one hot α one hot α one hot α one hot α one hot α γ parity circle γ circle 2 γ one hot β one hot β one hot β one hot β one hot β circle γ circle γ one hot β<br>r2: 95.7% r2: 96.0% r2: 96.8% r2: 98.7% r2: 99.0% r2: 96.8% r2: 60.2% r2: 60.9% r2: 47.8% r2: 51.6% r2: 48.3% r2: 48.2% r2: 48.2% r2: 48.1% r2: 35.4% r2: 34.2% r2: 43.6%<br>α + 1 = β α + 1 = β α + 1 = β α + 1 = β circle 2 γ γ parity circle 2 γ γ parity circle γ γ parity γ parity γ parity γ parity γ parity one hot β one hot β γ parity<br>r2: 96.4% r2: 96.5% r2: 97.2% r2: 98.9% r2: 99.2% r2: 98.0% r2: 64.7% r2: 68.0% r2: 62.5% r2: 59.5% r2: 57.9% r2: 56.9% r2: 55.6% r2: 55.1% r2: 56.8% r2: 56.5% r2: 54.4%<br>circle γ circle γ circle 2 γ γ parity circle 2 γ circle 2 γ circle 2 γ circle 2 γ circle 2 γ circle 2 γ circle 2 γ circle 2 γ<br>r2: 98.8% r2: 73.4% r2: 73.7% r2: 68.7% r2: 68.6% r2: 66.2% r2: 64.4% r2: 63.6% r2: 64.0% r2: 65.7% r2: 65.6% r2: 65.1%<br>one hot α one hot α one hot α one hot α one hot α one hot α one hot α one hot α one hot α one hot α one hot α<br>r2: 92.0% r2: 89.9% r2: 83.8% r2: 81.4% r2: 80.1% r2: 77.5% r2: 76.2% r2: 75.6% r2: 76.9% r2: 76.6% r2: 74.7%<br>one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β one hot α + β<br>r2: 95.2% r2: 94.3% r2: 91.4% r2: 90.6% r2: 90.9% r2: 90.8% r2: 90.7% r2: 90.6% r2: 90.9% r2: 91.1% r2: 92.4%<br><!-- End of picture text -->

Figure A.16: Iterative deconstruction of hidden state representations on the final token on Llama 3 8B, `Months` . 

64 

# **Chapter 3** 

# **Are Sparse Autoencoders Useful? A Case Study in Sparse Probing** 

## **3.1 Introduction** 

Dictionary learning is a popular method for interpreting LLM activations (Arora et al. 2018; Yun et al. 2021); most notably, Bricken, Templeton, et al. 2023; Cunningham et al. 2023b demonstrated the promise of sparse autoencoders (SAEs) and sparked considerable follow-up work. This includes papers focused on improving the downstream cross entropy loss of SAE reconstructions (Gao, Tour, et al. 2024a; Braun et al. 2024; Rajamanoharan, Conmy, et al. 2024b; Rajamanoharan, Lieberum, et al. 2024a), improving SAE training efficiency (Mudide et al. 2024; Gao, Tour, et al. 2024a), finding weaknesses in SAEs and proposing solutions (Chanin et al. 2024b; Ben Bussmann, Peter Leask, and Nanda 2024), using SAEs to understand model representation structure (Engels, Eric J. Michaud, et al. 2024; Engels, Riggs, and Tegmark 2024; Yuxiao Li et al. 2024), training large suites of SAEs (Lieberum et al. 2024a; Z. He, Shu, Ge, L. Chen, J. Wang, Zhou, F. Liu, Q. Guo, X. Huang, Zuxuan Wu, et al. 2024b), and developing benchmarks for SAEs (J. Huang et al. 2024; Karvonen, Wright, et al. n.d.; Karvonen, Rager, et al. 2024a). 

Unfortunately, we lack a “ground truth” to know whether SAEs truly extract the interpretable concepts used by language models. Prior work has largely avoided this limitation by evaluating SAEs with proxy metrics like reconstruction loss (Rajamanoharan, Conmy, et al. 2024b; Rajamanoharan, Lieberum, et al. 2024a; Gao, Tour, et al. 2024a; Olmo et al. 2024; Braun et al. 2024). These metrics are tractable to optimize for, but do not necessarily align with mechanistic interpretability’s (MI) goal of better understanding neural networks (see Bereska and Gavves 2024 and Sharkey et al. 2025 for surveys of MI). We argue that if SAEs truly advance MI’s goal, they should improve performance on a real, hard-to-fake model control or explainability task. 

However, despite the extensive body of work on SAEs, there are relatively few cases where SAEs have been shown to help on such a task: Marks, Rager, et al. 2024 show that SAE feature circuits can help identify and remove bias from classifiers; L. R. Smith and Brinkmann 2024 show that SAEs can identify misaligned features learned by a preference model; and Karvonen, Pai, et al. 2024 show that SAE feature ablations are better at preventing regex 

65 



<!-- Start of picture text -->
N/A<br>Standard  Data  Class  Label<br>OOD Interp.<br>Conditions Scarcity Imbalance Noise<br>Do SAE<br>Probes  ~<br>Help?<br><!-- End of picture text -->

Figure 3.1: SAE probes underperform the baseline of logistic regression in each regime when taking the mean across datasets. Additionally, we find that baseline methods can provide many of the interpretability insights of SAE probes. 

output than baselines in a setting with scarce supervised data. While these results are promising, they all study a single example in detail, and consider comparative baselines with varying levels of rigor. 

At the same time, there are also surprisingly few negative results finding that SAEs do _not_ help on downstream tasks; in fact, we are only aware of Chaudhary and Geiger 2024, which finds that SAE latents are worse than neurons at disentangling geographic data, and Farrell, Lau, and Conmy 2024, which finds that pinning related SAE latents is less effective than baselines for unlearning bioweapon knowledge. Thus, it is not clear whether SAEs are just one result away from being differentially useful, or if MI should seek fundamentally different methods. We defer a thorough examination of related work to Section B.1. 

In this work, we attempt to fairly evaluate the utility of SAEs by examining their competitive advantage on a concrete task: training probes from language model activations to targets (Alain 2016). Probing has two important qualities: 

1. **Probing is practically useful:** Probing has been used to investigate LLM representations (Gurnee and Tegmark 2023; Nanda, A. Lee, and Wattenberg 2023; Heinzerling and Inui 2024), detect safety relevant quantities (Zou et al. 2023), remove knowledge from models (Elazar et al. 2021), and catch synthetic sleeper agents (MacDiarmid et al. 2024). 

2. **SAEs might reasonably improve probing:** Theoretically, SAE latents are a more interpretable basis of model activations, and we hypothesize that this inductive bias will help train probes in difficult regimes. Recent work has also found some positive results for the utility of SAE probes (Bricken, Marcus, Mishra-Sharma, et al. 2024a; Gallifant et al. 2025). 

66 



<!-- Start of picture text -->
SAE Probe<br>i<br>            Select top Train logistic regression<br>k latents by avg. probe, L1 regularization<br>latent diff across<br>classes<br>KNN MLP<br>SAE Encoder<br>SAE<br>LogReg<br>Probe<br>PCA +<br>XGBoost<br>LogReg<br>Class 0  Activation Vectors Class 1  Activation Vectors<br><!-- End of picture text -->

Figure 3.2: **Left:** An illustration of our SAE probing method. We pass in training activation vectors from each class and train an _L_ 1 regularized logistic regression probe on the latents that differ the most between classes. **Right:** We ensure robustness of our results with the "quiver of arrows" approach (see Section 3.2.4): we add SAE regression into a set of methods, and see if the test accuracy of the best method (chosen by validation accuracy) increases. 

Thus, we curate 113 linear probing datasets from a variety of settings and train linear probes on corresponding SAE latent activations (see Fig. 3.2). We compare to a suite of baseline methods across 4 difficult probing regimes: 1) data scarcity, 2) class imbalance, 3) label noise, and 4) co-variate shift. Unfortunately, we find that **SAE probes fail to offer a consistent overall advantage when added to a simulated practitioner’s toolkit** . 

Further, we explore areas that Bricken, Marcus, Mishra-Sharma, et al. 2024a suggest SAEs may be valuable for, including detecting attributes distributed over multiple tokens and identifying dataset issues. Although SAEs initially seemed promising in these settings, we were able to achieve the same results with improved baselines. Our results underscore the necessity for MI works to rigorously design baselines when evaluating the utility of interpretability techniques. 

## **3.2 Methodology** 

We apply probes to the hidden states of two language models, Gemma-2-9B (Riviere et al. 2024) and Llama-3.1-8B (Grattafiori et al. 2024). Our main paper results use Gemma-2-9B. We replicate core results on Llama-3.1-8B in Section B.6. We use JumpReLU (Rajamanoharan, Lieberum, et al. 2024a) sparse autoencoders (SAEs) from Gemma Scope (Lieberum et al. 2024a) for Gemma-2-9B and TopK (Gao, Tour, et al. 2024a) SAEs from Llama Scope (Z. He, Shu, Ge, L. Chen, J. Wang, Zhou, F. Liu, Q. Guo, X. Huang, Zuxuan Wu, et al. 2024b) for Llama-3.1-8B. See Section B.1 for further background on SAEs. 

67 

Table 3.1: Example probing tasks. Only tasks with short prompts shown for conciseness. 

|**Dataset Name**|**Prompt**|**Target**|
|---|---|---|
|26_headline_frontpage|Sebelius’s Slow-Motion Resignation From the Cab-<br>inet.<br>MI5 References Emerge in Phone HackingLawsuit.|1<br>0|
|36_sciq_tf|Q: Binary fission is an example of which type of<br>production? A: asexual<br>Q: What occurs when light interacts with our at-<br>mosphere? A: prism effect|1<br>0|
|114_nyc_Manhattan|INWOOD BASKETBALL COURTS<br>PS 18 JOHN G WHITTIER|1<br>0|
|149_twt_happiness|All moved in to our new apartment. So exciting<br>is noow calmmm eating polvoron .. yuumm|1<br>0|
|155_athlete_basketball|Jimmy Butler<br>Steve Balboni|1<br>0|



### **3.2.1 Classification Datasets** 

We collect a diverse set of 113 binary classification datasets listed in Table B.1 (Section B.2), including datasets that we expect to be challenging for probes. For example, 26_headline_isfrontpage requires probes to identify front-page headlines and 136_glue_mnli _entailment requires probes to identify logically entailment. For other example datasets, refer to Table 3.1. All datasets are titled in the form ID_description. We originally collected datasets for other purposes and discarded some of them; thus, while we have 113 datasets, ID ranges from [5 _,_ 163]. 

All datasets contain prompts and targets. Probes are tasked with predicting the target from the model’s hidden activations when run on a prompt. We focus on binary classification, since SAEs are mostly thought of as representing binarized latents (see Bricken, Templeton, et al. 2023). Thus, target is either 0 or 1. The prompts in our datasets range in length from 5 tokens to a (left-truncated) maximum of 1024 tokens. 

### **3.2.2 Probing Strategy** 

To train a probe on a given dataset, we first run the model on all prompts from that dataset to generate model activations at each layer _l_ on the last token, _X−_<sup>_l_</sup> 1<sup>.</sup><sup>_X_</sup> _−_<sup>_l_</sup> 1<sup>isofshape</sup> (len(prompts) _,_ model_dim). We probe the last token to ensure the target information was present in the preceding context. For activation probes, we then train a probe _p_ to map from _X−_<sup>_l_</sup> 1<sup>_�→t_,where</sup><sup>_t_arethetargetsinourdataset.</sup> 

Our SAE probe training technique is summarized in Fig. 3.2. We first pass the training dataset _X−_<sup>_l_</sup> 1<sup>through the SAE encoder,resulting in a batch of vectors in the SAE latent space</sup> _Z_ = SAE( _X−_<sup>_l_</sup> 1<sup>)ofshape(len(prompts)</sup><sup>_, W_),where</sup><sup>_W_isthewidthoftheSAE.Wedonot</sup> train probes directly on the SAE latent space because we hypothesize that a small number of SAE latents encodes the desired concept. Instead, we create a basis of latents with the highest average absolute difference between the set of training prompts _T_ 1 with target = 1 

68 

and the set of training prompts _T_ 0 with target = 0. More formally, if SAE( _X−_<sup>_l_</sup> 1<sup>)</sup><sup>_∈_R</sup><sup>_W_, where</sup> _W ≫ d_ model, we choose _k ≪ W_ and find indices _I_ as follows: 



We then train a probe _p_ SAE to map from _Z_ [: _, I_ ] _�→ t_ . 

Note that previous work in SAE probing has aggregated SAE latents across tokens to provide a richer input space for SAE probes (Karvonen, Rager, et al. 2024a; Bricken, Marcus, Mishra-Sharma, et al. 2024b). However, most probing studies operate on the final token, which we choose to emulate. In Section 3.5, we present results considering probes on multiple tokens. 

Throughout this study, we use the area under the ROC curve (AUC) to evaluate the quality of a probe. AUC is the likelihood that the classifier assigns a higher probability of target = 1 to a target = 1 example than to a target = 0 example. Using AUC allows us to comprehensively assess probe performance agnostic to classification thresholds. For additional details on AUC, see Section B.3.1. 

Often, a probe _p_ has hyperparameters _hp_ we would like to optimize. We select _hp_ that has the maximal validation AUC using the cross-validation strategy described in Table B.2. We then test _p_ with optimal _hp_ on a held out test set to calculate AUC<sup>test</sup> _p_ . All datasets have at least 100 testing examples, with most having more (the average test set size is 1945). 

### **3.2.3 Probing Methods** 

We use the following 5 probing methods, with hyperparameters in parenthesis: 

1. Logistic Regression (L1 regularization for SAE probes, L2 otherwise). 

2. PCA Regression (# of PCA components to reduce _X−_<sup>_l_</sup> 1<sup>tobeforeusingunregularized</sup> logistic regression). 

3. K-Nearest Neighbors (KNN) (# of nearest neighbors). 

4. XGBoost (T. Chen and Guestrin 2016) (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, and min_child_weight). 

5. Multilayer Perceptron (MLP) (network depth, hidden state width, learning rate, and weight decay). 

We evaluate 10 hyperparameter values _hp_ for each probing method _p_ . For the first three probing methods, we use grid search, while for MLPs and XGBoost, we randomly sample from a hyperparameter grid to manage the larger search space. For additional details on hyperparameter ranges, see Section B.3.3. We use all five probing methods as baselines, but for SAE probes we only use logistic regression. 

### **3.2.4 Experimental Setup: Quiver of Arrows** 

To evaluate whether SAE probes provide an advantage over baselines, we use an evaluation metric we call the “Quiver of Arrows”: we ask whether adding SAE probes (a new “arrow”) to the set of existing probing methods available to a practitioner (the “quiver”) increases performance compared to a practitioner _without_ access to SAE probes. In other words, over a 

69 



<!-- Start of picture text -->
Layer 20, Width 16k<br>400<br>0.9<br>300<br>200<br>0.8<br>100<br>2 1 2 3 2 5 2 7 2 9<br>Number of Features Used in SAE Probe (k)<br>L0<br>Average Test AUC<br><!-- End of picture text -->

Figure 3.3: For a given width, using higher L0 and constructing probes with a larger basis of latents ( _k_ ) is more performant. 



<!-- Start of picture text -->
1.0<br>Mean : -0.003 ± 0.002<br>No Change<br>0.8<br>0.6<br>0.6 0.7 0.8 0.9 1.0<br>Test AUC (without SAE)<br>Test AUC (with SAE)<br><!-- End of picture text -->

Figure 3.4: In standard conditions, when SAE probes are added to the quiver, we find a slight decrease in performance. 

collection of methods we choose the best method by validation AUC and then report the test AUC of that method. We can then compare the marginal improvement of adding SAE probes to a practitioner’s toolkit by comparing the Quiver of Arrows AUC of baeline methods with and without SAE probes. We describe the Quiver of Arrows more formally in Section B.3.4. 

The quiver of arrows approach is designed as a robustness check to fairly evaluate the benefit of adding SAE probes to a practioner’s toolkit. If we instead constructed probes from a large set of SAEs and chose the best one based on test AUC, we could be tricked into thinking SAEs outperform baselines because we accessed the held-out test set. With the quiver of arrows approach, we emulate the information a real practioner has access to, allowing us to make a robust recommendation on the usefulness of SAE probes. 

## **3.3 Comparing Probing Techniques in Different Regimes** 

### **3.3.1 Standard Conditions** 

We initially assess probes under standard conditions, characterized by sufficient data for probe convergence and balanced classes. We find that baseline methods perform the best on layer 20 (see Fig. B.1a, Section B.4.1), so we run experiments with this layer. We train baselines on 1024 data points (or the maximum number of points in the dataset). 

We first conduct a preliminary investigation to cut down the large space of Gemma Scope SAEs. We train probes for all SAEs using _k_ latents (for logspaced values of _k_ ) on all datasets using 1024 training examples. We calculate the average test AUC for each SAE probe across all datasets. We find that SAE width is relatively unimportant to probe success, while larger L0s lead to more performant probes (Section B.4.2). Additionally, as we might expect, using a larger value of _k_ leads to better probe performance, as shown in Fig. 3.3. 

Thus, throughout the paper we train probes using the largest L0 for SAEs width = 16k _,_ width = 131k, and width = 1M. We use _k_ = 16 to construct easily interpretable probes that potentially overfit less and use _k_ = 128 for performance. 

Using this set of probes, we consider our quiver of arrows approach in standard conditions. SAEs are chosen as the “arrow” for 14/113 tasks, however, we see a slight _decrease_ in performance when they are added to the quiver in Fig. 3.4. 

It is unsurprising that SAE probes underperform baselines in standard conditions, as 

70 

their inductive bias is marginal given a large dataset. Thus, we now investigate more difficult settings to test if the SAE inductive bias translates into a competitive probing advantage. 



<!-- Start of picture text -->
Data Scarcity Class Imbalance Label Noise<br>Figure 3.5: For three of the datasets in Table 3.1, we visualize the performance when SAE<br>probes are in the quiver (dashed) versus when they are not (solid) for the regimes of data<br>scarcity (left), class imbalance (middle), and label noise (right). In all three regimes, we see<br>that on average (bottom row), SAEs do not help.<br>0.05<br>0.00<br>0.05<br>Number of Training Examples Ratio of Positive Class Fraction Corrupted<br>2 3 45 7 101420283853741031431982753825317371024 0.050.10.150.20.250.30.350.40.450.50.550.60.650.70.750.80.850.90.95 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5<br>SAE  (AUC)<br><!-- End of picture text -->

Figure 3.6: For the regimes of data scarcity, class imbalance, and label noise, we see no average improvement across datasets when adding SAEs to the quiver. Shading represents 95% confidence intervals. 

### **3.3.2 Data Scarcity, Class Imbalance, and Label Noise** 

In this section, we consider more difficult probing regimes: limiting the training data (Data Scarcity), changing the relative frequency of target = 1 examples (Class Imbalance), and randomly flipping a fraction of the targets (Label Noise). All of these settings are realistic workflows - for example, a researcher observing a rare model phenomena would like a probe that generalizes with few total examples or few examples of the desired class, while a researcher working with collected user data wants a generalizable probe even if some fraction of users respond randomly. 

Each regime is characterized by a parameter which we vary to compare SAEs and baselines. 

71 

- Data Scarcity - We use 20 values of _n_ logspaced in [2 _,_ 1024], where _n_ is the number of training examples. We use a standard test set. 

- Class Imbalance - We use 19 values of ratio linearly spaced between [0 _._ 05 _,_ 0 _._ 95], where ratio =<sup>_<u>n</u>_</sup> _n_<sup><u>1</u>and</sup><sup>_n_1isthenumberoftarget=1trainingexamples.Thetestsetusesthe</sup> same ratio. 

- Label Noise - We use 11 values of fraction linearly spaced between [0 _,_ 0 _._ 5], where fraction = _n_ corru _n_ <u>pted</u> . We use a standard test set (uncorrupted). 

We evaluate the first two regimes with the quiver of arrows method. However, the quiver of arrows method relies on the validation data being representative of the test data. This assumption fails for the label noise setting since the validation data is corrupted. Thus, a hypothetical practitioner would likely deploy their most performant method from other settings instead of allowing corrupted validation AUC to arbitrarily choose a method. To emulate this, we compare logistic regression and the width = 16k, _k_ = 128 SAE probe head-to-head in the setting of label noise. 

In Fig. 3.5, we visualize the SAE versus non-SAE quivers for a sample of datasets listed in Table 3.1 across the three regimes. Additionally, we visualize the average performance difference between the SAE and non-SAE quivers for each regime across all datasets in Fig. 3.6. At all parameter values in each regime, SAEs show no meaningful improvement over baselines. This is not because SAEs are not chosen from the quiver; in Section B.5 we see that SAEs are chosen for up to 40 datasets in each regime. SAEs simply underperform baselines in these settings. See Section B.5 for results for individual datasets. 

### **3.3.3 Covariate Shift** 

We now investigate if SAE probes are more resilient to distribution shifts in prompts. To model this covariate shift, we create or use 8 out-of-distribution (OOD) datasets: two preexisting GLUE-X datasets which are designed as “extreme” versions of tasks 87 and 90 to test grammaticality and logical entailment respectively; three datasets (tasks 66, 67, and 73) which we alter the language of; and three datasets (tasks 5, 6, and 7) where we use syntactical alterations to names or use cartoon characters instead of historical figures. We train probes on these datasets in standard settings and evaluate on 300 covariate shifted test examples. Like the label noise setting, our validation data is unfaithful to our test data. Thus, we compare logistic regression to a single SAE probe. We construct SAE probes from the width = 131k _,_ L0 = 114 SAE, as latent descriptions are available for this SAE on Neuronpedia (J. Lin 2023). Our results (Fig. 3.7) show that baselines outperform SAE probes when generalizing to covariate shifted data. 

## **3.4 Interpretability** 

While we find that SAE probes are not helpful in traditional probing settings, one intrinsic advantage of SAE probes is that their input basis is interpretable. To leverage this, we use the technique of automatic interpretation, or autointerp, to create natural language descriptions of top latents for each dataset (see Bills et al. 2023). We investigate three applications of labeled latents: 

72 



<!-- Start of picture text -->
1.0<br>0.8<br>0.6<br>0.4<br>0.2 Baseline AUC<br>SAE AUC<br>0.0<br>Dataset<br>66_living-room 87_glue_colaGLUE-X OOD90_glue_qnliGLUE-X OODAverage<br>Fictional Characters5_hist_fig_ismale6_hist_fig_isamericanRandom Letter Inserted7_hist_fig_ispoliticianReversed Name OrderTranslated to FrenchTranslated to Spanish67_social-securityTranslated to German73_control-group Mean across datasets<br>OOD Test AUC<br><!-- End of picture text -->

Figure 3.7: SAE probes often generalize worse than baseline logistic regression with covariate shift. 

1. **Probe Interpretability** : In Section 3.4.1, we investigate why SAE probes fail to perform well by pruning latents that o1 (OpenAI 2024b) ranks as spurious. 

2. **Latent Interpretability** : In Section 3.4.2, we generate autointerp explanations of each dataset’s top latent to find spurious latents and latents that fit well to our probes in a way not explained by the autointerp label. 

3. **Detecting Dataset Quality Issues** : In Section 3.4.3, we invesigate spurious dataset features and label errors using insights from the top latent descriptions and firing patterns. 

### **3.4.1 Probe Intepretability: Pruning and Latent Generalization** 

We first use autointerp to investigate why SAE probes fail to generalize to covariate shift. We have two initial hypotheses: 1) the latents SAE probes rely on leverage spurious correlations in the training data and 2) the latents are not spurious, but they are not robust to the distribution shift. 

First, we investigate hypothesis 1. Specifically, we attempt to improve SAE probes OOD performance by pruning latents deemed spurious by their autointerp description. We focus on three OOD tasks that SAE probes perform poorly on: 7_hist_fit_ispolitician, 66_living-room, and 90_glue_qnli (Fig. 3.7). We use the width = 131k _,_ L0 = 114 SAE. 

For each task, we take the _k_ = 8 top latents by mean difference and use Claude-3.5-Sonnet (Anthropic 2024a) to generate latent descriptions with autointerp. We then use OpenAI’s o1 model to rank the relevance of each latent’s description to the task. We also prompt o1 to downrank spurious latents given the OOD transformation. An example of this procedure for 

73 



<!-- Start of picture text -->
Latent 122774 on "Living Room" in Different Languages<br>40<br>20<br>0<br>Language<br>EnglishSpanishFrenchGermanItalianPortugueseRussianChineseJapaneseKoreanArabicHindiBengaliDutchTurkish<br>Latent Value<br><!-- End of picture text -->

Figure 3.8: Latent 122774 is most active on the English translation of “living room,” and does not fire on French. 

the task 66_living-room, which identifies if the phrase “living room” is in an English sentence, is shown in Table B.3, Section B.7.1. For this task, o1 ranked latent 12274, which identifies “mentions of living rooms” first, while ranking latent 51330, which identifies “objects and materials related to scientific experiments,” last. 

We then construct a probe with the top _k_ latents using o1’s relevance rank for _k ∈_ [1 _,_ 8]. If there are spurious latents in the _k_ = 8 probe, we expect a probe with _k <_ 8 to have better OOD generalization. We visualize each task’s performance by _k_ in Fig. B.8, Section B.7.1. For two of the datasets, 66_living-room and 90_glue_qnli, we see that pruning works, with a 0.024 and 0.052 increase in AUC between the _k_ = 8 and _k_ = 1 probe for each dataset. However, for 7_hist_fig_ispolitician, OOD test AUC _increases_ by 0.077 after pruning. This indicates that the task 7_hist_fig_ispolitician requires _k_ = 8 latents to represent. 

Our preliminary results show that pruning helps SAE probes generalize. However, the drop between in-distribution (ID) and OOD performance is much larger than the modest improvement from pruning. This indicates that spurious correlations are not the primary reason SAE probes fail to generalize. Thus, we consider the second hypothesis - that the underlying SAE latents do not generalize well OOD. 

On the task 66_living-room, latent 122774 has an ID test AUC of 0.99 while only having an OOD test AUC of 0.64. Since the OOD transformation involves translating the prompts to French, we hypothesize that this latent is active on the phrase “living room” in English but not other languages. We use GPT-4o to translate “living room” to 15 languages, and in Fig. 3.8 we indeed observe that latent 122774 is more active on the English translation than all other languages, and it is not active on the French translation at all. Thus, latent 122774 does not generalize OOD, which explains why the SAE probe also failed to generalize. 

74 

### **3.4.2 Latent Interpretability** 

We next generate autointerp descriptions for the most determinative latent for each dataset, allowing us to identify interesting categories of latents. Specifically, we consider each of the top 128 latents for each dataset for the width = 131k _,_ L0 = 114 SAE, and evaluate each latent’s _k_ = 1 SAE probe in standard settings. We then generate an autointerp explanation with GPT-4o for the best latent. 

Table B.3 (Section B.7.2) contains a breakdown of a selection of interesting top latents that we find. We see that some latents’ descriptions fit their tasks, like latent 81210 for 5_hist_fig_ismale, which activates on “references to female individuals.” Another interesting category is incorrect autointerp labels (e.g. for 125_wold_country_Italy, latent 50817 has 0.989 AUC, implying (correctly) that it fires on Italian concepts; however, the description reads “names of researchers and scientific authors”). However, some latents appear to be spurious and unrelated to the semantics of their dataset. For example, 22_headline_isobama, which targets headlines published during the Obama administration, is classified with 0 _._ 782 AUC by latent 10555, which activates on “strings of numbers, often with mathematical notations.” 

The spurious latent category seems especially promising because finding a spurious latent may help us identify spurious features in the dataset. However, in a cast study in Section 3.4.3, we find that similar findings may be possible using baseline classifiers: we apply a logistic regression probe to model hidden states on tokens from the Pile (Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, H. He, Thite, Nabeshima, et al. 2020b) and show that maximally activating examples also exhibit the spurious correlation. 

However, a practical advantage for SAEs is that the infrastructure to perform autointerp is pre-existing through platforms like Neuronpedia, and a theoretical advantage is that the baseline classifier can only identify the single most relevant coarse-grained feature, while the decomposability of SAE probes into latents allows for identifying many independent features of various importance. 

### **3.4.3 Detecting Dataset Quality Issues** 

We now investigate the top latents of two datasets, 87_glue_cola and 110_aimade_humangpt3. We find that although the latents identify errors in each dataset, baseline methods are also able to identify these dataset errors. 

#### **GLUE CoLA** 

We first examine 87_glue_cola, an established linguistic acceptability dataset. CoLA prompts are standard English sentences with target = 1 if the sentence is grammatically correct and 0 otherwise. Latent 369585 from the width = 1M SAE has a test AUC of 0.76 and appears to fire on ungrammatical text. Surprisingly, when we look at prompts that latent 369585 fires on, those that it “disagrees” with the labels on (ones that are labeled grammatical), often appear to be ungrammatical. Although we first found this result with SAE probes, we later found that logistic regression was also capable of the same identification. In Table B.5, we show the sentences that a baseline logistic regression classifier and latent 369585 mark as 

75 



<!-- Start of picture text -->
1.00<br>1.0<br>0.81 0.77 0.72 0.78  0.80 0.80  0.78<br>0.56  0.63<br>0.5 Original Labels 0.46<br>Baseline<br>SAE (k=128)<br>Latent 369585<br>0.00<br>0.0<br>CoLA Labels Clean Labels Mislabeled CoLA<br>Test Accuracy<br><!-- End of picture text -->

Figure 3.9: Latent 369585 outperforms a dense SAE probe and logistic regression when testing on mislabeled CoLA examples. 

ungrammatical while the label is grammatical; most such sentences are truly ungrammatical. Thus, both SAE and baseline methods lead us to hypothesize that the CoLA dataset is partially mislabeled. 

To test our hypothesis, we choose 3 LLMs - GPT-4o (OpenAI n.d.), Claude-3.5-SonnetNew, and Llama-3.1-405B Instruct (Grattafiori et al. 2024) - to judge the linguistic acceptability of 1000 random CoLA prompts. We take the LLM majority vote as the ensembled, clean label. This is analogous to the experiment performed in Warstadt, Singh, and Bowman 2019 that validated the CoLA dataset by ensembling the predictions of native English speakers, but with LLM judges instead of English speakers. We find that<sup>˜</sup> 25% of CoLA labels are mislabeled by this metric (higher than the 13% disagreement rate from Warstadt, Singh, and Bowman 2019). Notably, the Table B.5 sentences are identified as ungrammatical by the ensemble while being labeled as grammatical in CoLA. 

Next, we train a baseline logistic regression, a _k_ = 128 dense SAE probe, and the latent 369585 classifier on the original CoLA labels. We then test on a held-out set with the clean, ensembled predictions. In Fig. 3.9, we find that the original labels, the baseline classifier, the dense SAE probe, and latent 369585 all have about the same accuracy on the clean labels. Remarkably, when we test all classifiers on examples which were misclassified in CoLA (where the ensembled labels disagree with the original labels), we find that the latent 369585 classifier outperforms the dense SAE probe, which itself outperforms the baseline considerably. 

#### **AI vs Human** 

We also investigate 110_aimade_humangpt3, which tests if probes can distinguish between human and ChatGPT-3.5 generated text. Latent 105150 has an AUC of 0.82 on this task, yet appears to fire primarily on periods and punctuation.<sup>1</sup> Since this latent is syntactical and unrelated to the content of the dataset, we hypothesize it represents a spurious correlation in the dataset. 

We examine the final token of each prompt in the dataset and indeed find a spurious 

> 1See Neuronpedia for examples. 

76 



<!-- Start of picture text -->
Human<br>1500 AI<br>20<br>1000<br>10<br>500<br>0 0<br>105150: 116246: 3764: period quest. space excl.<br>periods spaces spaces mark point<br>Latent Last Token<br>(AI - Human)<br>Total Activation<br>|Mean Act Difference|<br><!-- End of picture text -->

Figure 3.10: **Left:** Top 3 latents (w/ descriptions) by mean activation difference between AI generated and human SAE latent activations. **Right:** Histograms of the identity of the 4 most common last tokens in AI generated and human text. 



<!-- Start of picture text -->
ILLUSION  FAIR<br>FAIR  Baseline = Last Both =<br>Both = Last SAE = Pool Quiver(Pool, Last)<br>60 56.5%<br>40 39.1% 34.8%<br>19.6%<br>20<br>8.7%<br>2.2%<br>0<br>Baseline > SAE SAE > Baseline<br>> 0.005  AUC<br>Win Rate Percentage<br><!-- End of picture text -->

Figure 3.11: Illustration of a potentially misleading result we find: when comparing activation probes to pooled SAE probes, SAE probe win rate increases considerably, but adding in a “pooled” attention-inspired probe brings down the SAE win-rate. 

feature: AI text is more likely to end in a period while human text is more likely to end in a space (see Fig. 3.10). Although we discovered this by investigating SAE latents, we could reach the same conclusion with baselines: in Table B.6 (Section B.7.4), we apply the logistic regression classifier to 2.8 million tokens from the Pile, and find that it is most active on punctuation tokens. 

## **3.5 Why Didn’t This Work: Illusions of SAE Probes** 

Bricken, Marcus, Mishra-Sharma, et al. 2024a report that SAE probes outperform baselines slightly. A natural question is why our experiments fail to replicate this result. One possible explanation is that Bricken, Marcus, Mishra-Sharma, et al. 2024a’s findings were based on a single dataset, whereas we evaluate across multiple datasets. We find that SAE probes outperform baselines on only a small subset of these datasets (2.2%, see Fig. 3.11). It is possible that the dataset used by Bricken, Marcus, Mishra-Sharma, et al. 2024a falls within this subset; however, without access to it, we cannot confirm this. 

A separate explanation involves a potential illusion due to insufficiently strong baselines. Unlike our work, Bricken, Marcus, Mishra-Sharma, et al. 2024a uses multi-token SAE probing: they aggregate the maximum value for each latent across all prompt tokens, while we only use last token latents. Crucially, they similarly max-pool each model dimension across prompt tokens for their comparative baseline, even though activations are unlikely to have privileged dimensions (in Table B.7 we show that pooling activations in this way works poorly). We implement multi-token SAE probing using max-aggregation on 60 random datasets from our list and _k_ = 128; see Table B.7 for full results on these datasets. We _also_ implement a better baseline: we train attention-pooled probes of the form 



for _q, v ∈_ R<sup>_d_</sup> . We find that when compared to the last-token baseline, max-pooled SAE probes 

77 

Table 3.2: Timeline of SAE Architecture Improvements. 

|**SAE Architecture**|**Publication Date**|
|---|---|
|ReLU (original)|October 4, 2023|
|Bricken, Templeton, et al. 2023||
|ReLU (updated)|April 26, 2024|
|Conerly et al. 2024||
|Gated|May 1, 2024|
|Rajamanoharan, Conmy, et al. 2024b||
|TopK|June 6, 2024|
|Gao, Tour, et al. 2024a||
|JumpReLU|July 19, 2024|
|Rajamanoharan, Lieberum, et al. 2024a||
|BatchTopK|July 19, 2024|
|Bart Bussmann, Patrick Leask, and Nanda 2024a||
|p-annealing<br>Karvonen, Wright, et al. n.d.|July 31, 2024|
|Matryoshka|December 19, 2024|
|Ben Bussmann, Peter Leask, and Nanda 2024||



win 19 _._ 6% of the time, a considerable improvement over win rate of last-token SAE probes (2.2%, see Fig. 3.11). However, when implementing attention-pooled baselines and using the quiver of arrows approach to select between pooled and last-token strategies for SAE probes and baselines, the SAE probe win rate drops more than 50% to 8 _._ 7% (see Fig. 3.11). 

## **3.6 Assessing Improvements in SAE Architectures** 

While SAE probes do not robustly outperform baselines, a separate interesting question is whether recent SAE architectural developments have improved SAE probing performance. We investigate this question by examining the performance of eight different SAE architectures released in the last two years, as outlined in Table 3.2. 

We use width = 16k Gemma-2-2B layer 12 SAEs with a variety of L0 values trained by Karvonen, Rager, et al. 2024a. We create _k_ = 16 _,_ 128 SAE probes on all regimes and datasets for each architecture, and additionally _k_ = 1 for standard conditions. In Fig. 3.12, we plot each SAE architecture’s _k_ = 16 probe performance in standard conditions. We find the average test AUC of each SAE and then take the max across L0 for each SAE architecture (using a single L0 value is somewhat noisy, see Section B.9). This metric tests how expressive individual SAE latents are for different SAE architectures. While we see a slight positive trend for probing performance with more recently released SAE architectures, the effect is not statistically significant. For plots in all regimes, see Section B.9; in other regimes there seems to be a possible slight improvement with newer SAE architectures. 

78 

AUC for SAE Architecture vs Baseline k=16 at Best L0 in Standard Conditions 



<!-- Start of picture text -->
0.00<br>0.05<br>0.10<br>0.15<br>0.20<br>Architecture (In Order of Release)<br>ReLU (orig.)ReLU (new) Gated TopKJump ReLUBatch TopK p-annealMatryoshka<br>SAE - Baseline AUC<br><!-- End of picture text -->

Figure 3.12: We see that there is a slight uptick in probing performance compared to baselines in standard conditions with more recent architectures. However, the spread of the data is considerably larger than this improvement. 

## **3.7 Conclusion** 

Our evaluation of sparse autoencoders (SAEs) reveals fundamental limitations in current SAE methodologies. Despite expectations that interpretable SAE latents would provide a competitive advantage in probing, we found no improvement over traditional methods across multiple regimes and over 100 datasets. Some failures expose critical weaknesses – for instance, SAE latents struggle to be robust to distributional shifts and fail to represent complex concepts – while others are more mundane – lower L0 SAEs create worse sparse probes. More broadly, our findings highlight the need for the field of mechanistic interpretability to evaluate techniques with rigorous baselines. While prior work reported advantages for SAE probes, our robust quiver of arrows methodology, use of stronger baselines, and evaluation on a large set of datasets demonstrate these advantages to be illusory. Even in our own analysis, initial 

79 

conclusions favoring SAE interpretability were later overturned when proper baselines were considered. We do not consider this work to be a wholesale critique of the SAE paradigm. Instead, we view it as a single rigorously evaluated datapoint to contextualize SAE utility and to motivate a more thorough examination of methods in mechanistic interpretability. 

**Limitations** We study the performance of SAE probes as a proxy for SAE utility. While we believe that probing performance is a more effective measurement than typical SAE evaluations like downstream cross-entropy loss or reconstruction error, it is ultimately still a proxy metric. We chose to study difficult probing regimes to try to find cases where the inductive bias of SAE latents outweighed imperfect SAE reconstruction, but it is possible that even a basis of “true” model representations would offer only a mild inductive bias advantage over traditional linear probing. We are thus excited for future work studying probing on toy models (e.g. models trained on board games (Karvonen, Wright, et al. 2024)) where the “true” model features are known. 

Finally, it is possible that further optimization of the SAE probe baseline might increase performance such that it beats baseline methods. For example, Gallifant et al. 2025 find that SAE probing with a number of modifications (multi-token probes, binarization of latents, and probing with _k_ = full SAE dimension) beats baseline methods on safety-relevant datasets (albeit only comparing to single-token baseline probes). However, we believe a main takeaway of our paper is that equivalent effort should be put into optimizing baselines. Indeed, we examine the effect of binarizing latents in Section B.10 and find that it does not significantly help. 

80 

# **Appendix B** 

# **Appendix: Are Sparse Autoencoders Useful? A Case Study in Sparse Probing** 

## **B.1 Related Work** 

### **B.1.1 Probing** 

Probing has a rich history in computational neuroscience, where linear decoders were used to study information representation in biological neural networks (Mur, Bandettini, and Kriegeskorte 2009). This technique was later adapted to study artificial neural networks by Alain 2016. Since then, probing has become a fundamental tool in neural network interpretability, revealing that many high-level concepts are linearly represented in model activations (Gurnee and Tegmark 2023; Heinzerling and Inui 2024; Nanda, A. Lee, and Wattenberg 2023). Similar to our work studying sparse probing, Gurnee, Nanda, et al. 2023 study sparse probing of activations and identify individual monosemantic neurons by setting _k_ = 1. Recent work has also used probing to study safety-relevant properties of language models, such as truthfulness (Marks and Tegmark 2023) and the presence of sleeper agents (MacDiarmid et al. 2024), without relying on potentially unreliable model outputs. Two recent works have investigated the utility of SAEs for probing. First, Bricken, Marcus, MishraSharma, et al. 2024a investigate a synthetic bioweapons dataset and show that SAE probes can sometimes offer an advantage against baseline probes when aggregated across multiple tokens; we discuss these results in our section on multi-token probing in Section 3.5. Second, Gallifant et al. 2025 use feature binarization, multi-token feature pooling, and probing on the entire SAE vector to report that SAE probes outperform baselines. We discuss Gallifant et al.’s (2025) results in our limitations section and in Section B.10. 

### **B.1.2 Challenges in Neural Network Interpretability** 

Our work connects to broader concerns about the reliability of neural network interpretation methods. Adebayo et al. 2018 studied saliency methods, a classical technique for understanding models, and found that randomizing the model and dataset labels did not change many aspects of saliency maps; thus, while the method produced plausible-looking explanations, they were not faithful to the true model and dataset. Similarly, (Bolukbasi et al. 2021) 

81 

demonstrated that seemingly interpretable neurons in BERT were artifacts of running on only a particular dataset rather than more general representations. For SAEs specifically, (Chanin et al. 2024b) identified feature absorption and splitting as fundamental challenges, Gao, Tour, et al. 2024a showed that SAEs have an irreducible error component, and in extremely recent work Heap et al. 2025 show that SAEs trained on random models also result in interpretable features. Our work extends these critiques by finding that many settings where SAE probes were thought to be helpful turn out not to be when compared to stronger baselines. 

### **B.1.3 SAE and SAE Applications** 

Sparse autoencoders (SAEs) provide a map from model activations to a sparse, higher dimensional latent space (Bricken, Templeton, et al. 2023; Cunningham et al. 2023b). Individual latents are hypothesized to represent mono-semantic concepts LLMs use for computation. An SAE is parameterized by its width, or the dimension of its latent space, and its L0, or how many latents are nonzero on average. While our work focuses on evaluating SAEs as probing tools, prior work has explored various downstream applications of SAEs. Templeton et al. 2024 first used SAE latents for steering, and in follow-up work Chalnev, Siu, and Conmy 2024 found that SAEs can help find better steering vectors (although see (Zhengxuan Wu et al. 2025) for a very recent work finding that SAEs are not competitive for steering). Karvonen, Rager, et al. 2024a implement a set of comprehensive benchmarks for evaluating SAE performance, including SHIFT (Marks, Rager, et al. 2024), sparse probing, unlearning, and feature absorption. Recent work has also used SAEs to interpret preference models (L. R. Smith and Brinkmann 2024) and prevent unwanted behavior in model output (Karvonen, Pai, et al. 2024), both specific applications where SAEs seem to be state of the art. 

## **B.2 Classification Datasets** 

Below we list in a (large) table all of the classification datasets we use, as well as their source. Note that datasets 161-163 are modified from their source -a mistake in our formatting reframes them as differentiating between news headlines and code samples. 

Table B.1: Binary classification tasks used. 

|**Citation**|**Dataset Name**|
|---|---|
|Gurnee and Tegmark 2024|5_hist_fig_ismale<br>6_hist_fig_isamerican<br>7_hist_fig_ispolitician<br>21_headline_istrump<br>22_headline_isobama<br>23_headline_ischina<br>24_headline_isiran<br>26_headline_isfrontpage<br>114_nyc_borough_Manhattan<br>115_nyc_borough_Brooklyn|
||Continued on next page|



82 

|**Citation**|**Dataset Name**|
|---|---|
||116_nyc_borough_Bronx|
||117_us_state_FL|
||118_us_state_CA|
||119_us_state_TX|
||120_us_timezone_Chicago|
||121_us_timezone_New_York|
||122_us_timezone_Los_Angeles|
||123_world_country_United_Kingdom|
||124_world_country_United_States|
||125_world_country_Italy|
||126_art_type_book|
||127_art_type_song|
||128_art_type_movie|
|Johannes Welbl 2017|36_sciq_tf|
|S. Lin, Hilton, and Evans 2022|41_truthqa_tf|
|Ben Zhou and Roth 2019|42_temp_sense|
||130_temp_cat_Frequency|
||131_temp_cat_Typical Time|
||132_temp_cat_Event Ordering|
|Bisk et al. 2020|44_phys_tf|
|Tafjord et al. "2019"|47_reasoning_tf|
|Hendrycks, Burns, Basart, Critch, et al. 2021|48_cm_correct|
||49_cm_isshort|
||50_deon_isvalid|
||51_just_is|
||52_virtue_is|
|Talmor et al. 2022|54_cs_tf|
|Gurnee, Nanda, et al. 2023|56_wikidatasex_or_gender|
||57_wikidatais_alive|
||58_wikidatapolitical_party|
||59_wikidata_occupation_isjournalist|
||60_wikidata_occupation_isathlete|
||61_wikidata_occupation_isactor|
||62_wikidata_occupation_ispolitician|
||63_wikidata_occupation_issinger|
||64wikidataoccupationisresearcher|
||___<br>65_high-school|
||66_living-room|
||67_social-security|
||68credit-card|
||_<br>69_blood-pressure|
||70_prime-factors|
||71_social-media|



Continued on next page 

83 

|**Citation**|**Dataset Name**|
|---|---|
||72_gene-expression|
||73_control-group<br>74_magnetic-field<br>75_cell-lines<br>76_trial-court|
||77_second-derivative|
||78_north-america<br>79_human-rights<br>80_side-effects|
||81_public-health|
||82_federal-government|
||83_third-party|
||84_clinical-trials|
||85mental-health|
|A. Wang et al. 2019|_<br>87_glue_cola|
||89_glue_mrpc|
||90_glue_qnli|
||91_glue_qqp|
||92_glue_sst2|
||136_glue_mnli_entailment|
||137_glue_mnli_neutral|
||138_glue_mnli_contradiction|
|Gerami n.d.|94_ai_gen|
|Z. Lin et al. 2023|95_toxic_is|
|AI and Ishii n.d.|96_spam_is|
|clmentbisaillon n.d.|100_news_fake|
|Kotari n.d.|105_click_bait|
|Davidson et al. 2017|106_hate_hate|
||107_hate_offensive|
|Gaggar, Bhagchandani, and Oza 2023|110_aimade_humangpt3|
|Pang and L. Lee 2005|113_movie_sent|
|AllenAI n.d.|129arithmcA|
|Rogers et al. 2020|___<br>133_context_type_Causality|
||134_context_type_Belief_states|
||135_context_type_Event_duration|
|Dublish n.d.|139_news_class_Politics|
||140_news_class_Technology|
||141newsclassEntertainment|
|Falgunipatel19 n.d.|___<br>142_cancer_cat_Thyroid_Cancer|
||143cancercatLungCancer|
||____<br>144_cancer_cat_Colon_Cancer|
|Kasaraneni n.d.|145_disease_class_digestive system diseases|
||146_disease_class_cardiovascular diseases|
||Continued on next page|



84 

|**Citation**|**Dataset Name**|
|---|---|
||147_disease_class_nervous system diseases|
|Gupta n.d.|148_twt_emotion_worry<br>149_twt_emotion_happiness<br>150_twt_emotion_sadness|
|Goh n.d.|151_it_tick_HR Support<br>152_it_tick_Hardware<br>153_it_tick_Administrative rights|
|Stathead n.d.|154_athlete_sport_football<br>155_athlete_sport_basketball<br>156_athlete_sport_baseball|
|Karvonen, Rager, et al. 2024a|157_amazon_5star<br>158_code_C<br>159_code_Python<br>160_code_HTML|
||161_agnews_0<br>162_agnews_1|
||163_agnews_2|



## **B.3 Probing Setup** 

### **B.3.1 AUC** 

The Area Under the Curve (AUC) quantifies the overall performance of a binary classifier using the Receiver Operating Characteristic (ROC) curve. An ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels, illustrating the trade-offs between correctly predicting positives and incorrectly predicting negatives. The AUC, ranging from 0 to 1, measures the entire two-dimensional area beneath this curve. An AUC of 1.0 signifies perfect classification, 0.5 indicates performance no better than random chance, and closer to 0 implies poor classification. This metric provides a single, aggregate measure of performance across all possible classification thresholds, making it particularly useful for comparing different classifiers. 

Technically, when choosing _k_ and the baseline SAE to use for the other section, we use test AUC on normal datasets, which constitutes a slight leakage of test data. However, we used trends observed over a large number of datasets, in the same way we might provide heuristics for constructing SAE probes to a future practitioner. 

### **B.3.2 Probing Method Validation Details** 

For each strategy, we use _hp_ that has the maximal average AUC across held out validation sets, AUC<sup>val</sup> _p_<sup>.WechooseavalidationmethodfromTableB.2basedonthedatasetsize</sup><sup>_n_</sup> (most of the time this is the last row in the table for large _n_ , except for the low data regime tests). 

85 

Table B.2: Selection methods to choose hyperparameters _hp_ when training on different dataset sizes 

|Data Size (_n_)|Selection method for probe _p_|
|---|---|
|_n ≤_3|Train _p_ with each _hp_ on all _n_<br>points; choose _hp_ which maxi-<br>mizes AUC on training set.|
|3_< n ≤_12|Use leave-two-out cross valida-<br>tion; train _p_ with each _hp_ on<br>all training splits of size _n −_2<br>and evaluate on the last 2 held<br>out points|
|12_< n ≤_128|Use 6-fold cross validation;<br>split _n_ in sixths, train _p_ with<br>each _hp_ on all sets of 5 splits,<br>and evaluate on the remaining<br>fold|
|_n >_128|Use 80%/20% training/valida-<br>tion split|



### **B.3.3 Probing Method Hyperparameter Details** 

In this section, we list each baseline method and the hyperparameters that we search over for that method. 

#### • **Logistic Regression** : 

**–** _C_ : Ranges logarithmically from 10<sup>5</sup> to 10<sup>_−_5</sup> . The _L_ 2/ _L_ 1 regularization is 1 _/C_ . 

- **PCA Regression** : 

   - Number of PCA Components: Varies logarithmically from 1 up to the minimum of the number of samples, latents, or 100. 

- **K-Nearest Neighbors (KNN)** : 

   - Number of Neighbors: Logarithmically spaced values up to the smaller of 100 or the number of samples minus one. 

#### • **XGBoost** : 

- n_estimators: Ranges from 50 to 250 in steps of 50. 

- max_depth: Ranges from 2 to 5. 

- learning_rate: Ranges logarithmically from 0.001 to 0.1. 

- subsample and colsample_bytree: Range from 0.7 to 1.0. 

- reg_alpha and reg_lambda: Range logarithmically from 0.001 to 10. 

86 



<!-- Start of picture text -->
Performance by Method Across Layers<br>0.95<br>0.90<br>0.85<br>0.80 Logistic Reg.<br>PCA<br>0.75 KNN<br>XGBoost<br>0.70 MLP<br>Combined Mean<br>0.65<br>embed 9 20 31 41<br>Layer<br>Test AUC<br><!-- End of picture text -->

(a) Layer 20 is best for baselines. 



<!-- Start of picture text -->
Best Performing Method by Layer with Test AUC<br>Logistic Reg.<br>100 PCA<br>KNN<br>80 XGBoost<br>MLP<br>60<br>40<br>20<br>0<br>embed 9 20 31 41<br>Layer<br>Number of Datasets<br><!-- End of picture text -->

(b) Logistic regression is often the most performant across datasets. 

Figure B.1: Analysis of layer performance and methods. (a) Shows the optimal layer for baseline performance. (b) Demonstrates the effectiveness of different methods across layers. 

   - min_child_weight: Ranges from 1 to 9. 

- **Multilayer Perceptron (MLP)** : 

   - Network depth: 1 to 3 hidden layers 

   - Hidden layer width: 16,32, or 64. 

   - learning_rate_init: Five values ranging logarithmically from 10<sup>_−_4</sup> to 10<sup>_−_2</sup> . 

   - alpha: Weight decay parameter, with 5 values ranging logarithmically from 10<sup>_−_5</sup> to 10<sup>_−_2</sup> . 

   - Activation function: ReLU. 

   - Optimizer: Adam. 

### **B.3.4 Formal Discussion of Quiver of Arrows** 

The quiver of arrows can be defined formally as follows. Given a set of probing methods _P_ = _p_ 1 _, . . . , pn_ (listed in Section 3.2.3), we find AUC<sup>val</sup> _pi_<sup>foreachmethodusingtheprocedure</sup> described in Section 3.2.2. We then choose the probing method _p∗_ with maximal AUC<sup>val</sup> _pi_<sup>and</sup> record its AUC<sup>test</sup> _P_ = AUC<sup>test</sup> _p∗_<sup>.WecanthencompareAUC</sup> _P_<sup>test</sup> to AUC<sup>test</sup> _P_<sup>_′_foradifferentsetof</sup> methods _P_<sup>_′_</sup> = _p_<sup>_′_</sup> 1<sup>_, . . . , p′_</sup> _n_<sup>.If we let</sup><sup>_P_be a set of baseline methods, and</sup><sup>_P ′_=</sup><sup>_P ∪{_SAEprobes</sup><sup>_}_,</sup> then AUC<sup>test</sup> _P_<sup>_′−_AUCtest</sup> _P_ directly represents the increase in test performance when adding SAE probes to a practitioner’s toolbox. We then aggregate AUC<sup>test</sup> _P −_ AUC<sup>test</sup> _P_<sup>_′_across many different</sup> datasets and testing regimes (e.g., dataset size) to give an overall sense of the improvement due to SAE probing. We note that adding a batch of SAE probes as additional arrows to the quiver is potentially disadvantageous for SAE probes. This is because using multiple SAE probes presents additional opportunities for these probes to overfit to the training data and be chosen by the quiver of arrows approach without properly generalizing to the test data. Regardless, we see the quiver of arrows as a proper counterfactual for a practitioner without SAE probes. Additionally, the quiver of arrows approach is not the reason we find negative 

87 



<!-- Start of picture text -->
SAE Performance by Width and Sparsity<br>Layer 20<br>10 6<br>0.88<br>0.86<br>0.84<br>10 5 SAE Probe Performance as a Function of Width<br>0.82 0.865 y = -0.00log2(x) + 0.91<br>R² = 0.966<br>0.860<br>0.80 0.855<br>0.850<br>0.845<br>10 1 10 2 0.840<br>L0 Sparsity<br>16 32 64 128 256 512 1024<br>Width (thousands)<br>(a) We find that L0 is important in creating<br>Width<br>Test AUC<br>Average Test AUC<br><!-- End of picture text -->

(a) We find that L0 is important in creating effective probes, while width plays a minimal role. Test AUC averaged over all values of _k_ and datasets. 

(b) When averaging over other factors, smaller width SAEs perform best as probes. However, the relationship is minimal. 

Figure B.2: Analysis of SAE width and L0 regularization effects on probe performance. (a) Shows the Pareto frontier of L0 vs width trade-offs. (b) Illustrates the relationship between SAE width and probe performance. 

results for SAE probes. For instance, in Fig. 3.1, we directly compare the test AUC of the most performant SAE probe and baseline in standard conditions across all regimes. Without using the quiver of arrows approach, we still find SAEs underperform baselines. 

## **B.4 Normal Conditions** 

### **B.4.1 Evaluating Baselines** 

We see that layer 20 is best for baselines, so we choose to use that layer moving forward (Fig. B.1a). Logistic regression most often has the highest test AUC across datasets in these conditions at layer 20 (Fig. B.1b). 

### **B.4.2 Choosing which SAEs to Test** 

When constructing a pareto curve based on the width and L0 of all available SAEs at layer 20 of Gemma-2-9B, we find that width is unimportant, while constructing probes from a higher L0 seems to improve probe performance (Fig. B.2a). 

To ensure that width plays a minimal role, we average over all datasets, _k_ , and L0s in Fig. B.2b. There is a clear negative trend indicating that smaller SAE widths are better, but with almost 0 slope. Because we aim to make as few decisions based off the test data 

88 

as possible, we consider three widths throughout our experiments, 16 _,_ 000, 131 _,_ 000, and 1 _,_ 000 _,_ 000. 

### **B.4.3 Plotting Dataset Performance vs. K** 

In Fig. B.3, we show the performance on all datasets vs. the number of latents we train the sparse probe on. Almost all datasets are monotonically increasing in _k_ ; some have a sharp increase at some _k_ value, but most increase relatively smooththly. 

## **B.5 Additional Results for Various Regimes** 

### **B.5.1 Data Scarcity** 

In the data scarce setting, we see that we choose mostly the SAE method from the quiver at smaller training fractions (Fig. B.5a). This is because we break ties by explicitly preferring the SAE method, and most methods have perfect validation AUC at small training fractions. Specifically, we first prefer the smallest width SAE, and then the SAE with the largest value of _k_ . In Fig. B.4, we visualize the improvement by using the quiver of arrows approach with SAE probes across all datasets. 

### **B.5.2 Class Imbalance** 

We show the improvement for all datasets in Fig. B.4 across all class imbalances. Generally, the SAE method is chosen at more extreme imbalance ratios (Fig. B.5b) 

### **B.5.3 Label Corruption** 

We show the performance of the SAE across datasets with all values of label corruption. Interestingly, the SAE still has highest test AUC at many places (Fig. B.5c shows the percent of time that the SAE test AUC is larger). However, it is clear from the per dataset imshow (Fig. B.4) that it struggles with more label noise. 

## **B.6 Reproducing Core Results on Llama-3.1-8b** 

We use SAEs provided by Llama Scope to replicate our core results, namely, that SAEs underperform baseline probes in normal, data scarce, class imbalance, and label noise settings when testing on Llama-3.1-8B. In Fig. B.6a, we show the results for baseline probes applied to various layers of Llama-3.1-8b in standard settings. We see that the layer roughly halfway through the model, layer 16, is best for probing experiments. In Fig. B.6b, we show that SAE probes in normal settings on layer 16 of Llama-3.1-8b are not more performant than baselines, although there are a few positive outliers we did not observe in Gemma-2-9b. Lastly, we use the quiver of arrows approach to show that SAE probes are not more performant in the conditions of data scarcity, class imbalance, and label noise (Fig. B.7). Because of the 

89 



<!-- Start of picture text -->
5 hist fig 6 hist fig 7 hist fig 21 headline 22 headline 23 headline 24 headline 26 headline 41 truthqa<br>ismale isamerican ispolitician istrump isobama ischina isiran isfrontpage 36 sciq tf tf<br>56<br>42 temp 47 reasoning 48 cm 49 cm 50 deon wikidatasex<br>sense 44 phys tf tf correct isshort isvalid 51 just is 52 virtue is 54 cs tf or gender<br>57 58 59 wikidata 60 wikidata 61 wikidata 62 wikidata 63 wikidata 64 wikidata<br>wikidatais wikidatapolitical occupation occupation occupation occupation occupation occupation 65 high 66 living<br>alive party isjourn isathle isactor ispolit issinge isresea school room<br>67 social 68 credit 69 blood 70 prime 71 social 72 gene 73 control 74 magnetic 75 cell 76 trial<br>security card pressure factors media expression group field lines court<br>77 second 78 north 79 human 80 side 81 public 82 federal 83 third 84 clinical 85 mental<br>derivative america rights effects health government party trials health 87 glue cola<br>100 news 105 click 106 hate<br>89 glue mrpc 90 glue qnli 91 glue qqp 92 glue sst2 94 ai gen 95 toxic is 96 spam is fake bait hate<br>114 nyc 115 nyc 120 us<br>107 hate 110 aimade 113 movie borough borough 116 nyc 117 us state 118 us state 119 us state timezone<br>offensive humangpt3 sent Manhattan Brooklyn borough Bronx FL CA TX Chicago<br>121 us 122 us 123 world 124 world<br>timezone New timezone Los country country 125 world 126 art type 127 art type 128 art type 129 arith mc 130 temp cat<br>York Angeles United Kingd United State country Italy book song movie A Frequency<br>133 context 134 context 135 context 136 glue 138 glue 139 news 140 news<br>131 temp cat 132 temp cat type type Belief type Event mnli 137 glue mnli class class<br>Typical Time Event Ordering Causality states duratio entailment mnli neutral contradiction Politics Technology<br>141 news 142 cancer 143 cancer 144 cancer 145 disease 146 disease 147 disease 149 twt 150 twt<br>class cat Thyroid cat Lung cat Colon class class class 148 twt emotion emotion<br>Entertainment Cancer Cancer Cancer digestive sy cardiovascul nervous syst emotion worry happiness sadness<br>154 athlete 155 athlete 156 athlete<br>151 it tick 152 it tick 153 it tick sport sport sport 157 amazon 159 code 160 code<br>HR Support Hardware Administrative rig football basketball baseball 5star 158 code C Python HTML<br>161 agnews 0 162 agnews 1 163 agnews 2<br>Number of Features in  (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)<br>Test AUC (0.5 - 1.0)<br><!-- End of picture text -->

Figure B.3: Test AUC on all datasets vs. _k_ = _|I|_ using the Gemma Scope layer 20 SAE with width 131k and _L_ 0 = 193. 

90 



<!-- Start of picture text -->
AUC Improvement AUC Improvement AUC Improvement<br>0.2 0.0 0.2 0.2 0.1 0.0 0.1 0.2 0.50 0.25 0.00 0.25 0.50<br>5_hist_fig_ismale<br>6_hist_fig_isamerican<br>7_hist_fig_ispolitician<br>21_headline_istrump<br>22_headline_isobama<br>23_headline_ischina<br>24_headline_isiran<br>26_headline_isfrontpage<br>36_sciq_tf<br>41_truthqa_tf<br>42_temp_sense<br>44_phys_tf<br>47_reasoning_tf<br>48_cm_correct<br>49_cm_isshort<br>50_deon_isvalid<br>51_just_is<br>52_virtue_is<br>54_cs_tf<br>56_wikidatasex_or_gender<br>57_wikidatais_alive<br>58_wikidatapolitical_party<br>59_wikidata_occupation_isjournalist<br>60_wikidata_occupation_isathlete<br>61_wikidata_occupation_isactor<br>62_wikidata_occupation_ispolitician<br>63_wikidata_occupation_issinger<br>64_wikidata_occupation_isresearcher<br>65_high-school<br>66_living-room<br>67_social-security<br>68_credit-card<br>69_blood-pressure<br>70_prime-factors<br>71_social-media<br>72_gene-expression<br>73_control-group<br>74_magnetic-field<br>75_cell-lines<br>76_trial-court<br>77_second-derivative<br>78_north-america<br>79_human-rights<br>80_side-effects<br>81_public-health<br>82_federal-government<br>83_third-party<br>84_clinical-trials<br>85_mental-health<br>87_glue_cola<br>89_glue_mrpc<br>90_glue_qnli<br>91_glue_qqp<br>92_glue_sst2<br>94_ai_gen<br>95_toxic_is<br>96_spam_is<br>100_news_fake<br>105_click_bait<br>106_hate_hate<br>107_hate_offensive<br>110_aimade_humangpt3<br>113_movie_sent<br>114_nyc_borough_Manhattan<br>115_nyc_borough_Brooklyn<br>116_nyc_borough_Bronx<br>117_us_state_FL<br>118_us_state_CA<br>119_us_state_TX<br>120_us_timezone_Chicago<br>121_us_timezone_New_York<br>122_us_timezone_Los_Angeles<br>123_world_country_United_Kingdom<br>124_world_country_United_States<br>125_world_country_Italy<br>126_art_type_book<br>127_art_type_song<br>128_art_type_movie<br>129_arith_mc_A<br>130_temp_cat_Frequency<br>131_temp_cat_Typical Time<br>132_temp_cat_Event Ordering<br>133_context_type_Causality<br>134_context_type_Belief_states<br>135_context_type_Event_duration<br>136_glue_mnli_entailment<br>137_glue_mnli_neutral<br>138_glue_mnli_contradiction<br>139_news_class_Politics<br>140_news_class_Technology<br>141_news_class_Entertainment<br>142_cancer_cat_Thyroid_Cancer<br>143_cancer_cat_Lung_Cancer<br>144_cancer_cat_Colon_Cancer<br>145_disease_class_digestive system diseases<br>146_disease_class_cardiovascular diseases<br>147_disease_class_nervous system diseases<br>148_twt_emotion_worry<br>149_twt_emotion_happiness<br>150_twt_emotion_sadness<br>151_it_tick_HR Support<br>152_it_tick_Hardware<br>153_it_tick_Administrative rights<br>154_athlete_sport_football<br>155_athlete_sport_basketball<br>156_athlete_sport_baseball<br>157_amazon_5star<br>158_code_C<br>159_code_Python<br>160_code_HTML<br>161_agnews_0<br>162_agnews_1<br>163_agnews_2<br>Fraction Corrupted Ratio of Positive Class Number of Training Examples<br>0.00.050.10.150.20.250.30.350.40.450.5 0.050.10.150.20.250.30.350.40.450.50.550.60.650.70.750.80.850.90.95 23457101420283853741031431982753825317371024<br>Dataset<br><!-- End of picture text -->

Figure B.4: Improvement from using SAE methods in the quiver across all datasets for various corruption ratios, ratios of positive classes, and number of training examples 

91 



<!-- Start of picture text -->
100<br>100 100<br>80<br>80 80<br>60 Non SAE ChosenSAE Chosen 60 60<br>40 40 40<br>20 20 Non SAE ChosenSAE Chosen 20 Logistic Regression BetterSAE Better<br>0 0 0<br>Number of Training Examples Ratio of Positive Class Fraction Corrupted<br>2 3 4 5 7 10 14 20 28 38 53 74 1031431982753825317371024 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5<br>% of Datasets<br>Number of Datasets Number of Datasets<br><!-- End of picture text -->

(a) SAE probes are chosen by our method at small training fractions because multiple methods have perfect validation AUC and we break ties by choosing an SAE probe. 

(b) SAE probes are generally chosen at more extreme imbalances. 

(c) Corruption fraction has a minor influence on the percent of time SAE probes or logistic regression probes are chosen. 

Figure B.5: Method selection analysis under different data conditions. (a) Selection under data scarcity. (b) Selection under class imbalance. (c) Selection under data corruption. 



<!-- Start of picture text -->
Performance by Method Across Layers<br>0.95<br>0.90<br>0.85<br>Logistic Reg.<br>0.80 PCA<br>KNN<br>0.75<br>XGBoost<br>0.70 MLP<br>Combined Mean<br>embed 8 16 24 31<br>Layer<br>Test AUC<br><!-- End of picture text -->

(a) When testing baseline methods at different layers of Llama-3.1-8b, we find the middle layer 16 works best by a slim margin. 



<!-- Start of picture text -->
1.0<br>Mean : -0.000 ± 0.001<br>No Change<br>0.8<br>0.6<br>0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00<br>Test AUC (without SAE)<br>Test AUC (with SAE)<br><!-- End of picture text -->

(b) In standard conditions, we see that SAE probes are not statistically more performant than baselines for Llama-3.1-8b 

Figure B.6: Analysis of Llama-3.1-8b probe performance. (a) Layer-wise performance of baseline methods. (b) Comparison between SAE and baseline probes under standard conditions. 

increased effort of testing OOD and interpretability, we do not replicate those results. Note that in all experiments, we use the provided width 128k, L0 55 SAE. 

## **B.7 Interpretability** 

### **B.7.1 Pruning** 

We demonstrate the latent ranking procedure for the task 66_living-room in Table B.3. We see that pruning works, for two of the tasks, 66_living-room and 90_glue_qnli, while failing for 7_hist_fig_ispolitician (Fig. B.8). However, even when pruning works, the improvement is marginal compared to the decrease in performance when moving from an in-distribution test set to an OOD test set. Thus, we hypothesize that the underlying latents are not 

92 



<!-- Start of picture text -->
0.0<br>0.1<br>Number of Training Examples Ratio of Positive Class Fraction Corrupted<br>2 3 45 7 101420283853741031431982753825317371024 0.050.10.150.20.250.30.350.40.450.50.550.60.650.70.750.80.850.90.95 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5<br>SAE  (AUC)<br><!-- End of picture text -->

Figure B.7: We find that Llama-3.1-8b SAE probes do not improve on baselines in the settings of data scarcity, class imbalance, and label noise. 

sufficiently expressive. For example, latent 122774 is most active on the English translation of “living room,” and not activate at all for the French translation (Fig. 3.8). This indicates this latent does not sufficiently represent the concept of “living room” to be robust to OOD transformations. 



<!-- Start of picture text -->
0.65<br>0.60<br>66_living-room<br>0.55 90_glue_qnli<br>7_hist_fig_ispolitician<br>0.50<br>0.45<br>1 2 3 4 5 6 7 8<br>Number of Features in Pruned Probe<br>OOD Test AUC<br><!-- End of picture text -->

Figure B.8: Two datasets improve with pruning, but one does not. Indicates that it needs more latents to construct answer. 

### **B.7.2 Latent Investigation** 

We generate natural language descriptions of all the latents with GPT-4o. We find that most are either relevant to the task or visibly spurious. We hypothesize that some latents have incomplete descriptions given their performance on the task. We list a selection of the most interesting of these latents from each category in Table B.4. 

Table B.4: Latent categorization, characteristics, and performance. 

|**Latent**<br>**Cat-**<br>**e-**<br>**gory**|**Dataset**|**Latent**<br>**Num-**<br>**ber**|**Test**<br>**AUC**|**Latent**<br>**scription**|**De-**|
|---|---|---|---|---|---|
|Description<br>Fits|5_hist_fig_ismale|81210|0.892|references <br>male indiv|to fe-<br>iduals,|
|Task||||especially <br>ing on pro<br>and names|focus-<br>nouns<br>.|



Continued on next page 

93 

|**Latent**|**Dataset**|**Latent**|**Test**|**Latent**<br>**De-**|
|---|---|---|---|---|
|**Cat-**||**Num-**|**AUC**|**scription**|
|**e-**||**ber**|||
|**gory**|||||
||61_wikidata_occupation_isactor|91505|0.830|names<br>of<br>well-<br>known<br>actors<br>and film industry<br>personalities.|
||66_living-room|122774|0.999|references<br>to|
|||||specific<br>rooms<br>in<br>a<br>house,<br>particularly<br>living rooms and<br>parlors.|
||155_athlete_sport_basketball|83267|0.943|references<br>to|
|||||basketball teams,|
|||||players,<br>and<br>related<br>sports|
|||||terms.|
|Clever|96_spam_is|35460|0.907|keywords<br>and|
|La-||||phrases<br>related|
|tents||||to text messag-|
|||||ing<br>and<br>SMS<br>communica-<br>tions.|
||100_news_fake|31726|0.968|references to con-<br>spiracy theories<br>and secretive or-<br>ganizations.|
|Potentially|21_headline_istrump|22915|0.812|numerical dates|
|Spu-||||and percentages|
|rious||||within a political<br>or historical con-<br>text.|
||22_headline_isobama|10555|0.782|strings of num-<br>bers, often with<br>mathematical no-<br>tations or identi-<br>fiers.|
||110_aimade_humangpt3|105150|0.816|statistical or nu-<br>merical data and<br>measurements.|



Continued on next page 

94 

|**Latent**|**Dataset**|**Latent**|**Test**|**Latent**<br>**De-**|
|---|---|---|---|---|
|**Cat-**||**Num-**|**AUC**|**scription**|
|**e-**||**ber**|||
|**gory**|||||
||133_context_type_Causality|89524|0.947|questions<br>or<br>questioning<br>|
|||||phrases,<br>par-<br>ticularly<br>those<br>beginning<br>with<br>"Why?".|
||134_context_type_Belief_states|113152|0.638|words related to<br>status or condi-<br>tion in technical<br>or medical con-<br>texts.|
||135_context_type_Event_duration|18897|0.876|questions<br>and<br>phrases<br>re-<br>lated to cost or<br>quantity.|
|Classified|i<br>49_cm_isshort|106376|1.000|sections of text|
|by||||related to math-|
|Con-||||ematical or pro-|
|text||||gramming nota-|
|Length||||tion and expres-|
|(Spu-<br>ri-||||sions.|
|ous)|||||
||161_agnews_0|106376|0.990|sections of text<br>related to math-<br>ematical or pro-<br>gramming nota-<br>tion and expres-<br>sions.|
||162_agnews_1|106376|0.988|sections of text<br>related to math-<br>ematical or pro-<br>gramming nota-|
|||||tion and expres-|
|||||sions.|
||Continued on next page||||



95 

|**Latent**|**Dataset**|**Latent**|**Test**|**Latent**|**De-**|
|---|---|---|---|---|---|
|**Cat-**||**Num-**|**AUC**|**scription**||
|**e-**||**ber**||||
|**gory**||||||
||163_agnews_2|106376|0.991|sections of <br>related to m<br>ematical or <br>gramming <br>tion and ex<br>sions.|text<br>ath-<br> pro-<br>nota-<br>pres-|
|Task|105_click_bait|78823|0.967|mentions of|spe-|
|Shows||||cific events|, ac-|
|La-||||tivities, or i|tems|
|tent||||in particula|r lo-|
|De-||||cations<br>or|set-|
|scrip-||||tings.||
|tion||||||
|is Im-||||||
|per-||||||
|fect||||||
||123_world_country_United_Kingdom|100153|0.978|information<br>related<br>to|pro-|
|||||fessional<br>|roles,|
|||||locations,<br>organization|and<br>s.|
||125worldcountryItaly|50817|0.989|names<br>of|re-|
||___|||searchers|and|
|||||scientific<br>thors.|au-|



### **B.7.3 Glue CoLA** 

In Table B.5, we show the top 5 examples where latent 369585 and the activation probe have the highest confidence that a prompt is ungrammatical, but it is (incorrectly) labeled as grammatical in the CoLA ground truth. 

### **B.7.4 AI Made** 

In Table B.6, we show the tokens that the 110_aimade_humangpt3 classifier activates on across the pile with the highest average activation. Like the SAE feature, the baseline classifier has top activations on punctuation tokens. 

96 

## **B.8 Multiple tokens results** 

In Table B.7, we show a comparison of all methods (including the attention based probes and multi-token SAE methods discussed in Section 3.5) on a random sample of 60 datasets. 

Table B.7: Comparison of different logistic regression (“base”) and SAE probing methods on a selection of 60 random datasets. Last is last token probing as in the rest of our paper, concat is concatenating the top 20 PCA dimensions of all tokens, mean is taking the mean across activation dimensions across the context, max is taking the max across activation dimensions across the context, and Attn-like probe is described in Section 3.5. 

|Dataset|Base<br>lt|Base<br>t|SAE<br>(last)|SAE<br>(mean)|SAE<br>(max)|SAE<br>(last)|SAE<br>(mean)|SAE<br>(max)|Attn<br>-Like|
|---|---|---|---|---|---|---|---|---|---|
||(as)|(conca)|l0=68|l0=68|l0=68|l0=408|l0=408|l0=408|Probe|
|6_hist_fig|**0.990**|0.977|0.982|0.914|0.984|0.983|0.883|0.985|0.987|
|7_hist_fig|0.750|0.698|0.738|0.661|0.739|0.746|0.613|0.738|**0.760**|
|21headlin|0.987|0.986|0.950|0.993|**1.000**|0.964|0.990|1.000|1.000|
|_<br>24headlin|0.991|0.979|0.892|0.995|0.997|0.947|0.994|**0.998**|0.996|
|_<br>44phystf|0.885|0.541|0.836|0.657|0.760|0.854|0.656|0.790|**0.903**|
|__<br>48_cm_corr|0.833|0.659|0.788|0.699|0.779|0.794|0.708|0.784|**0.860**|
|54cstf|0.695|0.554|0.689|0.588|0.680|**0.708**|0.580|0.697|0.697|
|__<br>59_wikidat|**0.961**|0.787|0.933|0.809|0.864|0.939|0.809|0.875|0.935|
|62_wikidat|**0.981**|0.878|0.948|0.892|0.923|0.959|0.898|0.925|0.954|
|63_wikidat|**0.959**|0.840|0.925|0.869|0.893|0.946|0.848|0.862|0.928|
|66_living-|**1.000**|0.985|1.000|0.968|0.999|1.000|0.950|0.998|0.998|
|67_social-|**1.000**|0.992|0.999|0.985|0.999|1.000|0.971|0.998|0.999|
|68_credit-|**1.000**|0.959|0.998|0.942|0.986|0.998|0.920|0.981|0.993|
|71_social-|0.998|0.985|**0.998**|0.969|0.993|0.998|0.950|0.989|0.995|
|73_control|**0.998**|0.970|0.995|0.945|0.984|0.998|0.935|0.986|0.982|
|78_north-a|**1.000**|0.933|0.998|0.907|0.990|0.999|0.883|0.976|0.995|
|79_human-r|**0.999**|0.974|0.995|0.964|0.988|0.996|0.947|0.990|0.992|
|89_glue_mr|**0.891**|0.761|0.785|0.784|0.796|0.823|0.796|0.777|0.867|
|90_glue_qn|0.926|0.729|0.886|0.830|0.909|0.910|0.859|0.909|**0.931**|
|94_ai_gen|**0.998**|0.996|0.993|0.997|0.996|0.996|0.996|0.997|0.996|
|96_spam_is|**0.999**|0.999|0.997|0.996|0.997|0.998|0.995|0.999|0.999|
|100_news_f|**1.000**|**1.000**|1.000|1.000|**1.000**|1.000|1.000|**1.000**|1.000|
|105_click_|1.000|**1.000**|1.000|0.999|1.000|0.999|1.000|1.000|1.000|
|106_hate_h|0.722|0.612|0.661|0.742|0.785|0.674|0.735|0.788|**0.812**|
|110_aimade|0.975|0.865|0.953|0.919|0.970|0.953|0.916|0.960|**0.979**|
|114_nyc_bo|0.780|0.755|0.740|0.716|0.797|0.745|0.733|0.805|**0.872**|
|119_us_sta|**0.997**|0.957|0.994|0.982|0.990|0.994|0.952|0.992|0.995|
|120_us_tim|0.944|0.803|0.942|0.785|**0.946**|0.936|0.751|0.940|0.941|
|121_us_tim|0.952|0.815|0.948|0.776|0.954|0.950|0.754|**0.957**|0.951|
|124_world_|0.999|0.999|**0.999**|0.965|0.999|0.998|0.968|0.999|0.999|



Continued on next page 

97 

|Dataset|Base<br>(last)|Base<br>(concat)|SAE<br>(last)<br>l0=68|SAE<br>(mean)<br>l0=68|SAE<br>(max)<br>l0=68|SAE<br>(last)<br>l0=408|SAE<br>(mean)<br>l0=408|SAE<br>(max)<br>l0=408|Attn<br>-Like<br>Probe|
|---|---|---|---|---|---|---|---|---|---|
|127_art_ty|0.913|0.872|0.894|0.834|0.889|0.905|0.845|0.900|**0.916**|
|129_arith_|**0.979**|0.903|0.880|0.867|0.951|0.938|0.861|0.975|0.977|
|132_temp_c|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|**1.000**|0.991|
|133_contex|0.984|0.982|0.975|0.973|0.979|0.968|0.973|0.980|**0.986**|
|134_contex|0.985|0.977|0.953|0.966|0.989|0.966|0.963|**0.994**|0.987|
|137_glue_m|**0.856**|0.551|0.788|0.632|0.734|0.807|0.658|0.742|0.844|
|139_news_c|**0.978**|0.950|0.944|0.972|0.953|0.962|0.967|0.973|0.965|
|141_news_c|0.951|0.942|0.917|0.967|0.968|0.933|0.965|**0.973**|0.965|
|144_cancer|**1.000**|1.000|0.981|**1.000**|**1.000**|0.991|**1.000**|**1.000**|1.000|
|145_diseas|0.618|0.677|0.501|0.618|0.678|0.563|0.606|**0.686**|0.655|
|149_twt_em|0.787|0.756|0.730|0.780|0.843|0.759|0.813|**0.853**|0.851|
|150_twt_em|0.710|0.612|0.652|0.700|0.730|0.672|0.717|0.746|**0.777**|
|155_athlet|0.995|0.988|0.990|0.970|0.988|0.988|0.960|0.985|**0.996**|
|158_code_C|**1.000**|0.998|1.000|0.999|1.000|1.000|0.999|1.000|0.997|
|162_agnews|**1.000**|**1.000**|**1.000**|1.000|**1.000**|**1.000**|**1.000**|**1.000**|1.000|
|163_agnews|**1.000**|1.000|1.000|1.000|**1.000**|**1.000**|**1.000**|**1.000**|1.000|



## **B.9 SAE Architectural Improvements** 

We test if improvements in SAE architectures have led to improvements in probing performance. We test eight SAE architectures on Gemma-2-2B in all regimes with _k_ = 16 _,_ 128 SAE probes. We plot the performance of SAE probes for all architectures in Fig. B.9 when averaging over all datasets. While there seems to be some improvement with later architectures, we note that there is significant variance in the average across datasets that is not visualized. 

98 



<!-- Start of picture text -->
Normal Setting (k=16) Normal Setting (k=128)<br>Baseline Test AUC = 0.927 Baseline Test AUC = 0.927<br>0.905<br>0.880<br>0.900<br>0.875<br>0.895<br>0.870<br>0.890<br>0.865<br>0.885<br>0.860 0.880<br>Scarcity Setting (k=16) Scarcity Setting (k=128)<br>Baseline Test AUC = 0.821 Baseline Test AUC = 0.821<br>0.795<br>0.805<br>0.790 0.800<br>0.785 0.795<br>0.780 0.790<br>0.775 0.785<br>0.780<br>Imbalance Setting (k=16) Imbalance Setting (k=128)<br>Baseline Test AUC = 0.912 Baseline Test AUC = 0.912<br>0.875<br>0.89<br>0.870<br>0.865<br>0.88<br>0.860<br>0.855 0.87<br>0.850<br>Corrupt Setting (k=16) Corrupt Setting (k=128)<br>Baseline Test AUC = 0.815 Baseline Test AUC = 0.815<br>0.805 0.79<br>0.800<br>0.795 0.78<br>0.790<br>0.77<br>0.785<br>50 100 150 200 250 300 350 400 50 100 150 200 250 300 350 400<br>SAE L0 SAE L0<br>Ood Setting (k=16) Ood Setting (k=128)<br>Baseline Test AUC = 0.795 Baseline Test AUC = 0.795<br>0.78 0.78<br>0.76<br>0.76<br>0.74<br>0.74<br>0.72<br>0.70 0.72<br>50 100 150 200 250 300 350 400 50 100 150 200 250 300 350 400<br>SAE L0 SAE L0<br>ReLU (orig.) ReLU (new) Gated TopK Jump ReLU Batch TopK p-anneal Matryoshka<br>Average Test AUC Average Test AUC<br>Average Test AUC Average Test AUC<br>Average Test AUC Average Test AUC<br>Average Test AUC Average Test AUC<br>Average Test AUC Average Test AUC<br><!-- End of picture text -->

Figure B.9: Across regimes, we see a slight positive trend in probing performance with more recent SAE architectures. 

99 



<!-- Start of picture text -->
ILLUSION  FAIR  ILLUSION  FAIR<br>FAIR  Baseline = Last Both = FAIR  Baseline = Last Both =<br>Both = Last SAE = Pool Quiver(Pool, Last) Both = Last SAE = Pool Quiver(Pool, Last)<br>60.9%<br>60 56.5% 60 56.5%<br>50.0%<br>40 39.1% 34.8% 40<br>19.6%<br>20 20<br>13.0%<br>2.2% 8.7% 4.3% 6.5%<br>0 0<br>Baseline > SAE SAE > Baseline Baseline > SAE SAE > Baseline<br>> 0.005  AUC > 0.005  AUC<br>Win Rate Percentage  Win Rate Percentage<br><!-- End of picture text -->

(a) Multi-token **non-binarized** experiments (the same as Fig. 3.11). 

(b) Multi-token **binzarized** experiments. 

Figure B.10: We run the multi-token experiments from Section 3.5 with and without binarizing aggregated latents. 

## **B.10 Assessing SAE Probe Binarization** 

Gallifant et al. 2025 find that binarizing latents with a threshold improves probe performance. Binarization entails setting a latent equal to 1 if its firing value is greater than a threshold, and setting it equal to 0 otherwise. Gallifant et al. 2025 use a threshold equal to 1. 

Gallifant et al. 2025 performs probing in the multi-token pooled setting, so to use as similar of a setup as possible we repeat our multi-token experiment with binarization and a threshold equal to 1. Following Gallifant et al. 2025, we binarize after the max-pooling aggregation. As shown in in Fig. B.10, we find that binarizing results in worse performance than not-binarizing. 

100 

Table B.3: Description of Latent Variables for Out-of-Distribution Detection. We find that this procedure ranks latents roughly according to their OOD test AUC, even though o1 did not have access to this information. However, the system is not perfect. For example, latent 51330’s natural language description is ranked last and seems unreleated to the task at hand. However, its OOD test AUC is the highest of all latents. We do not investigate this further, but hypothesize that either the latent descriptions require additional tuning, or the test set truly has some spurious quality that this latent identifies. 

|**Latent**|**o1**<br>**rank**|**Latent**<br>**Description**|**Val ID**<br>**AUC**|**Test OOD**<br>**AUC**|
|---|---|---|---|---|
|122774|1|mentions of living rooms or<br>parlors in houses or apart-<br>ments.|1.0|0.6402|
|98707|2|phrases and words related<br>to locations or settings, par-<br>ticularly indoor spaces like<br>rooms, parlors, or living ar-<br>eas, as well as time refer-<br>ences like afternoon or spe-<br>cific times.|0.8698|0.6177|
|100016|3|mentions of rooms or en-<br>closed spaces, particularly<br>when discussing physical lo-<br>cations or settings.|0.6895|0.4736|
|7498|4|locations within a house,<br>particularly upstairs, down-<br>stairs, basement, bathroom,<br>and kitchen areas.|0.7524|0.6206|
|78823|5|words and phrases related<br>to specific places, attrac-<br>tions, and experiences, par-<br>ticularly in the context of<br>travel and tourism.|0.6697|0.4526|
|116246|6|numbers, dates, and numer-<br>ical measurements, particu-<br>larly in scientific or technical<br>contexts.|0.7470|0.5398|
|40168|7|technical terms and compo-<br>nents related to computer<br>systems, engineering, and<br>scientific analysis.|0.9219|0.4513|
|51330|8|objects and materials re-<br>lated to scientific experi-<br>ments and laboratory equip-<br>ment, particularly those in-<br>volving fluids, particles, or<br>microscopic samples.|0.7320|0.6483|



|101|
|---|



Table B.5: **Left** We show the top 5 examples where latent 369585 is active while the CoLA prompt is labeled as grammatical. Most of these sentences are clearly ungrammatical, which illustrates that the CoLA data is corrupted. **Right** We then look at sentences a baseline classifier trained on CoLA marks as ungrammatical when the label is grammatical. We reach the same conclusion - the CoLA data is mislabeled. 

|**Prompt**<br>**(Labeled**<br>**Grammatical)**|**Latent 369585**|**Prompt**<br>**(Labeled**<br>**Grammatical)**|**P(y=0)**|
|---|---|---|---|
|I don’t remember what all|23.09|Rub the cloth on the baby|0.933|
|I said?||torn.||
|Aphrodite said he would|15.87|In front of them happen.|0.926|
|free the animals and free||||
|the animals he will||Who did you give pictures|0.911|
|Gilgamesh wanted to se-|14.92|of to friends of?||
|duce Ishtar, and seduce||||
|Ishtar he did.||An example of these sub-|0.893|
|An example of these sub-|14.85|stances be tobacco.||
|stances be tobacco.||Susan<br>hopes<br>herself<br>to|0.890|
|He will can go|14.43|sleep.||



Table B.6: Logistic regression classifier for 110_aimade_humangpt3’s top activating tokens when run on more than 2.5 million tokens from the Pile. The classifier predominantly activates on punctuation. Only tokens with at least 10 occurences shown. 

|**Token**|**Mean Activation**|**Occurrences**|
|---|---|---|
|<bos>|6.8863|7625|
|!).|6.2529|10|
|Q|6.2271|1436|
|”.|6.0338|144|
|.”|5.9111|975|
|.).|5.7334|24|
|␣|5.5035|17|
|."|5.4455|1057|
|“.|5.4132|319|
|}$.|5.3990|24|



102 

# **Chapter 4** 

# **Decomposing the Dark Matter of Sparse Autoencoders** 

## **4.1 Introduction** 

The ultimate goal for ambitious mechanistic interpretability is to understand neural networks from the bottom up by breaking them down into programs (“circuits") and the variables (“features”) that those programs operate on (Olah 2023). One recent successful technique for finding features in language models has been sparse autoencoders (SAEs), which learn a dictionary of one-dimensional representations that can be sparsely combined to reconstruct model hidden activations (Cunningham et al. 2023b; Bricken, Templeton, et al. 2023). However, as observed by Gao, Tour, et al. 2024b, the scaling behavior of SAE width (number of latents) vs. reconstruction mean squared error (MSE) is best fit by a power law with a constant error term. Gao, Tour, et al. 2024b speculate that this component of SAE error below the asymptote might best be explained by model activations having components with denser structure than simple SAE features (e.g. Gaussian noise). This is a concern for the ambitious agenda because it implies that there are components of model hidden states that are harder for SAEs to learn and which might not be eliminated by simple scaling of SAEs. 

Motivated by this discovery, in this work our goal is to specifically study the SAE error vector itself, and in doing so gain insight into the failures of current SAEs, the dynamics of SAE scaling, and possible distributions of model activations. Thus, our direction differs from the bulk of prior work that seeks to quantify SAE failures, as these mostly focus on downstream benchmarks or simple cross entropy loss (see e.g. Gao, Tour, et al. 2024b; Templeton et al. 2024; Anders and Bloom 2024). The structure of this paper is as follows: 1. In Section 4.4, we introduce the fundamental mystery that we will explore throughout the rest of the paper: SAE errors are shockingly predictable. To the best of our knowledge, we are the first to show that a large fraction of SAE error vectors can be explained with a linear transformation of the input activation, that the norm of SAE error vectors can be accurately predicted by a linear projection of the input activation, and that on a per-token level, error norms of large SAEs are linearly predictable from small SAEs. 

2. In Section 4.5, we investigate the linearly predictable and nonlinearly predictable components of SAE error. We find that although the nonlinear component affects downstream cross 

103 



<!-- Start of picture text -->
0.35 SAE Reconstruction, FVU 0.186 + 2.708W 0.317<br>SAE Reconstruction + Error Prediction, FVU 0.122<br>0.30<br>SAE Pursuit Reconstruction + Error Prediction<br>0.25<br>Absent Features<br>0.20<br>0.15 Linear Error<br>0.10<br>0.05 Nonlinear Error Encoder Error<br>0.00<br>10 5 10 6 10 7<br>SAE Width<br>Fraction Variance Unexplained (FVU)<br><!-- End of picture text -->

Figure 4.1: A breakdown of SAE dark matter for layer 20 Gemma 9B SAEs, with dotted lines assuming that observed trends continue for larger SAEs. See Section 4.4 for how we break down the overall fraction of unexplained variance into absent features, linear error, and nonlinear error. See Section 4.6.1 for further separating encoder error from nonlinear error. 

entropy loss in proportion to its norm, it is qualitatively different from linear error: as compared to linear error, nonlinear error is harder to learn SAEs for and has a norm that is harder to linearly predict from activations, suggesting that it consists of a smaller proportion of not-yet-learned linear features. 

3. In Section 4.6, we investigate methods for reducing nonlinear error. We show that inference time optimization increases the fraction of variance explained by SAEs, but only slightly decreases nonlinear error. Additionally, we show that we can use SAEs trained on previous components to decrease nonlinear error and total SAE error. 

## **4.2 Related Work** 

**Language Model Representation Structure:** The linear representation hypothesis (LRH) (Park, Choe, and Veitch 2023a; Elhage, Hume, et al. 2022) claims that language model hidden states can be decomposed into a sparse sum of linear feature directions. The LRH has seen recent empirical support with _sparse autoencoders_ (Makhzani and Frey 2013; Bricken, Templeton, et al. 2023; Cunningham et al. 2023b), which have succeeded in decomposing much of the variance of language model hidden states into such a sparse sum, as well as a long line of work that has used probing and dimensionality reduction to find causal linear representations for specific concepts (Alain 2016; Nanda, A. Lee, and Wattenberg 2023; Marks, Rager, et al. 2024; Gurnee 2024). On the other hand, some recent work has questioned whether the linear representation hypothesis is true: Engels, Liao, et al. 2024 find multidimensional circular representations in Mistral (A. Q. Jiang et al. 2023) and Llama (AI@Meta 2024), and Csordás 

104 

et al. 2024 examine synthetic recurrent neural networks and find “onion-like” non-linear features not contained in a linear subspace. This has inspired recent discussion about what a true model of activation space might be: Mendel 2024 argues that the linear representation hypothesis ignores the growing body of results showing the multi-dimensional structure of SAE latents, and L. Smith 2024b argues that we only have evidence for a “weak” form of the superposition hypothesis holding that only _some_ features are linearly represented (such a hypothesis has also been studied in the sparse coding literature (Tasissa, E. Theodosis, et al. 2020)). 

**SAE Errors and Benchmarking:** Multiple works have introduced techniques to benchmark SAEs and characterize their error: Bricken, Templeton, et al. 2023, Gao, Tour, et al. 2024b, and Templeton et al. 2024 use manual human analysis of features, automated interpretability, downstream cross entropy loss when SAE reconstructions are inserted back into the model, and feature geometry visualizations; Karvonen, Wright, et al. 2024 use the setting of board games, where the ground truth features are known, to determine what proportion of the true features SAEs learn; and Anders and Bloom 2024 use the performance of the model on NLP benchmarks when the SAE reconstruction is inserted back into the model. More specifically relevant to our main direction in this paper, Gurnee 2024 finds that SAE reconstruction errors are _pathological_ , that is, when SAE reconstructions are inserted into the model, they have a larger effect on cross entropy loss than random perturbations with the same error norm. Follow up work by Heimersheim and Mendel 2024 and D. Lee and Heimersheim 2024 find that this effect disappears when the random baseline is replaced by a perturbation in the direction of the difference between two random activations. 

**SAE Scaling Laws:** T. C. T. Anthropic 2024, Templeton et al. 2024, and Gao, Tour, et al. 2024b study how SAE MSE scales with respect to FLOPS, sparsity, and SAE width, and define scaling laws with respect to these quantities. Templeton et al. 2024 also study how specific groups of language features like chemical elements, cities, animals, and foods are learned by SAEs, and show that SAEs predictably learn these features in terms of their occurrence. Finally, Bart Bussmann, Patrick Leask, Bloom, et al. 2024 find that larger SAEs learn two new types of dictionary vectors as comparsed to smaller SAEs: features not present at all in smaller SAEs, and more fine-grained “feature split” versions of features in smaller SAEs. 

## **4.3 Notation** 

We consider neural network activations **x** _∈_ R<sup>_d_</sup> and sparse autoencoders `Sae` _∈_ R<sup>_d_</sup> _→_ R<sup>_d_</sup> . Sparse autoencoders map inputs to a latent space R<sup>_m_</sup> with _m ≫ d_ and then back into R<sup>_d_</sup> , while also requiring that only a sparse set of the _m_ latent dimensions are nonzero. SAEs have the general architecture 



105 

where _σ_ is an architecture dependent activation function, and seek to minimize 



where _S_ is an architecture dependent sparsity function. For convenience, we define `SaeError` ( **x** ) as the reconstruction error of the SAE and _L_ 0 as the average number of nonzeros in `hidden` ( **x** ): 





In this work, we study the relationship between **x** and `SaeError` ( **x** ) and are agnostic to the specifics of SAE architecture. Thus, we examine both TopK SAEs (Gao, Tour, et al. 2024b) and JumpRelu SAEs (Rajamanoharan, Lieberum, et al. 2024b). The TopK SAE is defined as 





Note that _L_ 0 = _k_ for a TopK SAE. The JumpRelu SAE is defined as 





where _θ_ is a learned bias vector and _λ_ is a sparsity parameter. 

## **4.4 Predicting SAE Error** 

In this section, we evaluate the extent to which SAE errors `SaeError` ( **x** ) can be predicted from input model activations **x** . 

We run experiments<sup>1</sup> on Gemma 2 2B and 9B (Gemma Team et al. 2024) and Llama 3.1 8B (AI@Meta 2024). We use the suite of Gemma Scope (Lieberum et al. 2024b) sparse autoencoders for Gemma 2 2B experiments and Llama Scope (Z. He, Shu, Ge, L. Chen, J. Wang, Zhou, F. Liu, Q. Guo, X. Huang, Zuxuan Wu, et al. 2024a) for Llama 3.1 8B. For experiments where we analyze the effect of _L_ 0 and SAE width, we focus on layer 12 of Gemma 2 2B and layer 20 of Gemma 9 9B in the main text as these layers have the most SAEs in Gemma Scope, and we include additional layers in the appendix (we do not run on Llama Scope for these experiments, as Llama Scope has just one SAE _L_ 0 and two SAE widths per layer). For experiments where we analyze the effect of layer, we show the set of SAEs with _L_ 0 closest to a target _L_ 0 across all layers for both Gemma Scope and Llama Scope suites of SAEs. 

We use 300 contexts of 1024 tokens from the uncopywrited subset of the Pile (Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, H. He, Thite, Nabeshima, et al. 2020a) and 

> 1Code at https://anonymous.4open.science/r/SAE-Dark-Matter-1163 

106 



<!-- Start of picture text -->
R 2  predicting error norm R 2  predicting error vector FVUnonlinear<br>0.144<br>0.920 0.60<br>0.136<br>0.904 0.55<br>0.128<br>0.888 0.50<br>10 2 0.872 0.45 0.120<br>0.112<br>0.856 0.40<br>0.104<br>0.840 0.35<br>0.096<br>0.824 0.30<br>0.088<br>10 1 0.808 0.25 0.080<br>10 5 10 6 10 5 10 6 10 5 10 6<br>SAE Width SAE Width SAE Width<br>(a) Layer 12 Gemma 2 2B SAEs from Gemma Scope. FVUnonlinearnonlinear is roughly constant for a fixed L 0..<br>R 2  predicting error norm R 2  predicting error vector FVUnonlinear<br>0.156<br>0.935 0.69<br>0.148<br>0.925 0.63<br>0.140<br>10 2 0.915 0.57<br>0.132<br>0.905 0.51<br>0.124<br>0.895 0.45<br>0.116<br>0.885 0.39<br>0.108<br>10 1 0.875 0.33<br>0.100<br>10 5 10 6 10 5 10 6 10 5 10 6<br>SAE Width SAE Width SAE Width<br>SAE L0<br>SAE L0<br><!-- End of picture text -->

(a) Layer 12 Gemma 2 2B SAEs from Gemma Scope. `FVUnonlinearnonlinear` is roughly constant for a fixed _L_ 0.. 

(b) Layer 20 Gemma 2 9B SAEs from Gemma Scope. `FVUnonlinear` is roughly constant for a fixed _L_ 0. 

Figure 4.2: **Left** : _R_<sup>2</sup> of predicting SAE error norms (see Eq. (4.10), **Mid** : _R_<sup>2</sup> of predicting SAE error vectors (see Eq. (4.11), **Right:** 1 _− R_<sup>2</sup> of predicting model activations given the SAE reconstruction and the SAE error vector prediction. We do a linear interpolation between SAEs (black dots) 

then filter to only activations of tokens after position 200 in each context, as Lieberum et al. 2024b find that earlier tokens are easier for sparse autoencoders to reconstruct, and we wish to ignore the effect of token position on our results. This results in a dataset of about 247k activations. For linear regressions, we use a random subset of size 150k as training examples (since all models have a dimension of less than 5000, this prevents overfitting) and report the _R_<sup>2</sup> on the other 97k activations. For linear transformations to a multi-dimensional output, we report the average _R_<sup>2</sup> across dimensions. We include bias terms in our linear regressions but omit them from equations for simplicity. 

### **4.4.1 Predicting SAE Error Norm** 

For our first set of experiments, we find the optimal linear probe **_a_**<sup>_∗_</sup> from **x** to _∥_ `SaeError` ( **x** ) _∥_ 2<sup>2.</sup> Formally (with a slight abuse of notation, since **x** is a random variable and not a dataset), 

107 

we solve for 



The _R_<sup>2</sup> of these probes are all extremely high: across all combinations of SAE width, _L_ 0, layer, and model, between 70% and 95% of the variance in SAE error norm is explained by the optimal linear probe. 

**Results across SAE width and** _L_ 0 **:** We plot the _R_<sup>2</sup> of probes from Eq. (4.10) across SAE width and SAE _L_ 0 for layer 9 of Gemma 2 2B and layer 20 of Gemma 2 9B as a contour plot (on the left) in Fig. 4.2. We also include plots for additional layers in Section C.1. Overall, sparser and wider SAEs have less predictable error norms, although interestingly for some layers, extremely dense SAEs have less predictable error norm as well. 

**Results across layers and models:** In Fig. 4.3 we plot the _R_<sup>2</sup> of probes from Eq. (4.10) for the SAEs with _L_ 0 closest to 50 on Gemma 2 2B and 9B and Llama Scope SAEs (which all have _k_ = _L_ 0 = 50). We find that the _R_<sup>2</sup> is low for the first few layers, increases rapidly to a value of about 0 _._ 8 to 0 _._ 95, remains at this level with a slight dip for mid-late layers, and then drops off steeply in the last few layers. In Section C.1, we compare the _R_<sup>2</sup> of this approach across layers versus other baselines like token identity; we find that using activations does much better, and thus it is not “easy” to predict error norm. 

### **4.4.2 Predicting SAE Error Vectors** 

We next examine the _R_<sup>2</sup> of the optimal linear transform **_b_**<sup>_∗_</sup> from **x** to `SaeError` ( **x** ): 



As we show in Fig. 4.2 (middle column) and in Section C.1 for additional layers, the _R_<sup>2</sup> of these transforms varies widely, between 15% and 72%; this is less than the _R_<sup>2</sup> for our norm prediction experiments, but still much higher than we might expect. Intuitively, this result implies that there are large linear subspaces that the SAE is mostly failing to learn. There is also a clear pattern across SAE _L_ 0 and width: _R_<sup>2</sup> decreases with increasing SAE width and _L_ 0. Interestingly, this pattern is _not_ the same as it was above for SAE error norm: the _R_<sup>2</sup> of error norm predictions increases with SAE _L_ 0, while it decreases for error vector predictions. One concern might be that **_b_**<sup>_∗_</sup> is mostly reversing feature shrinkage; in Section C.1.1, we show that this is not the case. 

### **4.4.3 Nonlinear FVU and Breaking Down SAE Dark Matter** 

Another related metric we are interested in is the total amount of the original activation **x** we fail to explain using _both_ `SAE` and a linear projection of **x** . That is, assuming we have found **_b_**<sup>_∗_</sup> as in Eq. (4.11), we are interested in the fraction of variance unexplained (FVU) by the sum of `SAE` and **_b_**<sup>_∗_</sup> _·_ **x** : 



108 



<!-- Start of picture text -->
1.0 1.00<br>16k<br>0.9<br>32k 0.95<br>0.8 65k<br>0.90<br>0.7 llama_3.1_8b 32k 131k<br>llama_3.1_8b 131k<br>262k<br>0.6 gemma_2_2b 16k 0.85<br>gemma_2_2b 65k 524k<br>0.5 gemma_2_9b 16k<br>gemma_2_9b 131k 1m 0.80<br>0.4<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Layer by depth fraction in model<br>SAE Width (y)<br>16k 32k 65k 131k262k524k 1m<br>SAE Width (x)<br>2 predicting error normR<br><!-- End of picture text -->

Figure 4.3: _R_<sup>2</sup> of SAE error norm predictions (see Eq. (4.10)) for Gemma Scope SAEs of width _L_ 0 _≈_ 50 and Llama Scope SAEs of width _L_ 0 = _k_ = 50. 

Figure 4.4: _R_<sup>2</sup> for linear probes of per token SAE errors of larger SAEs from smaller SAEs. Prediction accuracy decreases as the SAEs get farther apart in scale, but overall remains high. 

We label this quantity `FVUnonlinear` because it is intuitively the amount of the SAE’s unexplained variance that is not a linear projection of the input. Interestingly, we find that for middle layers (layer 12 Gemma 2 2B and layer 20 Gemma 2 9B) at a fixed _L_ 0, `FVUnonlinear` is approximately constant (see Fig. 4.2), while for other layers it decreases slightly as _L_ 0 increases (see Section C.1). That is, even though we can linearly predict a smaller portion of the error vector in larger SAEs, this effect is counteracted by the fact that the SAE error vector itself is getting smaller. In contrast, `FVUnonlinear` decreases as SAE _L_ 0 increases. 

As stated above, for middle layers we observe that `FVUnonlinear` is roughly constant at a fixed sparsity. Because we observe this across multiple orders of magnitude of SAE scale, we hypothesize that this will continue to hold as we scale the SAE. Using this assumption, we plot a horizontal fit for `FVUnonlinear` and a power law fit with a constant for Gemma SAE reconstructions as SAE width scales (choosing the SAEs closest to _L_ 0 _≈_ 60) in Fig. 4.1. The power law fit asymptotes above the horizontal fit, which implies the presence of linear error even at very large SAE width. The absent features component of the error in Fig. 4.1 comes from the following observation: if our hypothesis is correct that nonlinear error is roughly constant as the layer 20 SAEs scale, then the nonlinear features must reside in the linear error component, since this is the only component that decreases as the SAE scales and learns more features. We investigate this hypothesis further in Section C.2.2. 

### **4.4.4 Predicting SAE Per-Token Error Norms** 

We now examine per-token SAE scaling behavior. Given two SAEs, `SAE` 1 and `SAE` 2, we are interested in how much of the variance in error norms in `SAE` 2 is predictable from error norms 

109 



<!-- Start of picture text -->
16k SAE error 65k SAE error 262k SAE error 1m SAE error<br>32k SAE error 131k SAE error 524k SAE error<br>115<br>95<br>75<br>75<br>55<br>0 20 40 60 80 100<br>Token<br>' This'' article'' links'' hypothe''si' 'sed'' issues'' about'' impact'' evaluation'','<br>L2 Norm<br><!-- End of picture text -->

Figure 4.5: Per token scaling with average nonlinear error, layer 20 Gemma 9B SAEs from Gemma Scope closest to _L_ 0 = 60. 

in `SAE` 1. That is, we want to find **_c_**<sup>_∗_</sup> such that 



Note that in practice, although **_a_**<sup>_∗_</sup> also can predict the norm of SAE error, it requires training the target SAE to learn a probe. Here, on the other hand, although we formulate finding **_c_**<sup>_∗_</sup> as an optimization problem that requires a larger SAE, in practice we do not need to actually train the larger SAE to get interesting insights: since **_c_**<sup>_∗_</sup> has just one component, it simply measures how well small SAE error can be multiplied by a scalar to predict large SAE error. If the _R_<sup>2</sup> is high, we know that on tokens that small SAEs perform poorly on, larger SAEs will as well. In Fig. 4.4, we plot the _R_<sup>2</sup> of **_c_**<sup>_∗_</sup> probes on all pairs of Gemma Scope 9B layer 20 SAEs with _L_ 0 _≈_ 60 (restricting to pairs where `SAE` 2 is larger than `SAE` 2), and find that indeed, per token SAE errors are highly predictable. Additionally, we show concretely what these correlated SAE error norms looks like on a set of 100 tokens from the Pile in Fig. 4.5. 

## **4.5 Analyzing Components of SAE Error** 

In previous sections, we broke down the SAE error vector into a linearly predictable component and a non-linearly predictable component. A reasonable question is whether these error subsets meaningfully differ. Thus, in this section, we run experiments on these components to determine how the linear and non-linear components of `SaeError` ( **x** ) differ. For convenience, 

110 

given a probe **_b_**<sup>_∗_</sup> from Eq. (4.11), we write 

`LinearError` ( **x** ) := **_b_**<sup>_∗_</sup> _·_ **x** (4.14) `NonlinearError` ( **x** ) := `SaeError` ( **x** ) _−_ `LinearError` ( **x** ) = `SaeError` ( **x** ) _−_ **_b_**<sup>_∗_</sup> (4.15) 

In Section C.2.2 we discuss a hypothesis for SAE feature activations that might explain _why_ we observe these differences; this appendix section is not necessary for understanding the experiments in this section, and we will refer to it in the few cases where it provides additional intuition. 

We run the following experiments in this section to understand how the linear and nonlinear components of SAE error differ: 

1. In Section 4.5.1, we run the norm prediction test from Eq. (4.10) on different components of the error. We find that the norm of the nonlinear component is less predictable from activations than other components, implying that it might consist of fewer not-yet-learned features. 

2. We train new SAEs directly on the linear and nonlinear components of error. The SAE trained on nonlinear error converges to higher reconstruction loss and produces less interpretable features. 

3. We examine how much each component contributes to downstream model performance by intervening in the forward pass, and find that both components contribute proportionally to their size to increased cross entropy loss. 

Overall, these experiments suggest that our split of SAE dark matter into these two categories is indeed a meaningful one. 

### **4.5.1 Applying the Norm Prediction Test:** 

For our first experiment, we run the norm prediction test from Eq. (4.10) on five different random vectors: **x** , `SAE` , `SaeError` ( **x** ), `LinearError` ( **x** ), and `NonlinearError` ( **x** ). The results are shown as a violin plot for each component across all 329 SAEs we experiment with in Section 4.4 (layer 12 Gemma 2B SAEs, layer 20 Gemma 9B SAEs, all Gemma Scope SAEs with _L_ 0 closest to 60 across layers, and all Llama Scope SAEs) in Fig. 4.6 (the `SAE` bar can be seen as a summary of Fig. 4.2). There are one or two SAE with an outlier _R_<sup>2</sup> equal to or lower than 0; we omit these from the plot because they are likely due to numeric instability in the linear regression routine. 

The intuition behind this test is that if a random vector consists of a sum of almostorthogonal linear features from **x** , then its norm should be linearly predictable from **x** (see Section C.2 for more on this intuition). 

Firstly, we note that _∥_ **x** _∥_<sup>2</sup> 2<sup>canalmostbeperfectlypredictedfrom</sup><sup>**x**.Thisisreassuring</sup> news for the linear representation hypothesis, as it implies that **x** may be well modeled as the sum of many one-dimensional features, at least from the perspective of this test. We also find that `NonlinearError` ( **x** ) (and `SaeError` ( **x** ), which consists partly of `NonlinearError` ( **x** )) has a notably lower score on this test than `LinearError` ( **x** ). This result suggests that `NonlinearError` ( **x** ) does not consist of a sparse sum of linear features from **x** . 

111 



<!-- Start of picture text -->
1.0 0.8 SAE trained on nonlinear error<br>SAE trained on linear error<br>0.9<br>0.7<br>0.8<br>0.6<br>0.7<br>0.5<br>0.6<br>0.4<br>0.3<br>Decile<br>x Sae(x)<br>SaeError(x) Not Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10<br>LinearErrorNonlinearError(x) (x)<br>Accuracy<br>2Norm Prediction Test R<br><!-- End of picture text -->

Figure 4.6: Violin plot of norm prediction tests for all SAEs (layer 12 Gemma 2B SAEs, layer 20 Gemma 9B SAEs, all Gemma Scope SAEs with _L_ 0 closest to 60 across layers, and all Llama Scope SAEs). We plot the _R_<sup>2</sup> of a linear regression from **x** to to each random vector’s norm squared. 

Figure 4.7: Auto-interpretability results on SAEs trained on the linear and nonlinear components of `SaeError` ( **x** ) on a width 16k, _L_ 0 _≈_ 60, layer 20 Gemma Scope SAE. “Not” represents contexts that the SAE latent did not activate on, while each _Qi_ represents activating examples from decile _i_ . 

### **4.5.2 Training SAEs on** `SaeError` **(x) Components:** 

Another empirical test we run is training an SAE on `NonlinearError` ( **x** ) and `LinearError` ( **x** ). We choose a fixed Gemma 9B Gemma Scope layer 20 SAE with 16 _k_ latents and _L_ 0 _≈_ 60 to generate `SaeError` ( **x** ) from. This SAE has nonlinear and linear components of the error approximately equal in norm and _R_<sup>2</sup> of the total `SaeError` ( **x** ) they explain, so it presents a fair comparison. We train SAEs to convergence (about 100M tokens) on each of these components of error and find that the SAE trained on `NonlinearError` ( **x** ) converges to a fraction of variance unexplained an absolute 5 percent higher than the SAE trained on the linear component of SAE error ( _≈_ 0 _._ 59 and _≈_ 0 _._ 54 respectively). 

We additionally examine the _interpretability_ of the learned SAE latents using automated interpretability (this technique was first proposed by Bills et al. 2023 for interpreting neurons, and first applied to SAEs by Cunningham et al. 2023b). Specifically, we use the implementation introduced by Juang et al. 2024, where a language model (we use Llama 3.1 70b (AI@Meta 2024)) is given top activating examples to generate an explanation, and then must use only that explanation to predict if the feature fires on a test context. Our results in Fig. 4.7 show that the SAE trained on linear error produces latents that are about an absolute 5% more interpretable across all activation firing deciles (we average results across 1000 random features for both SAEs, where for each feature use 7 examples in each of the 10 feature activation deciles as well as 50 negative examples, and show 95% confidence intervals). 

112 



<!-- Start of picture text -->
R 2 Reconstruction + Linear Error<br>Norm / Total SAE Error Norm Reconstruction + Nonlinear Error<br>L0   60 Width 16k<br>100<br>75<br>50<br>25<br>0<br>L0<br>SAE Width<br>16k 32k 65k 131k 262k 524k 1m 11 20 36 68 138 310 427<br>Replacing Layer (%)<br>Percent Recovered When<br><!-- End of picture text -->

Figure 4.8: Results of intervening in the forward pass and replacing **x** with _SAE_ ( **x** ) + _NonlinearError_ ( **x** ) and _SAE_ ( **x** ) + _LinearError_ ( **x** ) during the forward pass on all layer 20 Gemma 9B SAEs with _L_ 0 _≈_ 60 or width of 16k. 

### **4.5.3 Downstream Cross Entropy Loss of** `SaeError` **(x) Components:** 

A common metric used to test SAEs is the percent of cross entropy loss recovered when the SAE reconstruction is inserted into the model in place of the original activation versus an ablation baseline (see e.g. Bloom 2024). We modify this test to specifically examine the different components of `SaeError` ( **x** ): we compare the percent of the cross entropy loss recovered when replacing **x** with `SAE` plus either `LinearError` ( **x** ) or `NonlinearError` ( **x** ) to the baseline of inserting just `SAE` in place of **x** on all layer 20 Gemma 9B SAEs with _L_ 0 _≈_ 60 or width of 16k. To estimate how much each component “should” recover, we use two metrics: the average norm of the component relative to the total norm of `SAE` and the percent of the variance that the component recovers between `SAE` and **x** . The results, shown in Fig. 4.8, show that for the most part these metrics are reasonable predictions for both types of error. That is, both `NonlinearError` ( **x** ) and `LinearError` ( **x** ) proportionally contribute to the SAE’s increase in downstream cross entropy loss, with possibly a slightly higher contribution than expected for `LinearError` ( **x** ). 

## **4.6 Reducing** `NonlinearError` **(x)** 

In past sections we identify `NonlinearError` ( **x** ) and `LinearError` ( **x** ) and argue that `NonlinearError` ( **x** ) likely does not consist of not-yet-learned linear features. This may be a problem for scaling SAEs farther: if a part of SAE error persistently consists of nonlinear transformations of linear features, these may never be able to be learned by SAEs. Thus, in this section, we investigate two approaches for reducing `NonlinearError` ( **x** ): 

1. Using a more powerful encoder at inference time. This approach has limited success, 

113 



<!-- Start of picture text -->
Pursuit SAE Reconstruction, FVU 0.136 + 2.342W 0.272<br>Pursuit SAE Reconstruction + Error Prediction, FVU 0.117<br>0.3<br>0.2<br>Unlearned Features<br>Dense Features / Linear Error<br>0.1<br>Nonlinear Error<br>(from Dictionary and Sparsity)<br>0.0<br>10 5 10 6 10 7<br>SAE Width<br>FVU<br><!-- End of picture text -->

(a) SAE error breakdown vs. SAE width for inference time optimized reconstructions from Gemma Scope _L_ 0 _≈_ 60 dictionaries. 



<!-- Start of picture text -->
Predicting Predicting Predicting<br>Nonlinear Error SAE-Error Linear Error<br>0.60<br>0.49<br>0.40<br>0.34 0.32<br>0.20 0.14 0.16  0.18<br>0.07 0.04 0.06 0.09 0.07 0.10<br>0.00<br>prev residualresidual MLPattention prev residualresidual MLPattention prev residualresidual MLPattention<br>R²<br><!-- End of picture text -->

(b) The _R_<sup>2</sup> of linearly predicting parts of SAE error from the SAE reconstructions of adjacent model components, layer 20 Gemma Scope _L_ 0 _≈_ 60, 16 _k_ SAE width. 

Figure 4.9: Investigations towards reducing nonlinear SAE error. 

reducing total FVU by 3-5% but leaving `FVUnonlinear` almost unchanged 

2. Attempting to predict `NonlinearError` ( **x** ) from SAE reconstructions of adjacent model components - this approach is more successful; on the SAE we test on, `SAE` predicts 50% of the nonlinear error, and previous model components predict between 4% and 14% of nonlinear error, although only between 6% and 10% of the overall SAE error. 

### **4.6.1 Using a More Powerful Encoder** 

Our first approach for reducing nonlinear error is to try improving the encoder. We use a recent approach suggested by L. Smith 2024a: applying a greedy inference time optimization (ITO) algorithm called gradient pursuit to a frozen learned SAE decoder matrix. We implement and run ITO on all layer 20 Gemma Scope 9 _b_ SAEs closest to _L_ 0 _≈_ 60. For each example **x** with reconstruction `SAE` , we use the gradient pursuit implementation with an _L_ 0 exactly equal to the _L_ 0 of **x** in the original `SAE` . 

Using these new reconstructions of **x** , we repeat Eq. (4.11) and do a linear transformation from **x** to the inference time optimized reconstructions. We then regenerate the same scaling plot as Fig. 4.1 and show this figure in Fig. 4.9a. Our first finding is that pursuit indeed decreases the total FVU of `SAE` by 3 to 5%; as L. Smith 2024a only showed an improvement on a small 1 layer model, to the best of our knowledge we are the first to show this result on state of the art SAEs. Our most interesting finding, however, is that the `FVUnonlinear` stays almost constant when compared to the original SAE scaling in Fig. 4.1. We hypothesize that this might happen because ITO reduces “easy” linear errors like feature shrinkage. In Fig. 4.1, we plot the additional reduction in `FVUnonlinear` as the contribution of encoder error; because `FVUnonlinear` stays almost constant, this section is very narrow. 

114 

### **4.6.2 Predicting SAE Errors Between SAEs:** 

For the Gemma 2 architecture at the locations the SAEs are trained on, each residual activation can be decomposed in terms of prior components: 



We focus on _layer_ = 20 and Gemma 2 9B, and thus use the layer 19 attention and MLP Gemma Scope SAEs. For all components we use the width 16 _k_ SAEs with _L_ 0 _≈_ 60. In Fig. 4.9b, we plot the _R_<sup>2</sup> of a regression from the SAE output corresponding to each of these right hand side components to each of the different components of an SAE trained on `Residlayer` ( `SaeError` ( **x** ), `LinearError` ( **x** ), and `NonlinearError` ( **x** )). 

We find that we can explain a small amount (up to _≈_ 10%) of total `SaeError` ( **x** ) using previous components, which may be immediately useful for circuit analysis. We also can explain large parts of `NonlinearError` ( **x** ) with prior components as well. While `SAE` explains 50% of the variance in the nonlinear error, this may be somewhat misleading, as the nonlinear error is partially a function of `SAE` : 



These results mean that we might be able to explain some of the SAE Error using a circuits level view, but that overall there are still large parts of each error component unexplained. 

## **4.7 Conclusion** 

The fact that SAE error can be predicted and analyzed at all is surprising. Thus, our findings are intriguing evidence that SAE errors, and not just SAE reconstructions, are worthy of analysis, and we hope that it inspires further work in decomposing SAE error. 

Concretely, we believe that our study of error has already discovered a number of promising directions for future SAE research. The discovery that SAE errors can be significantly predicted from model activations hints that some subspaces may resist being learned and points toward potential architectural innovations such as incorporating low-rank dense side channels. The predictable relationship between small and large SAE errors may also streamline experimentation with novel SAE architectures by allowing researchers to efficiently forecast scaling behavior through small-scale trials. Additionally, our demonstration that prior SAE outputs can explain part of SAE error has immediate practical implications for circuit analysis, which currently relies on large noise terms that complicate circuit interpretation. 

At a higher level, the presence of constant nonlinear error at a fixed sparsity as we scale implies that scaling SAEs may not be the only (or best) way to explain more of model behavior. Future work might explore alternative penalties besides sparsity or new ways to learn better dictionaries. Ultimately, we believe that there is still room to make SAEs better, not just bigger. 

115 

116 

# **Appendix C** 

# **Appendix: Decomposing the Dark Matter of Sparse Autoencoders** 

## **C.1 Extra Error Prediction Experiments** 

### **C.1.1 Note on Feature Shrinkage** 

Earlier SAE variants were prone to _feature shrinkage_ : the observation that `SAE` systematically undershot **x** . Current state of the art SAE variants (e.g. JumpReLU SAEs and TopK SAEs, which we examine in this work), are less vulnerable to this problem, although we still find that Gemma Scope reconstructions have about a 10% smaller norm than **x** . One potential concern is that the **b**<sup>_∗_</sup> in Eq. (4.11) that we learn is merely predicting this shrinkage. If this was the case, then the cosine similarity of the linear error prediction ( **_b_**<sup>_∗_</sup> )<sup>_T_</sup> _·_ **x** with **x** would be close to 1; however, in practice we find that it is around 0 _._ 5, so **_b_**<sup>_∗_</sup> is indeed doing more than predicting shrinkage. 

### **C.1.2 Additional Error Prediction Plots** 

In Section C.5 and Section C.5, we show the accuracy of SAE error norm predictions, SAE error vector predictions, and `FVUnonlinear` for layers 5 and 19 of Gemma 2 2B and layers 9 and 31 of Gemma 2 9B. As described in the main text, we find broadly similar results for SAE error norm and error vector prediction at these layers, but find that `FVUnonlinear` decreases some as we increase SAE width at a fixed _L_ 0, although this result is uncertain because of the sparsity of Gemma Scope SAEs at these layers. 

### **C.1.3 Norm Prediction Baselines** 

In Fig. C.2, we run a linear regression from different model variables to _∥_ `SaeError` ( **x** ) _∥_ across layers on Gemma Scope 9B for SAEs of size 131k. We find that is is not “easy” to predict SAE error norm, especially at later layers; the token identity, SAE L0, activation norm, and model loss all do significantly worse than using the full activation. It is interesting to note that at the first few layers, token identity does better at predicting SAE error than a probe 

117 



<!-- Start of picture text -->
Norm of Linearly Predictable Error Norm of Non-Linearly Predictable Error<br>100<br>SAE Width<br>90<br>16k<br>80 32k<br>65k<br>70<br>131k<br>60 262k<br>50 524k<br>1m<br>40<br>0 20 40 60 80 100 0 20 40 60 80 100<br>Token Token<br>L2 Norm<br><!-- End of picture text -->

Figure C.1: Per-token breakdown of linearly predictable and non-linearly predictable SAE error across SAE scale. We show the same tokens as in Fig. 4.5. The norm of linear error decreases with SAE width, whereas the norm of nonlinear error stays mostly constant. 

of the activations; this is perhaps not surprising, since recent results from e.g. Lad, Gurnee, and Tegmark 2024 show that very early layers primarily operate on a per token level. 

### **C.1.4 Breaking Apart Error Per Token** 

In Fig. C.1, we show the same subset of tokens as in Fig. 4.5, but now broken apart into linearly predictable and non-linearly predictable components. That is, we learn **_b_**<sup>_∗_</sup> for each SAE as in Eq. (4.11), and then plot the norm of **_b_**<sup>_∗_</sup> _·_ **x** at the norm of the linearly predictable error on the left, and plot the norm of `SaeError` ( **x** ) _−_ **_b_**<sup>_∗_</sup> _·_ **x** as the norm of the non-linearly predictable error on the right. We see that the linearly predictable error decreases as we scale SAE width, but the non-linearly predictable error mostly stays constant. This is especially interesting because the result in Fig. 4.1 just found this on an average level, whereas here we find the same result holds on a per-token level. 

## **C.2 Modeling Activations** 

In this section, we seek to explain why we see a difference between linear and nonlinear error in the main body of the paper. We will adopt the _weak_ linear hypothesis (L. Smith 2024b), a generalization of the linear representation hypothesis which holds only that _some_ features in language models are represented linearly. Thus we have 



for linear features _{_ **_y_** 1 _, . . . ,_ **_y_** _n}_ and random vector **w** _∈_ R<sup>_n_</sup> , where **w** is sparse ( _∥_ **w** _∥_ 1 _≪ d_ ) and `Dense` ( **x** ) is a random vector representing the dense component of **x** . `Dense` ( **x** ) might be Gaussian noise, nonlinear features as described by Csordás et al. 2024, or anything else not represented in a low-dimensional linear subspace. 

Before we proceed with our analysis, we note that there is a rich sparse coding literature studying dictionary learning in the setting with mixed sparse and dense signals. For example, 

118 



<!-- Start of picture text -->
Acts Model Loss Act Norm<br>Tokens L0<br>1.00<br>0.75<br>0.50<br>0.25<br>0.00<br>0 10 20 30 40<br>Layer<br>R^2<br><!-- End of picture text -->

Figure C.2: _R_<sup>2</sup> for linear regressions of SAE error norms with different regressors. We run on Gemma Scope 9B SAEs of size 131 _k_ with _L_ 0 _≈_ 60. Activations perform the best except on the first few layers. 



<!-- Start of picture text -->
(a * ) T feature<br>Running window average of (a * ) T feature<br>Average feature value when active<br>300<br>200<br>100<br>0<br>100<br>200<br>0 2500 5000 7500 10000<br>SAE Latent Index<br>Sorted by Average Active Value<br><!-- End of picture text -->

Figure C.3: Average SAE latent activation and dot product of the latent with the learned norm prediction vector **_a_**<sup>_∗_</sup> for the Gemma Scope layer 20, width 131 _k_ , _L_ 0 = 62 SAE. We also plot a smoothed version of this dot product with a smoothed window of 10. 

in a classic work Candès et al. 2011 propose Robust Principal Component Analysis for decomposing data matrices into sparse and dense components, although this is not directly applicable to our setting of trying to learn an autoencoder. More recently, Tasissa, M. Theodosis, et al. n.d. propose a dictionary learning technique for data exactly modeled as in Eq. (C.1). In our work, we focus on studying existing SAEs and hypothesizing why they fail, so these works are not immediately applicable, but we are excited to see future work that applies these combined sparse-dense autoencoders to language model activations.. 

Say our SAE has _m_ latents. Since by assumption `Dense` ( **x** ) cannot be represented in a low-dimensional linear subspace, the sparsity limited SAE will not be able to learn it. Thus, we will assume that the SAE learns only the _m_ most common features **_y_** 0 _, . . . ,_ **_y_** _m−_ 1. We will also assume that the SAE introduces some error when making this approximation, and instead learns **_y_** ˆ _i_ and wˆ _i_ . Thus we have 





119 

We finally define `Introduced` ( **x** ) :=<sup>�</sup><sup>_m_</sup> _i_ =0<sup>wˆ</sup><sup>_i_ˆ</sup><sup>**_y_**</sup><sup>_i −_�</sup><sup>_m_</sup> _i_ =0<sup>w</sup><sup>_i_</sup><sup>**_y_**</sup><sup>_i_,sowehave</sup> 



### **C.2.1 Analyzing Error Norm Prediction** 

We will first analyze Eq. (4.10), the learned probe from **x** to _∥_ `SaeError` ( **x** ) _∥_ 2<sup>2.First,weclaim</sup> that given a vector **x** , if **x** is a sparse sum of orthogonal vectors, then by basic linear algebra there exists a perfect prediction vector **_a_** such that **_a_**<sup>_T_</sup> **x** _≈|_ **x** _|_<sup>2</sup> 2<sup>(inotherwords,thenorm</sup> squared of **x** can be linearly predicted from **x** ). The proof of this claim is in Section C.3.1; the intuition is that we can set the probe vector **_a_**<sup>_∗_</sup> to the sum of the vectors **_y_** _i_ weighted by their average weight _E_ (w _i_ ). 

When **x** is instead a sparse sum of _non-orthogonal_ vectors, as it partly is in Eq. (C.1) and Eq. (C.4), this proof is no longer true, but we now argue that a similar intuition holds. If the **_y_** _i_ are almost orthogonal (formally _coherence < ϵ_ , see (Foucart and Rauhut 2013)) and do not activate much at the same time, then a probe vector again equal to the sum of vectors **_y_** _i_ weighted by their average value _E_ (w _i_ ) will be a good approximate prediction. Indeed, when we try predicting _∥_ `SAE` _∥_<sup>2</sup> 2<sup>from</sup><sup>`SAE`(whichisasparsesumofknownalmost</sup> orthogonal vectors of a similar distribution to the true SAE vectors), we find that indeed the linear probe that is learned is approximately equal to this sum (see Section C.3.2). 

Thus, we can now neatly explain why we can predict the norms of SAE errors: they mostly consist of almost orthogonal sparsely occuring not yet learned SAE features! We further can explain why larger SAEs have less predictable error norms: since _m_ is larger, there is a larger component in the error of not-as-linearly-predictable `Dense` ( **x** ) and `Introduced` ( **x** ). 

### **C.2.2 Analyzing Error Vector Prediction** 

We will now analyze Eq. (4.11), the learned transformation from **x** to `SaeError` ( **x** ), with our model of SAE error from Eq. (C.4). We assume that `Introduced` ( **x** ) cannot be approximated at all as a linear function of **x** . This is a reasonable assumption since the SAE is a nonlinear function of **x** , but if indeed some of `Introduced` ( **x** ) can be approximated in this way, then we will underestimate the amount of `Introduced` ( **x** ) and therefore overestimate the amount of `Dense` ( **x** ). 

If `Dense` ( **x** ) +<sup>�</sup><sup>_n_</sup> _i_ = _m_<sup>w</sup><sup>_i_</sup><sup>**_y_**</sup><sup>_i_iscontainedinalinearsubspaceof</sup><sup>**x**orthogonalto�</sup><sup>_m_</sup> _i_ =0<sup>w</sup><sup>_i_</sup><sup>**_y_**</sup><sup>_i_,</sup> then `Dense` ( **x** ) is part of the linearly explainable error, and the error of the transformation **_b_**<sup>_∗_</sup> exactly equals `Introduced` ( **x** ) (since the transformation is just exactly this orthogonal linear subspace). However, if such an orthogonal linear subspace does not exist, the optimal linear transform will only be able to reconstruct part of `Dense` ( **x** ) +<sup>�</sup><sup>_n_</sup> _i_ = _m_<sup>w</sup><sup>_i_</sup><sup>**_y_**</sup><sup>_i_,andthepercent</sup> of variance left unexplained by the regression will be an upper bound on the true variance explained by `Introduced` ( **x** ). We also note that if this linear transform indeed recovers `Dense` ( **x** ) and absent features but fails to recover `Introduced` ( **x** ), we can use it to estimate `Dense` ( **x** ): the difference between the variance explained by `SAE` and the variance explained by **x** _−_ ( `SAE` + **b**<sup>_∗_</sup> _·_ **x** ) will approach `Dense` ( **x** ) as _m →∞_ . 

120 

Thus, our ability to estimate `Introduced` ( **x** ) and `Dense` ( **x** ) using **_b_**<sup>_∗_</sup> depends on how well a linear transform works to predict `Dense` ( **x** ) and<sup>�</sup><sup>_n_</sup> _i_ = _m_<sup>w</sup><sup>_i_</sup><sup>**y**</sup><sup>_i_.Althoughwedonothave</sup> access to the ground truth vectors **_y_** _i_ , we _can_ replace **x**<sup>_′_</sup> with a similar distribution of vectors that we _do_ have access to, using the same trick as above. Given an SAE, we replace **x** with **x**<sup>_′_</sup> = `SAE` . **x**<sup>_′_</sup> has the useful property that it is a sparse linear sum of vectors (the ones that the SAE learned), and the distribution of these vectors and their weights are similar to that of the true features **_y_** _i_ . We now pass **x**<sup>_′_</sup> back through the SAE and can control all of the quantities we are interested in: we can vary _m_ by masking SAE dictionary elements, simulate `Dense(` **x**<sup>_′_</sup> `)` by adding Gaussian noise to **x**<sup>_′_</sup> , and simulate `Introduced` ( **x** ) by adding Gaussian noise to `Sae` ( **x**<sup>_′_</sup> ). 

We run this synthetic setup Table C.1: Correlation matrix between synthetic noise with a Gemma Scope layer 20 and estimated errors. SAE (width 16 _k_ , _L_ 0 _≈_ 68) in Section C.3.3, and find that inEstimated Estimated deed, estimated `Dense(` **x**<sup>_′_</sup> `)` is highly `Dense(` **x**<sup>_′_</sup> `) Introduced(` **x**<sup>_′_</sup> `)` correlated with the amount of **x**<sup>_′_</sup> Noise 0.9842 0.1417 Gaussian noise added to **x**<sup>_′_</sup> and `Sae` <u>(</u> **x**<sup>_′_</sup> <u>)</u> Noise 0.0988 0.9036 `Introduced(` **x**<sup>_′_</sup> `)` is highly correlated with the amount of Gaussian noise added to `Sae` ( **x**<sup>_′_</sup> ) (see Table C.1). However, note that because **x**<sup>_′_</sup> noise is also slightly correlated with estimated `Introduced(` **x**<sup>_′_</sup> `)` , it is possible that some of the contribution to the estimated nonlinear error is from `Dense(` **x**<sup>_′_</sup> `)` . 

Thus, we again now have a potential explanation for our initial results: we can predict error vectors because they consist in large part of not yet learned linear features in an almost orthogonal subspace of **x** , we can predict a smaller portion of larger SAE errors because the number of these linear features go down with SAE width, and the horizontal line in Fig. 4.1 is because `Dense` ( **x** ) and `Introduced` ( **x** ) are mostly constant. Furthermore, we can hypothesize from the correlations om Table C.1 that the linearly predictable component of SAE error consists mostly of not yet learned features and `Dense` ( **x** ), while the component that is not linearly predictable consists mostly of `Introduced` ( **x** ). We will explore this hypothesis in Section 4.5. 

### **C.2.3 Analyzing Per-Token Scaling Predictions** 

Finally, we provide a simple explanation for why per-token SAE errors are highly predictable between SAEs of different sizes. For this, we only need Eq. (C.4). Since `Dense` ( **x** ) and `Introduced` ( **x** ) stay mostly constant as _m_ increases, for large _m_ the SAE error stays mostly constant because it is primarily determined by these components. Thus, since _m_ = 16 _k_ is already large, a linear prediction that is just a slightly smaller version of the current error performs well. Additionally, this reasoning suggests a natural experiment: if we can predict `Introduced` ( **x** ) on a per-token level (which we hypothesize we can do with the non-linearly predictable component of SAE error), we may be able to better predict the floor of SAE scaling and therefore better predict larger SAE errors; we run this experiment in Section C.5, where we find an affirmative answer. 

121 

## **C.3 More Info on Modeling Activations** 

### **C.3.1 Proof of Claim from Section C.2.1** 

Say we have a set of _m_ unit vectors **_y_** 1 _,_ **_y_** 2 _, . . . ,_ **_y_** _m ∈_ R<sup>_d_</sup> . We will call these “feature vectors”. Define **_Y_** _∈_ R<sup>_d×m_</sup> as the matrix with the feature vectors as columns. We then define the Gram matrix **GY** _∈_ R<sup>_m×m_</sup> of dot products on **Y** : 



We now will define a random column vector **x** that is a weighted positive sum of the _m_ feature vectors, that is, **x** =<sup>�</sup> _i_<sup>w</sup><sup>_i_</sup><sup>**_y_**</sup><sup>_i_foranon-negativerandomvector</sup><sup>**w**</sup><sup>_∈_R</sup><sup>_m_.Wesayfeature</sup> vector **_y_** _i_ is active if w _i >_ 0. We now define the autocorrelation matrix **Rw** _∈_ R<sup>_m×m_</sup> for **w** as 



We are interested in breaking down **x** into its components, so we define a random matrix **X** as **X** _ij_ = _wj_ **Y** _ij_ , i.e. the columns of **_Y_** multiplied by **w** . We can now define the Gram matrix **GX** _∈_ R<sup>_m×m_</sup> : 



where _⊙_ denotes Schur (elementwise) multiplication. The intuition here is that the expected dot product between columns of **X** depends on the dot product between the corresponding columns of **Y** and the correlation of the corresponding elements of the random vector. 

We will now examine the L2 norm of **x** : 



We can also take the expected value: 



Our goal is to find a direction **_a_** _∈_ R<sup>_d_</sup> that when dotted with **x** predicts _∥_ **x** _∥_<sup>2</sup> 2<sup>.Inother</sup> words, we want to find **a** such that 



Combining equations, we want to find **a** such that 



122 

Let us first consider the simple case where for all _i̸_ = _j_ , _yi_ and _yj_ are perpendicular. Then our goal is to find **_a_** such that 



Since all of the _yi_ are perpendicular, WLOG we can write **_a_** =<sup>�</sup> _i_<sup>_bi_</sup><sup>**_y_**</sup><sup>_i_+</sup><sup>**_c_**foravector</sup> **_c_** _∈_ R<sup>_d_</sup> perpendicular to all **_y_** _i_ and a vector **_b_** _∈_ R<sup>_m_</sup> . Then we have 



Since ordinary least squares produces an unbiased estimator, we know that if we use ordinary least squares to solve for **_b_** , **E** ( **_b_**<sup>_T_</sup> **w** ) = **E** ( **w**<sup>_T_</sup> **w** ). Thus, 



Now that we have _bi_ , we can solve for the correlation coefficient between **_a_**<sup>_T_</sup> **x** = **_b_**<sup>_T_</sup> **w** and _∥_ **x** _∥_ 2<sup>2=</sup><sup>**w**</sup><sup>_T_</sup><sup>**w**.Thisgetsmessywhenusinggeneraldistributions,sowefocusonafewsimple</sup> cases. 

The first is the case where each w _i_ is a scaled independent Bernoulli distribution, so _wi_ is _si_ with probability _pi_ and 0 otherwise. Then _bi_ = _si_ . We also have that **E** ( **w**<sup>_T_</sup> **w** ) = **E** ( **_b_**<sup>_T_</sup> **w** ) =<sup>�</sup> _i_<sup>_s_</sup> _i_<sup>2</sup><sup>_pi_=</sup><sup>_µ_.</sup> 



That is, for Bernoulli variables, **x** =<sup>�</sup> _i_<sup>_si_</sup><sup>**_y_**</sup><sup>_i_isaperfectregressionvector.</sup> 

The second is the case when each w _i_ is an independent Poisson distribution with parameter _λi_ . Then **E** (w _i_ ) = _λi_ and **E** (w<sup>2</sup> _i_<sup>)=</sup><sup>_λ_2</sup> _i_<sup>+</sup><sup>_λi_,so</sup><sup>_bi_=</sup><sup>_λi_+ 1.Wealsohavethat</sup><sup>**E**(</sup><sup>**w**</sup><sup>_T_</sup><sup>**w**)=</sup> **E** ( **_b_**<sup>_T_</sup> **w** ) =<sup>�</sup> _i_<sup>_λ_</sup> _i_<sup>2+</sup><sup>_λi_=</sup><sup>_µ_.Finally,wewillusethefactthat</sup><sup>**E**(w3</sup> _i_<sup>)=</sup><sup>_λ_3</sup> _i_<sup>+ 3</sup><sup>_λ_2</sup> _i_<sup>+</sup><sup>_λi_and</sup> **E** (w<sup>4</sup> _i_<sup>) =</sup><sup>_λ_4 + 6</sup><sup>_λ_3 + 7</sup><sup>_λ_2 +</sup><sup>_λ_.Thenviaalgebrawehavethat</sup> 



For the special case _λi_ = 1, we then have 



123 

### **C.3.2 Empirical Norm Prediction** 

In this experiment, we aim to determine to what extent our analysis in Section C.2.1 holds true in practice on almost orthogonal true SAE features. Thus, we use a random vector that we can control: `SAE` . Specifically, we learn a probe **_a_**<sup>_∗_</sup> for the Gemma Scope layer 20, width 131 _k_ , _L_ 0 = 62 SAE as in Eq. (4.10), except with the regressor equal to `SAE` and the target equal to _∥_ `SAE` _∥_ : 



One important note is that we subtract the bias from `SAE` so that it is purely a sparse sum of SAE features (this makes analysis easier). For each SAE latent from the SAE, we then compute 



Finally, we plot this dot product against the average latent activation in Fig. C.3. If **_a_**<sup>_∗_</sup> indeed equals the sum of the latents weighted by their activation, as we predict in Section C.2.1, then these two quantities should be approximately equal, which we indeed see in the figure. 

### **C.3.3 Synthetic SAE Error Vector Experiments** 

The results for different Gaussian noise amounts versus percentage of features ablated are shown in Fig. C.4. On this distribution of vectors, the test works as expected; the variance explained by `SAE` + **_a_**<sup>_T_</sup> **x** is a horizontal line proportional to `Introduced` ( **x** ), while the gap between this horizontal line and the asymptote of the variance explained by `SAE` is proportional to `Dense` ( **x** ). 

We also tried running this test on a sparse sum of _random_ vectors, which did not work as well, possibly due to not including the structure of the SAE vectors (Giglemiani et al. 2024); see Section C.4 for more details. 

## **C.4 Synthetic Experiments with Random Data** 

For this set of experiments, we generated a random vector **x**<sup>_′_</sup> that was the sum of a power law of 100 _k_ random gaussian vectors in R<sup>4000</sup> with expected _L_ 0 of around 100. To simulate the SAE reconstruction and SAE error, we simply masked a portion of the vectors in the sum of **x**<sup>_′_</sup> . Unlike the more realistic synthetic data case we describe in Section C.3.3, this did not work as expected: even in the case with no noise added to **x**<sup>_′_</sup> or the simulated reconstruction, the variance explained by the sum of the linear estimate of the error plus the reconstructed vectors plotted against the number of features “ablated” formed a parabola (with minimum variance explained in the middle region), as opposed to a straight line as in Fig. C.4. 

We note that this result is not entirely surprising: other works have found that random vectors are a bad synthetic test case for language model activations. For example, in the setting of model sensitivity to perturbations of activations, Giglemiani et al. 2024 found they needed to control for both sparsity and cosine similarity of SAE latents to produce synthetic vectors that mimic SAE latents when perturbed. 

124 



<!-- Start of picture text -->
Sae(x ′ ) Noise Var = 1, x ′  Noise Var = 0.6 Sae(x ′ ) Noise Var = 0.6, x ′  Noise Var = 1<br>Linear Error f(Activation Noise)<br>0.1 0.1 Linear Error<br>f(Activation Noise)<br>Nonlinear Error<br>Nonlinear Error<br>f(Reconstruction Noise)<br>f(Reconstruction Noise)<br>0.0 0.0<br>50% 40% 30% 20% 10% 0% 50% 40% 30% 20% 10% 0%<br>% features ablated % features ablated<br>                                (FVU difference)Est Nonlinear Error Est Linear Error                    (FVU difference)<br>0.0 0.0 0.4<br>1.0<br>1.0 1.0<br>2.0 2.0 0.2<br>0.5<br>3.0 3.0<br>4.0 4.0<br>0.0 1.0 2.0 3.0 4.0 0.0 1.0 2.0 3.0 4.0<br>True Nonlinear Noise Var True Nonlinear Noise Var<br>FVU FVU<br>True Linear Noise Var True Linear Noise Var<br><!-- End of picture text -->

Figure C.4: **Top:** When controlled amounts of noise are added to synthetic data `Sae` ( **x**<sup>_′_</sup> ) and **x**<sup>_′_</sup> , the result is a plot similar to Fig. 4.1. **Bottom:** The nonlinear and linear error estimates (as shown at top) accurately correlate with the amount of noise added. The exact correlation between synthetic added noise and resulting estimated error components across these noise levels are shown in Table C.1 

## **C.5 Using** **_∥_** `NonlinearError` **(x)** **_∥_ to Predict Scaling:** 

Following up on our discussion in Section C.2.3, we are interested in whether `NonlinearError` ( **x** ) can help with predicting SAE per-token error norm scaling, as this might suggest that it contains a larger component of `Introduced` ( **x** ) than `LinearError` ( **x** ). Formally, we solve for 



To evaluate the improvement of **_d_**<sup>_∗_</sup> relative to **_c_**<sup>_∗_</sup> from Eq. (4.13), we report the percent decrease in FVU; see Fig. C.5. We find that using the norm of `NonlinearError` ( **x** ) provides a small but noticeable bump in the ability to predict larger SAE errors of up to a 5% decrease in FVU, validating this hypothesis. 

125 



<!-- Start of picture text -->
16k 6<br>32k<br>65k<br>4<br>131k<br>262k<br>2<br>524k<br>1m<br>. 0<br>SAE Width (y)<br>16k 32k 65k 131k262k524k 1m<br>SAE Width (x)<br><!-- End of picture text -->

Figure C.5: Percent decrease in FVU when additionally using the squared norms of nonlinear error to predict SAE error norm. 

126 



<!-- Start of picture text -->
R 2  predicting error norm R 2  predicting error vector FVUnonlinear<br>0.125<br>0.928<br>0.475<br>0.115<br>0.912<br>0.425<br>10 2 0.896 0.105<br>0.375<br>0.880 0.095<br>0.325<br>0.864 0.085<br>0.848 0.275 0.075<br>0.832 0.225<br>0.065<br>10 1 0.816 0.175<br>0.055<br>10 5 10 6 10 5 10 6 10 5 10 6<br>SAE Width SAE Width SAE Width<br>(a) Linear prediction results for layer 5 Gemma 2 2B SAEs from Gemma Scope.<br>R 2  predicting error norm R 2  predicting error vector FVUnonlinear<br>0.145<br>0.87 0.475<br>0.135<br>0.425<br>0.84<br>0.125<br>10 2 0.375<br>0.81 0.115<br>0.325<br>0.78 0.105<br>0.275<br>0.095<br>0.75<br>0.225<br>0.085<br>0.72 0.175<br>0.075<br>10 5 10 6 10 5 10 6 10 5 10 6<br>SAE Width SAE Width SAE Width<br>SAE L0<br>SAE L0<br><!-- End of picture text -->



<!-- Start of picture text -->
(b) Linear prediction results for layer 19 Gemma 2 9B SAEs from Gemma Scope.<br><!-- End of picture text -->

Figure C.6: Results of linearly predicting SAE error norm and SAE error from model activations on Gemma 2 2B layer 5 ( **top** ) and Gemma 2 2B layer 19 ( **bottom** ). The right plots show the _R_<sup>2</sup> of predicting SAE error norms (see Eq. (4.10), the middle plots show the _R_<sup>2</sup> of predicting SAE error vectors (see Eq. (4.11), and the right plots show 1 _− R_<sup>2</sup> of predicting model activations given the SAE reconstruction and the SAE error vector prediction. Unlike for the middle layers shown in the main body, `FVUnonlinear` decreases some as width is scaled, although the sparsity of the SAEs in the space makes this harder to verify. 

127 



<!-- Start of picture text -->
R 2  predicting error norm R 2  predicting error vector FVUnonlinear<br>0.944 0.135<br>0.575<br>0.928 0.125<br>10 2 0.525<br>0.912 0.475 0.115<br>0.425 0.105<br>0.896<br>0.375 0.095<br>0.880<br>0.325 0.085<br>0.864<br>0.275<br>10 1 0.075<br>0.848 0.225<br>0.065<br>10 5 10 6 10 5 10 6 10 5 10 6<br>SAE Width SAE Width SAE Width<br>(a) Linear prediction results for layer 9 Gemma 2 9B SAEs from Gemma Scope.<br>R 2  predicting error norm R 2  predicting error vector FVUnonlinear<br>0.915 0.625 0.144<br>0.900<br>0.575 0.136<br>10 2 0.885 0.525 0.128<br>0.870<br>0.855 0.475 0.120<br>0.840 0.425 0.112<br>0.825 0.104<br>0.375<br>0.810 0.096<br>0.325<br>0.795<br>0.088<br>10 1 0.780 0.275<br>0.080<br>10 5 10 6 10 5 10 6 10 5 10 6<br>SAE Width SAE Width SAE Width<br>SAE L0<br>SAE L0<br><!-- End of picture text -->



<!-- Start of picture text -->
(b) Linear prediction results for layer 31 Gemma 2 9B SAEs from Gemma Scope.<br><!-- End of picture text -->

Figure C.7: Results of linearly predicting SAE error norm and SAE error from model activations on Gemma 2 9B layer 9 ( **top** ) and Gemma 2 9B layer 31 ( **bottom** ). The right plots show the _R_<sup>2</sup> of predicting SAE error norms (see Eq. (4.10), the middle plots show the _R_<sup>2</sup> of predicting SAE error vectors (see Eq. (4.11), and the right plots show 1 _− R_<sup>2</sup> of predicting model activations given the SAE reconstruction and the SAE error vector prediction. Unlike for the middle layers shown in the main body, `FVUnonlinear` decreases some as width is scaled, although the sparsity of the SAEs in the space makes this harder to verify. 

128 

# **Chapter 5** 

# **Low-Rank Adapting Models for Sparse Autoencoders** 

## **5.1 Introduction** 



<!-- Start of picture text -->
2.70<br>TopK SAE<br>e2e SAE<br>TopK + LoRA<br>2.65<br>2.60<br>2.55<br>2.50<br>Original Model Loss (best achievable) = 2.4760<br>0 50 100 150 200 250 300<br>Training Time (hours)<br>CE Loss<br><!-- End of picture text -->

Figure 5.1: Cross entropy loss vs. training time over 2B tokens for Gemma-2-2B TopK SAEs with width = 18 _,_ 432 _,_ L0 = 64. We find that our method (TopK + LoRA in the plot) outperforms an e2e SAE and vanilla TopK SAE. 

Although language models demonstrate profound capabilities in tasks such as in-context learning, mathematics, and coding (Brown et al. 2020; OpenAI 2024c; Gemini Team et al. 2023; Bubeck et al. 2023; Anthropic 2024b), the mechanisms behind these behaviors remain largely inscrutable. _Mechanistic interpretability_ (MI) (Bereska and Gavves 2024) seeks to understand these mechanisms by reverse engineering them into human understandable algorithms. In this work, we focus on better understanding features, the “variables” of model computation (Olah et al. 2020; Mueller et al. 2024; Marks, Rager, et al. 2024). 

129 

A popular hypothesis in MI is the _Linear Representation Hypothesis_ (LRH) (Elhage, Hume, et al. 2022; Park, Choe, and Veitch 2023b), which posits that features are one-dimensional directions in activation space. Although some recent research has called aspects of this hypothesis into question (Engels, Eric J. Michaud, et al. 2024; Csordás et al. 2024; Engels, Riggs, and Tegmark 2024), the LRH has been empirically validated for many language model features “in the wild” (Nanda, A. Lee, and Wattenberg 2023; Heinzerling and Inui 2024). Inspired by these successes, _sparse autoencoders_ (SAEs) (Makhzani and Frey 2013) have recently been applied to decompose language model hidden states into many linear features (Cunningham et al. 2023a; Bricken, Templeton, et al. 2023). The latents SAEs learn are significantly more interpretable and monosemantic than the original neuron basis (Cunningham et al. 2023a; Bricken, Templeton, et al. 2023). 

While SAEs find interpretable latents, this comes at a cost: when SAE reconstructions are inserted back into the model and the forward pass is performed, the resulting cross entropy loss ( _L_ SAE) is significantly higher than the loss of the original model ( _L_ BASE). For example, when reconstructions from a TopK SAE are inserted into GPT-4, the resulting _L_ SAE is equivalent to the _L_ BASE of a model trained with just 10% of the pretraining compute of GPT-4 (Gao, Tour, et al. 2024a). Thus, previous work has extensively focused on optimizing SAE architectures to find Pareto improvements in the SAE sparsity vs. _L_ SAE frontier. This work includes TopK SAEs (Gao, Tour, et al. 2024a), Gated SAEs (Rajamanoharan, Conmy, et al. 2024a), JumpReLU SAEs (Rajamanoharan, Lieberum, et al. 2024a), ProLU SAEs (Taggart 2024), Switch SAEs (Mudide et al. 2024), and e2e SAEs (Braun et al. 2024). 

However, an unexplored question is whether language models themselves can be optimized after SAE training to gain an additional Pareto improvement in sparsity vs. _L_ SAE. In this work, we answer this question in the affirmative: we use _Low-Rank Adapters_ (LoRA) (Hu et al. 2021) to reduce the KL divergence between the original model’s logits and the model’s logits with an SAE inserted. The resulting model-SAE combination improves in _L_ SAE and on a diverse set of downstream SAE metrics. Compared to existing proposals for training more interpretable models with SAEs (Lai and Heimersheim 2024; Lai and W. Huang 2024), we estimate our LoRA method is 10<sup>7</sup> times faster<sup>1</sup> . Overall, we find that low-rank adapting models is a simple and efficient technique to improve the interpretability vs. performance trade-off. 

Our contributions include the following: 

1. To the best of our knowledge, we are the first to focus on improving the _model_ around an existing SAE. 

2. In Section 5.4.1, we analyze LoRA SAE training on the Gemma Scope (Lieberum et al. 2024b) family of SAEs. Across SAE width, SAE sparsity, language model size, LoRA rank, and language model layer, we find a 30% to 55% improvement in _L_ SAE–with final values between 0.01 and 0.17 nats–with especially large improvements in low sparsity regimes and larger models. 

3. In Section 5.4.2, we compare our method to e2e SAEs on training time vs. _L_ SAE on Gemma-2-2B (Gemma Team et al. 2024) and Llama-3.2-1B (AI@Meta 2024). We find that our method achieves the same _L_ SAE as e2e SAEs with between 2 _×_ and 20 _×_ less compute 

> 1Gemma-2-2B was trained with 6T tokens, whereas we use 15M tokens on up to 3% of the parameters; 6 _T/_ 15 _M/_ 0 _._ 03 = 1 _._ 3 _×_ 10<sup>7</sup> 

130 

and 130 _×_ fewer language model backward passes. 

4. In Section 5.4.3, we perform LoRA SAE training with multiple SAEs inserted into Llama-3.1-8B (AI@Meta 2024) and see large decreases in _L_ SAE, demonstrating the potential of our technique for helping circuit analysis. 

5. In Section 5.5, we show quantitative improvements on a diverse set of downstream tasks. 

6. In Section 5.6, we find that we can achieve much of the benefit of full-parameter LoRA by training an adapter only on the layer after the SAE and that our adapters achieve most improvement on tokens with low _L_ SAE. 

## **5.2 Related Work** 

**SAE Architecture Improvements.** Early SAEs for language models used a simple linear encoder, ReLU activation with an _L_ 1 penalty (approximating _L_ 0), and a linear decoder (Bricken, Templeton, et al. 2023; Cunningham et al. 2023a). The next generation introduced TopK and BatchTopK SAEs, which enforce sparsity by retaining only the _k_ largest activations (Gao, Tour, et al. 2024a; Bart Bussmann, Patrick Leask, and Nanda 2024b), and GatedSAEs and JumpReluSAEs, which use gating functions and straight-through estimators to approximate direct _L_ 0 optimization (Rajamanoharan, Conmy, et al. 2024a; Rajamanoharan, Lieberum, et al. 2024a). These methods improve the _L_ SAE vs. sparsity tradeoff, though no single approach is definitively superior on downstream tasks (Karvonen, Rager, et al. 2024b). Beyond sparsity penalties, Braun et al. 2024 optimize SAEs for KL divergence with the model’s logits to directly improve _L_ SAE, while Olmo et al. 2024 incorporate model gradients into TopK activations for more causal representations. However, gradient-based methods introduce computational overhead and have a large limitation: SAEs are typically trained on cached activations without available gradients. (Lieberum et al. 2024b). 

**Fine-tuning SAEs.** While this paper focuses on fine-tuning a model around an SAE, another research direction explores fine-tuning SAEs. Some work tailors SAEs to specific domains by oversampling certain contexts (Bricken, Marcus, Rivoire, et al. 2024) or fine-tuning on domain-specific activations (Drori 2024). Kissane et al. 2024 find that training SAEs on chat data captures the refusal latent, whereas training on the Pile (Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, H. He, Thite, Nabeshima, et al. 2020a) does not. Kutsyk, Mencattini, and Florea 2024 further analyze when base model SAEs generalize to a chat-tuned model, showing that it depends on the language model used. 

**Training interpretable models.** We are aware of two prior works that investigate training more interpretable models using SAEs: both (Lai and Heimersheim 2024) and (Lai and W. Huang 2024) train SAEs and models concurrently, and find that this improves _L_ SAE. However, because this requires training models from scratch, it is impractical to apply to existing models and is only shown to work in toy settings; in contrast, our method is extremely efficient, and we show it works on models up to 27B parameters. Many prior works also investigate this direction without SAEs. Elhage, Nanda, et al. 2022 introduce the softmax linear unit (SoLU) activation function, which increases the fraction of interpretable neurons at no cost on downstream performance; Z. Liu, Gan, and Tegmark 2023 propose a new loss term penalizing spatially distant connections in the network that leads to visually interpretable 

131 



<!-- Start of picture text -->
A A A A A A A A<br>B B B B B B B B<br>SAE<br>Low Rank Adapters<br>Low Rank Adapters<br>MLP MLP MLP MLP<br>Attention Attention Encoder Decoder Attention Attention<br><!-- End of picture text -->

Figure 5.2: Visual representation of our method, with a local SAE trained on layer 12 and low-rank adapters trained on MLP and attention components on all layers. 

networks; Z. Liu, Y. Wang, et al. 2024 introduce Kolmogorov-Arnold Networks, an alternative to standard MLPs with trainable activation functions that can be replaced by symbolic formulas; and Heimersheim 2024 fine-tune out the LayerNorm components in GPT-2 with a small downstream loss effect. 

**Parameter Efficient Fine Tuning.** _Parameter efficient fine tuning_ (PEFT) reduces the cost of full supervised fine tuning by updating fewer effective parameters. One of the simplest and most effective PEFT methods is low-rank adaptation (LoRA) (Hu et al. 2021). LoRA works as follows: for a frozen pretrained weight matrix _W_ 0 _∈_ R<sup>_d×k_</sup> , and low-rank matrices _A ∈_ R<sup>_d×r_</sup> , _B ∈_ R<sup>_r×k_</sup> with _r ≪_ min( _d, k_ ), the original forward pass _h_ ( _x_ ) = _W_ 0 _x_ becomes 



_A_ and _B_ can then be trained while the rest of the model is frozen, resulting in a low-rank update of the base model. 

## **5.3 Optimizing Models for Sparse Autoencoders** 

In this section, we formally describe existing methods for training SAEs and our method of adapting models for SAEs. For a decoder only transformer with _L_ layers and hidden dimension _d_ , input **_x_** 0, and output **_y_** , denote the activation after the _i_ th layer by **_x_** _i_ . Express the _i_ th transformer block as a function _hi_ such that the network can be expressed as 





### **5.3.1 TopK Sparse Autoencoders** 

SAEs learn an encoder **W** enc _∈_ R<sup>_m×d_</sup> for _m ≫ d_ , a decoder **W** dec _∈_ R<sup>_d×m_</sup> with unit norm columns, and biases **b** enc _∈_ R<sup>_m_</sup> _,_ **b** dec _∈_ R<sup>_d_</sup> . We call the m columns of **W** dec latents. For 

132 



<!-- Start of picture text -->
0.15 Rank 1 Rank 1 Rank 1<br>Rank 4 Rank 4 Rank 4<br>Rank 16 Rank 16 Rank 16<br>0.10 Rank 64 Rank 64 Rank 64<br>Rank 256 Rank 256 Rank 256<br>0.05<br>55<br>50<br>45<br>40<br>35<br>30<br>0 100 200 300 400 0 100 200 300 400 500 6 9 12 15 18<br>Sparsity (L0) Width (k) Layer<br>CE Improvement<br>CE Improvement (%)<br><!-- End of picture text -->

Figure 5.3: Cross entropy loss improvement ( **Top** : absolute, **Bottom:** percentage of CE loss gap closed) using our method for Gemma Scope SAEs on Gemma-2-2B. **Left:** Scaling across sparsity with fixed width=16k and layer=12, we see the largest effect by percentage of our method at lower sparsities, but still substantial effect at higher sparsities as well. **Middle:** Scaling across width with fixed _L_ 0 = 68 and layer=12, the highest effect by percentage is at low width but again this is not a large effect. **Right:** Scaling across layer with fixed _L_ 0 = 68 and width=16k, the highest effect of our method by percentage is at layer 9 but it is mostly unaffected by layer. 

activation **_x_** _l_ , the TopK SAE (Gao, Tour, et al. 2024a) reconstructs activation **_x_** ˆ _l_ as follows: 





During training, the SAE minimizes the reconstruction error _L_ = _∥_ **_x_** _l −_ **_x_** ˆ _l∥_<sup>2</sup> . We train TopK SAEs with _k_ = 64 for Gemma-2-2B and Llama-3.2-1B for 2B and 4B tokens, respectively, on the RedPajama dataset (Weber et al. 2024). 

### **5.3.2 End-to-End Sparse Autoencoders** 

In an e2e SAE (Braun et al. 2024), the SAE minimizes KL divergence with the base model instead of reconstruction error. Formally, if we have 



then the e2e SAE minimizes _L_ = KL( **_y_** ˆ _,_ **_y_** ). For both e2e and TopK SAEs, we use a TopK activation function with the same sparsity to allow for fair comparisons. 

133 

### **5.3.3 JumpReLU Sparse Autoencoders** 

We also evaluate our method on the Gemma Scope JumpReLU SAEs. Instead of the TopK function, JumpReLU SAEs (Rajamanoharan, Lieberum, et al. 2024a) use the JumpReLU activation function, 

JumpReLU _θ_ ( _z_ ) := _zH_ ( _z − θ_ ) _,_ 

where _H_ is the Heaviside step function and _θ >_ 0 is the JumpReLU’s threshold. The SAE is trained to minimize 



where **_z_** is defined from Eq. (5.4). 

### **5.3.4 Method for Low-Rank Adapting Models to SAEs** 

**Formulation.** We formally describe our method of optimizing models for SAEs, using notation from Equations (5.2)–(5.4). We insert a _frozen_ SAE immediately after layer _ℓ_ , and the reconstructed activation **_x_** ˆ _ℓ_ = SAE� **_x_** _ℓ_ � propagates through the remaining layers to produce **_x_** ˆ _i_ = _hi_ � **_x_** ˆ _i−_ 1� for _ℓ_ + 1 _≤ i ≤ L_ and **_y_** = softmax( **_x_** _L_ ). 

For JumpReLU SAEs we can only adapt layers _after_ the SAE to ensure average sparsity is unaffected, while for TopK SAEs we can train adapters on all layers by maintaining the TopK constraint during training. We add low-rank adapters of rank _r_ in each MLP and attention sublayer of every layer we are adapting. Concretely, for each frozen weight matrix _Wi ∈_ R<sup>_d_1</sup><sup>_×d_2</sup> , we add _Ai ∈_ R<sup>_d_1</sup><sup>_×r_</sup> and _Bi ∈_ R<sup>_r×d_2</sup> and modify the forward pass according to Eq. (5.1). 

We train only the low-rank adapters Θ = _{Ai} ∪{Bi}_ . For all experiments the training objective is the KL divergence between the next token probability distribution with and without the SAE inserted: 



By freezing both the SAE and the original model, this _parameter-efficient_ method aligns the SAE-enhanced model to the behavior of the original model with minimal additional cost. 

## **5.4 Results** 

In this section, we study how our method improves the cross entropy loss gap ( _L_ SAE _−L_ BASE) before and after LoRA on a wide variety of SAEs and language models. Unless otherwise specified, we use a layer 12 residual stream SAE. 

### **5.4.1 Scaling Laws for Downstream Loss** 

We first explore the scaling behavior of low-rank adapting models to SAEs across SAE sparsity, SAE width, language model size, LoRA rank, and model layer. Specifically, we use Gemma Scope’s JumpReLU SAEs (Rajamanoharan, Lieberum, et al. 2024a). To ensure we 

134 



<!-- Start of picture text -->
0.06<br>0.05<br>0.04<br>0.03<br>0.02<br>50<br>45<br>40<br>Rank 1<br>35 Rank 4<br>Rank 16<br>30<br>Rank 64<br>25 Rank 256<br>2B 9B 27B<br>Gemma Size<br>CE Improvement<br>CE Improvement (%)<br><!-- End of picture text -->

Figure 5.4: Cross entropy loss improvement ( **Top** : absolute, **Bottom:** percentage) for Gemma Scope SAEs of width 16 _k_ and _L_ 0 closest to 70 on Gemma-2-2B, 7B, and 27B. We find that our method works increasingly well on larger models. 

do not affect the average sparsity of these JumpReLU SAEs, we only finetune the layers _after_ the SAE. Over different sparsities, widths, and layers, we track the absolute and percent improvement in _L_ SAE _−L_ BASE after low-rank fine-tuning. We train on 15 _M_ random tokens of The Pile (uncopyrighted) dataset (Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, H. He, Thite, Nabeshima, et al. 2020a), and evaluate on a held out validation set of 1 _M_ random tokens. We report our findings in Fig. 5.4 for model size and in Fig. 5.3 for other scaling axes. 

We find that across all of the scaling regimes we test, we close the _L_ SAE _−L_ BASE gap by at least 30%, and sometimes by up to 55%. We find that using larger rank LoRA adapters reliably decreases the final _L_ SAE; this, combined with the fact that we adapt on only 15M tokens and do not see our adapters finish converging, implies that with more compute our method may be even more successful. 

We find that over varying layers, the improvement is largest for middle layers, although this result is not extremely strong (according to e.g. (Lad, Gurnee, and Tegmark 2024), this may arise from the fact that middle layers tend to have richer and more expressive representations that local SAEs may struggle reconstructing). We also find that the improvement is largest on lower sparsities, lower widths, and larger models; all of these results may be caused by these SAEs having a higher cross entropy loss gap to start with. We do still find it extremely promising that the effectiveness of our technique increases on larger models. 

### **5.4.2 Downstream Loss vs. Computational Cost** 

Next, we study the frontier of _L_ SAE versus training time for TopK SAEs, e2e SAEs, and low-rank adapted TopK SAEs. To do this, we need to train our own TopK and e2e SAEs to get their training curves. We also use checkpoints from the TopK training run to get the training curve for TopK SAEs + LoRA; after every 10% training checkpoint of the TopK 

135 

SAEs, we low-rank adapt the model checkpoint with rank 64 LoRA on all layers. 

We train on layer 12 of Llama-3.2-1B and Gemma-2-2B. We train TopK and e2e SAEs for 4B tokens on Llama-3.2-1B and for 2B tokens on Gemma-2-2B (similar to the number of tokens trained on for Gemma Scope SAEs). On each TopK SAE training checkpoint of Llama-3.2-1B we do LoRA finetuning for 100M tokens, while we finetune for 15M tokens on Gemma-2-2B TopK SAE checkpoints. 

We show the Pareto cross entropy frontiers for Gemma-2-2B and Llama-3.2-1B in Figures 5.1 and 5.5, respectively, where our method clearly dominates. Quantitatively, we show the speedup in wall clock time in achieving various cross entropy loss threshold when using TopK + LoRA versus e2e in Tables 5.1 and 5.2; our speedups ranging from 2 _×_ to 20 _×_ . Our approach (TopK + LoRA) also performs 130 _×_ fewer backward passes through the model than e2e SAEs on Gemma-2-2B and 40 _×_ fewer backward passes on Llama-3.2-1B. Finally, we do note, however, that e2e SAEs achieve a lower final CE loss than our method on Llama-3.2-1B (although not on Gemma-2-2B). 

### **5.4.3 Adapting Multiple SAEs** 

Inserting multiple SAEs at once into a language model causes _L_ SAE to grow extremely rapidly (e.g. as shown in Fig. 5.6, we find that inserting 5 SAEs leads to a cross entropy error of almost 10 nats, which is worse than a unigram model (Gao, Tour, et al. 2024a)). At the same time, inserting multiple SAEs at once is a very useful task for circuit analysis, since it allows one to determine dependencies between SAE latents (in practice, past SAE circuits work (Marks, Rager, et al. 2024) has used error terms to overcome this limitation, which results in less interpretable circuits). Thus, we adapt our procedure to work with multiple SAEs: we insert all SAEs at once during training, and otherwise follow Section 5.3.4. We measure the performance of our technique with the “Compound Cross Entropy Loss” (Lai and Heimersheim 2024), which is simply _L_ SAE with all SAEs inserted. 

We use the Llama Scope (Z. He, Shu, Ge, L. Chen, J. Wang, Zhou, F. Liu, Q. Guo, X. Huang, Zuxuan Wu, et al. 2024b) set of SAEs (width = 131 _,_ 072 _,_ L0 = 50) trained on Llama-3.1-8B. Because these are TopK SAEs, we can train the LoRA layers without worrying about violating sparsity constraints. We train with the following configurations of SAEs, chosen to maximize the distance between adjacent pairs of SAEs: 1 SAE at layers {16}; 3 SAEs at layers {10, 20, 30}; 5 SAEs at layers {6, 12, 18, 24, 30}; 7 SAEs at layers {4, 8, 12, . . . , 28}; 10 SAEs at layers {3, 6, 9, . . . , 30}; and 15 SAEs at layers {2, 4, 6, . . . , 30}. 

Our results (see Fig. 5.6) show that this method significantly reduces compound CE loss; for example, using LoRA, the compound CE loss for 7 SAEs goes from 7.83 nats to 2.78 nats, while the compound CE loss for 3 SAEs goes from 3.55 nats to 2.45 nats (which is lower than the original validation CE loss with a _single_ SAE and no LoRA). Thus, our technique seems extremely promising for circuit analysis. 

## **5.5 Downstream Improvements** 

A common criticism of previous SAE optimizations is the lack of grounded metrics for evaluating how good an SAE is. Prior work has largely relied on unsupervised metrics such 

136 

Table 5.1: Gemma-2-2B timing results and speedups, nearest hour 

|CE Loss|TopK + LoRA|TopK|e2e|Speedup|
|---|---|---|---|---|
|2.60|**12h**|59h|37h|**3.05x**|
|2.59|**12h**|—|79h|**6.53x**|
|2.58|**12h**|—|148h|**12.18x**|
|2.57|**12h**|—|243h|**20.01x**|
|2.55-2.57|**12h**–**107h**|—|—|—|



Table 5.2: Llama-3.2-1B timing results and speedups, nearest hour 

|CE Loss|TopK + LoRA|TopK|e2e|Speedup|
|---|---|---|---|---|
|2.73|**9h**|—|96h|**10.38x**|
|2.72|**12h**|—|113h|**9.08x**|
|2.71|**19h**|—|135h|**7.14x**|
|2.70|**70h**|—|156h|**2.23x**|
|2.67-2.70|—|—|**156h**–**213**|**h**<br>—|



as in Section 5.4. Recent work, however, has introduced evaluation metrics to measure a model and SAE according to their performance on downstream tasks (Karvonen, Rager, et al. 2024b; Pres et al. 2024). Thus, in this section, we evaluate our method on a diverse set of downstream benchmarks: 

1. In Section 5.5.1, we show that using LoRA on all layers improves downstream tasks on SAEBench. 

2. In Section 5.5.2, we introduce a novel steering metric and show that our method improves on it. We introduce a new metric because SAEBench metrics do not test the effects of SAEs on next token prediction. 

3. In Section 5.5.3, we show that our method improves overall model performance with the SAE inserted on MMLU, HellaSwag, and TruthfulQA. 

### **5.5.1 SAEBench** 

To address the core challenge of measuring how effectively a model and SAE work together, Karvonen, Rager, et al. 2024b introduce SAEBench, a benchmark of SAE metrics that are faithful to possible real world use cases. For the Gemma-2-2B TopK SAE ( _L_ 0 = 64) we trained in Section 5.4.2, we evaluate the model with the SAE and the model with the SAE + LoRA on SAEBench. 

Specifically, we look at spurious correlation removal (SCR), targeted probe perturbation (TPP), Sparse Probing, automated interpretability (Autointerp), and feature absorption (Absorption). SCR measures the separation of latents for different concepts, with higher scores indicating better ability to debias a classifier. TPP evaluates the impact of ablating specific latents on probe accuracy, where higher scores reflect well-isolated latents correspond- 

137 



<!-- Start of picture text -->
3.0<br>TopK SAE<br>e2e SAE<br>TopK + LoRA<br>2.9<br>2.8<br>2.7<br>2.6<br>Original Model Loss (best achievable) = 2.5481<br>0 25 50 75 100 125 150 175 200<br>Training Time (hours)<br>CE Loss<br><!-- End of picture text -->

Figure 5.5: Cross entropy loss vs. training time for Llama-3.2-1B with TopK SAEs of _L_ 0 = 64 and width 16384. Our method (TopK + LoRA) achieves lower CE loss sooner than e2e SAE or vanilla TopK SAEs 

ing to classes on a dataset. Sparse Probing tests the accuracy of a k-sparse probe trained on SAE latents, with higher scores indicating better feature learning. Autointerp, assessed using an LLM judge (GPT-4o-mini (OpenAI 2024a)), quantifies the interpretability of SAE latents. Absorption quantifies to what extent latents are “absorbed” together to improve sparsity. All metrics range from 0 to 1, with higher being better except for Absorption.<sup>2</sup> 

We display our results in Table 5.3, showing our low-rank adapted model outperforms the base model on TPP, SCR, and Sparse Probing, while very slightly underperforming on Autointerp and Absorption. 

### **5.5.2 Feature Steering** 

In this section, we demonstrate that the LoRA tuned model improves at activation steering– repressing or eliciting model behavior by scaling a steering vector–using its SAE latents. Given an SAE latent **v** _∈_ R<sup>_d_</sup> at layer _l_ , we steer via 



We assess steering effectiveness following (Pres et al. 2024), who evaluate steering by analyzing increases in likelihood for “positive” texts (aligned with the desired behavior) and decreases for “negative” texts (not aligned). We note that Olmo et al. 2024 also introduce an SAE steering evaluation, but it does not follow the best practices for steering laid out by (Pres et al. 2024) so we do not use it. 

> 2Excluded from our results are the RAVEL and Unlearning SAEBench metrics. RAVEL is not yet implemented in SAEBench and Unlearning is recommended for instruct models only. 

138 



<!-- Start of picture text -->
12<br>No SAEs 11.00<br>10 TopK 9.88 9.89<br>TopK + LoRA<br>8 7.83<br>6 5.78<br>4 3.55 3.36<br>2.602.25 2.45 2.94 2.78<br>2<br>0<br>1 3 5 7 10 15<br>Number of SAEs<br>Compound CE Val Loss<br><!-- End of picture text -->

Figure 5.6: Downstream cross entropy loss when multiple Llama Scope SAEs are inserted into Llama-3.1-8B at once. “Base” is the original loss without any fine-tuning, while “LoRA” is the loss after 15M tokens of LoRA training. 

Our method is slightly different than (Pres et al. 2024) because we are comparing different _models_ with the same steering method instead of comparing different steering methods on the same model. For a given SAE latent, we steer on a dataset of 500 positive and negative samples. The negative dataset consists of an equal mix of arabic tweets (Pain 2024), medical facts (MedAlpaca 2024), recipes (Corbt 2024), shakespearean quotes (Roudranil 2024), and law texts (GPT-4o-mini generated). We generate the positive datasets by selecting a latent about 1) machine learning, 2) San Fransisco, 3) Donald Trump, and 4) Covid-19. We then generate text samples where that latent fires using GPT-4o-mini. See Section D.0.1 for full prompt details. 

Following (Pres et al. 2024), we compute mean token log-likelihoods before and after steering, normalizing them so the original likelihoods span 0 to 100. We tune the hyperparameter _α_ in Eq. (5.8) by selecting a value that increases the likelihood of positive samples while minimizing likelihood increases on a validation subset of negative samples (medical facts). After tuning, we evaluate the effect of _α_ on a test consisting of the remaining negative samples. This tuning process is repeated for the base and LoRA models. 

To compare models, let ∆POSITIVE and ∆NEGATIVE represent the change in normalized likelihoods for positive and negative datasets when switching from the base to the LoRA model. The LoRA model is better suited for steering if ∆POSITIVE _>_ 0 and ∆NEGATIVE _≤_ 0. We compute 90% confidence intervals for ∆POSITIVE and ∆NEGATIVE across 500 examples for each of our four SAE latents. Results are summarized in Table 5.4. We show a histogram of changes across all examples after steering for the best performing latent, “Donald Trump”, in Fig. 5.7. 

139 

Table 5.3: Using the same TopK SAE trained on Gemma-2-2B, we compare the SAEBench metrics when the underlying model is low-rank adapted with rank 64. We see across most applicable metrics, the LoRA model shows meaningful improvement. Full results over various thresholds are displayed in Table D.1. 

|Downstream Metric|TopK + LoRA|TopK|
|---|---|---|
|SCR (max)|**0.526**|0.448|
|SCR (average)|**0.314**|0.289|
|TPP (max)|**0.412**|0.372|
|TPP (average)|**0.145**|0.111|
|Sparse Probing (top 1)|**0.760**|0.732|
|Sparse Probing (test)|**0.956**|0.955|
|Autointerp|0.830|**0.832**|
|Absorption|0.210|**0.205**|





<!-- Start of picture text -->
Positive Examples Negative Examples<br>Base Model 35<br>40 LoRA Model<br>30<br>30 25<br>20<br>20 15<br>10<br>10<br>5<br>0 0<br>0 10 20 30 0 10 20<br>Change in Normalized LL after Steering<br>Count<br><!-- End of picture text -->

Figure 5.7: The distribution of normalized log-likelihood change post-steering is more pronounced on positive examples for the LoRA model. Results shown for the SAE latent responsible for “Donald Trump”. 

### **5.5.3 General Language Model Capabilities** 

In addition to downstream tasks, we evaluate the model’s general language capabilities on MMLU (Hendrycks, Burns, Basart, Zou, et al. 2020), HellaSwag (Zellers et al. 2019), and TruthfulQA (S. Lin, Hilton, and Evans 2021) in the following 3 settings: 1) **Original** , the original model with no SAE, 2) **SAE** , the original model with the SAE, and 3) **SAE + LoRA** , a low rank adapted model with the SAE. Our results, shown in Table 5.5, show that our method improves on all datasets. 

140 

Table 5.4: For each SAE latent, ∆POSITIVE and ∆NEGATIVE denote the 90% CI improvement in normalized log-likelihood increase when using the LoRA model for steering on positive and negative examples, respectively. Because ∆POSITIVE _>_ 0 and ∆NEGATIVE _≤_ 0, we see the LoRA model is better at steering for a given SAE latent while not affecting other features. 

|SAE Feature|∆positive|∆negative|
|---|---|---|
|Machine Learning|**0**_._**86**_±_**0**_._**82**|_−_0_._84_±_0_._43|
|San Francisco|**0**_._**97**_±_**0**_._**76**|0_._06_±_0_._20|
|Donald Trump|**2**_._**50**_±_**0**_._**56**|0_._20_±_0_._40|
|COVID-19|**0**_._**44**_±_**0**_._**25**|_−_0_._01_±_0_._06|





<!-- Start of picture text -->
16000 Loss Improvement<br>Loss Degradation<br>14000<br>12000<br>10000<br>8000<br>6000<br>4000<br>2000<br>0<br>10 5 10 4 10 3 10 2 10 1 10 0 10 1<br>|Base Loss - LoRA Loss|<br># of Validation Tokens<br><!-- End of picture text -->

Figure 5.8: Distribution of loss improvements and loss degredations across validation tokens. We see that more tokens have a loss improvement than degredation (although a substantial number have a degradation) and most loss improvements and degredations happen in a range of about 0.01 to 1 ∆ _L_ SAE nats. 



<!-- Start of picture text -->
40<br>All layers after<br>0.04 Rank 64 30<br>20<br>0.02<br>10<br>0.00 0<br>2 4 6 8 10 12<br>Number of Layers after SAE<br>CE Improvement<br>CE Improvement (%<br><!-- End of picture text -->

Figure 5.9: Plot of _L_ SAE when running LoRA on just a single layer of Gemma Scope 2B. We find that LoRA on layers closer to the SAE layer do better, and that also LoRA on just the next layer achieves much of the loss reduction of training on all layers. 

## **5.6 Analyzing Why Our Method Works** 

### **5.6.1 Per Token Improvement Breakdown** 

In this experiment, we analyze how LoRA impacts _L_ SAE improvements. Figure 5.8 shows the distribution of ∆ _L_ SAE between the original and LoRA models across 15M validation tokens. The per-token loss change varies greatly, with the loss on 37% of tokens even getting worse (see the degradation histogram in the figure). Most of the overall improvement comes from small per-token decreases in loss (roughly 10<sup>_−_2</sup> to 1 nats), suggesting LoRA improves loss across many tokens rather than a more bimodal distribution. 

141 

Table 5.5: Comparisons of SAE, SAE + LoRA, and original model performance on standard NLP benchmarks. Error ranges represent one standard error; largest value between SAE and SAE + LoRA is bolded. 

|||**Gemma-**|**2-2B**||
|---|---|---|---|---|
|**Metric**|SAE|SAE + LoRA|Original|LoRA|
|MMLU|44_._2_±_0_._4|**45**_._**8**_±_**0**_._**4**|49_._3_±_0_._4|**50**_._**0**_±_**0**_._**4**|
|HellaSwag|50_._9_±_0_._5|**52**_._**1**_±_**0**_._**5**|55_._0_±_0_._5|**56**_._**0**_±_**0**_._**5**|
|BLEU|29_._9_±_1_._6|**30**_._**6**_±_**1**_._**6**|30_._4_±_1_._6|**32**_._**4**_±_**1**_._**6**|
|ROUGE-1|28_._2_±_1_._6|**28**_._**5**_±_**1**_._**6**|26_._9_±_1_._6|**30**_._**2**_±_**1**_._**6**|
|ROUGE-2|24_._8_±_1_._5|**26**_._**6**_±_**1**_._**5**|25_._6_±_1_._5|**29**_._**1**_±_**1**_._**6**|
|MC1|23_._1_±_1_._5|**23**_._**4**_±_**1**_._**5**|24_._1_±_1_._5|**24**_._**3**_±_**1**_._**5**|
|||**Gemma-**|**2-9B**||
|**Metric**|SAE|SAE + LoRA|Original|LoRA|
|MMLU|64_._2_±_0_._4|**65**_._**7**_±_**0**_._**4**|**70**_._**0**_±_**0**_._**4**|68_._8_±_0_._4|
|HellaSwag|58_._3_±_0_._5|**59**_._**6**_±_**0**_._**5**|61_._2_±_0_._5|**61**_._**9**_±_**0**_._**5**|
|BLEU|40_._9_±_1_._7|**42**_._**4**_±_**1**_._**7**|**43**_._**8**_±_**1**_._**7**|43_._6_±_1_._7|
|ROUGE-1|39_._0_±_1_._7|**40**_._**6**_±_**1**_._**7**|42_._7_±_1_._7|**43**_._**5**_±_**1**_._**7**|
|ROUGE-2|33_._4_±_1_._7|**36**_._**4**_±_**1**_._**7**|38_._3_±_1_._7|**38**_._**8**_±_**1**_._**7**|
|MC1|27_._1_±_1_._6|**28**_._**0**_±_**1**_._**6**|30_._5_±_1_._6|**31**_._**0**_±_**1**_._**6**|
|||**Gemma-2**|**-27B**||
|**Metric**|SAE|SAE + LoRA|Original|LoRA|
|MMLU|70_._9_±_0_._4|**71**_._**3**_±_**0**_._**4**|72_._1_±_0_._3|**72**_._**7**_±_**0**_._**3**|
|HellaSwag|61_._0_±_0_._5|**62**_._**7**_±_**0**_._**5**|65_._3_±_0_._5|**65**_._**5**_±_**0**_._**5**|
|BLEU|**40**_._**9**_±_**1**_._**7**|38_._9_±_1_._7|41_._1_±_1_._7|**41**_._**9**_±_**1**_._**7**|
|ROUGE-1|**41**_._**0**_±_**1**_._**7**|38_._3_±_1_._7|40_._9_±_1_._7|**41**_._**7**_±_**1**_._**7**|
|ROUGE-2|**37**_._**1**_±_**1**_._**7**|35_._3_±_1_._7|36_._2_±_1_._7|**37**_._**1**_±_**1**_._**7**|
|MC1|30_._2_±_1_._6|**31**_._**5**_±_**1**_._**6**|**33**_._**8**_±_**1**_._**7**|32_._9_±_1_._6|



### **5.6.2 Activation Distances** 

One concern identified by (Braun et al. 2024) is that optimizing towards KL divergence may lead the activations to be off distribution and follow a different computational path through the model. We find this is _not_ the case with our method: as shown in Fig. D.2, over a validation set of 500k tokens, our method slightly _decreases_ the distance between activations after the SAE and activations in the original model, while the cosine similarities slightly _increase_ . 

### **5.6.3 Single Layer Adapters** 

Another question we are interested in is which LoRA layers are most important for reducing _L_ SAE. In Fig. 5.9, we plot the results of an experiment where we train LoRA adapters on each individual layer after the layer with the inserted SAE. We find that the LoRA performance degrades as it gets farther from the original layer. Interestingly, we also find that training LoRA adapters on just the first layer after the SAE achieves 88.14% of the loss reduction in training LoRA adapters on _all_ the layers after the SAE, suggesting the loss improvement 

142 

mechanism may be reasonably simple. 

## **5.7 Conclusion** 

Low-rank adapting models for SAEs provides a fast, cheap, and effective path to producing interpretable combined model and SAE systems. Moreover, low-rank adapted models are “better” at using their SAEs for downstream tasks. Crucially, our work challenges the prevailing assumption that improving interpretability must rely solely on post-hoc model decomposition. We hypothesize that our method is much faster than e2e training because modifying the entire model gives many more degrees of freedom for the optimization procedure to work with; thus, our work suggests that focusing more on the larger space of possible language model modifications may be fruitful. We hope the results in this paper lay the groundwork for further such techniques. 

### **5.7.1 Limitations** 

Mechanistic interpretability work usually interprets a frozen model, while in our work we change it; for some applications, this might not be acceptable. However, we do note that we use LoRA to decrease the KL divergence with the original model, so if one is using SAEs anyways, our method creates a more faithful model. Another limitation is that it seems the e2e SAEs may not have finished converging in our experiments, so we cannot compare converged accuracy; however, we had already trained for more than a week before stopping, so training to full convergence may not be practical. We use the learning rates suggested by Gao, Tour, et al. 2024a, but it is possible that further tuning of the learning rate could make the e2e SAE train faster; for reasons of compute limitations, we do not experiment with learning rate. Finally, we note that if we had used LoRA for more tokens we may have gotten an additional improvement on the Gemma Scope scaling experiments; however, our results are still quite strong, and the fact that they were achieved with just 15M tokens shows the efficiency of our technique. 

## **Impact Statement** 

We believe that increasingly powerful language models and other AI systems pose many safety risks (e.g. deception, power seeking, misinformation, bias, CBRN risks; see (Slattery et al. 2024) for a complete summary). MI and other fields that try to better understand LLMs are motivated by reducing these risks (see (Bereska and Gavves 2024) and (Sharkey et al. 2025) for more in depth reviews of MI and AI safety and discussions of open problems). Thus, because the goal in our work is to train more interpretable systems at a fixed level of fidelity to the original (uninterpretable) model, we do not foresee negative consequences of our work; on the contrary, we believe our work is broadly beneficial for AI safety. 

143 

144 

# **Appendix D** 

# **Appendix: Low-Rank Adapting Models for Sparse Autoencoders** 

### **D.0.1 Steering** 

#### **Distribution Plots** 

In Fig. D.1, we plot histograms for the changes in normalized log-likelihoods for each of the four datasets from Table 5.4. 



<!-- Start of picture text -->
Positive Examples Negative Examples Positive Examples Negative Examples<br>Base Model Base Model<br>40 LoRA Model 40 50 LoRA Model 35<br>35 35<br>30<br>30 30 40<br>25<br>25 25<br>30 20<br>20 20<br>15<br>15 15 20<br>10 10 10<br>10<br>5 5 5<br>0 0 0 0<br>0 20 0 10 20 30 20 0 20 40 5 0 5<br>Change in Normalized LL after Steering Change in Normalized LL after Steering<br>Positive Examples Negative Examples Positive Examples Negative Examples<br>Base Model 35 70 Base Model 60<br>40 LoRA Model LoRA Model<br>30 60 50<br>30 25 50 40<br>20 40<br>30<br>20 15 30<br>20<br>10 20<br>10<br>5 10 10<br>0 0 0 0<br>0 10 20 30 0 10 20 5 0 5 10 15 0 2 4<br>Change in Normalized LL after Steering Change in Normalized LL after Steering<br>Count Count<br>Count Count<br><!-- End of picture text -->

Figure D.1: Distribution plots of the change in normalized log-likelihood after steering for various SAE latents. **Top left** is for machine learning (neuron 8421). **Top right** is for San Francisco (neuron 2195). **Bottom left** is for Donald Trump (neuron 13677). **Bottom right** is for COVID-19 (neuron 17811). 

145 

#### **Datasets** 

To generate our “positive” examples dataset, we generate examples eliciting the SAE feature with GPT-4o-mini. We prompt using the following chat template. 

```
prompt="""
```

```
Generate{num_examples}textexamplesthathavethefollowingfeature:
{feature_description}
```

```
Belowareexamplesoftextthathavethefeaturedescribedabove.
```

```
Examples:
{examples}
```

```
Eachtextexampleshouldbearound**twelve**wordslongandbeunique.
Trytobevariedinthecontentoftheexamples.
```

```
"""
```

### **D.0.2 SAEBench Metrics** 

Here, we define in more detail the SAEBench metrics defined by (Karvonen, Rager, et al. 2024b) and used in Section 5.5.1, along with the full results over different hyperparameter splits. 

**Absorption.** Feature absorption occurs when a latent representing concept _A_ implicitly encodes a related concept _B_ (e.g., _Elephant ⇒ gray_ ), leading to redundancy or loss of interpretability. This phenomenon disrupts feature disentanglement, as absorbed features may activate inconsistently, obscuring their semantic meaning. 

To measure absorption, (Karvonen, Rager, et al. 2024b) adapt the method of (Chanin et al. 2024a) using a first-letter classification task. A logistic regression probe is trained on residual stream activations to establish a ground-truth feature direction. They then perform _k_ -sparse probing on SAE latents, identifying primary latents responsible for the task. If increasing _k_ significantly improves F1 by some threshold, the new latent is classified as a feature split. 

They then detect absorption by identifying test cases where primary latents fail while the probe succeeds. A latent is flagged as absorbing the feature if it strongly aligns with the probe in cosine similarity and accounts for a sufficient fraction of the probe projection. 

**Spurious Correlation Removal.** The spurious correlation removal (SCR) metric evaluates whether the SAE captures separate latents for distinct concepts (e.g., gender vs. profession). A classifier is trained on a deliberately “biased” dataset (e.g., only male + professor, female + nurse), thereby picking up the spurious correlation, and then the latents most associated with the spurious feature (e.g., gender) are zero-ablated. 

During evaluation, the classifier is to be debiased. Choosing the top _n_ latents according to their probe attribution score, a modified classifier is defined in which all latents except for the spuriously correlated latent are zero ablated. Evaluated on a balanced dataset, this 

146 

modified classifier’s accuracy in classifying its concept is tracked, and the metric is defined as 



where _A_ abl is the probe accuracy after ablation, _A_ base is the original spurious probe’s accuracy, and _A_ oracle is the accuracy of a probe directly trained on the concept. This SHIFT score quantifies how much ablation improves accuracy (removing the spurious signal), relative. A higher score indicates better separation of the spurious feature and stronger debiasing. 

**Targeted Probe Perturbation.** SHIFT operates on datasets with correlated labels. To extend SHIFT to all multiclass NLP datasets, (Karvonen, Rager, et al. 2024b) introduce TPP, a method that identifies structured sets of SAE latents that disentangle dataset classes. This approach involves training probes on model activations and assessing the impact of ablating specific latent sets on probe accuracy. Ideally, removing a disentangled set of latents should only impact the corresponding class probe while leaving other class probes unaffected. 

Consider a dataset where each input is assigned a single label from a set of _m_ possible concepts, _C_ = _{c_ 1 _, c_ 2 _, ..., cm}_ . For each class indexed by _i ∈{_ 1 _, ..., m}_ , the most relevant latents _Li_ are determined using probe attribution scores. To evaluate their effect, the dataset is partitioned into instances belonging to the target concept _ci_ and a mixed subset containing randomly sampled instances from other labels. 

A linear classifier _Cj_ is defined to predict concept _cj_ with an accuracy of _Aj_ . Furthermore, let _Ci,j_ denote the classifier for _cj_ when latents in _Li_ are ablated. The accuracy of each classifier _Ci,j_ on the corresponding dataset partition for _cj_ is then computed as _Ai,j_ . The TPP metric is given by: 



This metric quantifies the extent to which ablating a disentangled set of latents selectively affects its corresponding class. A well-disentangled latent representation should cause a significant accuracy drop when _i_ = _j_ (i.e., ablating latents relevant to class _i_ in classifier _Ci_ ) while having minimal effect when _i̸_ = _j_ . 

**Sparse Probing.** To evaluate the SAE’s ability to learn specific features, SAEs are tested on diverse tasks (e.g., language ID, profession classification, sentiment analysis). Inputs are encoded with the SAE, mean-pooled over non-padding tokens, and the top-K latents are selected via maximum mean difference. A logistic regression probe is trained on these latents and evaluated on a held-out test set to assess how well the SAE captures the target features. A higher score reflects better feature representation (Karvonen, Rager, et al. 2024b). 

### **D.0.3 Activation Distances** 

In Fig. D.2 we show how low rank adapting the model affects the cosine similarity and _L_ 2 distance of the model activations with and without the SAE. 

147 

Table D.1: Using the same TopK SAE trained on Gemma-2-2B, we compare the SAEBench metrics when the underlying model is low-rank adapted with rank 64. The threshold hyperparameter for SCR and TPP denotes how many of the top _n_ latents are used in the modified classifier. 

|Downstream Metrics|LoRA Model|Base Model|
|---|---|---|
|SCR Metric @2|0.094|**0.097**|
|SCR Metric @5|**0.196**|0.177|
|SCR Metric @10|**0.260**|0.253|
|SCR Metric @20|**0.336**|0.327|
|SCR Metric @50|0.447|**0.448**|
|SCR Metric @100|**0.526**|0.400|
|SCR Metric @500|**0.342**|0.325|
|TPP Metric @2|**0.013**|0.007|
|TPP Metric @5|**0.023**|0.014|
|TPP Metric @10|**0.035**|0.023|
|TPP Metric @20|**0.085**|0.039|
|TPP Metric @50|**0.184**|0.128|
|TPP Metric @100|**0.266**|0.194|
|TPP Metric @500|**0.412**|0.372|
|Sparse Probing (Top 1)|**0.760**|0.732|
|Sparse Probing (Top 2)|**0.833**|0.832|
|Sparse Probing (Top 5)|0.875|0.875|
|Sparse Probing (Top 10)|**0.910**|0.907|
|Sparse Probing (Top 20)|0.930|0.930|
|Sparse Probing (Top 50)|0.946|0.946|
|Sparse Probing (Test)|**0.956**|0.955|
|Autointerp|0.830|**0.832**|
|Absorption|0.210|**0.205**|





<!-- Start of picture text -->
Rank 64 Rank 64<br>0.0<br>0.0015<br>0.5<br>0.0010<br>1.0<br>0.0005 1.5<br>2.0<br>0.0000<br>2.5<br>0.0005 3.0<br>0 2 4 6 8 10 12 0 2 4 6 8 10 12<br>Number of Layers after SAE Number of Layers after SAE<br>Change in Distance<br>Change in Cos Similarity<br><!-- End of picture text -->

Figure D.2: Change in average distance to original activations before and after applying LoRA; increases in cosine similarity ( **Left** ) and decreases in Euclidean distance ( **Right** ) are good. 

148 

# **References** 

- Achiam, Josh, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. (2023). “Gpt-4 technical report”. In: _arXiv preprint arXiv:2303.08774_ . 

- Adebayo, Julius, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim (2018). “Sanity checks for saliency maps”. In: _Advances in neural information processing systems_ 31. 

- AI, Team and Daisuke Ishii (n.d.). _Spam Text Message Classification — kaggle.com_ . [Accessed 21-01-2025]. url: https://www.kaggle.com/datasets/team-ai/spam-text-messageclassification. 

- AI@Meta (2024). _Llama 3 Model Card_ . url: https://github.com/meta-llama/llama3/blob/ main/MODEL_CARD.md. 

- Alain, Guillaume (2016). “Understanding intermediate layers using linear classifier probes”. In: _arXiv preprint arXiv:1610.01644_ . 

- AllenAI (n.d.). _allenai/basic_arithmetic · Datasets at Hugging Face — huggingface.co_ . [Accessed 21-01-2025]. url: https://huggingface.co/datasets/allenai/basic_arithmetic. 

- Alon, Noga (2003). “Problems and results in extremal combinatorics—I”. In: _Discrete Mathematics_ 273.1-3, pp. 31–53. 

- Anders, Evan and Joseph Bloom (2024). “Examining Language Model Performance with Reconstructed Activations using Sparse Autoencoders”. In: _LessWrong_ . url: https://www. lesswrong.com/posts/8QRH8wKcnKGhpAu2o/examining-language-model-performancewith-reconstructed. 

- Anthropic (2024a). _Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku_ . Accessed: 2025-01-30. url: %7Bhttps://www.anthropic.com/news/3-5-modelsand-computer-use%7D. 

- (2024b). _The Claude 3 Model Family: Opus, Sonnet, Haiku_ . Tech. rep. Anthropic. 

- Anthropic, Transformer Circuits Team (2024). _Circuits Updates April 2024_ . url: https: //transformer-circuits.pub/2024/april-update/index.html#scaling-laws. 

- Arora, Sanjeev, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski (2018). “Linear algebraic structure of word senses, with applications to polysemy”. In: _Transactions of the Association for Computational Linguistics_ 6, pp. 483–495. 

- Belrose, Nora, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt (2023). “Eliciting latent predictions from transformers with the tuned lens”. In: _arXiv preprint arXiv:2303.08112_ . 

149 

- Ben Zhou Daniel Khashabi, Qiang Ning and Dan Roth (2019). ““Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding”. In: _EMNLP_ . 

- Bereska, Leonard and Efstratios Gavves (2024). “Mechanistic Interpretability for AI Safety–A Review”. In: _arXiv preprint arXiv:2404.14082_ . 

- Bills, Steven, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders (2023). _Language models can explain neurons in language models_ . url: https://openaipublic.blob.core.windows.net/neuronexplainer/paper/index.html. 

- Bisk, Yonatan, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi (2020). “PIQA: Reasoning about Physical Commonsense in Natural Language”. In: _Thirty-Fourth AAAI Conference on Artificial Intelligence_ . 

- Black, Sid, Lee Sharkey, Leo Grinsztajn, Eric Winsor, Dan Braun, Jacob Merizian, Kip Parker, Carlos Ramón Guevara, Beren Millidge, Gabriel Alfour, et al. (2022). “Interpreting neural networks through the polytope lens”. In: _arXiv preprint arXiv:2211.12312_ . 

- Bloom, Joseph (2024). _Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2 Small_ . url: https://www.alignmentforum.org/posts/f9EgfLSurAiqRJySD/opensource-sparse-autoencoders-for-all-residual-stream. 

- Bolukbasi, Tolga, Adam Pearce, Ann Yuan, Andy Coenen, Emily Reif, Fernanda Viégas, and Martin Wattenberg (2021). “An interpretability illusion for bert”. In: _arXiv preprint arXiv:2104.07143_ . 

- Braun, Dan, Jordan Taylor, Nicholas Goldowsky-Dill, and Lee Sharkey (2024). _Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning_ . arXiv: 2405.12241 `[cs.LG]` . url: https://arxiv.org/abs/2405.12241. 

- Bricken, Trenton, Jonathan Marcus, Siddharth Mishra-Sharma, Meg Tong, Ethan Perez, Mrinank Sharma, Kelley Rivoire, and Thomas Henighan (Oct. 2024a). _Using Dictionary Learning Features as Classifiers_ . Ed. by Adam Jermyn. Accessed: 2025-01-23. url: https: //transformer-circuits.pub/2024/features-as-classifiers/index.html. 

- (Oct. 2024b). _Using Dictionary Learning Features as Classifiers_ . Ed. by Adam Jermyn. Transformer Circuits. url: https://transformer-circuits.pub/2024/features-as-classifiers/ index.html. 

- Bricken, Trenton, Jonathan Marcus, Kelley Rivoire, and Thomas Henighan (2024). _Transformer Circuits: September Update_ . Accessed: 2025-01-19. url: https://transformercircuits.pub/2024/september-update/index.html#oversampling. 

- Bricken, Trenton, Adly Templeton, et al. (2023). “Towards Monosemanticity: Decomposing Language Models With Dictionary Learning”. In: _Transformer Circuits Thread_ . https://transformer-circuits.pub/2023/monosemantic-features/index.html. 

- Brown, Tom B. et al. (2020). _Language Models are Few-Shot Learners_ . arXiv: 2005.14165 `[cs.CL]` . url: https://arxiv.org/abs/2005.14165. 

- Bubeck, Sebastien, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. (2023). “Sparks of artificial general intelligence: Early experiments with gpt-4”. In: _arXiv preprint arXiv:2303.12712_ . 

- Bussmann, Bart, Patrick Leask, Joseph Bloom, Curt Tigges, and Neel Nanda (2024). “Stitching SAEs of different sizes”. In: _AI Alignment Forum_ . url: https://www.alignmentforum.org/ posts/baJyjpktzmcmRfosq/stitching-saes-of-different-sizes. 

150 

- Bussmann, Bart, Patrick Leask, and Neel Nanda (2024a). _BatchTopK Sparse Autoencoders_ . eprint: arXiv:2412.06410. 

- (2024b). “Batchtopk sparse autoencoders”. In: _arXiv preprint arXiv:2412.06410_ . 

- Bussmann, Ben, Peter Leask, and Neel Nanda (2024). _Learning Multi-Level Features with Matryoshka SAEs_ . Accessed: 2025-01-23. url: https://www.lesswrong.com/posts/ rKM9b6B2LqwSB5ToN/learning-multi-level-features-with-matryoshka-saes. 

- Candès, Emmanuel J, Xiaodong Li, Yi Ma, and John Wright (2011). “Robust principal component analysis?” In: _Journal of the ACM (JACM)_ 58.3, pp. 1–37. 

- Chalnev, Sviatoslav, Matthew Siu, and Arthur Conmy (2024). “Improving steering vectors by targeting sparse autoencoder features”. In: _arXiv preprint arXiv:2411.02193_ . 

- Chanin, David, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, and Joseph Bloom (2024a). _A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders_ . arXiv: 2409.14507 `[cs.CL]` . url: https://arxiv.org/abs/2409.14507. 

- (2024b). “A is for absorption: Studying feature splitting and absorption in sparse autoencoders”. In: _arXiv preprint arXiv:2409.14507_ . 

- Chaudhary, Maheep and Atticus Geiger (2024). “Evaluating open-source sparse autoencoders on disentangling factual knowledge in gpt-2 small”. In: _arXiv preprint arXiv:2409.04478_ . 

- Chen, Tianqi and Carlos Guestrin (2016). “XGBoost: A Scalable Tree Boosting System”. In: _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ . KDD ’16. San Francisco, California, USA: Association for Computing Machinery, pp. 785–794. isbn: 9781450342322. doi: 10.1145/2939672.2939785. url: https://doi.org/10.1145/2939672.2939785. 

- clmentbisaillon (n.d.). _fake-and-real-news-dataset — kaggle.com_ . [Accessed 21-01-2025]. url: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset. 

- Conerly, Tom, Adly Templeton, Trenton Bricken, Jonathan Marcus, and Tom Henighan (2024). _Circuits Updates - April 2024 — transformer-circuits.pub_ . [Accessed 15-02-2025]. url: https://transformer-circuits.pub/2024/april-update/index.html#training-saes. 

- Conmy, Arthur, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso (2023). “Towards automated circuit discovery for mechanistic interpretability”. In: _Advances in Neural Information Processing Systems_ 36, pp. 16318–16352. 

- Corbt (2024). _All Recipes Dataset_ . https://huggingface.co/datasets/corbt/all- recipes. Accessed: 2025-01-30. 

- Csordás, Róbert, Christopher Potts, Christopher D Manning, and Atticus Geiger (2024). “Recurrent Neural Networks Learn to Store and Generate Sequences using Non-Linear Representations”. In: _arXiv preprint arXiv:2408.10920_ . 

- Cunningham, Hoagy, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey (2023a). _Sparse Autoencoders Find Highly Interpretable Features in Language Models_ . arXiv: 2309. 08600 `[cs.LG]` . url: https://arxiv.org/abs/2309.08600. 

- (2023b). “Sparse autoencoders find highly interpretable features in language models”. In: _arXiv preprint arXiv:2309.08600_ . 

- Dalrymple, David, Joar Skalse, Yoshua Bengio, Stuart Russell, Max Tegmark, Sanjit Seshia, Steve Omohundro, Christian Szegedy, Ben Goldhaber, Nora Ammann, et al. (2024). “Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems”. In: _arXiv preprint arXiv:2405.06624_ . 

151 

- Davidson, Thomas, Dana Warmsley, Michael Macy, and Ingmar Weber (2017). “Automated Hate Speech Detection and the Problem of Offensive Language”. In: _Proceedings of the 11th International AAAI Conference on Web and Social Media_ . ICWSM ’17. Montreal, Canada, pp. 512–515. 

- Drori, Jacob (2024). _Domain-Specific SAEs_ . url: https://www.lesswrong.com/posts/ ojERTvdGWW6XRZAqr/domain-specific-saes. 

- Dublish, Tanishq (n.d.). _Text classification documentation — kaggle.com_ . [Accessed 21-012025]. url: https://www.kaggle.com/datasets/tanishqdublish/text- classificationdocumentation. 

- Elazar, Yanai, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg (Mar. 2021). “Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals”. In: _Transactions of the Association for Computational Linguistics_ 9, pp. 160–175. issn: 2307-387X. doi: 10.1162/tacl_a_00359. url: http://dx.doi.org/10.1162/tacl_a_00359. 

- Elhage, Nelson, Tristan Hume, et al. (2022). “Toy Models of Superposition”. In: _Transformer Circuits Thread_ . https://transformer-circuits.pub/2022/toy_model/index.html. 

- Elhage, Nelson, Neel Nanda, Zachary Kernfeld, Tom Henighan, Catherine Olsson, and Nicholas Joseph (2022). _Softmax Linear Units_ . url: https://transformer-circuits.pub/2022/solu/ index.html. 

- Engels, Joshua, Isaac Liao, Eric J Michaud, Wes Gurnee, and Max Tegmark (2024). “Not All Language Model Features Are Linear”. In: _arXiv preprint arXiv:2405.14860_ . 

- Engels, Joshua, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark (2024). _Not All Language Model Features Are Linear_ . arXiv: 2405.14860 `[cs.LG]` . url: https://arxiv.org/ abs/2405.14860. 

- Engels, Joshua, Logan Riggs, and Max Tegmark (2024). “Decomposing The Dark Matter of Sparse Autoencoders”. In: _arXiv preprint arXiv:2410.14670_ . 

- Falgunipatel19 (n.d.). _Medical Text Dataset -Cancer Doc Classification — kaggle.com_ . [Accessed 21-01-2025]. url: https://www.kaggle.com/datasets/falgunipatel19/biomedicaltext-publication-classification. 

- Farrell, Eoin, Yeu-Tong Lau, and Arthur Conmy (2024). _Applying sparse autoencoders to unlearn knowledge in language models_ . eprint: arXiv:2410.19278. 

- Foucart, Simon and Holger Rauhut (2013). _A Mathematical Introduction to Compressive Sensing_ . Applied and Numerical Harmonic Analysis. New York: Springer. isbn: 978-08176-4947-0. doi: 10.1007/978-0-8176-4948-7. 

- Gaggar, Raghav, Ashish Bhagchandani, and Harsh Oza (2023). _Machine-Generated Text Detection using Deep Learning_ . eprint: arXiv:2311.15425. 

- Gallifant, Jack, Shan Chen, Kuleen Sasse, Hugo Aerts, Thomas Hartvigsen, and Danielle S. Bitterman (2025). _Sparse Autoencoder Features for Classifications and Transferability_ . eprint: arXiv:2502.11367. 

- Gao, Leo, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. (2020a). “The pile: An 800gb dataset of diverse text for language modeling”. In: _arXiv preprint arXiv:2101.00027_ . 

- Gao, Leo, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. (2020b). “The Pile: An 800GB Dataset of Diverse Text for Language Modeling”. In: _arXiv preprint arXiv:2101.00027_ . 

152 

- Gao, Leo, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu (2024a). _Scaling and evaluating sparse autoencoders_ . arXiv: 2406.04093 `[cs.LG]` . url: https://arxiv.org/abs/2406.04093. 

- (2024b). “Scaling and evaluating sparse autoencoders”. In: _arXiv preprint arXiv:2406.04093_ . 

- Gerami, Shayan (n.d.). _AI Vs Human Text — kaggle.com_ . [Accessed 21-01-2025]. url: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data. 

- Gershgorin, Semyon Aranovich (1931). “Uber die Abgrenzung der Eigenwerte einer Matrix”. In: _Izvestiya Rossii skoi akademii nauk. Seriya matematicheskaya_ 6, pp. 749–754. 

- Giglemiani, Giorgi, Nora Petrova, Chatrik Singh Mangat, Jett Janiak, and Stefan Heimersheim (2024). “Evaluating Synthetic Activations composed of SAE Latents in GPT-2”. In: _arXiv preprint arXiv:2409.15019_ . 

- Goh, Adison (n.d.). _IT Service Ticket Classification Dataset — kaggle.com_ . [Accessed 21-012025]. url: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classificationdataset. 

- Gould, Rhys, Euan Ong, George Ogden, and Arthur Conmy (2023). “Successor heads: Recurring, interpretable attention heads in the wild”. In: _arXiv preprint arXiv:2312.09230_ . 

- Grattafiori, Aaron, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. (2024). _The Llama 3 Herd of Models_ . arXiv: 2407.21783 `[cs.AI]` . url: https: //arxiv.org/abs/2407.21783. 

- Gupta, Pashupati (n.d.). _Emotion Detection from Text — kaggle.com_ . [Accessed 21-01-2025]. url: https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text. 

- Gurnee, Wes (2024). “Sae reconstruction errors are (empirically) pathological”. In: _AI Alignment Forum_ , p. 16. 

- Gurnee, Wes, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas (2023). “Finding Neurons in a Haystack: Case Studies with Sparse Probing”. In: _Transactions on Machine Learning Research_ . issn: 2835-8856. url: https://openreview. net/forum?id=JYs1R9IMJr. 

- Gurnee, Wes and Max Tegmark (2023). “Language models represent space and time”. In: _arXiv preprint arXiv:2310.02207_ . 

- (2024). “Language Models Represent Space and Time”. In: _The Twelfth International Conference on Learning Representations_ . url: https://openreview.net/forum?id= jE8xbmvFin. 

- Hanna, Michael, Ollie Liu, and Alexandre Variengien (2024). “How does gpt-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model”. In: _Advances in Neural Information Processing Systems_ 36. 

- He, Zhengfu, Wentao Shu, Xuyang Ge, Lingjie Chen, Junxuan Wang, Yunhua Zhou, Frances Liu, Qipeng Guo, Xuanjing Huang, Zuxuan Wu, et al. (2024a). “Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders”. In: _arXiv preprint arXiv:2410.20526_ . 

- He, Zhengfu, Wentao Shu, Xuyang Ge, Lingjie Chen, Junxuan Wang, Yunhua Zhou, Frances Liu, Qipeng Guo, Xuanjing Huang, Zuxuan Wu, et al. (2024b). _Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders_ . eprint: arXiv:2410. 20526. 

153 

- Heap, Thomas, Tim Lawson, Lucy Farnik, and Laurence Aitchison (2025). _Sparse Autoencoders Can Interpret Randomly Initialized Transformers_ . arXiv: 2501.17727 `[cs.LG]` . url: https: //arxiv.org/abs/2501.17727. 

- Heimersheim, Stefan (2024). _You can remove GPT2’s LayerNorm by fine-tuning_ . arXiv: 2409.13710 `[cs.CL]` . url: https://arxiv.org/abs/2409.13710. 

- Heimersheim, Stefan and Jake Mendel (2024). “Activation plateaus & sensitive directions in GPT2”. In: _LessWrong_ . url: https://www.lesswrong.com/posts/LajDyGyiyX8DNNsuF/ interim-research-report-activation-plateaus-and-sensitive-1. 

- Heinzerling, Benjamin and Kentaro Inui (2024). “Monotonic Representation of Numeric Properties in Language Models”. In: _arXiv preprint arXiv:2403.10381_ . 

- Hendrycks, Dan, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt (2021). “Aligning AI With Shared Human Values”. In: _Proceedings of the International Conference on Learning Representations (ICLR)_ . 

- Hendrycks, Dan, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (2020). “Measuring massive multitask language understanding”. In: _arXiv preprint arXiv:2009.03300_ . 

- Higham, Nicholas J. (May 2021). _Singular Value Inequalities_ . url: https://nhigham.com/ 2021/05/04/singular-value-inequalities. 

- Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021). _LoRA: Low-Rank Adaptation of Large Language Models_ . arXiv: 2106.09685 `[cs.CL]` . url: https://arxiv.org/abs/2106.09685. 

- Huang, Jing, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger (2024). “RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations”. In: _arXiv preprint arXiv:2402.17700_ . 

- Jiang, Albert Q, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. (2023). “Mistral 7B”. In: _arXiv preprint arXiv:2310.06825_ . 

- Jiang, Yibo, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, and Victor Veitch (2024). “On the Origins of Linear Representations in Large Language Models”. In: _arXiv preprint arXiv:2403.03867_ . 

- Johannes Welbl Nelson F. Liu, Matt Gardner (2017). “Crowdsourcing Multiple Choice Science Questions”. In. 

- Johnson, William B. and Joram Lindenstrauss (1984). “Extensions of Lipschitz mappings into a Hilbert space”. In: _Conference in modern analysis and probability (New Haven, Conn., 1982)_ . Vol. 26. Contemporary Mathematics. Providence, RI: American Mathematical Society, pp. 189–206. isbn: 0-8218-5030-X. doi: 10.1090/conm/026/737400. 

- Juang, Caden, Goncalo Paulo, Jacob Drori, and Nora Belrose (2024). _Open Source Automated Interpretability for Sparse Autoencoder Features_ . url: https://blog.eleuther.ai/autointerp/. 

- Kantamneni, Subhash and Max Tegmark (2025). “Language Models Use Trigonometry to Do Addition”. In: _arXiv preprint arXiv:2502.00873_ . 

- Karvonen, Adam, Dhruv Pai, Mason Wang, and Ben Keigwin (Dec. 2024). “Sieve: SAEs Beat Baselines on a Real-World Task (A Code Generation Case Study)”. In: _Tilde Research Blog_ . Blog post. url: https://www.tilderesearch.com/blog/sieve. 

154 

- Karvonen, Adam, Can Rager, et al. (Dec. 2024a). _SAEBench: A Comprehensive Benchmark for Sparse Autoencoders_ . Accessed: 2025-01-20. url: https://www.neuronpedia.org/saebench/info. 

- (2024b). _SAEBench: A Comprehensive Benchmark for Sparse Autoencoders_ . url: https: //www.neuronpedia.org/sae-bench/info. 

- Karvonen, Adam, Benjamin Wright, Can Rager, Rico Angell, Jannik Brinkmann, Logan Smith, Claudio Mayrink Verdun, David Bau, and Samuel Marks (2024). “Measuring progress in dictionary learning for language model interpretability with board game models”. In: _arXiv preprint arXiv:2408.00113_ . 

- (n.d.). “Measuring progress in dictionary learning for language model interpretability with board game models, 2024”. In: _URL https://arxiv. org/abs/2408.00113_ (), p. 16. 

- Kasaraneni, Chaitanya Krishna (n.d.). _Medical Text — kaggle.com_ . [Accessed 21-01-2025]. url: https://www.kaggle.com/datasets/chaitanyakck/medical-text. 

- Kissane, Connor, Robert Krzyzanowski, Neel Nanda, and Arthur Conmy (2024). _SAEs Are Highly Dataset Dependent: A Case Study on the Dataset_ . url: https : / / www . alignmentforum.org/posts/rtp6n7Z23uJpEH7od/saes-are-highly-dataset-dependent-acase-study-on-the. 

- Kotari, Rajesh (n.d.). _rkotari/clickbait · Datasets at Hugging Face — huggingface.co_ . [Accessed 21-01-2025]. url: https://huggingface.co/datasets/rkotari/clickbait. 

- Kutsyk, Taras, Tommaso Mencattini, and Ciprian Florea (2024). _Do Sparse Autoencoders (SAEs) Transfer Across Base and Target?_ url: https://www.alignmentforum.org/posts/ bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and. 

- Lad, Vedang, Wes Gurnee, and Max Tegmark (2024). “The Remarkable Robustness of LLMs: Stages of Inference?” In: _arXiv preprint arXiv:2406.19384_ . 

- Lai, Peter and Stefan Heimersheim (2024). _SAE regularization produces more interpretable models_ . url: https : / / www . lesswrong . com / posts / sYFNGRdDQYQrSJAd8 / sae - regularization-produces-more-interpretable-models. 

- Lai, Peter and Winston Huang (2024). _GPT-2 Circuits_ . Accessed: 2025-01-30. url: https: //peterlai.github.io/gpt-mri/. 

- Lattanzi, Silvio, Thomas Lavastida, Kefu Lu, and Benjamin Moseley (2020). “A framework for parallelizing hierarchical clustering methods”. In: _Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019, Würzburg, Germany, September 16–20, 2019, Proceedings, Part I_ . Springer, pp. 73–89. 

- Lee, Daniel and Stefan Heimersheim (2024). “Investigating Sensitive Directions in GPT2: An Improved Baseline and Comparative Analysis of SAEs”. In: _LessWrong_ . url: https://www.lesswrong.com/posts/dS5dSgwaDQRoWdTuu/investigating-sensitivedirections-in-gpt-2-an-improved. 

- Li, Kenneth, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg (2022). “Emergent world representations: Exploring a sequence model trained on a synthetic task”. In: _arXiv preprint arXiv:2210.13382_ . 

- Li, Yuxiao, Eric J Michaud, David D Baek, Joshua Engels, Xiaoqing Sun, and Max Tegmark (2024). “The geometry of concepts: Sparse autoencoder feature structure”. In: _arXiv preprint arXiv:2410.19750_ . 

- Lieberum, Tom, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda (2024a). 

155 

_Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2_ . eprint: arXiv:2408.05147. 

- Lieberum, Tom, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda (2024b). “Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2”. In: _arXiv preprint arXiv:2408.05147_ . 

- Lin, Johnny (2023). _Neuronpedia: Interactive Reference and Tooling for Analyzing Neural Networks_ . Software available from neuronpedia.org. url: https://www.neuronpedia.org. 

- Lin, Stephanie, Jacob Hilton, and Owain Evans (2021). “Truthfulqa: Measuring how models mimic human falsehoods”. In: _arXiv preprint arXiv:2109.07958_ . 

- (May 2022). “TruthfulQA: Measuring How Models Mimic Human Falsehoods”. In: _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Ed. by Smaranda Muresan, Preslav Nakov, and Aline Villavicencio. Dublin, Ireland: Association for Computational Linguistics, pp. 3214–3252. doi: 10.18653/v1/2022.acl-long.229. url: https://aclanthology.org/2022.acl-long.229/. 

- Lin, Zi, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, and Jingbo Shang (2023). _ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation_ . arXiv: 2310.17389 `[cs.CL]` . 

- Liu, Ziming, Eric Gan, and Max Tegmark (2023). _Seeing is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability_ . arXiv: 2305.08746 `[cs.NE]` . url: https://arxiv. org/abs/2305.08746. 

- Liu, Ziming, Ouail Kitouni, Niklas S Nolte, Eric Michaud, Max Tegmark, and Mike Williams (2022). “Towards understanding grokking: An effective theory of representation learning”. In: _Advances in Neural Information Processing Systems_ 35, pp. 34651–34663. 

- Liu, Ziming, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, and Max Tegmark (2024). _KAN: Kolmogorov-Arnold Networks_ . arXiv: 2404.19756 `[cs.LG]` . url: https://arxiv.org/abs/2404.19756. 

- MacDiarmid, Monte et al. (Apr. 23, 2024). _Simple probes can catch sleeper agents_ . url: https://www.anthropic.com/news/probes-catch-sleeper-agents. 

- Makhzani, Alireza and Brendan Frey (2013). “K-sparse autoencoders”. In: _arXiv preprint arXiv:1312.5663_ . 

- Marks, Samuel, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller (2024). “Sparse feature circuits: Discovering and editing interpretable causal graphs in language models”. In: _arXiv preprint arXiv:2403.19647_ . 

- Marks, Samuel and Max Tegmark (2023). “The geometry of truth: Emergent linear structure in large language model representations of true/false datasets”. In: _arXiv preprint arXiv:2310.06824_ . 

- MedAlpaca (2024). _Medical Meadow Medical Flashcards Dataset_ . https://huggingface.co/ datasets/medalpaca/medical_meadow_medical_flashcards. Accessed: 2025-01-30. 

- Mendel, Jake (2024). “SAE feature geometry is outside the superposition hypothesis”. In: _AI Alignment Forum_ . url: https://www.alignmentforum.org/posts/MFBTjb2qf3ziWmzz6/ sae-feature-geometry-is-outside-the-superposition-hypothesis. 

- Michaud, Eric J, Isaac Liao, Vedang Lad, Ziming Liu, Anish Mudide, Chloe Loughridge, Zifan Carl Guo, Tara Rezaei Kheirkhah, Mateja Vukelić, and Max Tegmark (2024). “Opening 

156 

the AI black box: program synthesis via mechanistic interpretability”. In: _arXiv preprint arXiv:2402.05110_ . 

- Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean (2013). “Distributed representations of words and phrases and their compositionality”. In: _Advances in neural information processing systems_ 26. 

- Mikolov, Tomáš, Wen-tau Yih, and Geoffrey Zweig (2013). “Linguistic regularities in continuous space word representations”. In: _Proceedings of the 2013 conference of the north american chapter of the association for computational linguistics: Human language technologies_ , pp. 746–751. 

- Morwani, Depen, Benjamin L Edelman, Costin-Andrei Oncescu, Rosie Zhao, and Sham Kakade (2023). “Feature emergence via margin maximization: case studies in algebraic tasks”. In: _arXiv preprint arXiv:2311.07568_ . 

- Mudide, Anish, Joshua Engels, Eric J. Michaud, Max Tegmark, and Christian Schroeder de Witt (2024). _Efficient Dictionary Learning with Switch Sparse Autoencoders_ . arXiv: 2410.08201 `[cs.LG]` . url: https://arxiv.org/abs/2410.08201. 

- Mueller, Aaron, Jannik Brinkmann, Millicent Li, Samuel Marks, Koyena Pal, Nikhil Prakash, Can Rager, Aruna Sankaranarayanan, Arnab Sen Sharma, Jiuding Sun, et al. (2024). “The quest for the right mediator: A history, survey, and theoretical grounding of causal interpretability”. In: _arXiv preprint arXiv:2408.01416_ . 

- Mur, Marieke, Peter A Bandettini, and Nikolaus Kriegeskorte (2009). “Revealing representational content with pattern-information fMRI—an introductory guide”. In: _Social cognitive and affective neuroscience_ 4.1, pp. 101–109. 

- Nanda, Neel and Joseph Bloom (2022). _TransformerLens_ . url: https : / / github . com / TransformerLensOrg/TransformerLens. 

- Nanda, Neel, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt (2023). “Progress measures for grokking via mechanistic interpretability”. In: _arXiv preprint arXiv:2301.05217_ . 

- Nanda, Neel, Andrew Lee, and Martin Wattenberg (2023). “Emergent linear representations in world models of self-supervised sequence models”. In: _arXiv preprint arXiv:2309.00941_ . 

- Olah, Chris (May 2023). “Interpretability Dreams”. In: _Transformer Circuits_ . url: https: //transformer-circuits.pub/2023/interpretability-dreams/index.html. 

- (2024). “What is a Linear Representation? What is a Multidimensional Feature?” In: _Transformer Circuits Thread_ . url: https://transformer-circuits.pub/2024/july-update/ index.html#linear-representations. 

- Olah, Chris, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter (2020). “Zoom In: An Introduction to Circuits”. In: _Distill_ . doi: 10.23915/distill. 00024.001. 

- Olmo, Jeffrey, Jared Wilson, Max Forsey, Bryce Hepner, Thomas Vin Howe, and David Wingate (2024). _Features that Make a Difference: Leveraging Gradients for Improved Dictionary Learning_ . arXiv: 2411.10397 `[cs.LG]` . url: https://arxiv.org/abs/2411.10397. 

- OpenAI (July 2024a). “GPT-4o mini: advancing cost-efficient intelligence”. In: url: https: //openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/. 

- (2024b). _Introducing OpenAI o1_ . [Accessed 29-01-2025]. url: https://openai.com/o1/. 

- (2024c). _Learning to Reason with Language Models_ . url: https://openai.com/index/ learning-to-reason-with-llms/. 

157 

- OpenAI (n.d.). _Hello GPT-4o_ . [Accessed 27-01-2025]. url: https://openai.com/index/hellogpt-4o/. 

- Pain (2024). _Arabic Tweets Dataset_ . https://huggingface.co/datasets/pain/Arabic-Tweets. Accessed: 2025-01-30. 

- Pang, Bo and Lillian Lee (2005). “Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales”. In: _Proceedings of the ACL_ . 

- Park, Kiho, Yo Joong Choe, and Victor Veitch (2023a). “The linear representation hypothesis and the geometry of large language models”. In: _arXiv preprint arXiv:2311.03658_ . 

- (2023b). “The linear representation hypothesis and the geometry of large language models”. In: _arXiv preprint arXiv:2311.03658_ . 

- Peng, Baolin, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao (2023). “Instruction tuning with gpt-4”. In: _arXiv preprint arXiv:2304.03277_ . 

- Pennington, Jeffrey, Richard Socher, and Christopher D Manning (2014). “Glove: Global vectors for word representation”. In: _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ , pp. 1532–1543. 

- Pres, Itamar, Laura Ruis, Ekdeep Singh Lubana, and David Krueger (2024). _Towards Reliable Evaluation of Behavior Steering Interventions in LLMs_ . arXiv: 2410.17245 `[cs.AI]` . url: https://arxiv.org/abs/2410.17245. 

- Radford, Alec, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. (2019). “Language models are unsupervised multitask learners”. In: _OpenAI blog_ . 

- Rajamanoharan, Senthooran, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, and Neel Nanda (2024a). _Improving Dictionary Learning with Gated Sparse Autoencoders_ . arXiv: 2404.16014 `[cs.LG]` . url: https://arxiv.org/abs/2404. 16014. 

- (2024b). “Improving dictionary learning with gated sparse autoencoders”. In: _arXiv preprint arXiv:2404.16014_ . 

- Rajamanoharan, Senthooran, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda (2024a). _Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders_ . arXiv: 2407.14435 `[cs.LG]` . url: https: //arxiv.org/abs/2407.14435. 

- (2024b). “Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders”. In: _arXiv preprint arXiv:2407.14435_ . 

- Riviere, Morgane, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, et al. (2024). _Gemma 2: Improving Open Language Models at a Practical Size_ . eprint: arXiv:2408.00118. 

- Rogers, Anna, Olga Kovaleva, Matthew Downey, and Anna Rumshisky (2020). “Getting Closer to AI Complete Question Answering: A Set of Prerequisite Real Tasks”. In: _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The ThirtySecond Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020_ . AAAI Press, pp. 8722–8731. url: https: //aaai.org/ojs/index.php/AAAI/article/view/6398. 

- Roudranil (2024). _Shakespearean and Modern English Conversational Dataset_ . https:// huggingface.co/datasets/Roudranil/shakespearean-and-modern-english-conversationaldataset. Accessed: 2025-01-30. 

158 

- Shai, Adam, Paul Riechers, Lucas Teixeira, Alexander Oldenziel, and Sarah Marzen (2024). _Transformers Represent Belief State Geometry in their Residual Stream_ . url: https: //www.alignmentforum.org/posts/gTZ2SxesbHckJ3CkF/transformers-represent-beliefstate-geometry-in-their. 

- Sharkey, Lee et al. (2025). _Open Problems in Mechanistic Interpretability_ . arXiv: 2501.16496 `[cs.LG]` . url: https://arxiv.org/abs/2501.16496. 

- Slattery, Peter, Alexander K Saeri, Emily AC Grundy, Jess Graham, Michael Noetel, Risto Uuk, James Dao, Soroush Pour, Stephen Casper, and Neil Thompson (2024). “The ai risk repository: A comprehensive meta-review, database, and taxonomy of risks from artificial intelligence”. In: _arXiv preprint arXiv:2408.12622_ . 

- Smith, Lewis (2024a). _Replacing SAE Encoders with Inference-Time Optimisation_ . url: https://www.alignmentforum.org/s/AtTZjoDm8q3DbDT8Z/p/C5KAZQib3bzzpeyrg. 

- (2024b). “The ‘strong’ feature hypothesis could be wrong”. In: _AI Alignment Forum_ . url: https://www.alignmentforum.org/posts/tojtPCCRpKLSHBdpn/the-strong-featurehypothesis-could-be-wrong. 

- Smith, Logan Riggs and Jannik Brinkmann (July 2024). _Interpreting Preference Models w/ Sparse Autoencoders_ . Accessed: 2025-01-23. url: https://www.alignmentforum.org/posts/ 5XmxmszdjzBQzqpmz/interpreting-preference-models-w-sparse-autoencoders. 

- Stathead (n.d.). _Stathead: Your all-access ticket to the Sports Reference database. | Stathead.com — stathead.com_ . [Accessed 21-01-2025]. url: https://stathead.com/. 

- Stolfo, Alessandro, Yonatan Belinkov, and Mrinmaya Sachan (2023). “A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis”. In: _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pp. 7035–7052. 

- Syed, Aaquib, Can Rager, and Arthur Conmy (2023). “Attribution Patching Outperforms Automated Circuit Discovery”. In: _arXiv preprint arXiv:2310.10348_ . 

- Tafjord, Oyvind, Matt Gardner, Kevin Lin, and Peter Clark ("2019"). “"QUARTZ: An Open-Domain Dataset of Qualitative Relationship Questions"”. In. 

- Taggart, Glen M. (2024). _ProLU: A Nonlinearity for Sparse Autoencoders_ . https://www. alignmentforum.org/posts/HEpufTdakGTTKgoYF/prolu-a-nonlinearity-for-sparseautoencoders. 

- Talmor, Alon, Ori Yoran, Ronan Le Bras, Chandra Bhagavatula, Yoav Goldberg, Yejin Choi, and Jonathan Berant (2022). “CommonsenseQA 2.0: Exposing the limits of AI through gamification”. In: _arXiv preprint arXiv:2201.05320_ . 

- Tasissa, Abiy, Emmanouil Theodosis, Bahareh Tolooshams, and Demba Ba (2020). “Towards improving discriminative reconstruction via simultaneous dense and sparse coding”. In: _arXiv preprint arXiv:2006.09534_ . 

- Tasissa, Abiy, Manos Theodosis, Bahareh Tolooshams, and Demba E Ba (n.d.). “Discriminative reconstruction via simultaneous dense and sparse coding”. In: _Transactions on Machine Learning Research_ (). 

- Team, Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. (2023). “Gemini: a family of highly capable multimodal models”. In: _arXiv preprint arXiv:2312.11805_ . 

- Team, Gemma, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, 

159 

et al. (2024). “Gemma 2: Improving open language models at a practical size”. In: _arXiv preprint arXiv:2408.00118_ . 

- Tegmark, Max and Steve Omohundro (2023). “Provably safe systems: the only path to controllable agi”. In: _arXiv preprint arXiv:2309.01933_ . 

- Templeton, Adly et al. (2024). “Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet”. In: _Transformer Circuits Thread_ . url: https://transformercircuits.pub/2024/scaling-monosemanticity/index.html. 

- Wang, Alex, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman (2019). “GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding”. In: _International Conference on Learning Representations_ . url: https: //openreview.net/forum?id=rJ4km2R5t7. 

- Wang, Kevin, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt (2022). “Interpretability in the wild: a circuit for indirect object identification in gpt-2 small”. In: _arXiv preprint arXiv:2211.00593_ . 

- Warstadt, Alex, Amanpreet Singh, and Samuel R. Bowman (2019). “Neural Network Acceptability Judgments”. In: _Transactions of the Association for Computational Linguistics_ 7. Ed. by Lillian Lee, Mark Johnson, Brian Roark, and Ani Nenkova, pp. 625–641. doi: 10.1162/tacl_a_00290. url: https://aclanthology.org/Q19-1040/. 

- Weber, Maurice et al. (2024). _RedPajama: an Open Dataset for Training Large Language Models_ . arXiv: 2411.12372 `[cs.CL]` . url: https://arxiv.org/abs/2411.12372. 

- Wu, Zhengxuan, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts (2025). _AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders_ . arXiv: 2501.17148 `[cs.CL]` . url: https://arxiv.org/abs/2501.17148. 

- Yedidia, Adam (2023a). _GPT-2’s Positional Embedding Matrix is a Helix_ . Accessed: 2024-0916. url: https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positionalembedding-matrix-is-a-helix. 

- (2023b). _The positional embedding matrix and previous-token heads: how do they actually work?_ url: https://www.lesswrong.com/posts/zRA8B2FJLtTYRgie6/the-positionalembedding-matrix-and-previous-token-heads-how. 

- Yun, Zeyu, Yubei Chen, Bruno A Olshausen, and Yann LeCun (2021). “Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors”. In: _arXiv preprint arXiv:2103.15949_ . 

- Zellers, Rowan, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi (2019). “Hellaswag: Can a machine really finish your sentence?” In: _arXiv preprint arXiv:1905.07830_ . 

- Zhang, Fred and Neel Nanda (2023). “Towards best practices of activation patching in language models: Metrics and methods”. In: _arXiv preprint arXiv:2309.16042_ . 

- Zhong, Ziqian, Ziming Liu, Max Tegmark, and Jacob Andreas (2024). “The clock and the pizza: Two stories in mechanistic explanation of neural networks”. In: _Advances in Neural Information Processing Systems_ 36. 

- Zou, Andy, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. (2023). “Representation engineering: A top-down approach to ai transparency”. In: _arXiv preprint arXiv:2310.01405_ . 

160 

