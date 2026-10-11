# The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks

**arXiv:** 2306.17844 · **Source:** https://arxiv.org/abs/2306.17844
**Route:** markdown copied from `~/Papers/Symbolic_Embeddings/corpus/markdown/literature/2306.17844.md`. That library does not record which PDF converter produced it. Single-column source, so no column interleaving. Equations and numbers resolve against the PDF.
**Authors:** Ziqian Zhong, Ziming Liu, Max Tegmark, Jacob Andreas. NeurIPS 2023.
**Corpus section:** Tyler Wheeler/interpretability
**Cited by:** NOT CITED BY WHEELER. Added for the audit phase of thread 01 (research/thread-01-amortized-inference/statement.md).
**Why it is here:** Networks trained on one task can implement different algorithms depending on seed and hyperparameters. Why a mechanism found in one ensemble member must be checked in the others.
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. Method literature, almost all on language models.
**Copied:** 2026-09-25

---

# **The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks** 

**Ziqian Zhong*, Ziming Liu*, Max Tegmark, Jacob Andreas** 

Massachusetts Institute of Technology `{ziqianz, zmliu, tegmark, jda}@mit.edu` 

## **Abstract** 

Do neural networks, trained on well-understood algorithmic tasks, reliably rediscover known algorithms for solving those tasks? Several recent studies, on tasks ranging from group arithmetic to in-context linear regression, have suggested that the answer is yes. Using modular addition as a prototypical problem, we show that algorithm discovery in neural networks is sometimes more complex. Small changes to model hyperparameters and initializations can induce discovery of qualitatively different algorithms from a fixed training set, and even parallel implementations of multiple such algorithms. Some networks trained to perform modular addition implement a familiar _Clock_ algorithm (previously described by Nanda et al. [1]); others implement a previously undescribed, less intuitive, but comprehensible procedure we term the _Pizza_ algorithm, or a variety of even more complex procedures. Our results show that even simple learning problems can admit a surprising diversity of solutions, motivating the development of new tools for characterizing the behavior of neural networks across their algorithmic phase space.<sup>1</sup> 

## **1 Introduction** 

Mechanistically understanding deep network models—reverse-engineering their learned algorithms and representation schemes—remains a major challenge across problem domains. Several recent studies [2, 3, 4, 5, 1] have exhibited specific examples of models apparently re-discovering interpretable (and in some cases familiar) solutions to tasks like curve detection, sequence copying and modular arithmetic. Are these models the exception or the rule? Under what conditions do neural network models discover familiar algorithmic solutions to algorithmic tasks? 

In this paper, we focus specifically on the problem of learning modular addition, training networks to compute sums like 8 + 6 = 2 (mod 12). Modular arithmetic can be implemented with a simple geometric solution, familiar to anyone who has learned to read a clock: every integer is represented as an angle, input angles are added together, and the resulting angle evaluated to obtain a modular sum (Figure 1, left). Nanda et al. [1] show that specific neural network architectures, when trained to perform modular addition, implement this _Clock_ algorithm. In this work, we show that the _Clock_ algorithm is only one part of a more complicated picture of algorithm learning in deep networks. In particular, networks structurally similar to the ones trained by Nanda et al. preferentially implement a qualitatively different approach to modular arithmetic, which we term the _Pizza_ algorithm (Figure 1, right), and sometimes even more complex solutions. Models exhibit sharp _algorithmic phase transitions_ [6] between the _Clock_ and _Pizza_ algorithms as their width and attention strength very, and often implement multiple, imperfect copies of the _Pizza_ algorithm in parallel. 

> *Equal contribution. 

> 1Code is available at https://github.com/fjzzq2002/pizza. 

37th Conference on Neural Information Processing Systems (NeurIPS 2023). 



<!-- Start of picture text -->
Input a ,  b Step 1 : Embed token  and  to a circle where  a b wk  = 2 πk / p  for some  k  ∈[1,2⋯,  p  −1]<br>a  → E a  ≡( E a , x ,  E a , y ) = (cos( wka ), sin( wka )),  b  → E b  ≡( E b , x ,  E b , y ) = (cos( wkb ), sin( wkb ))<br>Embed E a ,  E b<br>Clock  Algorithm Pizza  Algorithm<br>MLP or  E ab Step 2 : compute the angle sum using multiplication. Step 2.1 : compute the vector mean.<br>Transformer H ab E ab  ≡ ( EE abab ,, yx ) = ( EE aa ,, xx EE bb ,, yx  + −  EE aa ,, yy EE bb ,, yx ) = (cos(sin( wwkk (( aa  + +  b b ))))) Step 2.2E ab  = (: using      and nonlinearities to compute E a  +  E b E )/2 = (cos( ab wka ) + cos( wkb ), sin( wka ) + sin( H ab wkb ))/2<br>Unembed ⋅ U c H ab  =  E ab H ab  = | cos( wk ( a  − b )/2) | (cos( wk ( a  +  b )), sin( wk ( a  +  b )))<br>Logit Qabc Step 3 : score possible outputs  using a dot product. c<br>Qabc  =  U c  ⋅ H ab , U c  ≡( E c , x ,  E c , y ) = (cos( wkc ), sin( wkc ))<br>Qabc (Clock) = cos( wk ( a  +  b  − c )) Qabc (Pizza) = cos( wk ( a  − b )/2) cos( wk ( a  +  b  − c ))<br>12 12<br>11 1 11 1 E ab  = E a + 2  E b<br>910 θaθabθ b  = E  θaa E  + b θb 23 Same-label predictions collapse to a number on a  clock 910 EaEb 23 Same-label predictions  are spread over two  slices of 11+6 pizza 12+5 1+4 2+3=5 (mod 12)<br>8 E ab 4 8 4 9+810+7<br>7 6 5 2+3=1+4=12+5=11+6 5 7 6 5<br>=10+7=9+8=5 (mod 12)<br><!-- End of picture text -->

Figure 1: Illustration of the _Clock_ and the _Pizza_ Algorithm. 

Our results highlight the complexity of mechanistic description in even models trained to perform simple tasks. They point to characterization of algorithmic phase spaces, not just single algorithmic solutions, as an important goal in algorithm-level interpretability. 

**Organization** In Section 2, we review the _Clock_ algorithm [1] and show empirical evidence of deviation from it in models trained to perform modular addition. In Section 3, we show that these deviations can be explained by an alternative _Pizza_ algorithm. In Section 4, we define additional metrics to distinguish between these algorithms, and detect phase transitions between these algorithms (and others _Non-circular_ algorithms) when architectures and hyperparameters are varied. We discuss the relationship between these findings and other work on model interpretation in Section 5, and conclude in Section 6. 

## **2 Modular Arithmetic and the** **_Clock_ Algorithm** 

**Setup** We train neural networks to perform modular addition _a_ + _b_ = _c_ (mod _p_ ), where _a, b, c_ = 0 _,_ 1 _, · · · , p −_ 1. We use _p_ = 59 throughout the paper. In these networks, every integer _t_ has an associated embedding vector **E** _t ∈_ R<sup>_d_</sup> . Networks take as input embeddings [ **E** _a,_ **E** _b_ ] _∈_ R<sup>2</sup><sup>_d_</sup> and predict a categorical output _c_ . Both embeddings and network parameters are learned. In preliminary experiments, we train two different network architectures on the modular arithmetic task, which we refer to as: Model A and Model B. **Model A** is a one-layer ReLU transformer [7] with constant attention, while **Model B** is a standard one-layer ReLU transformer (see Appendix F.1 for details). As attention is not involved in Model A, it can also be understood as a ReLU MLP (Appendix G). 

### **2.1 Review of the** **_Clock_ Algorithm** 

As in past work, we find that after training both Model A and Model B, embeddings ( **E** _a,_ **E** _b_ in Figure 1) usually describe a circle [8] in the plane spanned by the first two principal components of the embedding matrix. Formally, **E** _a ≈_ [cos( _wka_ ) _,_ sin( _wka_ )] where _wk_ = 2 _πk/p_ , _k_ is an integer in [1 _, p −_ 1]. Nanda et al. [1] discovered a circuit that uses these circular embeddings to implement an interpretable algorithm for modular arithmetic, which we call the _Clock_ algorithm. 

|Algorithm|Learned Embeddings|Gradient Symmetry|Required Non-linearity|
|---|---|---|---|
|Clock|Circle|No|Multiplication|
|Pizza|Circle|Yes|Absolute value|
|Non-circular|Line,Lissajous-like curves,etc.|N/A|N/A|



Table 1: Different neural algorithms for modular addition 

2 

"If a meeting starts at 10, and lasts for 3 hours, then it will end at 1." This familiar fact is a description of a modular sum, 10+3 = 1 (mod 12), and the movement of a clock describes a simple algorithm for modular arithmetic: the numbers 1 through 12 are arranged on a circle in 360<sup>_◦_</sup> _/_ 12 = 30<sup>_◦_</sup> increments, angles of 10 _×_ 30<sup>_◦_</sup> and 3 _×_ 30<sup>_◦_</sup> are added together, then this angle is evaluated to determine that it corresponds to 1 _×_ 30<sup>_◦_</sup> . 

Remarkably, Nanda et al. [1] find that neural networks like our Model B implement this _Clock_ algorithm, visualized in Figure 1 (left): they represent tokens _a_ and _b_ as 2D vectors, and adding their polar angles using trigonometric identities. Concretely, the _Clock_ algorithm consists of three steps: In step 1, tokens _a_ and _b_ are embedded as **E** _a_ = [cos( _wka_ ) _,_ sin( _wka_ )] and **E** _b_ = [cos( _wkb_ ) _,_ sin( _wkb_ )], respectively, where _wk_ = 2 _πk/p_ (an everyday clock has _p_ = 12 and _k_ = 1). Then the polar angles of **E** _a_ and **E** _b_ are added (in step 2) and extracted (in step 3) via trigonometric identities. For each candidate output _c_ , we denote the logit _Qabc_ ; the predicted output is _c_<sup>_∗_</sup> = argmax _c Qabc_ . 

Crucial to this algorithm is the fact that the attention mechanism can be leveraged to perform multiplication. What happens in model variants when the attention mechanism is absent, as in Model A? We find two pieces of evidence of deviation from the _Clock_ algorithm in Model A. 

### **2.2 First Evidence for** **_Clock_ Violation: Gradient Symmetricity** 

Since the _Clock_ algorithm has logits: 



(see Figure 1) the gradients of _Qabc_ generically lack permutation symmetry in argument order: _∇_ **E** _a Qabc̸_ = _∇_ **E** _b Qabc_ . Thus, if learned models exhibit permutation symmetry ( _∇_ **E** _a Qabc_ = _∇_ **E** _b Qabc_ ), they must be implementing some other algorithm. 

We compute the 6 largest principal components of the input embedding vectors. We then compute the gradients of output logits (unnormalized log-probabilities from the model) with respect to the input embeddings. We then project them onto these 6 principal components (since the angles relevant to the _Clock_ and _Pizza_ algorithms are encoded in the first few principal components). These projections are shown in Figure 2. While Model B demonstrates asymmetry in general, Model A exhibits gradient symmetry. 



<!-- Start of picture text -->
(,b,c)=(7.49,33) (a,b.c)=(37,28,16)<br>“3324030 . “ “|| sS402 3 . Pay “)<br>E20 < #20 ’<br>—¥e<br>=10 “ 2 10 “<br>: °° : Pa<br>Zo “ Zoo wo<br>2 a 3<br>go m4 g710 Ae<br>Fj5-20 A “ 25-20 <a“<br>3£301,af € ©© ModelAPcAGradiéntsModelB PCA Gradients || 2| ag30] |-“4" ©@ ModelMode! 8APCA PCA Gradients,Gradients<br>i -20 «0 +2~=«20S AO io 00a<br>Gradients on 1st token embeddings E,<br>Gradients on 1st token embeddingsE,<br><!-- End of picture text -->

Figure 2: Gradients on first six principal components of input embeddings. ( _a, b, c_ ) in the title stands for taking gradients on the output logit _c_ for input ( _a, b_ ). x and y axes represent the gradients for embeddings of the first and the second token. The dashed line _y_ = _x_ signals a symmetric gradient. 

### **2.3 Second Evidence for** **_Clock_ Violation: Logit Patterns** 

Inspecting models’ outputs, in addition to inputs, reveals further differences. For each input pair ( _a, b_ ), we compute the output logit assigned to the correct label _a_ + _b_ . We visualize these _correct logits_ from Models A and B in Figure 3. Notice that the rows are indexed by _a − b_ and the columns by _a_ + _b_ . From Figure 3, we can see that the correct logits of Model A have a clear dependency on _a − b_ in that within each row, the correct logits are roughly the same, while this pattern is not observed in Model B. This suggests that Models A and B are implementing different algorithms. 

3 



<!-- Start of picture text -->
Correct Logits of Model A Correct Logits of Model B<br>a epesamact gs of Made A :<br>6(Se 6eatioct atorHemelaateiate P27<br>EE o Ae ] ; :<br>2 3 ed eee ee<br>E 40 oe £50 eon eee ee ee 2<br>Bc on . 4 hike<br><!-- End of picture text -->

Figure 3: Correct Logits of Model A & Model B. The correct logits of Model A (left) have a clear dependence on _a − b_ , while those of Model B (right) do not. 

## **3 An Alternative Solution: the** **_Pizza_ Algorithm** 

How does Model A perform modular arithmetic? Whatever solution it implements must exhibit gradient symmetricity in Figure 2 and the output patterns in Figure 3. In this section, we describe a new algorithm for modular arithmetic, which we call the _Pizza_ algorithm, and then provide evidence that this is the procedure implemented by Model A. 

### **3.1 The** **_Pizza_ Algorithm** 

Unlike the _Clock_ algorithm, the _Pizza_ algorithm operates _inside_ the circle formed by embeddings (just as pepperoni are spread all over a pizza), instead of operating on the circumference of the circle. The basic idea is illustrated in Figure 1: given a fixed label _c_ , for _all_ ( _a, b_ ) with _a_ + _b_ = _c_ (mod _p_ ), the points **E** _ab_ = ( **E** _a_ + **E** _b_ ) _/_ 2 lie on a line though the origin of a 2D plane, and the points closer to this line than to the lines corresponding to any other _c_ form two out of 2 _p_ mirrored “pizza slices”, as shown at the right of the figure. Thus, to perform modular arithmetic, a network can determine which slice pair the average of the two embedding vectors lies in. Concretely, the _Pizza_ algorithm also consists of three steps. Step 1 is the same as in the _Clock_ algorithm: the tokens _a_ and _b_ are embedded at **E** _a_ = (cos( _wka_ ) _,_ sin( _wka_ )) and **E** _b_ = (cos( _wkb_ ) _,_ sin( _wkb_ )), respectively. Step 2 and Step 3 are different from the _Clock_ algorithm. In Step 2.1, **E** _a_ and **E** _b_ are averaged to produce an embedding **E** _ab_ . In Step 2.2 and Step 3, the polar angle of **E** _ab_ is (implicitly) computed by computing the logit _Qabc_ for any possible outputs _c_ . While one possibility of doing so is to take the absolute value of the dot product of **E** _ab_ with (cos( _wkc/_ 2) _,_ sin( _wkc/_ 2)), it is not commonly observed in neural networks (and will result in a different logit pattern). Instead, Step 2.2 transforms **E** _ab_ into a vector encoding _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ (cos( _wk_ ( _a_ + _b_ )) _,_ sin( _wk_ ( _a_ + _b_ ))), which is then dotted with the output embedding _Uc_ = (cos( _wkc_ ) _,_ sin( _wkc_ )). Finally, the prediction is _c_<sup>_∗_</sup> = argmax _cQabc_ . See Appendix A and Appendix L for a more detailed analysis of a neural circuit that computes **H** _ab_ in a real network. 

The key difference between the two algorithms lies in what non-linear operations are required: _Clock_ requires multiplication of inputs in Step 2, while _Pizza_ requires only absolute value computation, which is easily implemented by the ReLU layers. If neural networks lack inductive biases toward implementing multiplication, they may be more likely to implement _Pizza_ rather than _Clock_ , as we will verify in Section 4. 

### **3.2 First Evidence for** **_Pizza_ : Logit Patterns** 

Both the _Clock_ and _Pizza_ algorithms compute logits _Qabc_ in Step 3, but they have different forms, shown in Figure 1. Specifically, _Qabc_ ( _Pizza_ ) has an extra multiplicative factor _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ compared to _Qabc_ ( _Clock_ ). As a result, given _c_ = _a_ + _b_ , _Qabc_ ( _Pizza_ ) is dependent on _a − b_ , but _Qabc_ ( _Clock_ ) is not. The intuition for the dependence is that a sample is more likely to be classified correctly if **E** _ab_ is longer. The norm of this vector depends on _a − b_ . As we observe in Figure 3, the logits in Model A indeed exhibit a strong dependence on _a − b_ . 

4 

### **3.3 Second Evidence for** **_Pizza_ : Clearer Logit Patterns via Circle Isolation** 

To better understand the behavior of this algorithm, we replace the embedding matrix **E** with a series of rank-2 approximations: using only the first and second principal components, or only the third and fourth, etc. For each such matrix, embeddings lie in a a two-dimensional subspace. For both Model A and Model B, we find that embeddings form a circle in this subspace (Figure 4 and Figure 5, bottom). We call this procedure _circle isolation_ . Even after this drastic modification to the trained models’ parameters, both Model A and Model B continue to behave in interpretable ways: a subset of predictions remain highly accurate, with this subset determined by the periodicity of the _k_ of the isolated circle. As predicted by the _Pizza_ and _Clock_ algorithms described in Figure 1, Model A’s accuracy drops to zero at specific values of _a − b_ , while Model B’s accuracy is invariant in _a − b_ . Applying circle isolation to Model A on the two principal components (one circle) yields a model with 32 _._ 8% overall accuracy, while retaining the first six principal components (three circles) yields an overall accuracy of 91 _._ 4%. See Appendix D for more discussion. By contrast, Model B achieves 100% when embeddings are truncated to the first six principal components. Circle isolation thus reveals an _error correction_ mechanism achieved via ensembling: when an algorithm (clock or pizza) exhibits systematic errors on subset of inputs, models can implement multiple algorithm variants in parallel to obtain more robust predictions. 



<!-- Start of picture text -->
Correct Logits of Model A after Circle Isolation<br>3g3<br>:— E é<br>Cite rom Principal Component 1,2 ice fam Principal Component 3,4 Cc from Principal Component 5, © Cle from Principal Component 11,12<br>)ss pt 0 FeBey LLne gl1 oe ne | _| poot Shain.My 03 eeePUR BE Py<br>a2] 3. Fs 5, - | ae ,,<br>A a] ool set elle 8) oo<br>% a) 8 FA BS 4<br>aa q a he PP4, ceoeenena $F} co sB creews. 002<br>ane& #7a | | onPoe |s* ol ,by, geroa|) oyis 2ra<br>ae BY ov gaae cel aan ot hited os te ty grbsat<br><!-- End of picture text -->

Figure 4: Correct logits of Model A ( _Pizza_ ) after circle isolation. The rightmost pizza is accompanying the third pizza (discussed in Section 3.4 and Appendix D). _Top:_ The logit pattern depends on _a − b_ . _Bottom:_ Embeddings for each circle. 

Using these isolated embeddings, we may additionally calculate the isolated logits directly with formulas in Figure 1 and compare with the actual logits from Model A. Results are displayed in Table 2. We find that _Qabc_ ( _Pizza_ ) explains substantially more variance than _Qabc_ ( _Clock_ ). 

**Why do we only analyze correct logits?** The logits from the _Pizza_ algorithm are given by _Qabc_ ( _Pizza_ ) = _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b − c_ )). By contrast, the _Clock_ algorithm has logits _Qabc_ ( _Clock_ ) = cos( _wk_ ( _a_ + _b − c_ )). In a word, _Qabc_ ( _Pizza_ ) has an extra multiplicative factor _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ compared to _Qabc_ ( _Clock_ ). By constraining _c_ = _a_ + _b_ (thus cos( _wk_ ( _a_ + _b − c_ )) = 1), the factor _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ can be identified. 

**(Unexpected) dependence of logits** _Qabc_ ( _Clock_ ) **on** _a_ + _b_ : Although our analysis above expects logits _Qabc_ ( _Clock_ ) not to depend on _a − b_ , they do not predict its dependence on _a_ + _b_ . In Figure 5, we surprisingly find that _Qabc_ ( _Clock_ ) is sensitive to this sum. Our conjecture is that Step 1 and Step 2 of the _Clock_ are implemented (almost) noiselessly, such that same-label samples collapse to the same point after Step 2. However, Step 3 (classification) is imperfect after circle isolation, resulting in fluctuations of logits. Inputs with common sums _a_ + _b_ produce the same logits. 

5 



<!-- Start of picture text -->
Correct Logits of Model & after Cirle Isolation<br>Cicle fom Pinal Component 1,2 __ Cle fom Pinal Camponent 3.4 Cle fam Pini Component. Cc om Pini Component 7.8<br>“| san?” Frseny, ” oe ener, oar,<br>“| ye Say, a “ig | oul ge *, - Sa,<br>of Bod f | | ff 4 * e,<br>ool B) ol # 4) 9) £ a) olf<br>col FA a OUR 3 # ard 4<br>circle #1,549 Hoo B— cuclew2.oa21 Bool ciclo #3,0-14 co ciceesonr 8<br>oe% BY 07) AhB) ool ae aodn esi<br>= Hoon, geet 06) eaves gent 7 Peeves <> Poy oped<br><!-- End of picture text -->

Figure 5: Correct logits of Model B ( _Clock_ ) after circle isolation. _Top:_ The logit pattern depends on _a_ + _b_ . _Bottom:_ Embeddings for each circle. 

|Circle|_wk_||_Qabc_(clock)FVE|_Qabc_(pizza)FVE|
|---|---|---|---|---|
|#1|2_π_|_/_59_·_17|75.41%|99.18%|
|#2|2_π_|_/_59_·_3|75.62%|99.18%|
|#3|2_π_|_/_59_·_44|75.38%|99.28%|



Table 2: After isolating circles in the input embedding, fraction of variance explained (FVE) of **all** Model A’s output logits (59 _×_ 59 _×_ 59 of them) by various formulas. Both model output logits and formula results’ are normalized to mean 0 variance 1 before taking FVE. _wk_ ’s are calculated according to the visualization. For example, distance between 0 and 1 in Circle #1 is 17, so _wk_ = 2 _π/_ 59 _·_ 17. 

### **3.4 Third Evidence for** **_Pizza_ : Accompanied & Accompanying Pizza** 

The Achilles’ heel of the _Pizza_ algorithm is antipodal pairs. If two inputs ( _a, b_ ) happen to lie antipodally, then their middle point will lie at the origin, where the correct “pizza slice” is difficult to identify. For example in Figure 1 right, antipodal pairs are (1,7), (2,8), (3,9) etc., whose middle points all collapse to the origin, but their class labels are different. Models cannot distinguish between, and thus correctly classify, these pairs. Even for odd _p_ ’s where there are no strict antipodal pairs, approximately antipodal pairs are also more likely to be classified incorrectly than non-antipodal pairs. 

Intriguingly, neural networks find a clever way to compensate for this failure mode. we find that pizzas usually come with “accompanying pizzas”. An accompanied pizza and its accompanying pizza complement each other in the sense that near-antipodal pairs in the accompanied pizza become adjacent or close (i.e, very non-antipodal) in the accompanying pizza. If we denote the difference between adjacent numbers on the circle as _δ_ and _δ_ 1, _δ_ 2 for accompanied and accompanying pizzas, respectively, then _δ_ 1 = 2 _δ_ 2 (mod _p_ ). In the experiment, we found that pizzas #1/#2/#3 in Figure 4 all have accompanying pizzas, which we call pizzas #4/#5/#6 (see Appendix D for details). However, these accompanying pizzas do not play a significant role in final model predictions<sup>2</sup> . We conjecture that training dynamics are as follows: (1) At initialization, pizzas #1/#2/#3 correspond to three different “lottery tickets” [9]. (2) In early stages of training, to compensate the weaknesses (antipodal pairs) of pizzas #1/#2/#3, pizzas #4/#5/#6 are formed. (3) As training goes on (in the presence of weight decay), the neural network gets pruned. As a result, pizzas #4/#5/#6 are not significantly involved in prediction, although they continue to be visible in the embedding space. 

> 2Accompanied pizzas #1/#2/#3 can achieve 99.7% accuracy, but accompanying pizzas #4/#5/#6 can only achieve 16.7% accuracy. 

6 

## **4 The Algorithmic Phase Space** 

In Section 3, we have demonstrated a typical _Clock_ (Model A) and a typical _Pizza_ (Model B). In this section, we study how architectures and hyperparametes govern the selection of these two algorithmic “phases”. In Section 4.1, we propose quantitative metrics that can distinguish between _Pizza_ and _Clock_ . In Section 4.2, we observe how these metrics behave with different architectures and hyperparameters, demonstrating sharp phase transitions. The results in this section focus _Clock_ and _Pizza_ models, but other algorithmic solutions to modular addition are also discovered, and explored in more detail in Appendix B. 

### **4.1 Metrics** 

We wish to study the distribution of _Pizza_ and _Clock_ algorithms statistically, which will require us to distinguish between two algorithms automatically. In order to do so, we formalize our observations in Section 2.2 and 2.3, arriving at two metrics: **gradient symmetricity** and **distance irrelevance** . 

### **4.1.1 Gradient Symmetricity** 

To measure the symmetricity of the gradients, we select some input-output group ( _a, b, c_ ), compute the gradient vectors for the output logit at position _c_ with respect to the input embeddings, and then compute the cosine similarity. Taking the average over many pairs yields the gradient symmetricity. **Definition 4.1** (Gradient Symmetricity) **.** _For a fixed set S ⊆_ Z<sup>3</sup> _p_<sup>_ofinput-outputpairs_3</sup><sup>_,define_</sup> **_gradient-symmetricity_** _of a network M with embedding layer E as_ 



_where sim_ ( _a, b_ ) = _|aa||·bb|_<sup>_is the cosine-similarity, Qabcis the logit for class c given input a and b.It is_</sup> _clear that sg ∈_ [ _−_ 1 _,_ 1] _._ 

As we discussed in Section 2.2, the _Pizza_ algorithm has symmetric gradients while the _Clock_ algorithm has asymmetric ones. Model A and Model B in Section 3 have gradient symmetricity 99 _._ 37% and 33 _._ 36%, respectively (Figure 2). 

### **4.1.2 Distance Irrelevance** 

To measure the dependence of correct logits on differences between two inputs, which reflect the distances of the inputs on the circles, we measure how much of the variance in the correct logit matrix depends on it. We do so by comparing the average standard deviation of correct logits from inputs with the same differences and the standard deviation from all inputs. 

**Definition 4.2** (Distance Irrelevance) **.** _For some network M with correct logit matrix L (Li,j_ = _Qij,i_ + _j), define its_ **_distance irrelevance_** _as_ 



_where_ std _computes the standard deviation of a set. It is clear that q ∈_ [0 _,_ 1] _._ 

Model A and Model B in Section 3 give distance irrelevance 0.17 and 0.85, respectively (Figure 3). A typical distance irrelevance from the _Pizza_ algorithm ranges from 0 to 0.4 while a typical distance irrelevance from _Clock_ algorithm ranges from 0.4 to 1. 

### **4.1.3 Which Metric is More Decisive?** 

When the two metrics have conflicting results, which one is more decisive? We consider distance irrelevance as the decisive factor of the _Pizza_ algorithm, as the output logits being dependent on the distance is highly suggestive of _Pizza_ . On the other hand, gradient symmetricity can be used to rule out the _Clock_ algorithm, as it requires multiplying (transformed) inputs which will result in asymmetric gradients. Figure 6 confirmed that at low distance irrelevance (suggesting pizza) the gradient symmetricity is almost always close to 1 (suggesting non-clock). 

> 3To speed-up the calculations, in our experiments _S_ is taken as a random subset of Z3 _p_<sup>of size 100.</sup> 

7 



<!-- Start of picture text -->
52> 10 7== eR oC IOEY EY isMNAAS Gtk<br>gz 05 Meee EES She NeRa pees<br>&<br>0.2 0.4 0.6 0.8 1.0<br>Distance Irrelevance<br><!-- End of picture text -->

Figure 6: Distance irrelevance vs gradient symmetricity over all the main experiments. 

### **4.2 Identifying algorithmic phase transitions** 

How do models “choose” whether to implement the _Clock_ or _Pizza_ algorithm? We investigate this question by interpolating between Model A (transformer without attention) and Model B (transformer with attention). To do so, we introduce a new hyperparameter _α_ we call the **attention rate** . 

For a model with attention rate _α_ , we modify the attention matrix _M_ for each attention head to be _M_<sup>_′_</sup> = _Mα_ + _J_ (1 _− α_ ). In other words, we modify this matrix to consist of a linear interpolation between the all-one matrix and the original attention (post-softmax), with the rate _α_ controlling how much of the attention is kept. The transformer with and without attention corresponds to the case where _α_ = 1 (attention kept) and _α_ = 0 (constant attention matrix). With this parameter, we can control the balance of attention versus linear layers in transformers. 

We performed the following set of experiments on transformers (see Appendix F.1 for architecture and training details). (1) One-layer transformers with width 128 and attention rate uniformly sampled in [0 _,_ 1] (Figure 7). (2) One-layer transformers with width log-uniformly sampled in [32 _,_ 512] and attention rate uniformly sampled in [0 _,_ 1] (Figure 7). (3) Transformers with 2 to 4 layers, width 128 and attention rate uniformly sampled in [0 _,_ 1] (Figure 11). 

**The** **_Pizza_ and the** **_Clock_ algorithms are the dominating algorithms with circular embeddings.** For circular models, most observed models either have low gradient symmetricity (corresponding to the _Clock_ algorithm) or low distance irrelevance (corresponding to the _Pizza_ algorithm). 

**Two-dimensional phase change observed for attention rate and layer width.** For the fixed-width experiment, we observed a clear phase transition from the _Pizza_ algorithm to the _Clock_ algorithm (characterized by gradient symmetricity and distance irrelevance). We also observe an almost linear phase boundary with regards to both attention rate and layer width. In other words, the attention rate transition point increases as the model gets wider. 

**Dominance of linear layers determines whether the** **_Pizza_ or the** **_Clock_ algorithm is preferred.** For one-layer transformers, we study the transition point against the attention rate and the width: 

- The _Clock_ algorithm dominates when the attention rate is higher than the phase change point, and the _Pizza_ algorithm dominates when the attention rate is lower than the point. Our explanation is: At a high attention rate, the attention mechanism is more prominent in the network, giving rise to the clock algorithm. At a low attention rate, the linear layers are more prominent, giving rise to the pizza algorithm. 

- The phase change point gets higher when the model width increases. Our explanation is: When the model gets wider, the linear layers become more capable while the attention mechanism receive less benefit (attentions remain scalars while outputs from linear layers become wider vectors). The linear layer therefore gets more prominence with a wider model. 

**Possibly hybrid algorithms between the** **_Clock_ and the** **_Pizza_ algorithms.** The continuous phase change suggests the existence of networks that lie between the _Clock_ and the _Pizza_ algorithms. This is achievable by having some principal components acting as the _Clock_ and some principal components acting as the _Pizza_ . 

8 



<!-- Start of picture text -->
Gradient Symmetricity Distance Irrelevance<br>0.2 0.4 0.6 08 0.2 0.4 0.6 08<br>1.0 : 1.0 Tike<br>a§ 0.6 otawe .the DeBeheite ” aewet] gE 06 HeereeeH: aSs a<br>3 0.4 foe sigs eee oe 204 : wed<br>5 aT eee s H . ee<br>3B 0.2 Wears re B 0.2 H "<br>ae Meo. 6 :<br>0.0 + 0.0 +<br>00 02 O04 06 08 10 00 02 O04 06 08 10<br>Attention Rate Attention Rate<br>Gradient Symmetricity Distance irrelevance<br>02 04 06 08 02 04 06 08<br>-. tee ar : oon ere<br>256 @fee aigeteee eeQe eT 256 & ‘ he eesa Tereee<br>= peewee ee, eke og oe] Bf tee eae es<br>Scie A CO oO a<br>Oe Oe ea Oty Aeoe<br>“fee32 @®) | “Lire32 OSYR®<br>00 ©=6020«(04) 06) B10 00 8402 04 06 08 10<br>Attention Rate Attention Rate<br><!-- End of picture text -->

Figure 7: Training results from 1-layer transformers. Each point in the plots represents a training run reaching circular embeddings and 100% validation accuracy. See Appendix C for additional plots. _Top:_ Model width fixed to be 128. _Bottom:_ Model width varies. The phase transition lines are calculated by logistic regression (classify the runs by whether gradient symmetricity _>_ 98% and whether distance irrelevance _<_ 0 _._ 6). 

**Existence of non-circular algorithms.** Although our presentation focuses on circular algorithms (i.e., whose embeddings are circular), we find non-circular algorithms (i.e., whose embeddings do not form a circle when projected onto any plane) to be present in neural networks. See Appendix B for preliminary findings. We find that deeper networks are more likely to form non-circular algorithms. We also observe the appearance of non-circular networks at low attention rates. Nevertheless, the _Pizza_ algorithm continues to be observed (low distance irrelevance, high gradient symmetricity). 

## **5 Related Work** 

**Mechanistic interpretability** aims to mechanically understand neural networks by reverse engineering them [2, 3, 5, 4, 10, 11, 1, 12, 13, 14]. One can either look for patterns in weights and activations by studying single-neuron behavior (superposition [11], monosemantic neurons [15]), or study meaningful modules or circuits grouped by neurons [4, 14]. Mechanistic interpretability is closely related to training dynamics [8, 13, 1]. 

**Learning mathematical tasks** : Mathematical tasks provide useful benchmarks for neural network interpretability, since the tasks themselves are well understood. The setup could be learning from images [16, 17], with trainable embeddings [18], or with number as inputs [19, 5]. Beyond arithmetic relations, machine learning has been applied to learn other mathematical structures, including geometry [20], knot theory [21] and group theory [22]. 

**Algorithmic phase transitions** : Phase transitions are present in classical algorithms [23] and in deep learning [6, 24, 25]. Usually the phase transition means that the algorithmic performance sharply 

9 

changes when a parameter is varied (e.g., amount of data, network capacity etc). However, the phase transition studied in this paper is _representational_ : both clock and pizza give perfect accuracy, but arrive at answers via different interal computations. These model-internal phase transitions are harder to study, but closer to corresponding phenomena in physical systems [24]. 

**Algorithm learning in neural networks** : Emergent abilities in deep neural networks, especially large language models, have recently attracted significant attention [26]. An ability is “emergent” if the performance on a subtask suddenly increases with growing model sizes, though such claims depend on the choice of metric [27]. It has been hypothesized that the emergence of specific capability in a model corresponds to the emergence of a modular circuit responsible for that capability, and that emergence of some model behaviors thus results from a sequence of quantized circuit discovery steps [5]. 

## **6 Conclusions** 

We have offered a closer look at recent findings that familiar algorithms arise in neural networks trained on specific algorithmic tasks. In modular arithmetic, we have shown that such algorithmic discoveries are not inevitable: in addition to the _Clock_ algorithm reverse-engineered by [1], we find other algorithms (including a _Pizza_ algorithm, and more complicated procedures) to be prevalent in trained models. These different algorithmic phases can be distinguished using a variety of new and existing interpretability techniques, including logit visualization, isolation of principle components in embedding space, and gradient-based measures of model symmetry. These techniques make it possible to _automatically_ classify trained networks according to the algorithms they implement, and reveal algorithmic phase transitions in the space of model hyperparameters. Here we found specifically that the emergence of a _Pizza_ or _Clock_ algorithm depends on the relative strength of linear layers and attention outputs. We additionally showed that these algorithms are not implemented in isolation; instead, networks sometimes ensemble multiple copies of an algorithm in parallel. These results offer exciting new challenges for mechanistic interpretability: (1) How to find, classify, and interpret unfamiliar algorithms in a systematic way? (2) How to disentangle multiple, parallel algorithm implementations in the presence of ensembling? 

**Limitations** We have focused on a single learning problem: modular addition. Even in this restricted domain, qualitatively different model behaviors emerge across architectures and seeds. Significant additional work is needed to scale these techniques to the even more complex models used in real-world tasks. 

**Broader Impact** We believe interpretability techniques can play a crucial role in creating and improving safe AI systems. However, they may also be used to build more accurate systems, with the attendant risks inherent in all dual-use technologies. It is therefore necessary to exercise caution and responsible decision-making when deploying such techniques. 

## **Acknowledgement** 

We would like to thank Mingyang Deng and anonymous reviewers for valuable and fruitful discussions and MIT SuperCloud for providing computation resources. ZL and MT are supported by the Foundational Questions Institute, the Rothberg Family Fund for Cognitive Science and IAIFI through NSF grant PHY-2019786. JA is supported by a gift from the OpenPhilanthropy Foundation. 

## **References** 

- [1] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [2] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. _Distill_ , 2020. https://distill.pub/2020/circuits/zoom-in. 

- [3] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, 

10 

Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. _Transformer Circuits Thread_ , 2022. https://transformer-circuits.pub/2022/in-context-learning-and-inductionheads/index.html. 

- [4] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [5] Eric J Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. _arXiv preprint arXiv:2303.13506_ , 2023. 

- [6] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing Systems_ , 30, 2017. 

- [8] Ziming Liu, Ouail Kitouni, Niklas S Nolte, Eric Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An effective theory of representation learning. _Advances in Neural Information Processing Systems_ , 35:34651–34663, 2022. 

- [9] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In _International Conference on Learning Representations_ , 2019. 

- [10] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. _Transformer Circuits Thread_ , 2021. https://transformer-circuits.pub/2021/framework/index.html. 

- [11] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. _Transformer Circuits Thread_ , 2022. 

- [12] Bilal Chughtai, Lawrence Chan, and Neel Nanda. A toy model of universality: Reverse engineering how networks learn group operations. In _The Fortieth International Conference on Machine Learning_ , 2023. 

- [13] Ziming Liu, Eric J Michaud, and Max Tegmark. Omnigrok: Grokking beyond algorithmic data. In _The Eleventh International Conference on Learning Representations_ , 2023. 

- [14] Arthur Conmy, Augustine N Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. _arXiv preprint arXiv:2304.14997_ , 2023. 

- [15] Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. _arXiv preprint arXiv:2305.01610_ , 2023. 

- [16] Yedid Hoshen and Shmuel Peleg. Visual learning of arithmetic operation. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , 2016. 

- [17] Samuel Kim, Peter Y Lu, Srijon Mukherjee, Michael Gilbert, Li Jing, Vladimir Ceperi´c, and<sup>ˇ</sup> Marin Soljaˇci´c. Integration of neural network-based symbolic regression in deep learning for scientific discovery. _IEEE Transactions on Neural Networks and Learning Systems_ , 32(9):4166– 4177, 2020. 

- [18] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. _arXiv preprint arXiv:2201.02177_ , 2022. 

- [19] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: SGD learns parities near the computational limit. _Advances in Neural Information Processing Systems_ , 35:21750–21764, 2022. 

11 

- [20] Yang-Hui He. Machine-learning mathematical structures. _International Journal of Data Science in the Mathematical Sciences_ , 1(01):23–47, 2023. 

- [21] Sergei Gukov, James Halverson, Fabian Ruehle, and Piotr Sułkowski. Learning to unknot. _Machine Learning: Science and Technology_ , 2(2):025035, 2021. 

- [22] Alex Davies, Petar Veliˇckovi´c, Lars Buesing, Sam Blackwell, Daniel Zheng, Nenad Tomašev, Richard Tanburn, Peter Battaglia, Charles Blundell, András Juhász, et al. Advancing mathematics by guiding human intuition with ai. _Nature_ , 600(7887):70–74, 2021. 

- [23] Alexander K Hartmann and Martin Weigt. _Phase transitions in combinatorial optimization problems: basics, algorithms and statistical mechanics_ . John Wiley & Sons, 2006. 

- [24] Lorenza Saitta, Attilio Giordana, and Antoine Cornuejols. _Phase transitions in machine learning_ . Cambridge University Press, 2011. 

- [25] Ekdeep Singh Lubana, Eric J Bigelow, Robert P Dick, David Krueger, and Hidenori Tanaka. Mechanistic mode connectivity. In _International Conference on Machine Learning_ , pages 22965–23004. PMLR, 2023. 

- [26] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. _Transactions on Machine Learning Research_ , 2022. Survey Certification. 

- [27] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? _arXiv preprint arXiv:2304.15004_ , 2023. 

- [28] Michael Hassid, Hao Peng, Daniel Rotem, Jungo Kasai, Ivan Montero, Noah A. Smith, and Roy Schwartz. How much does attention actually attend? questioning the importance of attention in pretrained transformers. In _Findings of the Association for Computational Linguistics: EMNLP 2022_ , pages 1403–1416, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. 

- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Conference on Learning Representations_ , 2019. 

12 

# **Supplementary material** 

## **A Mathematical Analysis and An Example of Pizza Algorithm** 

In the pizza algorithm, we have _Eab_ = cos( _wk_ ( _a − b_ ) _/_ 2) _·_ (cos( _wk_ ( _a_ + _b_ ) _/_ 2) _,_ sin( _wk_ ( _a_ + _b_ ) _/_ 2)), as cos _x_ +cos _y_ = cos(( _x − y_ ) _/_ 2)(2 cos(( _x_ + _y_ ) _/_ 2)) and sin _x_ +sin _y_ = cos(( _x − y_ ) _/_ 2)(2 sin(( _x_ + _y_ ) _/_ 2)). 

To get _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ (cos( _wk_ ( _a_ + _b_ )) _,_ sin( _wk_ ( _a_ + _b_ ))), we generalize this to _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b − u_ )) (the two given cases correspond to _u_ = 0 and _u_ = _π/_ 2 _/wk_ ). 

_|_ (cos( _wku/_ 2) _,_ sin( _wku/_ 2)) _· Eab|_ = _|_ cos( _wk_ ( _a − b_ ) _/_ 2) cos( _wk_ ( _a_ + _b − u_ ) _/_ 2) _|_ 

_|_ ( _−_ sin( _wku/_ 2) _,_ cos( _wku/_ 2)) _· Eab|_ = _|_ cos( _wk_ ( _a − b_ ) _/_ 2) sin( _wk_ ( _a_ + _b − u_ ) _/_ 2) _|_ 

thus their difference will be equal to 

_|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ ( _|_ cos( _wk_ ( _a_ + _b − u_ ) _/_ 2) _| −|_ sin( _wk_ ( _a_ + _b − u_ ) _/_ 2) _|_ ) _._ 

Now notice _|_ cos( _t_ ) _| −|_ sin( _t_ ) _| ≈_ cos(2 _t_ ) for any _t ∈_ R (Figure 8), so the difference is approximately _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b − u_ )). 



<!-- Start of picture text -->
1.00<br>0.75<br>0.50<br>02s<br>0.00<br>~0.25<br>0.50<br>0.75 — cos(2t)<br>1.00 — |costt)}-|sin(t)}<br>0 1 2 3 3 5 6<br><!-- End of picture text -->

Figure 8: _|_ cos( _t_ ) _| −|_ sin( _t_ ) _|_ is approximately cos(2 _t_ ) for any _t ∈_ R 

Plugging in _u_ = 0 and _u_ = _π/_ 2 _/wk_ as mentioned, we get the following particular implementation of the pizza algorithm. 

### **Algorithm: Pizza, Example** 

Step 1 On given input _a_ and _b_ , circularly embed them to two vectors on the circumference (cos( _wka_ ) _,_ sin( _wka_ )) and (cos( _wkb_ ) _,_ sin( _wkb_ )). 

Step 2 Compute: 

_α_ = _|_ cos( _wka_ ) + cos( _wkb_ ) _|/_ 2 _−|_ sin( _wka_ ) + sin( _wkb_ ) _|/_ 2 _≈|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b_ )) _β_ = _|_ cos( _wka_ ) + cos( _wkb_ ) + sin( _wka_ ) + sin( _wkb_ ) _|/_ (2 _<u>√</u>_ 2) _−|_ cos( _wka_ ) + cos( _wkb_ ) _−_ sin( _wka_ ) _−_ sin( _wkb_ ) _|/_ (2 _√_ 2) = _|_ cos( _wka − π/_ 4) + cos( _wkb − π/_ 4) _|/_ 2 _−|_ sin( _wka − π/_ 4) + sin( _wkb − π/_ 4) _|/_ 2 _≈|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b_ ) _− π/_ 2) = _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ sin( _wk_ ( _a_ + _b_ )) 

Step 3 Output of this pizza is computed as a dot product. 

_Q_<sup>_′_</sup> _abc_<sup>=</sup><sup>_α_cos(</sup><sup>_wkc_) +</sup><sup>_β_sin(</sup><sup>_wkc_)</sup><sup>_≈|_cos(</sup><sup>_wk_(</sup><sup>_a −b_)</sup><sup>_/_2)</sup><sup>_|_cos(</sup><sup>_wk_(</sup><sup>_a_+</sup><sup>_b −c_))</sup> 

Similar circuits are observed in the wild, but instead of the above two-term approximation, a more complicated one is observed. See Appendix L for details. 

The extra _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ term is not a coincidence. We can generalize our derivation as the following. 

13 

**Lemma A.1.** _A symmetric function f_ ( _x, y_ ) _that is a linear combination of_ cos _x,_ sin _x,_ cos _y,_ sin _y_<sup>4</sup> _can always be written as_ cos(( _x − y_ ) _/_ 2) _g_ ( _x_ + _y_ ) _for some function g._ 

_Proof._ Notice cos _x_ + cos _y_ = cos(( _x − y_ ) _/_ 2)(2 cos(( _x_ + _y_ ) _/_ 2)) and sin _x_ + sin _y_ = cos(( _x − y_ ) _/_ 2)(2 sin(( _x_ + _y_ ) _/_ 2)), so _α_ (cos _x_ + cos _y_ ) + _β_ (sin _x_ + sin _y_ ) = cos(( _x − y_ ) _/_ 2)(2 _α_ cos(( _x_ + _y_ ) _/_ 2) + 2 _β_ sin(( _x_ + _y_ ) _/_ 2)). ~~ao~~ 

This is why we consider the output pattern with the _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ terms rather than the actual computation circuits as the determinant feature of the pizza algorithm. 

## **B Non-Circular algorithms** 

One thing that further complicates our experiment is the existence of non-circular embeddings. While only circular algorithms are reported in the previous works [8, 1], many non-circular embeddings are found in our experiments, e.g., 1D lines or 3D Lissajous-like curves, as shown in Figure 9. We leave the detailed analysis of these non-circular algorithms for future study. Since circular algorithms are our primary focus of study, we propose the following metric **circularity** to filter out non-circular algorithms. The metric reaches maximum 1 when the principal components aligns with cosine waves. 



<!-- Start of picture text -->
os<br>-050.0 —Principal Component 1<br>16°50°25. 01734 9 43 18.52.27. 2 36 11 45<br>20/54294 38 13 47 22 56 31 6 40 15 49 24 58<br><!-- End of picture text -->



<!-- Start of picture text -->
§°57,g fF we Bost,6 . 5 054% Soe<br>E . . +. {8 : . g e<br>os oe a € of, ee eo Lh ee<br>§ 00 zat al | B00) wy Rea te Y<br>Sos} | ste | Bosye . ~ # | Bos vee .<br>i tee? tt] Eno . . + | E10 *:<br>-05 00 05 -05 00 05 “05 00 05<br>Principal Component 1 Principal Component 1 Principal Component 2<br><!-- End of picture text -->

Figure 9: Visualization of the principal components of input embeddings for two trained non-circular models. _Top:_ A line-like first principal component. Notice the re-arranged x axis (token id). _Bottom:_ First three principal components forming a three-dimensional non-circular pattern. Each point represents the embedding of a token. 

**Definition B.1** (Circularity) **.** _For some network, suppose the l-th principal component of its input embeddings is vl,_ 0 _, vl,_ 1 _, · · · , vl,p−_ 1 _, define its_ **_circularity_** _based on first four components as_ 



_where_ **_i_** _is the imaginary unit. c ∈_ [0 _,_ 1] _by Fourier analysis. c_ = 1 _means first four components are Fourier waves._ 

Both Model A and Model B in Section 3 have a circularity around 99 _._ 8% and we consider models with circularity _≥_ 99 _._ 5% **circular** . 

## **C More Results from the Main Experiments** 

Here we provide Figure 7 with non-circular networks unfiltered (Figure 10). We can see more noise emerging in the plot. We also provide the training results from multi-layer transformers (Figure 11). 

> 4The actual neural networks could be more complicated - even if our neural network is locally linear and symmetric, locally they could be asymmetric (e.g. _|x|_ + _|y|_ could locally be _x − y_ ). Nevertheless, the pattern is observed in our trained networks. 

14 



<!-- Start of picture text -->
Gradient Symmetricity Distance Irrelevance<br>02 0.4 0.6 0.8 02 04 06 08<br>a =” eageae] ot<br>1.0 a — TE 1.0 a<br>8 oe Bian,rs Piet S| Bb Satestreaties<br>eRe eaeesineerys, e- ieee<br>o a a on Pea<br>Zo25  itesda gameieaneh tacraentenegenetetenttes =circular* gB02 @ circular 2 Bey<br>° ees ‘m= non-circular | © non-circular<br>0.0 0.0<br>00 02 O04 06 08 10 00 02 O04 06 O08 10<br>Attention Rate Attention Rate<br><!-- End of picture text -->



<!-- Start of picture text -->
Gradient Symmetricity Distance Irrelevance<br>02 04 «406 08 02 «04 «206 08<br>512 yee =e a ae,3" 512 » Tele eae, ye<br>SeahSeead Tie peneeseeAE Be ahty EAT<br>52 MOTE SPA sheeee foeaebt ees a rg 256 fatS Theeeoe inaig ot ragt ahsRy<br>S hepsi ial tee 8 ble Sine erat,<br>=3 we #8Fa ogtel saepee ssooSeayteeds tas eycs| |SSeptoehee eeeeeeeDak"xa1<br>= foaled ee a eee at oad, § eeeeee ty<br>4 drtmt i ane non-circular| 64 yrs SETS SGEE, mnon-circular<br>Se EE ese. circular wig AES ye circular |<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Attention Rate Attention Rate<br><!-- End of picture text -->

Figure 10: Training results from 1-layer transformers. Each point in the plots represents a training run reaching 100% validation accuracy. Among all the trained 1-layer transformers, 34.31% are circular. _Top:_ Model width fixed to be 128. _Bottom:_ Model width varies. 

## **D Pizzas Come in Pairs** 

Cautious readers might notice that the pizza algorithm is imperfect - for near antipodal points, the sum vector will have a very small norm and the result will be noise-sensitive. While the problem is partially elevated by the use of multiple circles instead of one, we also noticed another pattern emerged: **accompanying pizzas** . 

The idea is the following: suppose the difference between adjacent points is 2 _k_ mod _p_ , then the antipodal points have difference _±k_ . Therefore, if we arrange a new circle with a difference _k_ for adjacent points, we will get a pizza that works **best** for formerly antipodal points. 

### **Algorithm: Accompanying Pizza** 

- Step 1 Take _wk_ as of the accompanied pizza. On given input _a_ and _b_ , circularly embed them to two vectors on the circumference (cos(2 _wka_ ) _,_ sin(2 _wka_ )) and (cos(2 _wkb_ ) _,_ sin(2 _wkb_ )). 

Step 2 Compute the midpoint: 



Step 3 Output of this pizza is computed as a dot product. 



This is exactly what we observed in Model A (Table 3, Figure 13). With the six circles (pizzas and accompanying pizzas) included in the embedding, Model A also gets 100% accuracy. 

15 



<!-- Start of picture text -->
02 Gradient04 symmetricity06 08 02 Distance04 Irrelevance06 08 0.4 Gradient0.6 Symmetricity08<br>10ee— — Logeenee 10en<br>8 0.8 fiete Bebet Se A Sosy. ¢ .<br>BO eR a soap) E ee .<br>S06 {th eased cae so] E06 Sr $06) . . .<br>g FORSTER Wyte sc] § * | é :<br>Poe ec ie ee ed 20.4 sts) Soa, .<br>Bo2ti° patatages te.)*em. circularnon-circular'| oOBo2 mw@ = circularnoncicuir,| °BortiP neywt ilwe @ =ttcircular<br>0.000 02 04 06 08 10 0.000 02 04 06 08 10 0.000 02 04 06 08 10<br>Attention Rate Attention Rate Attention Rate<br><!-- End of picture text -->



<!-- Start of picture text -->
02Gradient04 Symmetricity06 08 02 Distance04 irrelevance06 08 oa Gradient0.6 Symmetricity0.8<br>10eeSRSte — 10en—— 10en<br>SnSP ET rE meres ;<br>BOONB Seer e wtSiete]ree| 2BOSEPeFLT eeeGheeeT E087;. eewee :<br>B06ae {ttyglthtretre teoeseetTD] Eos MNMamie | $0636) Ft. we<br>oa kageOakee a ee Zoag awangay ESO4qeye etaee, F<br>Bo2qr te, Ape a nomcircular | B02 wm non-circular} 2o2]2 +5 peSiwett<br>0.000 02 04 @ =06circular08 10 Oo 0.000 02 04 @ 06circular08 104 0.000 02 04 06@ circular08 10<br>Attention Rate Attention Rate Attention Rate<br><!-- End of picture text -->



<!-- Start of picture text -->
Gradient Symmetrcity Distance irrelevance Gradient Symmetrcity<br>00°02 04 06 08 02 04 06 08 02 04 06 08<br>10SE SESE: lS:<br>5 oe 10 — 1.0 =<br>2LosiSo ARTEPe Te,Peeg t-aeeEy TS]Fedo | opFog Pefe atestiePee Sy FestiPaige. ne algLos ° a. * °<br>Soe} fees ers regeR) BME Se ties] 8 .<br>tree tatet ee |] Eoefs tee “TERE | S06 tte<br>goss sR se | Zod is Cray $04 . .<br>£024 7 ee gh sm noncircular | 80.2 so, @ non-ciecularf 202] "a yt 4. *<br>0.000 02 we 04 @ =06circular08 10 oO 0.000 02 04 @ =06circular—-—08 10 0.000 02 . 04 06@ = circular08 10<br>Attention Rate Attention Rate Attention Rate<br><!-- End of picture text -->

Figure 11: Training results from transformers with 2, 3 and 4 layers. Among all the trained transformers with 2, 3 and 4 layers, 9.95%, 11.55% and 6.08% are circular, respectively. 



<!-- Start of picture text -->
Accompanied and Accompanying Pizza (p = 11)<br>0 0<br>9 t/owt 2 10 ows\ OH > e]<br>“direction<br>8 045; 3 4 7<br>Hi - direction<br>7A49<br>65 3 8<br>(0+1 —+ larger norm —+ more likely to be correct +1 —» sum points at a point + negative result<br>045 —+ smaller norm —* more likely to be wrong _0+5 —» sum points between points —* positive result<br><!-- End of picture text -->

Figure 12: An Illustration on the Accompanying Pizza Algorithm 

16 

|Circle|_wk_||_Ac_ FVE|
|---|---|---|---|
|#4(accompanying#1)|2_π/_|59_·_17|97.56%|
|#5(accompanying#2)|2_π/_|59_·_3|97.23%|
|#6 (accompanying #3)|2_π/_|59_·_44|97.69%|



Table 3: After isolating accompanying circles in the input embedding, fraction of variance explained (FVE) of **all** Model A’s output logits by various formulas. Both model output logits and formula results’ are normalized to mean 0 variance 1 before taking FVE. Accompanying and accompanied pizza have the same _wk_ . 



<!-- Start of picture text -->
Correct Logits of Model A after Accompanying Circle Isolation<br>0 3000 a3 40 5 0 3 ws ee ee<br>D3 6 9 2usteziza2730s536sbHzasdasisesy O36 9 12151671242730353630"24540515457 03 6 9 12151871242730393639424540515657<br>(a+b) mod 59 (a+b) mod 59 (a+b) mod 59<br>Circle from Principal Component 6, 7 Circle from Principal Component 8, 9 Circle from Principal Component 10, 12<br>oa03 as 2¥°h, ows 56 RE sayy os 270 86,,02F7  6,<br>BS "D1 ages 03 4 3 i 5)<br>oo) ed,an e | af 3Pro ea] 3) 8owt ss3,<br>os| aa? B] ° of aul™ #,| 4 7 #<br>6<br>oo} gxo Circle #4, 6-26 Boo 3 Circle #5,5=10 458] 00 2 Circle #6, 5=2 E<br>-o2}-01] ‘52 aMa) or] B ° oe| 0] 50, Fax<br>“4s, + “0 = £ 3 a<br>03 cy S582 48 "30 02} “S4p5 a<br># ayo oe ‘ ep as? do # 2<br>mae “ 2 oe 3 95% gx3sc0t<br>osa YY Cr Yr rer eT)<br><!-- End of picture text -->

Figure 13: Correct logits of Model A ( _Pizza_ ) after circle isolation. Only accompanying pizzas are displayed. Notice the complementing logit pattern (Figure 4). 

## **E Results in Other Linear Architectures** 

While this is not the primary focus of our paper, we also ran experiments on the following four different linear model setups (see Section F.2 for setup details). 

- For all the models, we first encode input tokens ( _a, b_ ) with a trainable embedding layer _WE_ : _x_ 1 = _WE,a, x_ 2 = _WE,b_ (positional embedding removed for simplicity). _L_ 1 _, L_ 2 _, L_ 3 are trainable linear layers. The outmost layers (commonly referred as unembed layers) have no biases and the other layers have biases included for generality. 

- Model _α_ : calculate output logits as _L_ 2(ReLU( _L_ 1( _x_ 1 + _x_ 2))). 

- Model _β_ : calculate output logits as _L_ 3(ReLU( _L_ 2(ReLU( _L_ 1( _x_ 1 + _x_ 2))))). 

- Model _γ_ : calculate output logits as _L_ 3(ReLU( _L_ 2(ReLU( _L_ 1( _x_ 1) + _L_ 1( _x_ 2))))). 

- Model _δ_ : calculate output logits as _L_ 2(ReLU( _L_ 1([ _x_ 1; _x_ 2]))) ([ _x_ 1; _x_ 2] stands for the concatenation of _x_ 1 and _x_ 2) 

The results are shown in Figure 14. Rather surprisingly, Model _α_ , Model _β_ and Model _δ_ gave radically different results. Model _β_ and Model _γ_ are very similar, and in general they are more pizza-like 

17 



<!-- Start of picture text -->
rotsSe:Sa gs ts a. = 10 [oo vem we<br>os} Parta ge veekies,eect : . B08g<br>sof we atthe eegel teEe ot ‘ FoeE<br>2 yee Tt te | Poa<br>© 0.77 Model Type cae Ble Model Type<br>ea “ee . ha) 3 ea<br>os . “ G es<br>0.6 tole 3 | B02<br>ey6 : oof @ 6 &<br>Os e . e fe<br>02 03 04 05 06 07 08 09 10 02 03 04 05 06 07 08 09 10<br>Distance Irelevancy Distance Irelevancy<br><!-- End of picture text -->



<!-- Start of picture text -->
35 mE Modela | 40 40 mm Modelo | 5. lm Model a’<br>30 30<br>30 30<br>225 25<br>37 20 20 20<br>#15 45<br>10 10 10 10<br>55<br>00 0 0<br>04 05 06 07 04 06 08 092 094 0.96 098 04 05 06 0.7<br>Distance Irrelevancy Distance Irrelevancy Distance Irrelevancy Distance lrrelevancy<br><!-- End of picture text -->

Figure 14: Training results from linear models. Each point in the first-row plots represents a training run. The second row are histograms for distance irrelevancy of each model type. 

than Model _α_ , with lower distance irrelevancy and higher circularity. This could be explained by the addition of an extra linear layer. 

However, Model _δ_ gave very different results from Model _α_ although they are both one-layer linear models. It is more likely to be non-circular and have very high distance irrelevancy in general. In other words, concatenating instead of adding embeddings yields radically different behaviors in one-layer linear model. This result, again, alarmed us the significance of induction biases in neural networks. 

We also want to note that using different embeddings on two tokens of Model _α_ doesn’t resolve the discrepancy. The following model 

- Model _α_<sup>_′_</sup> : calculate output logits as _L_ 2(ReLU( _L_ 1( _x_ 1 + _x_ 2))) where _x_ 1 = _WE,a_<sup>_A,x_2=</sup> _WE,b_<sup>_B_on input (</sup><sup>_a, b_) and</sup><sup>_W A_</sup> _E_<sup>_, W_</sup> _E_<sup>_B_are different embedding layers.</sup> 

gives roughly the same result as of Model _α_ (Figure 14, lower right corner). 

Figure 15 shows the correct logits after circle isolation (Section 3.3) of a circular model from Model _β_ implementing the pizza algorithm. Figure 16 shows the correct logits after circle isolation (Section 3.3) of a circular model from Model _δ_ . We can see the pattern is similar but different from the one of clock algorithm (Figure 5). We leave the study of such models to future work. 

## **F Architecture and Training Details** 

### **F.1 Transformers** 

Here we describe our setup for the main experiments. See Appendix E and Appendix I for experiments on different setups. 

**Architecture** We train bidirectional transformers (attention unmasked) to perform modular addition mod _p_ where _p_ = 59. To calculate ( _a_ + _b_ ) mod _p_ , the input is provided to the model as a sequence 

18 

Figure 15: Correct logits from Model _β_ after circle isolation. 



<!-- Start of picture text -->
o 2 4 6 8 w 2 we 0 2 4 6 8 we we 0 2 8 6 8 lw<br>22<br>g& 8<br>(a+b) mod 59 (ab) mod 59 (ab) mod 59<br>1.00 Circle from Principal Component 1, 2 Circle from Principal Component 3, 4 Circle from Principal Component 5, 6<br>075 ° 8B sotn 1.00 pate co6a gs 1.00 got2d Bos<br>oo) ofe 20 26| onsoso] aP ryMy | 075onl shon" 8,“<br>ons) & x Ses 2<br>Ph o2s| 7 253) 0254 age %<br>-025co! gf3. Circle#1, 6=2 ®2] 000} gar Circle #2,=27 #) O o oo |") R& Circle #3, 6=21 oy<br>osos} | Peg,3 a2ae | -050|) or, v4ae] 025] &3 é<br>30! aa e -050 ‘8 2og<br>nos) 2 075) a gw gsassoe “078 Foon a0<br>100 Se a0 49 S839 “uo re o eC) “bs730 ah00os<br><!-- End of picture text -->

Figure 16: Correct logits from Model _δ_ after circle isolation. 



<!-- Start of picture text -->
40 45 50 55 60 65 70 75 24 8 6 a8 sw<br>B ii Fie Fit = = an qth<br>| ihe > = 3 fath i<br>3PUBial Wilei] TeEe ee E:: Bz eaerat fe|<br>mn - g a = < 1 | i<br>gcat) ili Heth)Hie, 28 BeHe3 == &3 P|ATs ek eae<br>O36 Sazisienizazys03aa6s0ezesaesises7 0-3 912151621242730333630¢24840515657 0-3 6 9 12151421242730333630424540515657<br>(a+b) mod 59 (a+b) mod 59 (a+b) mod 59<br>Circle from Principal Component 1, 2 Circle from Principal Component 3, 4 Circle from Principal Component 5, 6<br>10 50 apr87 sp, 8 79 20, wy al<br>2 “, 10 # . P85. 10 % e, “ )<br>af sy z 2 ne ag<br>03} 2 ; 2 ous eee<br>oo| Seé , a|2 95] # e2 os} es“ya a orgse<br>2 Circle #1, 6=23 aoq 004 °"go Circle #2,6=29 |7 094 esa2 Circle#3, 5=24 ah<br>-o5) %3 Fe 05} * 4 2a -05 “ oF<br>“to we, 5Ps 4, ? we 2 wo 4 8s 3<br>3. # cso] MBs ral hae oan” es<br>Po%p2g9 gE" "ys dg ye @ -10) ’ 2<br>“io 8s oo 0ST “Io 6s doo 0 “io bs o> ~~<br><!-- End of picture text -->

of two tokens [ _a, b_ ]. The output logit at the last token is considered as the output of the model. For a transformer with “width” _d_ , the input embedding and the residue stream will be _d_ -dimensional, 4 attention heads of _⌊d/_ 4 _⌋_ dimensions will be employed, and the MLP will be of 4 _d_ hidden units. By default _d_ = 128 is chosen. ReLU is used as the activation function and layer normalization isn’t applied. The post-softmax attention matrix is interpolated between an all-one matrix and original as 

19 

specified by the attention rate (Section 4.2). We want to point out that the setup of constant-attention transformers is also considered in the previous work [28]. 

**Data** Among all possible data points ( _p_<sup>2</sup> = 3481 of them), we randomly select 80% as training samples and 20% as validation samples. This choice (small _p_ and high training data fraction) helps accelerating the training. 

**Training** We used AdamW optimizer [29] with learning rate _γ_ = 0 _._ 001 and weight decay factor _β_ = 2. We do not use minibatches and the shuffled training data is provided as a whole batch in every epoch. For each run, we start the training from scratch and train for 20 _,_ 000 epoches. We removed the runs that did not reach 100% validation accuracy at the end of the training (majority of the runs reached 100%). 

### **F.2 Linear Models** 

Here we describe our setup for the linear model experiments (Appendix E). 

**Architecture** We train several types of linear models to perform modular addition mod _p_ where _p_ = 59. The input embedding, residue stream and hidden layer are all _d_ = 256 dimensional. ReLU is used as the activation function. The actual structures of network types are specified in Appendix E. 

**Data & Training** Same as in the previous section (Section F.1). 

### **F.3 Computing Resources** 

A total of 226 GPU days of NVidia V100 is spent on this project, although we expect a replication would take significantly fewer resources. 

## **G Mathematical Description of Constant-Attention transformer** 

In this section, we examine the structure of constant-attention transformers loosely following the notation of [10]. 

Denote the weight of embedding layer as _WE_ , the weight of positional embedding as _W_ pos, the weight of the value and output matrix of the _j_ -th head of the _t_ -th layer as _WV_<sup>_t,j_</sup> and _WO_<sup>_t,j_, the weights</sup> and biases of the input linear map of MLP in the _t_ -th layer as _W_ in<sup>_t_and</sup><sup>_bt_</sup> in<sup>, the corresponding weights</sup> and biases of the output linear map as _W_ out<sup>_t_and</sup><sup>_bt_</sup> out<sup>, and the weight of the unembedding layer as</sup><sup>_WU_.</sup> Notice that the query and the key matrices are irrelevant as the attention matrix is replaced with an all-one matrix. Denote _x_<sup>_j_</sup> as the value of residue stream vector after the first _j_ layers and denote _ci_ as the character in the _i_ -th position. We use subscripts like _xt_ to denote taking a specific element of vector. 

We can formalize the logit calculation as the following: 

- Embedding: _x_<sup>0</sup> _i_<sup>=</sup><sup>_WE,c_</sup> _i_<sup>+</sup><sup>_W_pos</sup><sup>_,i_.</sup> 

- For each layer _t_ from 1 to _n_ layer: 

   - **Constant** Attention: _wi_<sup>_t_=</sup><sup>_xt_</sup> _i_<sup>_−_1</sup> +<sup>�</sup> _j_<sup>_W t,j_</sup> _O_<sup>_W t,j_</sup> _V_ � _k_<sup>_xt_</sup> _k_<sup>_−_1</sup> . 

   - **–** MLP: _x_<sup>_t_</sup> = _w_<sup>_t_</sup> + _b_<sup>_t_</sup> out<sup>+</sup><sup>_W t_</sup> out<sup>ReLU(</sup><sup>_bt_</sup> in<sup>+</sup><sup>_W t_</sup> in<sup>_wt_).</sup> 

- Output: _O_ = _WU x_<sup>_n_layer</sup> . 

In the particular case where the input length is 2, the number of layer is 1, and we focus on the logit of the last position, we may restate as the following (denote _z_ as _x_<sup>1</sup> and _y_ as _w_<sup>1</sup> ): 

- Embedding: _x_ 1 = _WE,c_ 1 + _W_ pos _,_ 1 _, x_ 2 = _WE,c_ 2 + _W_ pos _,_ 2. 

- Constant Attention: _y_ = _x_ 2 +<sup>�</sup> _j_<sup>_W j_</sup> _O_<sup>_W j_</sup> _V_<sup>(</sup><sup>_x_1 +</sup><sup>_x_2).</sup> 

- • MLP: _z_ = _y_ + _b_<sup>_t_</sup> out<sup>+</sup><sup>_W t_</sup> out<sup>ReLU(</sup><sup>_bt_</sup> in<sup>+</sup><sup>_W t_</sup> in<sup>_y_).</sup> 

20 

• Output: _o_ = _WU z_ . 

If we remove the skip connections, the network after embedding could be seen as 



where _L_<sup>_j_</sup> _V_<sup>_, Lj_</sup> _O_<sup>_, L_in</sup><sup>_, L_out</sup><sup>_, LU_are a series of linear layers corresponding to the matrices.</sup> 

## **H Pizza with Attention** 

Extrapolating from Figure 7, we trained transformers with width 1024 and attention rate 1 (normal attention). After several tries, we are able to observe a trained circular model with distance irrelevance 0 _._ 156 and gradient symmetricity 0 _._ 995, which fits our definition of _Pizza_ (Figure 17). 



<!-- Start of picture text -->
Circle om Principal Component 2,2 Cele rom Principal Component 34 Circe fom Principal Component 5.6 Cele rm Principal Component 7.8<br>fae)os 4, | °8 laa] jp | on garblan Man | Sarayof as) 5<br>% i 7<br>Be < By 2 oe) | ty<br>ol 4) of a 4 oe | od |<br>cof crctem.o=12 Foo BF cnc wz.o-8 oe z crenen ; vo crater enn Fe<br>ae i) 3, cor] on 7) 714 3"<br>a8 My opewer™a ce %,8 atzpavue 3 oe  ae0)rates* -“ ByDias go as #8<br><!-- End of picture text -->

Figure 17: Correct logits of the trained model in Section H after circle isolation (Section 3.3). 

## **I Results on Slightly Different Setups** 

We considered the following variations of our setups (Appendix F.1, Section 4), for which the existence of pizzas and clocks as well as the phase changes are still observed. 

**GeLU instead of ReLU** We conducted the same 1-layer transformer experiment with activation function GeLU instead of ReLU. Very similar results are observed (Figure 18). 

**Encode Two Tokens Differently** We conducted the 1-layer transformer experiments but with different embedding for the two tokens. Again very similar results are observed (Figure 19). We also discovered that the two tokens’ embeddings are often aligned to implement the _Pizza_ and _Clock_ algorithm (Figure 20). 

**Adding Equal Sign** We conducted the 1-layer transformer experiment with an equal sign added. Very similar results are observed (Figure 21). 

## **J Pizza Occurs Early in the Clock Training** 

We plotted intermediate states during the training of a model with attention (attention rate 1). Pizzalike pattern was observed early in the training, but the pattern gradually disappeared during the run (Figure 22). 

21 

Figure 18: Training results from 1-layer transformers with GeLU instead of ReLU as the activation function. Each point in the plots represents a training run that reached 100% validation accuracy. 



<!-- Start of picture text -->
Gradient Symmetricity Gradient Symmetricity<br>-0.2 00 02 04 06 08 04 0.6 08<br>1.0 aE 1.0<br>.gieee meDrlow aeee. wate)eeaf. gg . . gt . et<br>> . . . att > . "a"<br>a “e ae 3 Se<br>$06] 6 YoogheeT. 5 06 so<br>= . eth = “Oe .<br>§047 2". ee g 0.4 . fee<br>§ oo ‘ ee . 5 : wat<br>oO20.2-entRR Regat, oe 3m@ circularnon-circular OODor.|e Mh« ateee ee . @ circular<br>0.0 0.0<br>00 02 O04 06 08 10 00 02 O04 06 08 10<br>Attention Rate Attention Rate<br>Distance irrelevance Circularity<br>0.2 0.4 0.6 0.8 0.4 0.6 08<br>1.0 — 1.0 EF<br>~ SES aie<br>2 = . -—<br>£08a feys,s “ 28 :te . » oo §eu"Daekee te<br>oO a £08), Ce Do soa?<br>£08 _*, am. .. 2$ 064 eat. aseagheeae<br>2 0.4 st. =$047 oie.. eesi, as<br>B02 @ non-circular B02 foes ob $0 m non-circular<br>o @ circular nt Se See oF @ circular<br>0.0 2 0.0<br>00 O02 O04 06 08 10 00 02 04 06 08 10<br>Attention Rate Attention Rate<br><!-- End of picture text -->

Figure 19: Training results from 1-layer transformers where the two tokens use different embeddings (feed [ _a, b_ + _p_ ] to the model on input ( _a, b_ ); 2 _p_ tokens are handled in the embedding layer). Each point in the plots represents a training run that reached 100% validation accuracy. We did not use circularity to filter the result because it is no longer well-defined. 



<!-- Start of picture text -->
Gradient Symmetricity Distance irrelevance<br>-0.25 0.00 0.25 0.50 0.75 0.2 0.4 0.6 08<br>1.0 1.0 =<br>rh* ose ra As<br>o<br>2087,.gE ue oo. eeltigaebeigdS| 2S08 theshewentg* aie,”oo oe<br>gt.<br>Fr a eewatsteS) § waparigEfond<br>EP gy feels ote FA “ahd ote y<br>GO47s ee ee a . 20.4 ee<br>& [eee Oo eee og .° 5 Seated<br>£02 inseries Jegera, B02 id<br>ar ee fc} . :<br>0.0 0.0<br>00 02 04 06 08 10 00 02 04 06 08 10<br>Attention Rate Attention Rate<br><!-- End of picture text -->

22 

Figure 20: Correct logits after circle isolation from a trained model where two tokens use different embeddings. The blue points represent the embeddings for the first token and the green points represent the embeddings for the second token. The model is implementing the _Pizza_ algorithm. The correct logit pattern is shifted comparing to the previous patterns because the embeddings of two tokens do not line up exactly. For example, the third circle has near-maximum correct logit for _a_ = 6 _, b_ = 3 (the two points lining up on the top) and ( _a − b_ ) _/_ 18 _≡_ 10 (mod 59). This is the reason that the correct logit pattern appears to be shifted 10 units down. 



<!-- Start of picture text -->
os ws wm ms 0 0 Sts co 25 50 75 190 1s 150<br>10:3 6 9 sbubieai24z70333630424549515457 si@ 9 iaisteaizenrs033e304z4548515457 0 3.6 9 12151071242730333639424540513457<br>(ab) mod 59 (a+b) mod 59 (ab) mod 59<br>Circle from Principal Component 1, 2 Circle from Principal Component 3, 4 Circle from Principal Component 5, 6<br>os ’at, ; oa >tPH o4 PF eliat on<br># 3 e Re Fa ot,<br># i 4 a, Z ay<br>oo oe#d Circle #1, 6=19 0S004 # Circle #2,6=17 s| 004 &382 Circle#3, 5=18 is38<br>% &i aom #‘ beya #ao<br>=) #| | Bo] Be #<br>bys, gf Bis iv oy ot<br>veal Play a 2 t:Sp oat By atl®:<br>Sette tans aa 293, exes ~oa Scscicsccial<br>“ba wooo “ie ooo “ie cop do<br><!-- End of picture text -->

Figure 21: Training results from 1-layer transformers where an equal sign is added (feed [ _a, b,_ =] to the model on input ( _a, b_ ) where = is a special token; _p_ + 1 tokens are handled in the embedding layer; context length of the model becomes 3). Each point in the plots represents a training run that reached 100% validation accuracy. We did not use circularity to filter the result because it is no longer well-defined. 



<!-- Start of picture text -->
Gradient Symmetricity Gradient Symmetricity<br>0.90 0.92 0.94 0.96 0.98 0.90 0.92 0.94 0.96 0.98<br>1.0 7 7. 7 he 1.0 7 . os<br>goof208".“one« 4myereeat aedTh aZoey.ol . . . ~ t.tes. :<br>$06,= ar“"t atea",et weray sat $06] 2) . aesee .<br>oa all". anetpea? 8 oF =804 wo.. raewe .<br>D 0.2 + toatl «= * Mm non-circular | 3 0.2 Soothe”<br>° Bite @ circular ° Reedsa @ circular<br>0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 08 1.0<br>Attention Rate Attention Rate<br><!-- End of picture text -->

## **K Accompanying Pizza Occurs Early in the Pizza Training** 

We plotted intermediate states during the training of a model without attention (attention rate 0). We observed the early emergence of a pattern similar to accompanying pizza in training runs (Figure 

23 



<!-- Start of picture text -->
S0Steps  pci2 PCA.3 Po1a Pc2.3 Poza Pca.4<br>OreNa eceins. of Matwe tel Pet‘ err os Ptcae eA i<br>oof Spee] oof all] oof "s MELE] oo oo) tee} oof Serer,<br>cosf as PRT]: gf OE . EA wot"ge Ssst | os wos slote ~0s watesee<br>500 05 5 00 05 5 00 05 “35 00 05 “35 00 05 —t5 00 05<br>Iso. Correct Logits ‘Iso. Correct Logits «Iso. Correct Logits _—so. Correct Logits ‘Iso. Correct Logits_Iso. Correct Logits<br>g ie. 8 g 10 pf po 8 -10<br>z g 0 8 g g sf<br>eel, : § aa H A<br>ay ° 3 o 3 ° 3 ° 3 o & °<br>3 tik 2 2 3 8 2 2<br>210 Steps(a0)pc1,2mod 59 (a+b)PC1, mod 593 (a+b)Pca. mod 4 59 (a+b)P23 mod 59 (a+b)PC moa 2,4 59 (248)Pc 3.4 mod 59<br>oof; ene” | coy, foods | oof te Ba] oof te Pe] oof nets<br>cosfs-is 00si 05 | os -65 00 05 cos "-6sili00 05| 0s 05 a0ee05| “is bedo os vos “isCes00 05<br>g bao 8 -2 8 g -20 & ba<br>FYIso. Corect Logits ‘Iso. Correct Logits «Iso. Correct Logits Iso. Correct Logits Iso. Correct Logits _Iso. Correct Logits<br>3 » 3 ne a (> 3 i 0<br>} i wt vf wf wf »<br>g2go<br>*é° *3 ° *3 -10 3& ° 3= 0 3£ -10<br>(4b) mod 59 (a+b) mod 59 (948) mod 59 (a+b) moa 59 (+8) moa 59 (040) mod 59<br>300 Steps pc1,2 PCL. PCL 4 PC2.3 P24 PC 3,4<br>ofe: of. = of e osfy,t us osf rls osf Aye<br>ao ET] conta TK ook ET] oo eye tye | ool et ef oop tl Ar<br>cos{t ts Mere00 os| -osf 5 00Ye] os is [hh 0005Vee | sf'h-is 00fe 05 wos tsoA00 05 no <o5 00a 05<br>Iso. Correct Logits Iso. Correct Logits _—‘so. Correct Logits Iso. Correct Logits _—_tso. Correct Logits _Iso. Correct Logits<br>iE 8 g g S eeeRe | Lo<br>3 2» g » g3 » 3g » g aa3 °<br>(a+b) moa 39 (a0) mod 59 (a+b) moa 59 (a+b) moa 59 (+2) mod 59 (a0) mod 59<br>510 Steps pc1,2 PC 1.3 Pca PC 2,3 Pc2.4 PC3.4<br>OS) | eee ete | OPEL ae] 05] ete] One oa cae<br>wosfe3500AS 0s| os 3500ce" 0s| cosfhris 000si | -os “35 00ot os | os| “i500Oe 0s os. =35 00ann 05<br>Iso. Correct Logits Igo. Correct Logits ‘Iso. Correct Logits so. Correct Logits Iso. Correct Logits _1s0. Correct Logits<br>g -0 8 js “0 “0 3 40<br>? » E » E nF nt » ! 20<br>g g Fa 8 8 g<br>32 o a8 o 2 ogFy oF3 o 5a °<br>840 Steps(240)pci,2 mod 59 (246)°c1,3 mod 59 (4b) modpc1.4 59 (9+) mod°c2.3 58 (9+) mod°c2.4 58 (a4)°c3.4 mod 59<br>oof fy See tay! oof LEA] 00 re iec i 2<br>“is 00 0s ts 00 05 is 00 05 “is 00 05 “is co os “is 00 05<br>Iso. Correct Logits Iso. Correct Logits _—so. Correct Logits ‘iso. Correct Logits ‘Iso. Correct Logits _Iso. Correct Logits<br>g8 & <0 & -0 3 bio 2 8 L<br>3? lo20 z& 2wo z8 20 ia 20 gé= Lao» g&E 402<br>g3 °F= 0 g8 ° g3 ° &3 ° g8 o<br>2 2 a 3 -20 2 a<br>(a+b) moa 59 (040) mod 59 (4b) mod 59 (a+0) moa 59 (a0) mod 59 (040) mod 59<br><!-- End of picture text -->

Figure 22: For a 1-layer transformer with attention, correct logits after principal component (possibly non-circle) isolations at various states during the training. The pizza-like pattern gradually _desolved_ . 

23) and removing that circle brings accuracy down from 99.7% to 97.9%. They are less helpful later in the network (removing accompanying pizzas in trained Model A only brings accuracy down to 99.7%). 

24 



<!-- Start of picture text -->
ow Da <3 -20 10 «60 = 20<br>2 wee =e 3<br>a a o a<br>8° Soa:ehehsis3 - |<br>© 3 6 9 12151821242730333639424548515457 0 3 6 9 171518712427303336394245.48515457<br>(a+b) mod 59 (a+b) mod 59<br>Circle from Principal Component 1, 2 Circle from Principal Component 13, 14<br>03 a7 . BF8 os * =<br>6 Be<br>02 ge 7 ee? 34<br>2 23 4 5 Sad<br>on ao 8& , on asst 3 s 2<br>;<br>oo} e Pye a x<br>$1 Circle #1,6=20 gr49 poo! go 2cirgarea¥6 = 10 o<br>cal Ay?Be we | |ea a0‘Se * 50<br>-02 we 2 7 os ,<br>es 95 00 Sg Pe)<br>-0 5 fF oe 55 56 02 ” ao °<br>“03-02 -01 00 01 02 03 “02-01. -«09°~«=01 ~~<br><!-- End of picture text -->

Figure 23: Immediate state after 600 epochs of training for a 1-layer transformer with constant attention. 

## **L A Closer Look at a Linear Pizza Model** 

In this section, we provide a full picture of the linear model shown in Figure 15 by investigating the actual weights in the model. 

### **L.1 Model Structure** 

As described in Appendix E, on input ( _a, b_ ), the output logits of the model is computed as 



Denote the weight of embedding layer as _WE_ , the weight of the unembedding layer ( _L_ 3) as _WU_ , and the weights and biases of _L_ 1 and _L_ 2 as _W_ 1 _, b_ 1 and _W_ 2 _, b_ 2, respectively, then the output logits on input ( _a, b_ ) can be written as 



### **L.2 General Picture** 

We first perform principal component visualizations on the embedding and unembedding matrices. From Figure 24, we can see that the embedding and unembedding matrices formed _matching circles_ (circles with the same gap _δ_ between adjacent entries). 

We now give the general overview of the circuit. Each pair of matching circles forms an instance of _Pizza_ and they operate independently (with rather limited interference). Specifically for each pair, 

> • The embedding matrix first places the inputs _a, b_ on the circumference: _WE_<sup>_′_[</sup><sup>_a_]</sup><sup>_≈_</sup> (cos( _wka_ ) _,_ sin( _wka_ )) and _WE_<sup>_′_[</sup><sup>_b_]</sup><sup>_≈_(cos(</sup><sup>_wkb_)</sup><sup>_,_sin(</sup><sup>_wkb_))(</sup><sup>_wk_=2</sup><sup>_πk/p_forsomeinte-</sup> ger _k ∈_ [1 _, p −_ 1] as in Section 2.1; _WE_<sup>_′_stands for the two currently considered principal</sup> components of _WE_ ; rotation and scaling omitted for brevity). 

25 

- The embeddings are added to get 

      - (cos( _wka_ ) + cos( _wkb_ ) _,_ sin( _wka_ ) + sin( _wkb_ )) 

      - = cos( _wk_ ( _a − b_ ) _/_ 2) _·_ (cos( _wk_ ( _a_ + _b_ ) _/_ 2) _,_ sin( _wk_ ( _a_ + _b_ ) _/_ 2)) 

- It is then passed through the first linear layer _L_ 1. Each result entry pre-ReLU will thus be a linear combination of the two dimensions of the aforementioned vectors, i.e. cos( _wk_ ( _a − b_ ) _/_ 2) _·_ ( _α_ cos( _wk_ ( _a_ + _b_ ) _/_ 2) + _β_ sin( _wk_ ( _a_ + _b_ ) _/_ 2)) for some _α, β_ , which will become _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _||α_ cos( _wk_ ( _a_ + _b_ ) _/_ 2) + _β_ sin( _wk_ ( _a_ + _b_ ) _/_ 2)) _|_ after ReLU. 

- These values are then passed through the second linear layer _L_ 2. Empirically the ReLU is not observed to be effective as the majority of values is positive. The output entries are then simply linear combinations of aforementioned outputs of _L_ 1. 

- The unembedding matrix is finally applied. In the principal components we are considering, _WU_<sup>_′_[</sup><sup>_c_]</sup><sup>_≈_(cos(</sup><sup>_wkc_)</sup><sup>_,_sin(</sup><sup>_wkc_)).(</sup><sup>_W ′_</sup> _U_<sup>standsforthetwocurrentlyconsideredprincipal</sup> components of _WU_ ; rotation and scaling omitted for brevity) and these two principal components correspond to a linear combination of the output entries of _L_ 2, which then correspond to a linear combination of the outputs of _L_ 1 (thanks to the non-functional ReLU). 

- Similar to the formula _|_ sin( _t_ ) _| −|_ cos( _t_ ) _| ≈_ cos(2 _t_ ) discussed in Appendix A, these linear combinations provide good approximations for _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b_ )) and _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ sin( _wk_ ( _a_ + _b_ )). Finally we arrive at 

   - _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ (cos( _wkc_ ) cos( _wk_ ( _a_ + _b_ )) + sin( _wkc_ ) sin( _wk_ ( _a_ + _b_ ))) 

   - = _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b − c_ )) 



<!-- Start of picture text -->
10 Embedding: Cirle fom PC1.2. Embedding Circle fom PC3.4 _Unembedding: Circle rom PC1,2 _Unembedding: Circle fram PC3,4<br>os aeSB 50 10 aeDPys 4: | 10 Poade 10 ee» Sages<br>00} gh#®MeCrce#1.o=2 “hes232] Os00) &pr We cele 2.6-27 S|gf co/'g psWsCice#1.0=2°” “e| 00os oeWiScirclesf #2, 0=é245%ee<br>03) Fi ..|% Pho tet ee<br>vaso io ae.osronsago*000sa 30 meanestLo é jshe oe0p oefet0s 0 7-1o =o bseweoyeo> ees0S *5s Togo OP)|, eeTo oe,-osearlvo 0s3 $eio<br><!-- End of picture text -->

Figure 24: Visualization of the principal components of the embeddings and unembedding matrices. 

### **L.3 Aligning Weight Matrices** 

We first verify that the ReLU from the second layer is not functional. After removing it, the accuracy of the model remains 100% and the cross-entropy loss actually _decreased_ from 6 _._ 20 _×_ 10<sup>_−_7</sup> to 5 _._ 89 _×_ 10<sup>_−_7</sup> . 

Therefore, the model output can be approximately written as 

_WU_ ( _b_ 2+ _W_ 2ReLU( _b_ 1+ _W_ 1( _WE_ [ _a_ ]+ _WE_ [ _b_ ]))) = _WU b_ 2+ _WU W_ 2ReLU( _b_ 1+ _W_ 1( _WE_ [ _a_ ]+ _WE_ [ _b_ ])) _._ 

We now “align” the weight matrices _W_ 1 and _W_ 2 by mapping through the directions of the principal components of the embeddings and unembeddings. That is, we calculate how these matrices act on and onto the principal directions (consider _W_ 1 _v_ for every principal direction _v_ in _WE_ and _v_<sup>_T_</sup> _W_ 2 for every principal direction _v_ in _WU_ ). We call the other dimension of aligned _W_ 1 and _W_ 2 output and source dimensions, respectively (Figure 25). 

In the aligned weight matrices, we can see a clear domino-like pattern: in most output or source dimensions, only two principal components have significant non-zero values, and they correspond to a pair of matching circle, or a pizza. In this way, every immediate dimension serves for exactly one pizza, so the pizzas do not interfere with each other. 

26 



<!-- Start of picture text -->
Ew<br>Bt “10<br>30<br>Algned Wy: We acting ort the Fst principal components of Wi<br>3 $10<br>2 05<br>S “10<br>zo<br><!-- End of picture text -->

Figure 25: Visualization of the aligned _W_ 1 and _W_ 2. 

### **L.4 Approximation** 

Everything becomes much clearer after realigning the matrices. For a pizza and its two corresponding principal embedding / unembedding dimensions, _WE_<sup>_′_[</sup><sup>_a_] +</sup><sup>_W ′_</sup> _E_<sup>[</sup><sup>_b_]</sup><sup>_≈_cos(</sup><sup>_wk_(</sup><sup>_a −b_)</sup><sup>_/_2)</sup><sup>_·_(cos(</sup><sup>_wk_(</sup><sup>_a_+</sup> _b_ ) _/_ 2) _,_ sin( _wk_ ( _a_ + _b_ ) _/_ 2)) will be mapped by realigned _W_ 1 into its corresponding columns (which are different for every pizza), added with _b_ 1 and apply ReLU. The result will then be mapped by the realigned _W_ 2, added with _realigned b_ 2, and finally multipled by (cos( _wkc_ ) _,_ sin( _wkc_ )). 

For the first two principal dimensions, realigned _W_ 1 has 44 corresponding columns (with coefficients of absolute value _>_ 0 _._ 1). Let the embedded input be ( _x, y_ ) = _WE_<sup>_′_[</sup><sup>_a_] +</sup><sup>_W ′_</sup> _E_<sup>[</sup><sup>_b_]</sup><sup>_≈_cos(</sup><sup>_wk_(</sup><sup>_a −b_)</sup><sup>_/_2)</sup><sup>_·_</sup> (cos( _wk_ ( _a_ + _b_ ) _/_ 2) _,_ sin( _wk_ ( _a_ + _b_ ) _/_ 2)), the intermediate columns are 

ReLU([0 _._ 530 _x −_ 1 _._ 135 _y_ +0 _._ 253 _, −_ 0 _._ 164 _x −_ 1 _._ 100 _y_ +0 _._ 205 _,_ 1 _._ 210 _x −_ 0 _._ 370 _y_ +0 _._ 198 _, −_ 0 _._ 478 _x −_ 1 _._ 072 _y_ + 0 _._ 215 _, −_ 1 _._ 017 _x_ + 0 _._ 799 _y_ + 0 _._ 249 _,_ 0 _._ 342 _x −_ 0 _._ 048 _y_ + 0 _._ 085 _,_ 1 _._ 149 _x −_ 0 _._ 598 _y_ + 0 _._ 212 _, −_ 0 _._ 443 _x_ +1 _._ 336 _y_ +0 _._ 159 _, −_ 1 _._ 580 _x −_ 0 _._ 000 _y_ +0 _._ 131 _, −_ 1 _._ 463 _x_ +0 _._ 410 _y_ +0 _._ 178 _,_ 1 _._ 038 _x_ + 0 _._ 905 _y_ + 0 _._ 190 _,_ 0 _._ 568 _x_ + 1 _._ 188 _y_ + 0 _._ 128 _,_ 0 _._ 235 _x −_ 1 _._ 337 _y_ + 0 _._ 164 _, −_ 1 _._ 180 _x_ + 1 _._ 052 _y_ + 0 _._ 139 _, −_ 0 _._ 173 _x_ +0 _._ 918 _y_ +0 _._ 148 _, −_ 0 _._ 200 _x_ +1 _._ 060 _y_ +0 _._ 173 _, −_ 1 _._ 342 _x_ +0 _._ 390 _y_ +0 _._ 256 _,_ 0 _._ 105 _x−_ 1 _._ 246 _y_ + 0 _._ 209 _,_ 0 _._ 115 _x_ + 1 _._ 293 _y_ + 0 _._ 197 _,_ 0 _._ 252 _x_ + 1 _._ 247 _y_ + 0 _._ 140 _, −_ 0 _._ 493 _x_ + 1 _._ 252 _y_ + 0 _._ 213 _,_ 1 _._ 120 _x_ + 0 _._ 262 _y_ + 0 _._ 239 _,_ 0 _._ 668 _x_ + 1 _._ 096 _y_ + 0 _._ 205 _, −_ 0 _._ 487 _x −_ 1 _._ 302 _y_ + 0 _._ 145 _,_ 1 _._ 134 _x −_ 0 _._ 862 _y_ + 0 _._ 273 _,_ 1 _._ 143 _x_ + 0 _._ 435 _y_ + 0 _._ 171 _, −_ 1 _._ 285 _x −_ 0 _._ 644 _y_ + 0 _._ 142 _, −_ 1 _._ 454 _x −_ 0 _._ 285 _y_ + 0 _._ 218 _, −_ 0 _._ 924 _x_ +1 _._ 068 _y_ +0 _._ 145 _, −_ 0 _._ 401 _x_ +0 _._ 167 _y_ +0 _._ 106 _, −_ 0 _._ 411 _x−_ 1 _._ 389 _y_ +0 _._ 249 _,_ 1 _._ 422 _x−_ 0 _._ 117 _y_ + 0 _._ 227 _, −_ 0 _._ 859 _x −_ 0 _._ 778 _y_ + 0 _._ 121 _, −_ 0 _._ 528 _x −_ 0 _._ 216 _y_ + 0 _._ 097 _, −_ 0 _._ 884 _x −_ 0 _._ 724 _y_ + 0 _._ 171 _,_ 1 _._ 193 _x_ +0 _._ 724 _y_ +0 _._ 131 _,_ 1 _._ 086 _x_ +0 _._ 667 _y_ +0 _._ 218 _,_ 0 _._ 402 _x_ +1 _._ 240 _y_ +0 _._ 213 _,_ 1 _._ 069 _x−_ 0 _._ 903 _y_ + 0 _._ 120 _,_ 0 _._ 506 _x −_ 1 _._ 042 _y_ + 0 _._ 153 _,_ 1 _._ 404 _x −_ 0 _._ 064 _y_ + 0 _._ 152 _,_ 0 _._ 696 _x −_ 1 _._ 249 _y_ + 0 _._ 199 _, −_ 0 _._ 752 _x −_ 0 _._ 880 _y_ + 0 _._ 106 _, −_ 0 _._ 956 _x −_ 0 _._ 581 _y_ + 0 _._ 223]) _._ 

For the first principal unembedding dimension, it will be taken dot product with 

[1 _._ 326 _,_ 0 _._ 179 _,_ 0 _._ 142 _, −_ 0 _._ 458 _,_ 1 _._ 101 _, −_ 0 _._ 083 _,_ 0 _._ 621 _,_ 1 _._ 255 _, −_ 0 _._ 709 _,_ 0 _._ 123 _, −_ 1 _._ 346 _, −_ 0 _._ 571 _,_ 1 _._ 016 _,_ 1 _._ 337 _,_ 0 _._ 732 _,_ 0 _._ 839 _,_ 0 _._ 129 _,_ 0 _._ 804 _,_ 0 _._ 377 _,_ 0 _._ 078 _,_ 1 _._ 322 _, −_ 1 _._ 021 _, −_ 0 _._ 799 _, −_ 0 _._ 339 _,_ 1 _._ 117 _, −_ 1 _._ 162 _, −_ 1 _._ 423 _, −_ 1 _._ 157 _,_ 1 _._ 363 _,_ 0 _._ 156 _, −_ 0 _._ 165 _, −_ 0 _._ 451 _, −_ 1 _._ 101 _, −_ 0 _._ 572 _, −_ 1 _._ 180 _, −_ 1 _._ 386 _, −_ 1 _._ 346 _, −_ 0 _._ 226 _,_ 1 _._ 091 _,_ 1 _._ 159 _, −_ 0 _._ 524 _,_ 1 _._ 441 _, −_ 0 _._ 949 _, −_ 1 _._ 248] _._ 

Call this function _f_ ( _x, y_ ). When we plug in _x_ = cos( _t_ ) _, y_ = sin( _t_ ), we get a function that wellapproximated 8 cos(2 _t_ + 2) (Figure 26). Therefore, let _t_ = _wk_ ( _a_ + _b_ ) _/_ 2, the dot product will be approximately 8 _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b_ ) + 2), or _|_ cos( _wk_ ( _a − b_ ) _/_ 2) _|_ cos( _wk_ ( _a_ + _b_ )) if we ignore the phase and scaling. This completes the picture we described above. 

27 



<!-- Start of picture text -->
af — eosto sine)<br>el — acost2e+2)<br>— 20.5cosK).0:5sint)<br>24<br>-24<br>ot 2 9 4 5 6<br><!-- End of picture text -->

Figure 26: _f_ (cos( _t_ ) _,_ sin( _t_ )) well-approximates 8 cos(2 _t_ + 2). 

28 

