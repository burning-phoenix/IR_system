# Curve Circuits

**Source:** https://distill.pub/2020/circuits/curve-circuits/ · **DOI:** 10.23915/distill.00024.006
**Authors:** Nick Cammarata, Gabriel Goh, Shan Carter, Chelsea Voss, Ludwig Schubert, Chris Olah. Distill, 2021-01-30. Part of the Circuits thread (Distill 2020).
**License:** Text and diagrams CC-BY 4.0 per the article's Reuse section, except figures marked "Figure from ...". Reproduced here with attribution.
**Route:** the page builds its content with JavaScript, so a plain fetch returns almost nothing. Rendered with headless Chrome on 2026-09-25 and the resulting DOM saved. Article body only (the d-article element), converted with html2text. Figures are interactive and not reproduced. MathML replaced by its TeX annotation. In-text citations kept as [cite: key].
**NO PDF.** Markdown only, by decision on 2026-09-25. The reader cannot resolve citations into this file. Check quotations against the source URL.
**Corpus section:** Tyler Wheeler/interpretability
**Cited by:** NOT CITED BY WHEELER. Added for the audit phase of thread 01 (research/thread-01-amortized-inference/statement.md).
**Why it is here:** The circuit that builds curve detectors, read off the weights and re-implemented by hand. The standard for what "understanding a feature" can mean, and the companion to Curve Detectors.
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. Vision-model interpretability (InceptionV1), not language models.
**Copied:** 2026-09-25

---

# Curve Circuits
### Contents

Overview of the Curve Circuit

The Components of Curve Detectors

  * 3a Curve Neuron Family
  * 3a Line Neuron Family
  * Cosmetic Neurons

An _Artificial_ Artificial Neural Network

Downstream

Closing Thoughts

This article is part of the Circuits thread, an experimental format collecting invited short articles and critical commentary delving into the inner workings of neural networks.

High-Low Frequency DetectorsVisualizing Weights

Many approaches to interpretability give high level explanations, but it's hard to know if they're true. Circuits takes the opposite approach: building up from individual neurons and individual weights to easily verifiable explanations of tiny slices of neural networks. But it faces a different question: can we ever hope to understand a full neural network this way?

This paper uses the Circuits approach to reverse-engineer curve detectors, a neuron family we studied in a previous article. We find that although curve detection involves more than 50,000 parameters, those parameters actually implement a simple algorithm that can be read off the weights and described in just a few English sentences. Based on this understanding, we re-implement curve detectors by hand, writing out weights to create an _artificial_ artificial neural network that mimics curve detectors.

The reason the curve circuit is relatively simple is because of the equivariance motif. Rotational equivariance reduces the complexity by a factor of 10-20x while scale equivariance reduces it by an additional 2-3x, for a total reduction of ~50x [footnote: A small portion involving the interactions of color-contrast detectors with line detectors is further reduced by a factor of ~8x due to hue equivariance, but doesn't significantly reduce the complexity of the overall circuit.] . We find this an exciting example of how motifs can dramatically simplify circuits. In this circuit the main motif is equivariance, but in others different motifs may be more important.

While the curve circuit in InceptionV1 is quaint next to the 175b weights in GPT-3, it's a big step up from the tiny circuits in Zoom In. We think the surprising simplicity of the curve circuit is a glimmer of hope that the Circuits approach may scale to letting us reverse-engineer big neural networks into small verifiable explanations.

* * *

## The Curve Detector Algorithm

In this article we'll reverse-engineer the 10 curve neurons we studied in Curve Detectors [cite: cammarata2020curve] , which reside in the fifth convolutional layer of InceptionV1. We’ll focus on one curve neuron, 3b:379, then show how they all follow a similar pattern.

3b:379

To get a high level understanding of the first four layers in the context of curve detectors, we use a method we call **decomposed feature visualization**. Decomposed feature visualization renders a grid of feature visualization that shows the expanded weights of an upstream layer to a downstream neuron of interest (in this case 3b:379) [footnote: Specifically, we calculate the gradient from each layer to the 3b curve family using a linear approximation of InceptionV1 where ReLUs are replaced with the identity. Then we use feature visualization with the objective of maximizing the dot product between each image and each spatial position of the gradient.] . Intuitively, this is like decomposing the curve into simpler shapes, with simpler shapes the further back we go, from early curves, to lines, and eventually Gabor filters. Each shape is oriented along the tangent of the curve in 3b:379.

conv2d0

Gabor Filters line the tangent with secondary influence from the Color Contrast family. In InceptionV1 these two families are not as cleanly differentiated as in some models.

conv2d1

The tangent is lined primarily by Complex Gabor neurons, which resemble Gabor Filters but are invariant to the exact positions of which side is dark or light.

conv2d2

The model constructs a large family of well-developed lines family. We will discuss these more in lines in conv2d2.

mixed3a

The tangent of 3a is primarily lined by the Early Curve family as well as a second Line family. We’ll explore this layer in depth throughout this article.

We use decomposed feature visualization to show how the first four layers of InceptionV1 incrementally build towards curve detectors in 3b. To increase legibility, we render each feature visualization with alpha transparency and grayscale. Since curve detectors are invariant to color, the vibrant colors feature visualization produces don't give us much information, and they can be distracting. Instead we visualize attribution. To give us a sense of the strength of each position we set the opacity relative to its magnitude. We see the highest magnitude weights follow the tangent of 3b:379.

We can take the highest magnitude position from each layer above and see which neuron families contribute to it to get a sense for how each shape is built [footnote: Specifically, we take the highest magnitude spatial position from each layer and look at which neuron families according from Overview of Early Vision contribute most to it.] [footnote: When doing this decomposition, it really matters which metric you choose to measure how neurons contribute to the vector, such as average or total weight, since there is a long tail of neurons with low magnitude that collectively make up a large fraction of the total weight. Training models with weight sparsity would reduce this long tail.] .

layer

conv2d0

=

Gabor Filters

65%

+

Color Contrast

21%

+

Other Units

14%

layer

conv2d1

=

Complex Gabor

54%

+

Gabor Like

23%

+

Low Frequency

19%

...

layer

conv2d2

=

Line

47%

+

Color Contrast

19%

+

Shifted Line

9%

...

layer

3a

=

Curves

22%

+

Lines

16%

+

Angles

16%

...

We decompose the highest magnitude spatial position in each layer into neuron families to see which families contribute most to curve detection. This is similar to a semantic dictionary interface in The Building Blocks of Interpretability. [cite: olah2018building] 

For each family we can also visualize how they connect to each other, showing us a birds-eye view of the curve detection algorithm. Again, we see the layers progressively building more complex shapes, with shapes that closely resemble curve detectors like early curves and well-defined lines being built in 3a, only one layer before curve detectors.

We can visualize how neuron families connect to see a high-level view of the curve detection algorithm of InceptionV1. In addition to giving us a sense of how these families are built, it lets us see how information flows to 3b curves. For instance, we see that conv2d2 lines contribute to 3b curves through both 3a lines and 3a early curves. While it may seem like this throws away lots of information, in a sense this is a natural way to understand a circuit. It’s much easier to think about a line family in conv2d2 than 33 individual line neurons that vary in orientation.

Now that we know which neuron families are most important for curve detection, we can invest in understanding the weights connecting them. Luckily this is sometimes easy, since weights connecting families tend to follow patterns that we call motifs. Many families in this diagram implement the rotational equivariance motif, meaning that each neuron in a family is approximately a rotated version of another neuron in that family. We can see this in the weights connecting 3a early curves and 3b curves.

The weights connecting 3a early curves with 3b:379 implement rotational equivariance. It turns out there are curves of curves literally inscribed in the weights of neural networks. We’ll discuss equivariance later in this paper, and likely again in the Circuits thread in a dedicated paper.

When neuron families implement rotational equivariance we learn a lot looking at the strongest positive and negative neuron connections, because the others are just rotated versions of them. In the weights we see a general pattern, as each layer builds a contrast detector by tiling a simpler contrast detector along its tangent. When the two shapes are aligned the weight is most positive, and most negative when they are perpendicular.

Some of the strongest positive and negative weights between neuron families in the curve circuit. We see two patterns. First, the most positive weights tend to be where shapes are aligned, and the most negative are where shapes are in opposing orientations. Secondly, we see the same transitions can occur multiple times across different layers, such as the transition from lines to curves.

So far we’ve looked at the neuron families that are most important to curve detection, and also at the circuit motifs connecting them. This high-level “circuit schematic” is useful for seeing a complex algorithm at a glance, and it tells us the main components we'd need to build if we wanted to reimplement the circuit from scratch.

The circuit schematic also makes it easy to describe a few sentence English story of how curve detection works. Gabor filters turn into proto-lines which build lines and early curves. Finally, lines and early curves are composed into curves. In each case, each shape family (eg conv2d2 line) has positive weight across the tangent of the shape family it builds (eg. 3a early curve). Each shape family implements the rotational equivariance motif, containing multiple rotated copies of approximately the same neuron.

In the next few sections we'll zoom in from this high level description to a weight-level analysis of how each of the 3a families we've looked at so far contribute to curve detectors. [footnote: To read more about the idea and benefits of fluidly navigating different levels of abstraction, we recommend the interactive essay Up and Down the Ladder of Abstraction by Bret Victor.] .

* * *

## The Components of Curve Detectors

### 3a Early Curve Family

Early curves in 3a contribute more to the construction of curve detectors than any other neuron family. At every position curve neurons are excited by early curves in a similar orientation and inhibited by ones in the opposing orientation, closely following the general pattern we saw earlier.

The raw weights connecting an early curve neuron and curve neuron of the same orientation contains a curve

This can be interpreted as looking for “tangent curves” at each point along the curve

At every position curve neurons are excited by early curves in a similar orientation and inhibited by ones in the opposing orientation. If you look closely you can see the weights shift slightly over the course of the curve to track the change in local curvature.

Curves are excited by early curves in **similar orientations**

...and inhibited by early curves in **opposing orientations**

There are a couple other subtle things you can find if you look closely. Firstly, you can see that the weights shift slightly over the course of the curve, to track the change in local curvature. Secondly, curve detectors in the same orientation are slightly inhibitory when they’re located off the curve. This makes sense, as seeing an early curve located anywhere but the tangent indicates a different shape than a pure curve.

The weight matrix connecting early curves and curves shows a striking set of positive weights lining the diagonal where the two shapes have similar orientations. This band of positive weights is surrounded by negative weights where the early curve and curve orientations differ. The transition is smooth — if a curve strongly wants to see an early curve it wants to see its neighbor a bit too.

Curve detectors are excited by earlier detectors in **similar orientations**

They are inhibited by earlier detectors in **opposing orientations**

Why do we see that curve detectors which are rotated 180 degrees from each other are inhibitory, but not ones rotated 90 degrees? Recall from our previous article that early curve detectors respond slightly to curves that are 180 degrees rotated from their prefered orientation, which we called an echo. This makes sense: a curve in the opposite orientation runs tangent to the curve we’re trying to detect in the middle of the receptive field, causing similar edge detectors to fire.

These negative weights are the reason curve neurons in 3b have no echoes, which we can validate by using circuit editing to remove them and confirming that echoes appear.

Unedited Model

Each curve responds sharply to a range of orientations and does not have echoes

Early Curves Only

Responses are lower resolution than the original model but do not have echoes

Early Curves, Positive Weights

Responses are low resolution and have echoes

The curve family of 3b uses negative weights to avoid the problem of echoes. By removing these negative weights with circuit editing we see echoes reappear.

Using negative weights in this way follows a more general trend: our experience is that negative weights are used for things that are in some way similar enough that they could be mistaken. This seems potentially analogous to lateral inhibition in biological neural networks.

### 3a Line Families

Overview of Early Vision separates line-like shapes in 3a into several families: lines, line misc, angles, thick lines, and line ends. Since they have similar roles in the context of curve detectors we’ll discuss them together, while pointing out their subtle differences.

Like early curves, lines align to the tangent of curve detectors, with more positive weights when the neurons have the same orientation. However, this pattern is more nuanced and discontinuous with lines than early curves. A line with a similar orientation to a curve will excite it, but a line that's rotated a little may inhibit it. This makes it hard to see a general pattern by looking directly at the weight matrix.

Instead, we can study which line orientations excite curve detectors using synthetic stimuli. We can take a similar approach to decomposed feature visualization to see which lines different spatial positions respond to. This shows us that each 3b curve neuron is excited by edges along its tangent line with a tolerance of between about 10° to 45°.

Measuring responses to synthetic edge stimuli to see where curves are looking for edges.

This tolerance isn’t always symmetric, which we can see in 3b:406 below. On its left side it is most excited by lines about 10° upwards. If the line is oriented above 10° it is still excited, but if it is less than 10° it switches to being inhibited.

3b:406 looks for slightly upward tilted lines on the left and slightly downward tilted lines on the right.

Upwards and downwards tilted lines are relatively similar in orientation but have opposite weights.

This view tells us how each curve detector responds to different orientations of lines in general. We can connect this back to individual line neurons by studying which orientations those line neurons respond to using a radial tuning curve.

#### Lines

There are 11 simple line neurons in 3a that mostly fire to one orientation, although some activate more weakly to an "echo" 90 degrees away.

#### Texture Lines

Five neurons in 3a respond to lines that are perpendicular to the orientation where they are longest. These neurons mostly detect fur, but they also contribute to 3b curves. 

#### Cliffs

Finally, there are five line neurons with curiously sharp transitions. These lines want an orientation facing a particular direction, and tolerate error in that direction, but definitely don't want it going the other way, even if slightly. In curve detection, this is useful for handling imperfections, like bumps.

We find cliff-like line neurons an interesting example of non-linear behavior. We usually think of neurons as measuring distance from some ideal. For instance, we may expect car neurons to prefer [footnote: Or we could imagine a more sophisticated metric, such as style. Since ImageNet contains classes for sports cars and non-sports cars, there may be neurons that measure the “sportiness" of different cars. We think that studying neurons that correspond to cars is an interesting research direction, since it’s easy to accumulate datasets of cars with labelled properties such as year, country of origin, and price.] cars of a certain size, and have weaker activations to cars that are bigger or smaller than its ideal. But this line family provides a counter-example, accepting error on one side while not firing at all on the other.

#### 

#### Lines in conv2d2

The different types of line neurons we looked at above each have different behaviors, which is part of why the weight matrix between 3a lines and 3b curves is indecipherable. However, if we go back one more layer and look at how conv2d2 lines connect to 3b curves, we see structure.

The weights connecting conv2d2 lines and 3b curves implement rotational equivariance.

We think this points to an interesting property of both curves and lines in InceptionV1. The line family in conv2d2 are roughly "pure" line detectors, detecting lines in an elegant pattern. The next layer (3a) builds lines too, but they're more applied and nuanced, with seemingly-awkward behavior like cliff-lines where the network finds them useful. Similarly, the 3b curve detectors are surprisingly elegant detectors of curves behaviorally, and mostly follow clean patterns in their construction. In contrast, the curves in the next layer (4a) are more applied and nuanced, mostly corresponding to 3d geometry and seemingly specialized for detecting real-world objects like the tops of cups and pans. Perhaps this points to a yet unnamed motif of pure shape detectors directly followed by applied ones.

### Cosmetic Neurons

In Curve Detectors we saw how curve neurons seem to be robust to several cosmetic properties, with similar behavior across textures like metal and wood in a variety of lighting. How do they do it?

We believe this reflects a more widespread phenomenon in early vision. As progressively sophisticated shapes are built in each layer, new shapes incorporate cosmetic neuron families like colors and texture. For instance, 3b curve neurons are built primarily from the line and early curve neuron families, but they also incorporate a family of 65 texture neurons. This means they both inherit the cosmetic robustness of the line and early curve neurons, as well as strengthen it by including more textures.

While we won't do a detailed weight-level analysis of how cosmetic robustness propagates through the shapes of early vision in this article, which is a broader topic than curve detection, we think this is an exciting direction for future research.

* * *

## An Artificial Artificial Neural Network

How do we know this story about the mechanics of curve detectors is true? One way is to use it to reimplement curve detectors from scratch. We manually set the weights of a blank neural network to implement the neuron families and circuit motifs from this article, crafting an _artificial_ artificial neural network, and made the python code available and runnable in this Colab notebook. This was initially a few hour process for one person (Chris Olah), and they did not look at the original neural network’s weights when constructing it, which would go against the spirit of the exercise. Later, before publishing they tweaked the weights and in particular added negative weights to remove echoes in the activations.

To compare our artificial curve detectors against InceptionV1's naturally learned ones we have the full palette of techniques we used in Curve Detectors available to us. We'll choose three: feature visualization, dataset examples, and synthetic stimuli. From there we'll run two additional comparisons by leveraging circuits and model editing.

First we'll look at feature visualization and responses to synthetic curve stimuli together. We see the feature visualizations indeed render curves, except they are grayscale since we never include cosmetic features such as color-contrast detectors in our artificial curves. We see their response to curve stimuli approximates the natural curve detectors across a range of radii and all orientations. One difference is our artificial neurons have a slight echo [footnote: Activations on the opposite of the curve's most preferred orientation, which we discussed in more depth in Curve Detectors.] , which we believe could be removed by more carefully setting negative weights.

Artificial Curve Neurons

Natural InceptionV1 Curve Neurons

We can compare the curve detectors in a neural network we hand-crafted with the curve detectors in InceptionV1 by measuring how they activate to synthetic curve stimuli. We see that across a range of radii and orientations, our artificial curve neurons approximate the natural ones.

We can also get a qualitative sense for the differences by looking at a saliency map of dataset examples that cause artificial curve detectors to fire strongly [footnote: Since our artificial curve detectors don't respond to color, there are likely many false-negatives where natural curve detectors correctly catch that our artificial ones don't. Our goal in building artificial curve detectors isn't perfection, but to show that we can use the circuit patterns described in this article to roughly approximate the curve detection algorithm in InceptionV1.] .

Dataset SampleArtificial Curve DetectorsNatural Curve Detectors

Next we can compare the weights for the circuits connecting neuron families in the two models, alongside feature visualizations for each of those families. We see they follow approximately the same circuit structure.

conv2d0

The first later constructs a family of Gabor Filters using a 7x7 convolution.

conv2d1

A 1x1conv builds Complex Gabor Filters, invariant to small translations.

conv2d2

Complex Gabor Filters are tiled to create a Line family.

3a

Lines are composed to create Early Curves.

3b

Finally, both models tile lines and curves from 3a to construct curve detectors.

Artificial AANN

Natural InceptionV1

A comparison of the most important family in each layer across our artificial artificial neural network with InceptionV1 trained on ImageNet.

We can also zoom into specific weight matrices that we've already studied in this article. We see the raw weights between early curves and curves as well as lines to curves look approximately like the ones in InceptionV1, but cleaner since we set them programmatically.

3a Early Curves → 3b CurvesArtificial AANN

3a Lines → 3b CurvesArtificial AANN

The weights of the artificial artificial neural network closely resemble the equivariant weight in the naturally trained network, except their pattern is cleaner because the weights are programmetically generated. equivariance.

Finally, a preliminary experiment suggests that adding artificial curve detectors helps recover some of the loss in classification accuracy across the dataset of removing them from the model entirely [footnote: Specifically, when we remove InceptionV1's 3b curve detectors entirely, the model's top-5 accuracy across 2000 validation set images drops from 88.6% to 86.3% (46 fewer correct classifications). When we replace 3b curve detectors with our artificial curve detectors more than half of this drop is recovered, with an accuracy of 87.6% (reducing the gap from 46 incorrect classifications to 20). We suspect the remaining gap is due to three factors. First, our artificial neurons are grayscale, since we are not implementing color contrast neurons or texture neurons. Secondly, the receptive fields of artificial neurons may not be exactly centered the same as natural curve detectors. Thirdly, we may not have scaled the activations of curve detectors optimally. Additionally, there are two caveats worth mentioning about our experimental setup. First, our ImageNet evaluation likely doesn't mimic the exact conditions the model was trained under (eg. data preprocessing), since the original model was trained at Google using a precursor Tensorflow. Secondly, the reason we ran it on less than the full validation set was operational, not a result of cherry-picking. We initially ran it on a small set in a prototype experiment to validate our hypothesis. We planned to run it on the full set before our publication date, but due OpenAI infrastructure changes our setup broke and we were unable to reimplement it in time. For this reason we emphasize that the experiment is preliminary, although we suspect it's likely to work on the full validation set as well.] .

Overall, we believe these five experiments show our artificial curve detectors are roughly analogous to the naturally trained ones. Since they are nearly a direct translation from the neuron families and circuit motifs we've described in this article into Python code for setting weights, we think this is strong evidence these patterns accurately reflect the underlying circuits that construct curve detectors.

* * *

## Downstream

While this article focused mostly on how curve detection works upstream of the curve detector, it's also worth briefly considering how curves are used downstream in the model. It's easiest to see their mark on the next layer, 4a, where they're used to construct more sophisticated shapes.

The curve neurons in 3b are used to build more complex and specific shape detectors such as sophisticated curves, circles, S-shapes, spirals, divots, and “evolutes" (a term we’ve repurposed to describe units detecting curves facing away from the middle). 

Many of these shapes look for individual curves at different spatial positions, such as circles and evolutes. These shapes often reappear across different branches of 4a, such as the 3x3 and 5x5 branch. 

Layer 4a also constructs a series of curve detectors, mostly in the 5x5 branch that specializes in 3d geometry. However, we believe they should be thought of less as pure abstract shapes and more as corresponding to specific worldly objects, like 4a:406 which often detects the top of cups and pans.

This article is part of the Circuits thread, an experimental format collecting invited short articles and critical commentary delving into the inner workings of neural networks.

High-Low Frequency DetectorsVisualizing Weights
