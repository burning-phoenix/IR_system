# Naturally Occurring Equivariance in Neural Networks

**Source:** https://distill.pub/2020/circuits/equivariance/ · **DOI:** 10.23915/distill.00024.004
**Authors:** Chris Olah, Nick Cammarata, Chelsea Voss, Ludwig Schubert, Gabriel Goh. Distill, 2020-12-08. Part of the Circuits thread (Distill 2020).
**License:** Text and diagrams CC-BY 4.0 per the article's Reuse section, except figures marked "Figure from ...". Reproduced here with attribution.
**Route:** HTML fetched from distill.pub on 2026-09-25. Article body only (the d-article element), converted with html2text. Figures are interactive and not reproduced. Their text, including SVG labels, is kept as one [Figure text] paragraph per figure. MathML replaced by its TeX annotation. In-text citations kept as [cite: key].
**NO PDF.** Distill publishes HTML only. The reader cannot resolve citations into this file against a PDF text layer. Check quotations against the source URL.
**Corpus section:** Tyler Wheeler/interpretability
**Cited by:** NOT CITED BY WHEELER. Added for the audit phase of thread 01 (research/thread-01-amortized-inference/statement.md).
**Why it is here:** Networks learn transformed copies of features when a symmetry is not built in, and the authors predicted which features form when it is. Grounds the thread-01 test of whether pose standardization collapses orientation families in a TPC encoder.
**Note:** REFERENCE MATERIAL, NOT WHEELER'S WORK. Vision-model interpretability (InceptionV1), not language models.
**Copied:** 2026-09-25

---

# Naturally Occurring Equivariance in Neural Networks
This article is part of the Circuits thread, an experimental format collecting invited short articles and critical commentary delving into the inner workings of neural networks. 

Curve Detectors High-Low Frequency Detectors

### Contents

Equivariant Features

  * Rotation, Scale, Hue, Hue-Rotation, Reflection, Miscellaneous

Equivariant Circuits

  * High-Low, Contrast→Center, BW-Color, Line→Circle/Divergence, Curve→Circle/Evolute, Human-Animal, Invariant Dog Head, Hue→Hue, Curve→Curve, Contrast→Line

Equivariant Architectures

Conclusion

Convolutional neural networks contain a hidden world of symmetries within themselves. This symmetry is a powerful tool in understanding the features and circuits inside neural networks. It also suggests that efforts to design neural networks with additional symmetries baked in (eg. [cite: bergstra2011statistical,cohen2016group,dieleman2016exploiting,cohen2016steerable,thomas2018tensor,winkels20183d] ) may be on a promising track. 

To see these symmetries, we need to look at the individual neurons inside convolutional neural networks and the circuits that connect them. It turns out that many neurons are slightly transformed versions of the same basic feature. This includes rotated copies of the same feature, scaled copies, flipped copies, features detecting different colors, and much more. We sometimes call this phenomenon “equivariance,” since it means that switching the neurons is equivalent to transforming the input. [footnote: The standard definition of equivariance in group theory is that a function $f$ f is equivariant if for all $g\in G$ g ∈ G , it’s the case that $f(g\cdot x) = g\cdot f(x)$ f ( g ⋅ x ) = g ⋅ f ( x ) . At first blush, this doesn’t seem very relevant to transformed versions of neurons. Before we talk about the examples introduced in this article, let’s talk about how this definition maps to the classic example of equivariance in neural networks: translation and convolutional neural network nets. In a conv net, translating the input image is equivalent to translating the neurons in the hidden layers (ignoring pooling, striding, etc). Formally, $g\in Z^2$ g ∈ Z 2 and $f$ f maps images to hidden layer activations. Then $g$ g acts on the input image $x$ x by translating spatially, and acts on the activations by also spatially translating them. Now let’s consider the case of curve detectors (the first example in the Equivariant Features section), which have ten rotated copies. In this case, $g\in Z_{10}$ g ∈ Z 1 0 ​ and $f(x) = (\mathrm{curve}_1(x), …)$ f ( x ) = ( c u r v e 1 ​ ( x ) , . . . ) maps a position at an image to a ten dimensional vector describing how much each curve detector fires. Then $g$ g acts on the input image $x$ x by rotating it around that position and $g$ g acts on the hidden layers by reorganizing the neurons so that their orientations correspond to the appropriate rotations. This satisfies, at least approximately, the original definition of equivariance. This transformed neuron form of equivariance is a special case of equivariance. There are many ways a neural network could be equivariant without having transformed versions of neurons. Conversely, we’ll also see a number of examples of equivariance that don’t map exactly to the group theory definition of equivariance: some have “holes” where a transformed neuron is missing, while others consist of a set of transformations that have a weaker structure than a group or don’t correspond to a simple action on the image. But this general structure remains.] 

Equivariance can be seen as a kind of ”circuit motif,” an abstract recurring pattern across circuits analogous to motifs in systems biology [cite: alon2019introduction] . It can also be seen as a kind of larger-scale “structural phenomenon” (similar to weight banding and branch specialization), since a given equivariance type is often widespread in some layers and rare in others.

In this article, we’ll focus on examples of equivariance in InceptionV1 [cite: szegedy2015going] trained on ImageNet [cite: deng2009imagenet] , but we’ve observed at least some equivariance in every model trained on natural images we’ve studied.   
  

* * *

  
  

## Equivariant Features

**Rotational Equivariance:** One example of equivariance is rotated versions of the same feature. These are especially common in early vision, for example curve detectors,  high-low frequency detectors, and line detectors. 

[Figure text] Rotational Equivariance Some rotationally equivariant features wrap around at 180 degrees due to symmetry. (There are even units which wrap around at 90 degrees, such as hatch texture detectors.) Curve Detectors Rotational Equivariance (mod 180) High-Low Frequency Detectors Edge Detectors Line Detectors 

One can test that these are genuinely rotated versions of the same feature by taking examples that cause one to fire, rotating them, and checking that the others fire as expected. The article on curve detectors tests their equivariance through several experiments, including rotating stimuli that activate one neuron and seeing how the others respond. 

One way to verify that units like curve detectors are truly rotated versions of the same feature is to take stimuli that activate one and see how they fire as you rotate the stimuli. Learn more.

**Scale Equivariance:** Rotated versions aren’t the only kind of variation we see. It’s also quite common to see the same feature at different scales, although usually the scaled features occur at different layers. For example, we see circle detectors across a huge variety of scales, with the small ones in early layers and the large ones in later layers. 

[Figure text] Scale Equivariance Circle Detectors 

**Hue Equivariance:** For color-detecting features, we often see variants detecting the same thing in different hues. For example, color center-surround units will detect one hue in the center, and the opposing hue on around it. Units can be found doing this up until the seventh or even eighth layer of InceptionV1. 

[Figure text] Hue Equivariance Color Center -Surround 

**Hue-Rotation Equivariance:** In early vision, we very often see color contrast units. These units detect one hue on one side, and the opposite hue on the other. As a result, they have variation in both hue and rotation. These variations are particularly interesting, because there’s an interaction between hue and rotation. But cycling hue by 180 degrees flips which hue is on which side, and is so is equivalent to rotating by 180 degrees. 

In the following diagram, we show orientation rotating the whole 360 degrees, but hue only rotating 180. At the bottom of the chart, it wraps around to the top but shifts by 180 degrees. 

[Figure text] Rotational Equivariance Hue Equivariance Color Contrast Detectors (hue+180, orientation) = (hue, orientation+180) 

**Reflection Equivariance:** As we move into the mid layers of the network, rotated variations become less prominent, but horizontally flipped pairs become quite prevalent. 

[Figure text] Horizontal Flip Dog snout detectors S-curve detectors Human beside animal Horizontal Flip Horizontal Flip 

**Miscellaneous Equivariance:** Finally, we see variations of features transformed in other miscellaneous ways. For example, short vs long-snouted versions of the same dog head features, or human vs dog versions of the same feature. We even see units which are equivariant to camera perspective (found in a Places365 [cite: zhou2016places] model). These aren’t necessarily something that we would classically think of as forms of equivariance, but do seem to essentially be the same thing. 

[Figure text] Snout Length Human vs Dog Perspective   
  

* * *

  
  

## Equivariant Circuits

The equivariant behavior we observe in neurons is really a reflection of a deeper symmetry that exists in the weights of neural networks and the circuits they form. 

We’ll start by focusing on rotationally equivariant features that are formed from rotationally invariant features. This “invariant→equivariant” case is probably the simplest form of equivariant circuit. Next, we’ll look at “equivariant→invariant” circuits, and then finally the more complex “equivariant→equivariant” circuits.   
  

**High-Low Circuit:** In the following example, we see high-low frequency detectors get built from a high-frequency factor and a low-frequency factor (both factors correspond to a combination of neurons in the previous layer). Each high-low frequency detector responds to a transition in frequency in a given direction, detecting high-frequency patterns on one side, and low frequency patterns on the other. Notice how the same weight pattern rotates, making rotated versions of the feature. 

[Figure text] positive (excitation) negative (inhibition) respond to a high-frequency neuron factor on one side and low frequency on the other. Notice how the weights rotate: High-low frequency detectors This makes them rotationally equivariant.   
  

**Contrast→Center Circuit:** This same pattern can be used in reverse to turn rotationally equivariant features back into rotationally invariant features (an “equivariant→invariant” circuit). In the following example, we see several green-purple color contrast detectors get combined to create green-purple and purple-green center-surround detectors. Compare the weights in this circuit to the ones in the previous one. It’s literally the same weight pattern transposed. 

[Figure text] Rotational equivariance can be turned into invariance with the transpose of an invariant -> equivariant circuit. Here, we see (rotationally equivariant) combine to make (rotationally invariant). Again, notice how the weights rotate, forming the same pattern we saw above with high-low frequency detectors, but with inputs and outputs swapped. color contrast units color center surround units positive (excitation) negative (inhibition) 

Sometimes we see one of these immediately follow the other: equivariance be created, and then immediately partially used to create invariant units.   
  

**BW-Color Circuit:** In the following example, a generic color factor and a black and white factor are used to create black and white vs color features. Later, the black and white vs color features are combined to create units which detect black and white at the center, but color around, or vice versa. 

[Figure text] But then one major use of the equivariant units is combining them into rotationally invariant center surround units. First, rotationally equivariant “black and white vs color” units are formed from mostly invariant features. positive (excitation) negative (inhibition)   
  

**Line→Circle/Divergence Circuit:** Another example of equivariant features being combined to create invariant features is very early line-like complex Gabor detectors being combined to create a small circle unit and diverging lines unit. 

[Figure text] positive (excitation) negative (inhibition) A is created by detecting early edges to a normal line. circle detector perpendicular A is created by detecting early edges to a normal line. diverging line detector parallel   
  

**Curve→Circle/Evolute Circuit:** For a more complex example of rotational equivariance being combined to create invariant units, we can look at curve detectors being combined to create circle and evolute detectors. This circuit is also an example of scale equivariance. The same general pattern which turns small curve detectors into a small circle detector turns large curve detectors into a large circle detector. The same pattern which turns medium curve detectors into a medium evolute detector turns large curves into a large evolute detector. 

[Figure text] conv2d -> mixed3a Small circle from curves mixed3a -> mixed3b Medium evolute from curves mixed3b -> mixed4a Large circle and evolute from curves   
  

**Human-Animal Circuit:** So far, all of the examples we’ve seen of circuits have involved rotation. These human-animal and animal-human detectors are an example of horizontal flip equivariance instead: 

[Figure text] positive (excitation) negative (inhibition) Human detectors excite the human side of each unit and inhibit the other. Other units (mainly dog detectors) inhibit the human side of each unit and excite the other. 

**Invariant Dog Head Circuit:** Conversely, this example (part of the broader oriented dog head circuit) shows left and right oriented dog heads get combined into a pose invariant dog head detector. Notice how the weights flip. 

### “Equivariant→Equivariant” Circuits

The circuits we’ve looked at so far were either “invariant→equivariant” or “equivariant→invariant.” Either they had invariant input units, or invariant output units. Circuits of this form are quite simple: the weights rotate, or flip, or otherwise transform, but only in response to the transformation of a single feature. When we look at “equivariant→equivariant” circuits, things become a bit more complex. Both the input and output features transform, and we need to consider the relative relationship between the two units.   
  

**Hue→Hue Circuit:** Let’s start with a circuit connecting two sets of hue-equivariant center-surround detectors. Each unit in the second layer is excited by the unit selecting for a similar hue in the previous layer. 

[Figure text] Each unit is excited by the unit with the same hue in the previous layer. The weights between all selected color center-surround units. Units are excited by units with the same, and tend to be inhibited by those with slightly different hues. (In early layers, very different hues inhibit; by later layers very different colors are already distinguished and inhibition focuses on similar colors.) 

To understand the above, we need to focus on the relative relationships between each input and output neuron — in this case, how far the hues are apart on the color wheel. When they have the same hue, the relationship is excitatory. When they have close but different hues, it’s inhibitory. And when they are very different, the weight is close to zero. [footnote: The units used to illustrate hue equivariance here were selected to have a straightforward circuit. Other units may have more complex relationships. For example, some units respond to a range of hues like yellow-red and have correspondingly more complex weights.]   
  

**Curve→Curve Circuit:** Let’s now consider a slightly more complex example, how early curve detectors connect to late curve detectors. We’ll focus on four curve detectors that are 90 degrees rotated from each other. [footnote: Again, the curve detectors presented were selected to make the circuit as simple and pedagogical as possible. They have clean weights and even spacing between them, which will make the pattern easier to see. A forthcoming article will discuss curve circuits in detail.] 

If we just look at the matrix of weights, it’s a bit hard to understand. But if we focus on how each curve detector connects to the earlier curve in the same and opposite orientations, it becomes easier to see the structure. Rather than each curve being built from the same neurons in the previous layer, they shift. Each curve is excited by curves in the same orientation and inhibited by those in the opposite. At the same time, the spatial structure of the weights also rotate. 

[Figure text] Weights from each of the four curve detectors displayed to each of the other four. Notice how the diagonal is excitatory ( ), while the off-diagonal is inhibitory ( ). The weights shown to the left are a subset of these. Each curve inhibits the curve in the opposite orientation in the next layer along its tangent. Notice how the weights rotate: Each curve excites the curve in the same orientation in the next layer along its tangent. Notice how the weights rotate:   
  

**Contrast→Line Circuit:** For a yet more complex example, let’s look at how color contrast detectors connect to line detectors. The general idea is line detectors should fire more strongly if there are different colors on each side of the line. Conversely, they should be inhibited by a change in color if it is perpendicular to the line. 

Note that this is an “equivariant→equivariant” circuit with respect to rotation, but “equivariant→invariant” with respect to hue. 

[Figure text] roughly arranged by orientation and hue. Color contrast detectors One of the downstream roles of color contrast detectors is to make line detectors respond to changes in color across the line. In this circuit, we will see how this is done. A (slightly tilted) is excited by horizontal color contrasts of all hues, and inhibited by vertical ones. horizontal line detector A is excited by vertical color contrasts of all hues, and inhibited by horizontal ones. vertical line detector   
  

* * *

  
  
  

## Equivariant Architectures

Equivariance has a rich history in deep learning. Many important neural network architectures have equivariance at their core, and there is a very active thread of research around more aggressively incorporating equivariance. But the focus is normally on designing equivariant architectures, rather than “natural equivariance” we’ve discussed so far. How should we think about the relationship between “natural” and “designed” equivariance? As we’ll see, there appears to be quite a deep connection. 

Historically, there has been some interesting back and forth between the two. Researchers have often observed that many features in the first layer of neural networks are transformed versions of one basic template. [footnote: Features in the first layer of neural networks are much more often studied than in other layers. This is because they are easy to study: you can just visualize the weights to pixel values, or more generally to input features.] This naturally occurring equivariance in the first layer has then sometimes been — and in other cases, easily could have been — inspiration for the design of new architectures. 

For example, if you train a fully-connected neural network on a visual task, the first layer will learn variants of the same features over and over: Gabor filters at different positions, orientations, and scales. Convolutional neural networks changed this. By baking the existence of translated copies of each feature directly into the network architecture, they generally remove the need for the network to learn translated copies of each feature. This resulted in a massive increase in statistical efficiency, and became a cornerstone of modern deep learning approaches to computer vision. But if we look at the first layer of a well-trained convolutional neural network, we see that other transformed versions of the same feature remain: 

[Figure text] The weights for the units in the first layer of the TF-Slim [cite: silberman2018tf] version of InceptionV1 [cite: szegedy2015going] . [footnote: We show the first layer conv weights of the tf-slim version of InceptionV1 rather than the canonical one because its weights are cleaner. This is likely due to the inclusion of batch-norm in the slim variant, causing cleaner gradients.] Units are sorted by the first principal component of the adjacency matrix between the first and second layers. Note how many features are similar except for rotation, scale, and hue. 

Inspired by this, a 2011 paper subtitled “One Gabor to Rule Them All” [cite: bergstra2011statistical] created a sparse coding model which had a single Gabor filter translated, rotated, and scaled. In more recent years, a number of papers have extended this equivariance to the hidden layers of neural networks, and to broader kinds of transformations [cite: cohen2016group,dieleman2016exploiting,cohen2016steerable,winkels20183d] . Just as convolutional neural networks enforce that the weights between two features be the same if they have the same relative position:  $W_{(x_1,~y_1,~a) ~\to~ (x_2,~y_2,~b)} ~~=~~ W_{(x_1+\Delta x,~y_1 +\Delta y,~a) ~\to~ (x_2+\Delta x,~y_2+\Delta y,~b)}$ W(x1​, y1​, a) → (x2​, y2​, b)​ = W(x1​+Δx, y1​+Δy, a) → (x2​+Δx, y2​+Δy, b)​

… these more sophisticated equivariant networks make the weights between two neurons equal if they have the same relative relationship under more general transformations: [footnote: For our purposes, it suffices to know that these equivariant neural networks have the same weights when there is the same relative relationship between neurons. This footnote is for the benefit of readers who may wish to engage more deeply in the enforced equivariance literature, and can be safely skipped. Group theory is an area of mathematics that gives us tools for describing symmetries and sets of interacting transformations. To build equivariant neural networks, we often borrow an idea from group theory called a group convolution. Just as a regular convolution can describe weights that correctly respect translational equivariance, a group convolution can describe weights that respect a complex set of interacting transformations (the group it operates over). Although you could try to work out how to tie the weights to achieve this from first principles, it’s easy to make mistakes. (One of the authors participated in many conversations with researchers in 2012 where people made errors on whiteboards about how sets of rotated and translated features should interact, without using convolutions.) Group convolutions can take any group you describe and give you the correct weight tying. For an approachable introduction to group convolutions, we recommend this article . If you dig further, you may begin to see papers discussing something called a group representation instead of group convolutions. This is a more advanced topic in group theory. The core idea is analogous to the Fourier transform. Recall that the Fourier transform turns convolution into pointwise multiplication (this is sometimes used to accelerate convolution). Well, the Fourier transform has a version that can operate over functions on groups, and also maps convolution to pointwise multiplication. And when you apply the Fourier transform to a group, the resulting coefficients correspond to something called a group representation, which you can think of as being analogous to a frequency in the regular Fourier transform.]  $W_{a~\to~ b} ~~=~~ W_{T(a) \to T(b)}$ Wa → b​ = WT(a)→T(b)​

This is, at least approximately, what we saw conv nets naturally doing when we look at equivariant circuits! The weights had symmetries that caused neurons with similar relationships to have similar weights, much like an equivariant architecture would force them to. 

Given that we have neural network architectures which mimic the natural structures we observe, it seems natural to wonder what features and circuits such models learn. Do they learn the same equivariant features we see naturally form? Or do they do something entirely different? To answer these questions, we trained an equivariant model roughly inspired by InceptionV1 on ImageNet. We made half the neurons rotationally equivariant [cite: cohen2016group] (with 16 rotations), and made the others rotationally invariant. Since we put no effort into tuning it, the model achieved abysmal test accuracy but still learns interesting features. [footnote: Here are the full set of features learned by the equivariant model. Half are forced to be rotationally equivariant, while half are forced to be rotationally invariant.] Looking at mixed3b, we found that the equivariant model learned analogues of many large rotationally equivariant families from InceptionV1, such as curve detectors, boundary detectors, divot detectors, and oriented fur detectors: 

[Figure text] respond to curves in various orientations. Curve detectors Naturally ocurring rotationally equivariant features Analogous features found in a model where some units are forced to have 16 rotated copies Oriented fur detectors detect fur parting in a particular way. Boundary detectors use multiple cues to detect oriented boundaries of objects. Divot detectors look for sharp curves sticking out. 

The existence of analogous features in equivariant models can be seen as a successful prediction of interpretability. As researchers engaged in more qualitative research, we should always be worried that we may be fooling ourselves. Successfully predicting which features will form in an equivariant neural network architecture is actually a pretty non-trivial prediction to make, and a nice confirmation that we’re correctly understanding things. 

Another exciting possibility is that this kind of feature and circuit analysis may be able to help inform equivariance research. For example, the kinds of equivariance that naturally form might be helpful in informing what types of equivariance we should design into different layers of a neural network. 

* * *

## Conclusion

Equivariance has a remarkable ability to simplify our understanding of neural networks. When we see neural networks as families of features, interacting in structured ways, understanding small templates can actually turn into understanding how large numbers of neurons interact. Equivariance is a big help whenever we discover it. 

We sometimes think of understanding neural networks as being like reverse engineering a regular computer program. In this analogy, equivariance is like finding the same inlined function repeated throughout the code. Once you realize that you’re seeing many copies of the same function, you only need to understand it once. 

But natural equivariance does have some limitations. For starters, we have to find the equivariant families. This can actually take us quite a bit of work, poring through neurons. Further, they may not be exactly equivariant: one unit may be wired up slightly differently, or have a small exception, and so understanding it as equivariant could leave gaps in our understanding. 

We’re excited about the potential of equivariant architectures to make the features and circuits of neural networks easier to understand. This seems especially promising in the context of early vision, where the vast majority of features seem to be equivariant to rotation, hue, scale, or a combination of those. 

One of the biggest — and least discussed — advantages we have over neuroscientists in studying vision in artificial neural networks instead of biological neural networks is translational equivariance. By only having one neuron for each feature instead of tens of thousands of translated copies, convolutional neural networks massively reduce the complexity of studying artificial vision systems relative to biological ones. This has been a key ingredient in making it at all plausible that we can systematically understand InceptionV1. 

Perhaps in the future, the right equivariant architecture will be able to shave another order of magnitude of complexity off of understanding early vision in neural networks. If so, understanding early vision might move from “possible with effort” to “easily achievable.” 

This article is part of the Circuits thread, a collection of short articles and commentary by an open scientific collaboration delving into the inner workings of neural networks.  

Curve Detectors High-Low Frequency Detectors
