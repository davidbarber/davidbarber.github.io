---
layout: post

authors:
- David Barber

title: "AI Coworking"
date: "2018-09-12"
slug: "AI as a coworker"
description: "AI coworker"
category:
  - views
  - featured
# tags will also be used as html meta keywords.
tags:
  - deep learning
  - nlp
  - natural language processing
  - latent variable models
  - translation
  - neural machine translation
  - semi supervised learning


show_meta: true
comments: true
mathjax: true
gistembed: true
published: true
noindex: false
nofollow: false
# hide QR code, permalink block while printing.
hide_printmsg: false
# show post summary or full post in RSS feed.
summaryfeed: false
## for twitter summary card with squared image and page description or page excerpt:
# imagesummary: 
## for twitter card with large image:
# imagefeature: 
## for twitter video card: (active for this page)
# videofeature: 
# imagefeature: 
# videocredit: 
---

What's wrong with current machine learning approaches in practice?

<!--more-->

<iframe width="420" height="315" src="https://www.youtube.com/watch?v=1A-Nf3QIJjM" frameborder="0" allowfullscreen></iframe>






One approach to achieve this is to require models to be good, not only at translation, but additional tasks, such as question answering.  A parallel direction is to encourage models to internally focus on the meaning of the sentence. This post summarises our approach to this latter direction, published at NIPS 2018[^SB2018] ([arxiv version](https://arxiv.org/abs/1806.05138)).



## Learning meaningful representations of data
{:.no_toc}


One possible method to learn meaningful representations of sentences is to use a latent variable model. Latent variable models are based on the hypothesis that for each sentence, there is a 'hidden' (or 'latent') vector which represents the sentence's meaning, and that the sentence itself is a textual manifestation of that meaning.

Latent variable models in natural language processing typically posit the following generative process for sentences:

- The 'hidden' or 'latent' representation of the sentence's meaning is randomly generated according to a prior distribution.
- The sentence itself is then generated conditioned on this latent representation.


## Generative Neural Machine Translation (GNMT)
{:.no_toc}

With Generative Neural Machine Translation (GNMT)[^SB2018], we use a single shared latent representation to model the same sentence in multiple languages. The latent variable is a language agnostic representation of the sentence; by giving it the responsiblity for modelling the same sentence in multiple languages, it is encouraged to learn the semantic meaning.

For each data point $$n$$ in a data set, typical latent variable architectures model the joint distribution of the latent representation $$\mathbf{z}^{(n)}$$ and the observed sentence $$\mathbf{x}^{(n)}$$ as:

$$p(\mathbf{z}^{(n)},\mathbf{x}^{(n)}) = p(\mathbf{z}^{(n)}) p_{\theta}(\mathbf{x}^{(n)}\vert \mathbf{z}^{(n)})$$

where $$\theta$$ are trainable parameters of the model.

{:.text-center img}
![SGVB]({{ site.urlimg }}/fig_1.png "SGVB")

Instead of modelling a single sentence per latent representation, GNMT uses a shared latent representation to model the same sentence both in the source and target languages. GNMT models the joint distribution of the latent representation, source sentence $$\mathbf{x}^{(n)}$$ and target sentence $$\mathbf{y}^{(n)}$$ as:

$$p(\mathbf{z}^{(n)},\mathbf{x}^{(n)},\mathbf{y}^{(n)}) = p(\mathbf{z}^{(n)}) p_{\theta}(\mathbf{x}^{(n)}\vert \mathbf{z}^{(n)}) p_{\theta}(\mathbf{y}^{(n)}\vert \mathbf{z}^{(n)},\mathbf{x}^{(n)})$$

{:.text-center img}
![GNMT]({{ site.urlimg }}/fig_2.png "GNMT")

This set up means that $$\mathbf{z}^{(n)}$$ models the commonality between the source and target sentences, which is the semantic meaning.

For full details on the neural networks used to model the distributions of the source $$p_{\theta}(\mathbf{x}^{(n)}\vert\mathbf{z}^{(n)})$$ and target $$p_{\theta}(\mathbf{y}^{(n)}\vert \mathbf{z}^{(n)},\mathbf{x}^{(n)})$$, see [^SB2018].

One may argue that it would be better if the target sentence were dependent only on the latent representation and not directly on the source sentence, giving the model $$p(\mathbf{x}\vert \mathbf{z})p(\mathbf{y}\vert \mathbf{z})p(\mathbf{z})$$ -- forcing the latent representation to be fully responsible for generating both the source and target sentence. However, we found that the generated translations weren't sufficiently syntactically coherent.


## Training the model

We use the Stochastic Gradient Variational Bayes (SGVB)[^KW2014][^R2014] algorithm to train the model described above. SGVB introduces a 'recognition' model $$q_{\phi}(\mathbf{z}^{(n)}\vert \mathbf{x}^{(n)})$$ which acts as an approximation to the true but intractable posterior $$p(\mathbf{z}^{(n)}\vert \mathbf{x}^{(n)})$$, thus forming the following lower bound on the log likelihood of the observed data:

$$\mathcal{L}(\mathbf{x}^{(n)}) = \mathbb{E}_{q_{\phi}(\mathbf{z}^{(n)}\vert \mathbf{x}^{(n)})} [\log p(\mathbf{z}^{(n)},\mathbf{x}^{(n)}) - \log q_{\phi}(\mathbf{z}^{(n)}\vert \mathbf{x}^{(n)})]$$

The model parameters $$\theta$$ and recognition parameters $$\phi$$ are then jointly learned by performing gradient ascent on this lower bound.


## Generating translations - the 'banana trick'

Suppose the model has been trained, and then we are given a sentence in the source language ($$\mathbf{x}$$) and asked to find a translation of that sentence. In this scenario, we want to find the most likely target sentence conditioned on the given source sentence. The natural objective is therefore:

$$\mathbf{y}^{*} = \arg \max_{y} p(\mathbf{y}\vert \mathbf{x}) = \arg \max_{y} \int p_{\theta}(\mathbf{y}\vert \mathbf{z},\mathbf{x}) p(\mathbf{z}\vert \mathbf{x}) d\mathbf{z}$$

However this integral is intractable, and so we cannot perform this maximisation exactly. Instead, we perform approximate maximisation by iteratively refining a 'guess' for the target sentence. We first make a random guess for the target sentence, and then iterate between the following two steps:

1. Draw samples of the latent representation from the approximate posterior, using the source sentence and the latest guess for the target sentence.
2. Update the guess for the target sentence based on the latent representation samples from step 1. This update is done by choosing $$\mathbf{y}$$ to maximise $$p_{\theta}(\mathbf{y}\vert \mathbf{z},\mathbf{x})$$.

Intuitively, this procedure computes the values of the latent representation that are likely to have generated both the source sentence and the latest guess for the target sentence. It then improves the guess based on those values of the latent variable. Mathematically, this iteratively increases a lower bound on $$\log p(\mathbf{y}\vert \mathbf{x})$$ until convergence and is an application of the Expectation Maximisation algorithm, here $$\mathbf{y}$$ playing the role of parameters.

Note: in our group we refer to this as the 'banana trick' because we are first aware of its usage in exercise 5.7 in [^B2012], discussing a fictitious protein sequence in bananas.

Below, we show an example of a long sentence translated from French to English by GNMT. The long range coherence of the translation is a good indicator of the model's ability to capture semantic information about the sentence within the latent representation.

**Source**: Dans ce décret, il met en lumière les principales réalisations de la République d'Ouzbékistan dans le domaine de la protection et de la promotion des droits de l'homme et approuve le programme d'activités marquant le soixantième anniversaire de la déclaration universelle des droits de l'homme.

**Target**: The decree highlights major achievements by the Republic of Uzbekistan in the field of protection and promotion of human rights and approves the programme of activities devoted to the sixtieth anniversary of the universal declaration of human rights.

**GNMT**: In this decree, it highlights the main achievements of the Republic of Uzbekistan on the protection and promotion of human rights and approves the activities of the sixtieth anniversary of the universal declaration of human rights.


## Dealing with missing words

Because GNMT's latent representation captures information about the meaning of the sentence rather than just the syntax, it is able to produce good translations even when there are missing words in the source sentence. The procedure for generating translations is similar to that described above --  however in this scenario we also have to refine a guess for the missing words in the source sentence. 

We first make random guesses for the missing words in the source sentence and for the target sentence, and then iterate between the following three steps:

1. Draw samples of the latent representation from the approximate posterior, using the latest guesses for the source and target sentences.
2. Update the guess for the source sentence based on the latent representation samples from step 1. This update is done by choosing $$\mathbf{x}$$ to maximise $$p_{\theta}(\mathbf{x}\vert \mathbf{z})$$.
3. Update the guess for the target sentence based on the latent representation samples from step 1 and on the updated guess for the source sentence from step 2. This update is done by choosing $$\mathbf{y}$$ to maximise $$p_{\theta}(\mathbf{y}\vert \mathbf{z},\mathbf{x})$$.

Below is an example of a sentence translated from Spanish to English, where the struck through words in the source sentence are considered missing. Using its latent representation, the model does remarkably well at imputing what the missing words may be and translating them accordingly.

**Source**: Expresando su ~~satisfacción~~ por ~~la~~ asistencia que han ~~prestado~~ a los territorios no autónomos algunos ~~organismos~~ especializados y ~~otras~~ organizaciones del sistema de las naciones ~~unidas~~, especialmente el ~~programa~~ de las naciones unidas ~~para~~ el desarrollo.

**Target**: Welcoming the assistance extended to non-self-governing territories by certain specialized agencies and other organizations of the United Nations system, in particular the United Nations development programme.

**GNMT**: Expressing its gratitude for the assistance given to non-self-governing territories by some specialized agencies and other organizations of the United Nations system, in particular from the development programmes of the United Nations.


## Cross-language parameter sharing

With the architecture described above, if we wanted to translate between, say, English (EN), Spanish (ES) and French (FR), we would have to train 6 separate models for EN → ES, ES → EN, EN → FR, etc. However because all three of these languages share somewhat similar structures, we may not lose much performance by sharing parameters. We therefore add two indicator variables to the model, one for the input language ($$l_{x}$$) and another for the output language ($$l_{y}$$). By doing this, we only have to train a single model to translate between three languages, instead of having 6 separate models.

The joint distribution of the latent representation, source sentence and target sentence becomes:

$$p(\mathbf{z}^{(n)},\mathbf{x}^{(n)},\mathbf{y}^{(n)}\vert l_{x},l_{y}) = p(\mathbf{z}^{(n)}) p_{\theta}(\mathbf{x}^{(n)}\vert \mathbf{z}^{(n)},l_{x}) p_{\theta}(\mathbf{y}^{(n)}\vert \mathbf{z}^{(n)},\mathbf{x}^{(n)},l_{x},l_{y})$$

{:.text-center img}
![GNMT-Multi]({{ site.urlimg }}/fig_3.png "GNMT-Multi")

We refer to this version of the model as GNMT-Multi.

Overfitting is a phenomenon whereby a model is too closely fit to a particular set of data points. In machine translation, this often occurs when there aren't enough paired sentences for the model to learn from. However, the cross language parameter sharing used for GNMT-Multi helps to mitigate this issue. This is because 6 separate models are essentially condensed into a single model, meaning that there aren't enough parameters to allow the model to memorise the training data.

Below, we plot the BLEU scores comparing GNMT-Multi against 6 separate GNMT models, trained with only 400,000 pairs of translated sentences. BLEU is a measure of how well the generated translations match the true target sentences; higher is better. Clearly GNMT-Multi with a limited amount of available training data performs significantly better than each of the 6 separate GNMT models.

{:.text-center img}
![GNMT_vs_GNMT-Multi_1]({{ site.urlimg }}/gnmt_vs_gnmt_multi_1.png "GNMT vs. GNMT-Multi (trained with 400,000 sentence pairs)")

When there is a large amount of training data available, we find that there is very little difference in the performance of GNMT and GNMT-Multi. Below, we plot the BLEU scores for the models trained with 4,000,000 sentence pairs; we find that there is no degradation in performance due to sharing parameters across languages!

{:.text-center img}
![GNMT_vs_GNMT-Multi_2]({{ site.urlimg }}/gnmt_vs_gnmt_multi_2.png "GNMT vs. GNMT-Multi (trained with 4,000,000 sentence pairs)")


## Semi-supervised learning

Suppose, as in the previous section, that we only have access to a limited number of paired sentences, but that we now have available lots of untranslated sentences in each language. To learn from untranslated sentences, we can set the input language $$l_{x}$$ and output language $$l_{y}$$ to the same value, so that the model learns to reconstruct the sentence instead of translating it. Intuitively, this should further help the model to learn the structure and style of each language and thus produce more coherent translations at test time. We refer to this version of the model as GNMT-Multi-SSL.

Below we plot the BLEU scores of GNMT-Multi-SSL, GNMT-Multi and the 6 separate GNMT models. They are trained first with 400,000 then with 4,000,000 pairs of translated sentences. In both cases, they are also trained with approximately 20,900,000 untranslated English sentences, 2,700,000 untranslated Spanish sentences and 4,500,000 untranslated French sentences. GNMT-Multi-SSL clearly helps to mitigate overfitting when there are limited paired sentences available. In fact, GNMT-Multi-SSL trained with only 400,000 paired sentences performs about as well as each of the 6 separate GNMT models trained with 4,000,000 sentences! GNMT-Multi-SSL also produces higher BLEU scores even when there is lots of paired data; this verifies our intuition that adding monolingual data helps the model to develop a better understanding of each language individually, and output more coherent sentences accordingly.

{:.text-center img}
![GNMT_vs_GNMT-Multi_vs_GNMT-Multi-SSL_1]({{ site.urlimg }}/gnmt_vs_gnmt_multi_vs_gnmt_multi_ssl_1.png "GNMT vs. GNMT-Multi (trained with 400,000 sentence pairs)")

{:.text-center img}
![GNMT_vs_GNMT-Multi_vs_GNMT-Multi-SSL_2]({{ site.urlimg }}/gnmt_vs_gnmt_multi_vs_gnmt_multi_ssl_2.png "GNMT vs. GNMT-Multi (trained with 4,000,000 sentence pairs)")


## Summary
{:.no_toc}

We introduce Generative Neural Machine Translation (GNMT), which is a latent variable model that uses sentences with the same meaning in multiple languages to learn representations which better understand the semantics of the text. It can be used to translate a source sentence by iteratively refining a guess for the target sentence and updating the latent representation accordingly. Because it captures the meaning of the sentence, GNMT is particularly effective at producing translations when there are missing words in the source sentence. We also introduce GNMT-Multi, which is a single unified model (instead of one per language pair) to mitigate overfitting when there is limited paired data available. Finally, we leverage large amounts of untranslated sentences to help the model to further learn the structure and style of each language and produce more coherent translations.


### References
{:.no_toc}

[^B2012]: D. Barber. Bayesian Reasoning and Machine Learning. Cambridge University Press, 2012.

[^KW2014]: D. Kingma and M. Welling. Auto-Encoding Variational Bayes. In International Conference on Learning Representations, 2014.

[^R2014]: D. Rezende et al. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In Proceedings of the 31st International Conference on Machine Learning, PMLR 32, pages 1278–1286, 2014.

[^SB2018]: H. Shah and D. Barber. Generative Neural Machine Translation. In Advances in Neural Information Processing Systems, 2018.