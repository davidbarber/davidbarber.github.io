---
layout: post
title: "Thinking Fast and Slow with Deep Learning and Tree Search"
date: "2017-10-30"
slug: "Thinking Fast and Slow with Deep Learning and Tree Search"
description: "Reinforcement Learning"
category: 
  - views
  - featured
# tags will also be used as html meta keywords.
tags:
  - deep learning
  - Monte Carlo Tree Search
  - Hex
  - reinforcement learning 
  - AlphaGo
  - Dual Process Theory


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
# imagesummary: foo.png
## for twitter card with large image:
# imagefeature: "http://img.youtube.com/vi/VEIrQUXm_hY/0.jpg"
## for twitter video card: (active for this page)
#videofeature: "https://www.youtube.com/embed/iG9CE55wbtY"
#imagefeature: "http://img.youtube.com/vi/iG9CE55wbtY/0.jpg"
#videocredit: tedtalks
---

How to train powerful reinforcement learning agents by Thinking Fast and Slow. 


<!--more-->

* TOC
{:toc}

$$\newcommand{\sq}[1]{\left[#1\right]}$$
$$\newcommand{\ave}[1]{\mathbb{E}\sq{#1}}$$



## Dual Process Theory
{:.no_toc}

According to [dual-process theory](https://en.wikipedia.org/wiki/Dual_process_theory) human reasoning consists of two different kinds of thinking.
System 1, is a fast, unconscious and automatic mode of thought, also known as intuition or heuristic process. System 2, an evolutionarily recent process unique to humans, is a slow, conscious, explicit and rule-based mode of reasoning.


{:.text-center img}
![dual process]({{ site.urlimg }}/behaviour-design-predicting-irrational-decisions-12-638.jpg "dual process theory")
[image credit](https://www.slideshare.net/AshDonaldson/behaviour-design-predicting-irrational-decisions)


When learning to complete a challenging planning task, such as playing a board game, humans exploit both processes: strong intuitions allow for more effective analytic reasoning by rapidly selecting interesting lines of play for consideration. Repeated deep study gradually improves intuitions. Stronger intuitions feedback to stronger analysis, creating a closed learning loop. In other words, humans learn by thinking fast and slow.


<!--### What's wrong with Deep RL?
{:.no_toc}-->

In current Deep Reinforcement Learning algorithms such as Policy Gradients[^Williams] and DQN[^DQN], neural networks make action selections with no lookahead; this is analogous to System 1. Unlike human intuition, their training does not benefit from a ‘System 2’ to suggest strong policies. Another issue is that state-of-the-art RL board playing algorithms require an initial database of human expert play[^AlphaGo]. Making a state-of-the-art board game player _ex nihilo_ is a major challenge for AI.

## Expert Iteration (ExIt)
{:.no_toc}

In our [NIPSexitpaper](), we introduced Expert Iteration (ExIt), which is a new and general framework for learning.  

{:.text-center img}
![ExIt]({{ site.urlimg }}/ExIt.png "Expert Iteration")

ExIt can be viewed as an extension of Imitation Learning (IL) methods to domains where the best known experts are unable to achieve satisfactory performance. In stanadard IL an apprentice is trained to imitate the behaviour of an expert.  In ExIt, we extend this to an iterative learning process.  Between each iteration, we perform an Expert Improvement step, where we bootstrap the (fast) apprentice policy to increase the performance of the (comparatively slow) expert.

To give some intuition around this idea, consider playing a board game such as chess. Here the expert is analogous to a strong (but slow) chess player, and the apprentice is analogous to an initially novice (but quick thinking) chess player.  

Initially the expert player plays some games against an opponent. The expert is a strong player, thinking deep (and slow) about each move. The apprentice observes the chess board state and each eventual move made by the expert and then tries his best to learn to quickly imitate the move made by the expert in each of the observed board positions . In an algorithm, this imitation could be done, for example, by fitting a neural network to the move made by the expert from a game position.  The apprentice learns a fast policy that is able to quickly imitate the play of the expert on the moves seen so far.  A key point here is that the apprentice learns in such a way that they are able to generalise and then apply their quick intution on positions not previously seen -- the neural network thus plays the role of both generalising and imitating the play of the expert. 

Now that the apprentice has learned a fast imitation of the expert (on the moves seen so far), she can try to be of use to the expert.  One approach would be that when the expert now wishes to make a move, a small set of candidate moves are suggested very quickly by the apprentice which the expert can then consider in depth, possibly also guided during this slow thought process by other quick suggestions by the apprentice.

At the end of this phase, the expert will have made a set of apprentice-aided moves, with each move being typically  much stronger than the apprentice could have made with the deep (but slow) thought of the expert.  

The above process now repeats, with the apprentice retraining on the moves suggested by the expert. This completes one full iteration of the learning phase of the apprentice and we iterate this process until we either run out of time or the apprentice learns to perform at roughly the same level as the expert. 

From a Dual Process perspective, the Imitation Learning step is analogous to a human improving their intuition for the task by studying example problems, while the Expert Improvement step is analogous to a human using the their improved intuition to guide future analysis


### Tree Search and Deep Learning 
{:.no_toc}

Exit is a very general strategy for learning and the apprentice and expert can be specified in a variety of ways. In board games Monte Carlo Tree Search is a strong playing strategy[^MCTS] and is a natural candidate to play the role of the expert.  Deep Learning has been shown to be a successfull method to imitate the play of strong players and it is therefore a natural candidate to play the role of the apprentice[^AlphaGo].

The expert policy is calculated using a tree search algorithm. We use the apprentice to improve such an expert by using the apprentice policy to direct search effort towards promising moves, or by evaluating states encountered during search more quickly and accurately using our apprentice, effectively reducing the search breadth and depth. In other words, we bootstrap the knowledge acquired by Imitation Learning back into the planning algorithm.

 
<!--The role of the expert is to perform exploration, and thereby to accurately determine strong move sequences, from a single position. The role of the apprentice is to generalise the policies that the expert discovers across the whole state space, and to provide rapid access to that strong policy for bootstrapping in future searches. The canonical choice of expert is a tree search algorithm. Search considers the exact dynamics of the game tree local to the state under consideration, and can be considered analogous to the lookahead human games players engage in when planning their moves. The bootstrap policy can be used to bias search towards promising moves, aid node evaluation, or both. By employing search, we can find promising sequences potentially far away from the bootstrap policy, accelerating learning in complex scenarios. Possible tree search algorithms include Monte Carlo Tree Search[^MCTS]-->



### The Boardgame Hex
{:.no_toc}

[Hex](https://en.wikipedia.org/wiki/Hex_(board_game)) is a classic two-player boardgame played on an $$n\times n$$ hexagonal grid. The players, denoted by colours black and white, alternate placing stones of their colour in empty cells. The black player wins if there is a sequence of adjacent black stones connecting the North edge of the board to the South edge. White wins if he achieves a sequence of adjacent white stones running from the West edge to the East edge.


{:.text-center img}
![hex]({{ site.urlimg }}/hexBW.png "Hex")

The above represents play on a $$5\times 5$$ board, with white winning (reproduced from [^Hex]).   Hex has deep strategy, making it challenging for machines to play and its large action set and connection-based rules means it shares similar challenges for AI to Go. Compared to Go, however, the rules are simpler and there can be no draws, making it an ideal testbed for AI. 

Because the rules of Hex are so simple, the game is amenable to mathematical analysis and the current best machine player MOHEX[^MOHEX] uses a combination of Monte Carlo Tree Search and smart mathematical insights.  MOHEX has won every Computer Games Olympiad Hex tournament since 2009. MOHEX is also trained on datasets of human expert play. 

We wanted to see if we can outperform MOHEX by using our ExIt strategy, learning trained tabula rasa, without game-specific knowledge or human example play, beside the rules of the game. To do this, our expert is a MCTS player that is guided by the apprentice neural network.  Our neural network is based on a deep convolutional network, as shown below

{:.text-center img}
![NN]({{ site.urlimg }}/NN.png "NN")

The expert improvement step is then modelled by using the modified MCTS formula 

$$
UCT(s,a) + w \frac{\hat{\pi}(a|s)}{n(s,a)+1}
$$

This formula is closely related to one found in Gelly and Silver[^OnlineOffline]. Here $$s$$ is the state of the Hex board, $$a$$ is a possible action (ie move) from $$s$$. The term $$UCT(s,a)$$ represents the classical Upper Confidence Tree formula[^MCTS] used in MCTS.

The additional term denotes apprentice  helping guide the search to more promising moves, with $$\hat{\pi}$$ being the suggestion of the apprentice, and $$n(s,a)$$ the number of visits currently made by the search algorithm through state $$s$$ and taking action $$a$$; $$w$$ is an empirically chosen weighting factor that balances the slow thinking of the expert with the fast intuition of the apprentice. 


To generate the data for traiing the apprentice (during each Imitation Learning phase), the batch approach generates data afresh, discarding all data from previous iterations. In the online version we consider instead a buffer of the most recent moves generted and we also consider an exponentially weighted version that favours more recent moves generated (since the more recent moves will be be from a stronger player).  A comparison of these different approaches is given below in which we compare the strength (the [Elo Rating](https://en.wikipedia.org/wiki/Elo_rating_system) of the learned algorithm against a measure of training time.  We also show a more classical approach, known as REINFORCE (see [NIPSexitpaper]() for details)

{:.text-center img}
![results]({{ site.urlimg }}/BatchOnline.png "results")


What the figure shows is that the ExIt approach is considerably more effective than classical approaches.  Indeed, after training, our apprentice-aided MCTS player outperforms the best known machine Hex player, namely MOHEX, beating it in 71% of games played on a $$9\times 9$$ board. 


Need some plots of game play


### Relation to AlphaGo Zero
{:.no_toc}

AlphaGo Zero [? ], published shortly after this work was first published, also implements the ExIt algorithm, and shows that it is able to achieve state-of-the-art performance in Go. Blah.....


## Summary
{:.no_toc}

ExIt is great...... Blah


### References
{:.no_toc}

[^Williams]: R. J. Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Mach. Learn., 8(3-4):229–256, 1992.

[^DQN]: V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-Level Control through Deep Reinforcement Learning. Nature, 518(7540):529–533, 2015.

[^AlphaGo]: D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 529(7587):484–489, 2016.

[^MCTS]: L. Kocsis and C. Szepesvári. Bandit Based Monte-Carlo Planning. In European Conference on Machine Learning, pages 282–293. Springer, 2006.

[^MOHEX]: S.-C. Huang, B. Arneson, R. Hayward, M. Müller, and J. Pawlewicz. MoHex 2.0: A Pattern-Based MCTS Hex Player. In International Conference on Computers and Games, pages 60–71. Springer, 2013.

[^OnlineOffline]: S. Gelly and D. Silver. Combining Online and Offline Knowledge in UCT. In Proceedings of the 24th International Conference on Machine learning, pages 273–280. ACM, 2007.