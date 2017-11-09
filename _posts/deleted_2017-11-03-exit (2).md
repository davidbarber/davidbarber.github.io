---
layout: post
title: "Learning From Scratch by Thinking Fast and Slow with Deep Learning and Tree Search"
date: "2017-11-3"
slug: "Learning From Scratch by Thinking Fast and Slow with Deep Learning and Tree Search"
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

Training powerful reinforcement learning agents from scratch by Thinking Fast and Slow.


<!--more-->

* TOC
{:toc}


## Dual Process Theory
{:.no_toc}

According to [dual process theory](https://en.wikipedia.org/wiki/Dual_process_theory) human reasoning consists of two different kinds of thinking.
System 1 is a fast, unconscious and automatic mode of thought, also known as intuition. System 2 is a slow, conscious, explicit and rule-based mode of reasoning is believed to  be an evolutionarily recent process.

{:.text-center img}
![dual process]({{ site.urlimg }}/behaviour-design-predicting-irrational-decisions-12-638.jpg "dual process theory")
[image credit](https://www.slideshare.net/AshDonaldson/behaviour-design-predicting-irrational-decisions)


When learning to complete a challenging planning task, such as playing a board game, humans exploit both processes: strong intuitions allow for more effective analytic reasoning by rapidly selecting interesting lines of play for consideration. Repeated deep study gradually improves intuitions. Stronger intuitions feedback to stronger analysis, creating a closed learning loop. In other words, humans learn by thinking fast and slow[^TFAS].


<!--### What's wrong with Deep RL?
{:.no_toc}-->

In current Deep Reinforcement Learning algorithms such as Policy Gradients[^Williams] and DQN[^DQN], neural networks make action selections with no lookahead; this is analogous to System 1. Unlike human intuition, their training does not benefit from a ‘System 2’ to suggest strong policies.

A criticism of some AI algorithms such as AlphaGo[^AlphaGo] is that they use a database of human expert play[^AlphaGo].   In the initial phase of training the RL agent mimics the moves of a human expert -- only after this initial phase does it begin to learn potentially more powerful super-human play. This is somewhat unsatisfactory since the resulting algorithm may be heavily biased toward a human style of playing, blind to potentially more powerful lines of play. Whilst, in areas such as game playing, it is natural to assume that there will be a database of human expert play available, in other settings in which we wish to train an AI machine, no such database may be available. Therefore, showing how to train a state-of-the-art board game player _ex nihilo_ is a major challenge for AI.

## Expert Iteration (ExIt)
{:.no_toc}

Expert Iteration[^NIPSpaper] (ExIt) is a general framework for learning that we introduced in July 2017 and can result in powerful AI machines, without needing to mimic human strategies.

{:.text-center img}
![ExIt]({{ site.urlimg }}/ExIt.png "Expert Iteration")

ExIt can be viewed as an extension of Imitation Learning (IL) methods to domains where the best known experts are unable to achieve satisfactory performance. In standard IL an apprentice is trained to imitate the behaviour of an expert.  In ExIt, we extend this to an iterative learning process.  Between each iteration, we perform an Expert Improvement step, where we bootstrap the (fast) apprentice policy to increase the performance of the (comparatively slow) expert.

To give some intuition around this idea, consider playing a board game such as chess. Here the expert is analogous to chess player playing on slow time controls (having lots of time to decide on his move), and the apprentice is playing on blitz time controls (having little time to decide which move to make).

During independent study, the player considers multiple possible moves from a position, thinking deeply (and slowly) about each possible move move. She discovers which moves are and are not successful in this position. When she encounters a similar board state in the future, her study will have given her an intuitive understanding of what moves are likely to be good, allowing her to play well, even under blitz time controls. Her intuition is imitating the strong play she calculated via deep thinking. Humans do not become become excellent chess players by only playing blitz matches, deeper study is an essential part of the learning process.

<!--Initially the expert player plays some games against an opponent. The expert is a strong player, thinking deeply (and slowly) about each move. The apprentice observes each chess board state and the eventual move made by the expert for each of those states. She then tries her best to learn to quickly imitate the move made by the expert in each of the observed board positions.-->


For an AI game playing machine, this imitation could be achieved, for example, by fitting a neural network to the move made by another `machine expert' from a game position.  The apprentice learns a fast policy that is able to quickly imitate the play of the expert on the moves seen so far.  A key point here is that, assuming that there is structure underlying the game,  Machine Learning enables the apprentice to generalise their intuitive to take quick decisions on positions not previously seen. That is, the apprentice isn't just a creating a look-up-table of moves made by the human from a fixed database of positions. The neural network thus plays the role of both generalising and imitating the play of the expert.

Now that the apprentice has learned a fast imitation of the expert (on the moves seen so far), she can try to be of use to the expert. When the expert now wishes to make a move, a small set of candidate moves are suggested very quickly by the apprentice which the expert can then consider in depth, possibly also guided during this slow thought process by other quick insights from the apprentice.

At the end of this phase, the expert will have made a set of apprentice-aided moves, with each move being typically  much stronger than either the apprentice or expect could have made alone.

The above process now repeats, with the apprentice retraining on the moves suggested by the expert. This completes one full iteration of the learning phase and we iterate this process until the apprentice converges.

From a Dual Process perspective, the Imitation Learning step is analogous to a human improving their intuition for the task by studying example problems, while the Expert Improvement step is analogous to a human using their improved intuition to guide future analysis.


### Tree Search and Deep Learning
{:.no_toc}

Exit is a general strategy for learning and the apprentice and expert can be specified in a variety of ways. In board games Monte Carlo Tree Search (MCTS) is a strong playing strategy[^MCTS] and is a natural candidate to play the role of the expert.  Deep Learning has been shown to be a successful method to imitate the play of strong players[^AlphaGo] which we therefore use as the apprentice.

At the Expert Improvement phase we use the apprentice to direct the Monte Carlo Tree Search algorithm toward promising moves, effectively reducing the game tree search breadth and depth. In this way, we bootstrap the knowledge acquired by Imitation Learning back into the planning algorithm.


<!--The role of the expert is to perform exploration, and thereby to accurately determine strong move sequences, from a single position. The role of the apprentice is to generalise the policies that the expert discovers across the whole state space, and to provide rapid access to that strong policy for bootstrapping in future searches. The canonical choice of expert is a tree search algorithm. Search considers the exact dynamics of the game tree local to the state under consideration, and can be considered analogous to the lookahead human games players engage in when planning their moves. The bootstrap policy can be used to bias search toward promising moves, aid node evaluation, or both. By employing search, we can find promising sequences potentially far away from the bootstrap policy, accelerating learning in complex scenarios. Possible tree search algorithms include Monte Carlo Tree Search[^MCTS]-->

We use our apprentice to improve the expert a second way. The fast move choices allow us to simulate many high level games cheaply. We can summarise this experience by learning a value function, which will aid our expert by rapidly evaluating whether states found in the tree are desirable, and so reduce the required depth of search.


### The board game Hex
{:.no_toc}

[Hex](https://en.wikipedia.org/wiki/Hex_(board_game)) is a classic two-player board game played on a $$n\times n$$ hexagonal grid. The players, denoted by colours black and white, alternate placing stones of their colour in empty cells. The black player wins if there is a sequence of adjacent black stones connecting the North edge of the board to the South edge. White wins if he achieves a sequence of adjacent white stones running from the West edge to the East edge.


{:.text-center img}
![hex]({{ site.urlimg }}/hexBW.png "Hex")

The above represents play on a $$5\times 5$$ board, with white winning (reproduced from [^MOHEX]).   Hex has deep strategy, making it challenging for machines to play and its large action set and connection-based rules means it shares similar challenges for AI to Go. Compared to Go, the rules are simpler and there can be no draws, making it an ideal testbed for AI game play learning strategies.

Because the rules of Hex are so simple, the game is relatively amenable to mathematical analysis (compared for example to Go) and the current best machine player MoHex[^MOHEX] uses a combination of Monte Carlo Tree Search and smart mathematical insights.  MoHex has won every Computer Games Olympiad Hex tournament since 2009. It is noteworthy that MoHex uses a rollout policy trained on datasets of human expert play.

We wanted to see if we can use our ExIt training strategy to learn an AI player than can outperform MoHex, without using any game-specific knowledge or human example play (beside the rules of the game). To do this, our expert is a MCTS player that is guided by the apprentice neural network.  Our neural network is a form of deep convolutional network with two output policies -- ones for black play and one for white (see [^NIPSpaper] for details). 

<!--{:.text-center img}
![NN]({{ site.urlimg }}/NN.png "NN")-->

Expert Improvement is achieved by using the modified MCTS formula[^OnlineOffline]

$$
UCT(s,a) + w \frac{\hat{\pi}(a|s)}{n(s,a)+1}
\tag{1}\label{eq:uct}
$$

Here $$s$$ is the state of the Hex board, $$a$$ is a possible action (ie move) from $$s$$. The term $$UCT(s,a)$$ represents the classical Upper Confidence Tree formula[^MCTS] used in MCTS.

The additional term helps the neural network apprentice  guide the search to more promising moves. Here $$\hat{\pi}$$ is the policy (suggested relative strength of each possible action $$a$$ from the board state $$s$$) of the apprentice and $$n(s,a)$$ the number of visits currently made by the search algorithm through state $$s$$ and taking action $$a$$; $$w$$ is an empirically chosen weighting factor that balances the slow thinking of the expert with the fast intuition of the apprentice.


To generate the data for training the apprentice (during each Imitation Learning phase), the batch approach generates data afresh, discarding all data from previous iterations. We also consider an online version in which we instead keep a running buffer of the most recent moves generated. We also consider an exponentially weighted online version that favours more recent moves generated (which correspond to the strongest, most recent play).  A comparison of these different approaches is given below in which we compare the strength (measured in terms of the [ELO](https://en.wikipedia.org/wiki/Elo_rating_system) score) of each learned policy network against a measure of training time.  

{:.text-center img}
![results]({{ site.urlimg }}/BatchOnline.png "results")

We also show the result of using a more traditional Reinforcement Learning approach in which a policy $$\hat{\pi}(a\vert s)$$ is learned only through self play (ie no MCTS). This is essentially the method used within AlphaGo[^AlphaGo] to train their policy network. The figure shows that the ExIt training approach is considerably more effective than more classical approaches. 

Ater training, the policy network is used to create the final player, which is a policy guided MCTS, as in \eqref{eq:uct}. Our apprentice-aided MCTS player outperforms the best known machine Hex player, MoHex, beating it in 71% of games played on a $$9\times 9$$ board.



### Why does ExIt work so well?
{:.no_toc}

Imitation Learning is generally appreciated to be easier than Reinforcement Learning, and this partly explains why ExIt is more successful than model-free methods like REINFORCE.

Furthermore, for MCTS to recommend a move, it must be unable to find any weakness with its search. Effectively, therefore, a move played by MCTS is good against a large selection of possible opponents. In contrast, in regular self play (in which the opponent move is made by the network playing as the opposite colour), moves are recommended if they beat only this single opponent under consideration. This is, we believe, a key insight into why ExIt works well (when using MCTS as the expert) --- the apprentice effectively learns to play well against many opponents.


### Relation to AlphaGo Zero
{:.no_toc}

AlphaGo Zero[^AlphaGoZero] (published a few months after the first version of our work appeared[^AXpaper]) also implements an ExIt style algorithm and shows that it possible to achieve state-of-the-art performance in Go without the use of human expert play.  There are subtle differences in the details of the training approach and neural network architecture used, but the approach is essentially the same. 


## Summary
{:.no_toc}

Expert Iteration is a new Reinforcement Learning algorithm, motivated by the dual process theory of human thought. ExIt decomposes the Reinforcement Learning into the separate subproblems of generalisation and planning. Planning is performed on a case-by-case basis, and only once a strong plan is found is the resultant policy generalised. This allows for long-term planning and results in faster learning and state-of-the-art final performance, particularly for challenging problems. This training strategy is powerful enough to learn state-of-the-art board game AI players without requiring any examples of expert human play.

<!--We show that this algorithm significantly outperforms a variant of the REINFORCE algorithm in learning to play the board game Hex, and that the resultant tree search algorithm comfortably  defeats the state-of-the-art in Hex play, despite being trained tabula rasa.-->


### References
{:.no_toc}


[^TFAS]: D. Kahneman. Thinking, Fast and Slow. Macmillan, 2011.

[^Williams]: R. J. Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Mach. Learn., 8(3-4):229–256, 1992.

[^DQN]: V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-Level Control through Deep Reinforcement Learning. Nature, 518(7540):529–533, 2015.

[^AlphaGo]: D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 529(7587):484–489, 2016.

[^MCTS]: L. Kocsis and C. Szepesvári. Bandit Based Monte-Carlo Planning. In European Conference on Machine Learning, pages 282–293. Springer, 2006.

[^MOHEX]: S.-C. Huang, B. Arneson, R. Hayward, M. Müller, and J. Pawlewicz. MoHex 2.0: A Pattern-Based MCTS Hex Player. In International Conference on Computers and Games, pages 60–71. Springer, 2013.

[^OnlineOffline]: S. Gelly and D. Silver. Combining Online and Offline Knowledge in UCT. In Proceedings of the 24th International Conference on Machine learning, pages 273–280. ACM, 2007.

[^NIPSpaper]: T. Anthony, and T. Zheng, and D. Barber. Thinking Fast and Slow with Deep Learning and Tree Search, Neural Information Processing Systems (NIPS 2017). In Press.

[^AXpaper]: T. Anthony, and T. Zheng, and D. Barber. Thinking Fast and Slow with Deep Learning and Tree Search. [arXiv CoRR:abs/1705.08439](http://arxiv.org/abs/1705.08439), May 2017.

[^AlphaGoZero]: D. Silver,  etal.  Mastering the game of Go without human knowledge. Nature 550:354–359, October 2017.