---
title:
- "DarkrAI: a Pareto ε-greedy policy"
subtitle:
- Improving Pokémon AI Traning With NSGA-II
aspectratio:
- 1609
institute:
- University of Trento
author: 
- \href{mailto:simone.alghisi-1@studenti.unitn.it}{Simone Alghisi} 
- \href{mailto:samuele.bortolotti@studenti.unitn.it}{Samuele Bortolotti} 
- \href{mailto:massimo.rizzoli@studenti.unitn.it}{Massimo Rizzoli}\newline
- \href{mailto:erich.robbi@studenti.unitn.it}{Erich Robbi}
date:
- \today
lang:
- en-US
section-titles:
- false
theme:
- Copenhagen
colortheme:
- default
navigation:
- horizontal
logo:
- assets/unitn.pdf
fontsize: 9pt
link-citations: true
nocite: |
  @*
---

# Introduction

::: {.columns align=center}

:::: {.column width=50%}

## Pokémon

Pokémon uses a turn-based system: at the start of each turn, both sides can choose to attack, use an item, switch the Pokémon for another in their party. The Pokemon who strikes first is determined by the Move's Priority and the Pokémon Speed. Each Pokémon uses moves to reduce their opponent's HP until one of them faints, i.e. their HP reach 0. If all of a player's Pokémon faint, the player loses the battle.

::::

:::: {.column width=50%}

\centering
\begin{figure}
\animategraphics[autoplay,loop,width=\linewidth]{5}{./assets/pokemonbattle-}{0}{20}
\caption{Pokémon battle}
\end{figure}
::::

:::

# Reinforcement Learning
*Reinforcement learning (RL)* is an area of Machine Learning where an agent receives a reward based on the action it has performed. Actions allow the agent to transition from a state to another. The final objective is to learn a policy to reach a terminal state with the best reward achievable.

## Deep Q-Learning
The reinforcement learning technique we have employed is called *Deep Q-Learning*, which maps input states to a pair of actions and Q-values using an Artificial Neural Network. *Q-Learning* is based on the *Q-function*, namely $Q : S \times A \rightarrow R$, which returns - given a state-action pair $(s, a) \in S \times A$ - the expected discounted reward $(r \in R)$ for future states.

# NSGA-II
*NSGA-II* is a Evolutionary Algorithm that allows to produce *Pareto-equivalent* (or non-dominated) solutions of a multi-objective optimisation problem.

## General idea
The idea is that, given that the search space is very big - there are $10^{354}$ different ways a Pokémon battle can start, and each turn has at most $306$ different outcomes (and only for a single player) - we would like to positively bias our model with a controlled search, removing particularly useless moves, i.e. consider for the most Pareto-equivalent solutions.

# Genotype representation
Generally, in a Pokémon battle two actions are possible, i.e. performing a move or a switch. Moreover, depending on the type of battle, it may be necessary to specify the target of the move. To encode such a thing, we came up with the following genotype: each Pokémon is represented using two genes, i.e. action and target (optional) $(a, t)$. The whole genotype tells us who is going
to perform what on whom.

\begin{figure}
\centering
\includegraphics[width=\linewidth]{./assets/genotype_representation}
\caption{Genotype representation}
\end{figure}

# Genetic operators - Mutation
Mutation is performed for each gene in a genotype with probability $\mathbb{P}_{m} = 10\%$: both the action and the target may be mutated, meaning that it is possible to go from a move to a switch (and vice-versa).

\begin{figure}
\centering
\includegraphics[scale=.45]{./assets/mutation}
\caption{Mutation}
\end{figure}

# Genetic operators - Recombination
Instead, we used Uniform Crossover in a particular way: given that each Pokémon is represented by a valid $(a, t)$ pair, we perform crossover by selecting the whole pair from one of the parents to avoid inconsistencies. Furthermore, crossover is performed with $\mathbb{P}_{c} = 100\%$, and $\mathbb{P}_{bias} = 50\%$ (i.e. the bias towards a certain offspring).

\begin{figure}
\centering
\includegraphics[width=\linewidth]{./assets/crossover}
\caption{Recombination}
\end{figure}

# Objective & Optimisation
Concerning the *Pareto front* we have considered four variables with the following optimisation problem:
$$\underline{x} = (x_1,x_2,x_3,x_4) \in \mathbb{R}^4 \quad \text{where} \;\; \mathbb{R}^4 = \{(x_1,x_2,x_3,x_4) : 0 \leq x_1,x_2,x_3,x_4 \leq 100\}$$
where $x_1$ is the damage dealt by the ally Pokémons to the opponents, $x_2$ is the damage dealt by the opponents' Pokémons to the allies, $x_3$ is the health points remaining of the player's Pokémons and $x_4$ is the health points remaining of the opponent's Pokémons.

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.85\linewidth]{./assets/pareto_front}
\caption{Multi-level diagram plot of the Pareto Front}
\end{figure}

# Architecture details
The agent architecture is a four-layer deep *Multilayer Perceptron (MLP)*, which employs *ReLU* as activation function. In particular:

- input and output layers size depend on the type of battle the network is facing (e.g. a $4 \text{ VS } 4$ battle implies a size of $244$ input neurons);
- two hidden hidden layers of size $256$ and $128$, respectively.

\begin{figure}
\centering
\includegraphics[width=\linewidth]{./assets/ann_io}
\caption{Forward pass of the Artificial Neural Network}
\end{figure}

# Players
The standard agent uses a simple $\varepsilon$-greedy policy:

- it starts from a probability $\mathbb{P}_r=1.0$ to perform a random action;
- it linearly decreases to $\mathbb{P}_r=0.1$ in the first $40\%$ of the training;
- for the remaining $60\%$ of the training it linearly decreases to $\mathbb{P}_r=0.01$.

::: {.columns align=top}

:::: {.column width=55%}

## ParetoPlayer
*ParetoPlayer* embeds the Pareto search of non-dominated moves: it performs 

- a random action using *NSGA-II* with $\mathbb{P}_{pareto} = 0.7$;
- a completely random move with $\mathbb{P}_{random} = 0.3$.

::::

:::: {.column width=45%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=\linewidth]{./assets/eps_threshold}
\caption{eps-threshold value during the training}
\end{figure}

::::

:::

# Program structure

\begin{figure}
\centering
\includegraphics[width=\linewidth]{./assets/program_structure_extended}
\caption{Program structure}
\end{figure}

# Experiments
All agents were trained by having them fight against *MaxDamagePlayer*, i.e. a bot which always chooses the combination of moves that deals the highest amount of damage.

Several situations were considered, such as:

- 2 VS 2 battle with static teams;
- 2 VS 2 battle with the opponent team sampled randomly from a pool of possible Pokémons.

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.8\linewidth]{./assets/battles_setup}
\caption{Different battle settings}
\end{figure}

# Statistical tests

We have tested both the normality and the statistical significance of the proposed solution with the employment of the following graphical and analytical tools:

::: {.columns align=center}

:::: {.column width=50%}

## Normality
* Quantile-Quantile plot
* Shapiro-Wilk test
* Kolmogorov-Smirnov nomality test

::::

:::: {.column width=50%}

## Statistical significance
* Box plot
* t-test
* Wilcoxon rank-sum test

::::

:::

# Empirical results - Fixed teams

::: {.columns align=center}

:::: {.column width=50%}

*  We expect the episode reward of ParetoPlayer to be higher than the episode reward of Player ($p \leq 2.2 \cdot 10^{-16}$).
*  Training runs of ParetoPlayer tends to produce higher reward values ($p \leq 2.886 \cdot 10^{-12}$), but in some cases the rewards are almost equivalent.
*  During evaluation ParetoPlayer tends to win more ($p \leq 3.048 \cdot 10^{-5}$)
::::

:::: {.column width=50%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=\linewidth]{./assets/reward_on_episode}
\caption{Row-mean reward per episode for Pareto and ParetoPlayer}
\end{figure}

::::

:::

# Empirical results - Fixed Teams

\centering
\movie[
  width=\linewidth,
  height=0.6\linewidth,
  showcontrols,
  poster
]{}{./assets/2v2_fixed.mp4}

# Empirical results - Sampled Teams

::: {.columns align=center}

:::: {.column width=50%}

*  We still expect the episode reward of ParetoPlayer to be higher than the episode reward of Player 
*  ParetoPlayers' reward distributions have a significant shift location to the right w.r.t to the Player's distribution ($p \leq 0.002278$ and $p \leq 0.01931$)
*  The winning percentage is not always in favour of ParetoPlayer (0.716 and 0.673 vs 0.694)
::::

:::: {.column width=50%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=\linewidth]{./assets/rewardasfunctionofepisode2}
\caption{Row-mean reward per episode for Pareto and ParetoPlayer (sampled teams)}
\end{figure}

::::

:::

# Difficulties
The main difficulties we have encountered concern:

* Damage calculator
* Hyperparameters selection and topology search
* Pokémon double battles
* Pokémon battle switches

# Conclusions
- ParetoPlayer is able to positively bias the training by providing higher rewards
- when the search space is small enough and a single win condition is presented, Player outperforms ParetoPlayer

## Future work
- perform better topology and hyperparameters search
- reduce NSGA-II performance bottleneck (time-consuming operations)
- use another network to properly address forced switch

#
\begin{center}
  \LARGE{Thanks for your attention!}
\end{center}

# Resources

## Repositories

* [pareto-epsilon-greedy-RL](https://github.com/Simone-Alghisi/pareto-epsilon-greedy-RL)
* [poke-env (modified)](https://github.com/Simone-Alghisi/poke-env)
* [Pokemon_info](https://github.com/massimo-rizzoli/Pokemon_info)

## Collaborators' Github

* [Simone Alghisi](https://github.com/Simone-Alghisi)
* [Samuele Bortolotti](https://github.com/samuelebortolotti)
* [Massimo Rizzoli](https://github.com/massimo-rizzoli)
* [Erich Robbi](https://github.com/erich-r)

<!-- # Appendix
## Appendix content
The appendix contains the topics we are not able to discuss during the oral examination
-->


# Normality - Fixed teams 

::: {.columns align=center}

:::: {.column width=50%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.8\linewidth]{./assets/Normality_QQ_Pareto}
\caption{Quantile-Quantile plot episode reward computed on $1000$ battles during \textcolor{red}{ParetoPlayer} model evaluation}
\end{figure}

::::

:::: {.column width=50%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.8\linewidth]{./assets/Normality_QQ_Random}
\caption{Quantile-Quantile plot episode reward computed on $1000$ battles during \textcolor{blue}{Player} model evaluation}
\end{figure}

::::

:::

# Additional results - Box plots

::: {.columns align=center}

:::: {.column width=50%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.8\linewidth]{./assets/box_plot_2v2}
\caption{Box plot computed on $1000$ battles during ParetoPlayer and Player model evaluation}
\end{figure}

::::

:::: {.column width=50%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.8\linewidth]{./assets/box_plot_2v2_sampled}
\caption{Box plot computed on $1000$ battles during ParetoPlayer and Player model evaluation (\textbf{with variable enemy team})}
\end{figure}

::::

:::

# Unknown Moves

::: {.columns align=top}

:::: {.column width=60%}

At the beginning of the battle the agent does not know which moves the opponent Pokémon have, thus we rely on Pikalytics in order to get the most probable moves in competitive settings.

To assign the most probable moves to a Pokémon we:

0. normalise the probabilities obtained from Pikalytics;
1. draw a random number;
2. sample the corresponding move;
3. repeat from 0 until we have a total of 4 moves.

::::

:::: {.column width=40%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=\linewidth]{./assets/pikalytics}
\caption{Zacian's possible moves on Pikalytics}
\end{figure}

::::

:::

# State Description

Among all the possible information, we focused on the following:

- the percentage of Pokémons alive;
- the weather;
- the field condition;
- for each Pokémon we considered:
  - type (e.g. fire, grass, ect.);
  - HP percentage;
  - statistics (normalised);
  - status (e.g. asleep, poisoned, ect.);
  - for each of its move, we considered: id, priority, type, and damage it deals to the opponent active Pokémons.

# Fitness Evaluation

In order to get a good fitness evaluation of our turn, we perform the following:

0. analyse the previous turn;
1. estimate the statistics of the opponent;
2. predict a possible turn order based on the Moves Priority and Pokémons' Speed;
3. prepare the field by handling switches;
4. compute the damage by either:
   - sending a request to the server;
   - retrieve a previous result.

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=0.9\linewidth]{./assets/damage_calculator}
\caption{Damage calculator request}
\end{figure}

# Previous Turn Analysis

::: {.columns align=center}

:::: {.column width=60%}

To have a better estimate of the next possible turns, the previous turn is analysed to extract unknown information (e.g. Pokemon's stats, item, etc.). In particular we

0. retrieve the previous turn;
1. extract the actions performed;
2. simulate the actions execution using our current knowledge;
3. compare the actual turn with the one estimated;
4. adjust our believes if needed.

::::

:::: {.column width=40%}

\begin{figure}
\captionsetup{justification=centering}
\centering
\includegraphics[width=\linewidth]{./assets/base_stats_range}
\caption{A Pokémon's base stats}
\end{figure}

::::

:::

# Shapiro-Wilk test

The Shapiro-Wilk test is a test of normality which is frequent in statistics and it is based on the expected values of the order statistics. Its null hypothesis is whether a sample $\{x_1, \dots, x_n\}$ came from a **normally distributed population**:

- Thus, if the $p-$value is less than the chosen $\alpha$ level, then the null hypothesis is rejected and there is evidence that the data tested are not normally distributed;
- On the other hand, if the $p-$value is greater than the chosen $\alpha$ level, then the null hypothesis (that the data came from a normally distributed population) can not be rejected.

# Kolmogorov-Smirnov test

In statistics, the Kolmogorov-Smirnov test is a nonparametric test of the equality between two distribution, namely it does not assume anything about the underlying data distribution. Its null hypothesis is whether the two set of samples were drawn from the **same probability distribution**:

- Thus, if the $p-$value is less than the chosen $\alpha$ level, then the null hypothesis is rejected and there is evidence that the data tested are not drawn from the same distribution, namely one group stochastically dominates the other;
- On the other hand, if the $p-$value is greater than the chosen $\alpha$ level, then the null hypothesis (that the data came from the same distribution) can not be rejected.

# Wilcoxon rank-sum test

The Wilcoxon rank-sum test is one of the most powerful non-parametric test which is used to compare two groups of continuous measures. Its null hypothesis is whether the two populations have the **same distribution and the same median**:

- Thus, if the $p-$value is less than the chosen $\alpha$ level, then the null hypothesis is rejected and there is evidence that the data tested are not drawn from the same distribution, namely one distribution is shifted to the left or right of the other;
- On the other hand, if the $p-$value is greater than the chosen $\alpha$ level, then the null hypothesis (that the data came from the same distribution) can not be rejected.

# Empirical results - Sampled teams [First Game]
\movie[
  width=\linewidth,
  height=0.6\linewidth,
  showcontrols,
  poster
]{}{./assets/2v2_sampled.mp4}


# Empirical results - Sampled teams [Second Game]
\centering
\movie[
  width=\linewidth,
  height=0.6\linewidth,
  showcontrols,
  poster
]{}{./assets/2v2_sampled_2.mp4}

<!-- # Guidelines 

1. introducing pokemon, and RL in a few words
2. describe why the training becomes very difficult (THE NUMBERS MASON, WHAT DO THEY MEAN)
3. propose the solution with NSGA-II in order to have a controlled search rather than a completely random one
4. describe the multi-objective problem:
  - genotype representation
  - mutation and recombination strategy 
  - search strategy
  - defining objectives and the optimisation
5. specify what kind of tests have been conducted, and why (IDK EITHER il re bomba)
6. live demo
7. analysis of the results
  - pareto front (we show both plots of the same pareto front)
  - training convergence
8. difficulties
9. contributions
-->

<!-- # Gif in PDF
comandi totali:
convert -coalesce something.gif something.png -> questo fa in modo di avere una png per frame
magick identify -verbose something.gif | grep 'Delay' -> questo fa in modo di ritornare il framerate della gif
Aprire il pdf con Okular o Adobe Acrobat
https://tex.stackexchange.com/questions/240243/getting-gif-and-or-moving-images-into-a-latex-presentation per più info
Parametri: FPS - preambolo dei png - 0 perche si e 12 sono i frame totali

\centering\animategraphics[autoplay,loop,width=0.5\linewidth]{10}{./assets/lol-}{0}{12}
-->