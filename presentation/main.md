---
title:
- IL RE BOMBA
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
bibliography: bibliography.bib
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
The reinforcement learning technique we have employed is called *Deep Q-Learning*, which maps input states to a pair of actions and Q-values using an Artificial Neural Network. *Q-Learning* is based on the *Q-function*, namely $Q : S \times A \rightarrow R$, which returns - given a state-action pair ($s, a \in S \times A$) - the expected discounted reward ($r \in R$) for future states.

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
\centering
\includegraphics[width=0.9\linewidth]{./assets/pareto_front}
\caption{Recombination}
\end{figure}

# Architecture details
The agent architecture is a four-layer deep *Multilayer Perceptron (MLP)*, which employs *ReLU* as activation function. In particular:

- input and output layers size depend on the type of battle the network is facing (e.g. a $4 \text{ VS } 4$ battle implies a size of $244$ input neurons);
- two hidden hidden layers of size $256$ and $128$, respectively.

# Players
The standard agent uses a simple $\varepsilon$-greedy policy:

- it starts from a probability $\mathbb{P}_r=1.0$ to perform a random action;
- it linearly decreases to $\mathbb{P}_r=0.1$ in the first $40\%$ of the training;
- for the remaining $60\%$ of the training it linearly decreases to $\mathbb{P}_r=0.01$.

## ParetoPlayer
*ParetoPlayer* embeds the Pareto search of non-dominated moves:

- it performs a random move chosen from the ones returned by *NSGA-II* with $70\%$ probability;
- a completely random one with $30\%$ probability.

# Program structure

\begin{figure}
\centering
\includegraphics[width=\linewidth]{./assets/program_structure}
\caption{Program structure}
\end{figure}

# Experiments
All agents were trained by having them fight against *MaxDamagePlayer*, i.e. a bot which always chooses the combination of moves that deals the highest amount of damage.

Several situations were considered, such as:

- 2 VS 2 battle with a static teams;
- 2 VS 2 battle with a single victory condition;
- 2 VS 2 battle with teams sampled randomly from a pool of possible Pokémons.

::: {.columns align=center}

:::: {.column width=50%}

## A

::::

:::: {.column width=50%}

## B

::::

:::

# Live Demo

\centering
\movie[
  width=0.7\linewidth,
  height=0.6\linewidth,
  showcontrols,
  poster
]{}{./assets/rock.mp4}

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
* Wilcoxon rank mean test

::::

:::

# Empirical results


# Difficulties
The main difficulties we have encountered concern:

* Damage calculator
* Hyperparameters selection and topology search
* Pokémon double battles
* Pokémon battle switches

# Contributions

<!-- # Gif in PDF
comandi totali:
convert -coalesce something.gif something.png -> questo fa in modo di avere una png per frame
magick identify -verbose something.gif | grep 'Delay' -> questo fa in modo di ritornare il framerate della gif
Aprire il pdf con Okular o Adobe Acrobat
https://tex.stackexchange.com/questions/240243/getting-gif-and-or-moving-images-into-a-latex-presentation per più info
Parametri: FPS - preambolo dei png - 0 perche si e 12 sono i frame totali

\centering\animategraphics[autoplay,loop,width=0.5\linewidth]{10}{./assets/lol-}{0}{12}
-->

# Conclusions

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