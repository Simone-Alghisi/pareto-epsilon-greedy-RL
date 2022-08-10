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
fontsize:
- 10mm
bibliography: bibliography.bib
link-citations: true
nocite: |
  @*
---

# Introduction

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

# Second slide

::: {.columns align=center}

:::: column

## A

::::

:::: column

::::

:::

# Top text

::: {.columns align=center}

:::: column

## A

::::

:::: column

## B

::::

:::

# Gif in PDF

<!-- comandi totali: -->
<!-- convert -coalesce something.gif something.png -> questo fa in modo di avere una png per frame --> 
<!-- magick identify -verbose something.gif | grep 'Delay' -> questo fa in modo di ritornare il framerate della gif --> 
<!-- Aprire il pdf con Okular o Adobe Acrobat, forse su Chrome? -->
<!-- https://tex.stackexchange.com/questions/240243/getting-gif-and-or-moving-images-into-a-latex-presentation per più info -->
<!-- Parametri: FPS - preambolo dei png - 0 perche si e 12 sono i frame totali -->
\centering\animategraphics[autoplay,loop,width=0.5\linewidth]{10}{./assets/lol-}{0}{12}

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

# Conclusions

\begin{center}
  \LARGE{Thanks for your attention!}
\end{center}

<!--# Appendix-->
<!---->
<!--## Appendix content-->
<!--The appendix contains the topics we are not able to discuss during the oral examination-->

# References {.allowframebreaks}