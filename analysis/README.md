# Analysis

This folder contains some statistical tests which are useful for corroborating the thesis of this project, namely whether the improvement due to the introduction of Pareto optimal solutions is statistically significant for a reinforcement learning technique which plays Pokémon.

Here we present simple and brief descriptions of each of the method employed.

## Normality test

### Q-Q plots

In statistics, Q-Q(quantile-quantile) plots play a very vital role to graphically analyze and compare two probability distributions by plotting their quantiles against each other. If the two distributions which we are comparing are exactly equal then the points on the Q-Q plot will perfectly lie on a straight line y = x.

Q-Q plots are used to find the type of distribution for a random variable whether it be a Gaussian Distribution, Uniform Distribution, Exponential Distribution or even Pareto Distribution, etc. 

You can tell the type of distribution using the power of the Q-Q plot just by looking at the plot. In general, we are talking about Normal distributions.
Therefore, we know how much of the data lies in the range of first standard deviation, second standard deviation and third standard deviation from the mean.

We plot the theoretical quantiles or basically known as the standard normal variate (a normal distribution with mean=0 and standard deviation=1)on the x-axis and the ordered values for the random variable which we want to find whether it is Gaussian distributed or not, on the y-axis. Which gives a very beautiful and a smooth straight line like structure from each point plotted on the graph.

Now we have to focus on the ends of the straight line. If the points at the ends of the curve formed from the points are not falling on a straight line but indeed are scattered significantly from the positions then we cannot conclude a relationship between the x and y axes which clearly signifies that our ordered values which we wanted to calculate are not Normally distributed.

If all the points plotted on the graph perfectly lies on a straight line then we can clearly say that this distribution is Normally distribution because it is evenly aligned with the standard normal variate which is the simple concept of Q-Q plot.

Note that when the data points are pretty less the Q-Q plot does not perform very precisely and it fails to give a conclusive answer but when we have ample amount of data points and then we plot a Q-Q plot using a large data set then it gives us a significant result to conclude any result about the type of distribution.

An example of a Q-Q plot is the following:

![Example Q-Q plot](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Weibull_qq.svg/1024px-Weibull_qq.svg.png)


### Shapiro-Wilk test

The Shapiro-Wilk test is a statistical test for the hypothesis that a group of values come from a normal distribution. (The mean and variance of this normal distribution need not be 0 or 1 respectively.) Empirically, this test appears to have the best power (among tests that test for normality).

The basis idea behind the Shapiro-Wilk test is to estimate the variance of the sample in two ways: 
- the regression line in the QQ-Plot allows to estimate the variance
- the variance of the sample can also be regarded as an estimator of the population variance. Both estimated values should approximately equal in the case of a normal distribution and thus should result in a quotient of close to 1.0. If the quotient is significantly lower than 1.0 then the null hypothesis (of having a normal distribution) should be rejected.

### Kolmogorov-Smirnov nomality test

In statistics, the Kolmogorov–Smirnov test (K-S test or KS test) is a nonparametric test of the equality of continuous (or discontinuous), one-dimensional probability distributions that can be used to compare a sample with a reference probability distribution (one-sample K–S test), or to compare two samples (two-sample K–S test). 

In essence, the test answers the question "What is the probability that this collection of samples could have been drawn from that probability distribution?" or, in the second case, "What is the probability that these two sets of samples were drawn from the same (but unknown) probability distribution?".

The Kolmogorov–Smirnov test can be modified to serve as a goodness of fit test. In the special case of testing for normality of the distribution, samples are standardized and compared with a standard normal distribution. This is equivalent to setting the mean and variance of the reference distribution equal to the sample estimates, and it is known that using these to define the specific reference distribution changes the null distribution of the test statistic.

## Statistical significance

### t-test

It is a type of inferential statistic used to study if there is a statistical difference between two groups. Mathematically, it establishes the problem by assuming that the means of the two distributions are equal ($H_0: \mu_1=\mu_2$). If the t-test rejects the null hypothesis ($H_0: \mu_1=\mu_2$), it indicates that the groups are highly probably different.

The most frequently used t-tests are one-sample and two-sample tests:
- A one-sample location test of whether the mean of a population has a value specified in a null hypothesis.
- A two-sample location test of the null hypothesis such that the means of two populations are equal. 

In this case, since we are comparing two different approaches we are employing a two-sample location t-test.

### Wilcoxon rank mean test

The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used either to test the location of a population based on a sample of data, or to compare the locations of two populations using two matched samples.

The one-sample version serves a purpose similar to that of the one-sample Student's t-test. For two matched samples, it is a paired difference test like the paired Student's t-test (also known as the "t-test for matched pairs" or "t-test for dependent samples"). The Wilcoxon test can be a good alternative to the t-test when population means are not of interest; for example, when one wishes to test whether a population's median is non-zero, or whether there is a better than 50% chance that a sample from one population is greater than a sample from another population.

### Boxplot

In descriptive statistics, a box plot or boxplot is a method for graphically demonstrating the locality, spread and skewness groups of numerical data through their quartiles. In addition to the box on a box plot, there can be lines (which are called whiskers) extending from the box indicating variability outside the upper and lower quartiles, thus, the plot is also termed as the box-and-whisker plot and the box-and-whisker diagram. Outliers that differ significantly from the rest of the dataset may be plotted as individual points beyond the whiskers on the box-plot.

Box plots are non-parametric: they display variation in samples of a statistical population without making any assumptions of the underlying statistical distribution. The spacings in each subsection of the box-plot indicate the degree of dispersion (spread) and skewness of the data, which are usually described using the five-number summary. In addition, the box-plot allows one to visually estimate various L-estimators, notably the interquartile range, midhinge, range, mid-range, and trimean. Box plots can be drawn either horizontally or vertically. 

An example of a box plot is the following:

![Example box plot](https://upload.wikimedia.org/wikipedia/commons/2/2a/Boxplots_with_skewness.png)

# Sources

- [towardsdatascience](https://towardsdatascience.com/)
- [wikipedia](https://en.wikipedia.org/)
- [statistical odds and ends](https://statisticaloddsandends.wordpress.com/)