# ========== IMPORTS ========================
library(ggplot2)
suppressMessages(library(dplyr))
library(tibble)

# just to make this example reproducible
set.seed(32)

# ========= import the data =================

# **NOTE:** here since we do not have the data I assume it is mtcars with mpg as column
data(mtcars) 

df <- mtcars %>% 
  dplyr::select(mpg) %>%
  dplyr::rename(winrate = mpg) %>%
  tibble::as_tibble()

# ========= normality tests =================

# Q-Q plot
file_name <- "Normality_QQ.pdf"
print(paste("Creating Q-Q plot in", file_name))
pdf(file_name)
qq <- ggplot(df, aes(sample = winrate)) +
  stat_qq() +
  stat_qq_line() +
  labs(
    title = "Normal Q-Q plot",
    subtitle = "Winrate normality visual test"
    y = "Sample Quantiles",
    x = "Theoretical Quantiles"
  ) +
  theme_minimal()
print(qq)
invisible(dev.off())

# shapiro-wilk test
print("Shapiro Normality test [expected p-value to be >> 0.05]")
shapiro.test(df$winrate)
# If the p-value is much greater than 0.05 we cannot reject that the null hypothesis which states that the sample distribution is normal distributed 
# (note, this does not imply our data is normal distributed, for this reason I have run more tests)
# what the test does is:
# H0: sample distribution is normal
# HA: sample distribution is not normal

# kolmogorov-smirnov normality test
print("Kolmogorov Smirnov test [expected p-value to be >> 0.05]")
ks.test(df$winrate, 'pnorm', mean(df$winrate), sd(df$winrate))
# If the p-value is much greater than 0.05 we cannot reject that the null hypothesis which states that the sample distribution is normal distributed 
# (note, this does not imply our data is normal distributed, for this reason I have run more tests)
# what the test does is:
# H0: sample distribution is normal
# HA: sample distribution is not normal

#### Few notes on data normality #####
# How to Handle Non-Normal Data
#
# If a given dataset is not normally distributed, we can often perform one of the following transformations to make it more normally distributed:
#
# 1. Log Transformation: Transform the values from x to log(x).
#
# 2. Square Root Transformation: Transform the values from x to √x.
#
# 3. Cube Root Transformation: Transform the values from x to x1/3.
#
# By performing these transformations, the dataset typically becomes more normally distributed.
# Read this tutorial to see how to perform these transformations in R.
# https://www.statology.org/transform-data-in-r/