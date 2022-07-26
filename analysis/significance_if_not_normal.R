# Wilcoxon's rank sum test
# Wilcoxon's rank sum test **does not assumes** that the samples being tests is drawn from a normal distribution

# ========== IMPORTS ========================
library(ggplot2)
suppressMessages(library(dplyr))
library(tibble)
set.seed(32)

# ========= import the data =================
# **NOTE:** here since we do not have the data I assume it is not sampled from a normal distribution
winrate_with_pareto_data <- rgamma(20, shape = 1, rate = 3) # gamma distribution
winrate_without_pareto_data <- rbeta(20, shape1 = 2, shape2 = 4) # Beta distribution

winrate_with_pareto <- tibble(method = rep("With Pareto", length(winrate_with_pareto_data)), rate = winrate_with_pareto_data)
winrate_without_pareto <- tibble(method = rep("Without Pareto", length(winrate_without_pareto_data)), rate = winrate_without_pareto_data)
winrate <- dplyr::bind_rows(winrate_with_pareto, winrate_without_pareto)

# ========= plot the data ===================
file_name <- "box_plot.pdf"
print(paste("Creating box plot in", file_name))
pdf(file_name)
g <- ggplot(winrate, aes(x = method, y = rate)) + 
  geom_boxplot() + 
  theme_minimal() +
  labs(
    title = "Box plot", 
    subtitle = "Reinforcement Learning with and without NSGA2 solution warm up",
    y = "Winrate",
    x = "Method"
  )
plot(g)
invisible(dev.off())

# Wilcoxon rank mean test
print("Wilcoxon rank mean test [expected p-value to be << 0.05]")
wilcox.test(winrate_without_pareto_data, winrate_with_pareto_data, alternative = "g") 

# if the p-value is less than 0.05 then we can reject the null-hypothesis -> statistical significance is proven
# what the test does is:
# H0: µ1 = µ2 (the two means are equal)
# HA: µ1 ≠ µ2 (the two means are not equal)