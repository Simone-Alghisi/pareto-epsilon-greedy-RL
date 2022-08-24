# Wilcoxon's rank sum test
# Wilcoxon's rank sum test **does not assumes** that the samples being tests is drawn from a normal distribution

# ========== IMPORTS ========================
library(ggplot2)
suppressMessages(library(dplyr))
library(tibble)
set.seed(32)

# ========= import the data =================

name_test = 'reward'

df_pareto_raw <- read.csv('./data/reward_598.csv') %>% 
  dplyr::rename(value = mean_reward) %>%
  tibble::as_tibble()

df_random_raw <- read.csv('./data/reward_596.csv') %>% 
  dplyr::rename(value = mean_reward) %>%
  tibble::as_tibble()

print(paste("Mean outcomes of the Pareto battles [1 win, 0 lose]", mean(df_pareto_raw$won)))
print(paste("Mean outcomes of the Random battles [1 win, 0 lose]", mean(df_random_raw$won)))

df_pareto <- tibble(method = rep("Pareto", nrow(df_pareto_raw)), value = df_pareto_raw$value)
df_random <- tibble(method = rep("Random", nrow(df_random_raw)), value = df_random_raw$value)
df <- dplyr::bind_rows(df_pareto, df_random)
df <- df %>%
  group_by(method) %>%
  mutate(outlier = (value > median(value) + IQR(value) * 1.5 | value < median(value) - IQR(value) * 1.5)) %>% 
  ungroup

# ========= plot the data ===================

file_name <- "box_plot.pdf"
print(paste("Creating box plot in", file_name))
pdf(file_name)
g <- ggplot(df, aes(x = method, y = value)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_point(data = function(x) dplyr::filter(x, outlier == TRUE), position = 'jitter') +
  theme_minimal() +
  labs(
    title = "Box plot", 
    subtitle = "Reinforcement Learning with and without NSGA2 solution warm up",
    y = name_test,
    x = "Method"
  )
plot(g)
invisible(dev.off())

# Wilcoxon rank mean test
print("Wilcoxon rank mean test [expected p-value to be << 0.05]")

wilcox.test(df_pareto_raw$won, df_random_raw$won,alternative="g") 

# if the p-value is less than 0.05 then we can reject the null-hypothesis -> statistical significance is proven
# what the test does is:
# H0: µ1 = µ2 (the two means are equal)
# HA: µ1 ≠ µ2 (the two means are not equal)


