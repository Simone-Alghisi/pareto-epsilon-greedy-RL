library(tidyverse)

runs.random <- read.csv("./data/random_run.csv")
runs.pareto <- read.csv("./data/pareto_run.csv")

runs.pareto <- runs.pareto %>% select(contains("ep_reward") & -contains("MIN") & -contains("MAX") | contains("episode"))
runs.random <- runs.random %>% select(contains("ep_reward") & -contains("MIN") & -contains("MAX"))

runs.pareto <- runs.pareto %>% mutate(mean_pareto = rowMeans(select(.,-contains("episode"))))
runs.random <- runs.random %>% mutate(mean_random = rowMeans(select(.,-contains("episode"))))

export <- bind_cols(runs.random,runs.pareto)

ggplot(data=export,aes(x=episode)) +
  geom_line(aes(y=mean_pareto,color="pareto"),alpha=0.2) + 
  geom_line(aes(y=mean_random,colour="random"),alpha=0.2) + 
  geom_smooth(aes(y=mean_pareto,colour="pareto"),method = "loess",span=0.2,formula=y ~ x) +
  geom_smooth(aes(y=mean_random,colour="random"),method = "loess",span=0.2,formula=y ~ x) + 
  theme_minimal() + 
  labs(x="Episode",y="Episode Reward") +
  scale_color_manual(name = c("Pareto", "Random"), 
                     values = c("pareto" = "blue", "random" = "red"))+
  guides(col=guide_legend(title="Method"))

# are mean pareto and mean random sampled from the same distribution?
ks.test(export$mean_pareto,export$mean_random)

#is pareto variable dominating over random variable?
ks.test(export$mean_pareto,export$mean_random,alternative = "l")
