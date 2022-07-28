# Imports
library(ggplot2)
library(tibble)
library(tidyr)
suppressMessages(library(ggpmisc))
suppressMessages(library(dplyr))
library(optparse)
 
# option parser
option_list = list(
    make_option(c("-f", "--file"), type="character", default=NULL, 
              help="dataset file name", metavar="character"),
    make_option(c("-c", "--column"), type="character", default=NULL,
              help="step column name", metavar="character"),
    make_option(c("-y", "--response"), type="character", default=NULL,
              help="response variable name", metavar="character")
) 
 
# Option parser
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser);

# control the arguments
if (is.null(opt$file) | is.null(opt$column) | is.null(opt$response)){
  print_help(opt_parser)
  stop("Three arguments must be supplied (inputfile, columnname, response)", call.=FALSE)
}

# Open the dataset
df <- as_tibble(read.csv(opt$file))
column = opt$column
response = opt$response

# clean data
df <- df %>% pivot_longer(
  -column,
  names_to = "run", 
  values_to = "values"
)

df <- df %>%
  dplyr::filter(run %in% c("X3v3.40.10.pt.expl.0.7.0.5k.tu.mem.fill.5k.1.5k.eps.369...winrate",
  "X3v3.0.75k.tu.3k.eps.365...winrate")) %>%
  na.omit(values)

# visualize the content
file_name <- paste0("regression_on_",response, ".pdf")
print(paste("Training curve chart", file_name))
pdf(file_name)
g <- ggplot(df, aes(x=column, y=values, color = run)) + 
    geom_smooth(method = "lm", formula = y ~ x) +
    stat_poly_line(formula = y ~ x) +
    stat_poly_eq(aes(label = paste(
      after_stat(eq.label),
      after_stat(rr.label), sep = "*\", \"*",
      after_stat(p.value.label))
    )) +
    coord_cartesian(ylim = c(0, 1.10)) +
    geom_point() +
    geom_line() +
    labs(title="Time Series Chart", 
         subtitle=paste(column, "%"), 
         y=response) +
    theme_minimal()+
    theme(plot.margin=unit(c(1,1,4,0.5),"cm")) +
    theme(legend.position=c(0.55,-0.2))
plot(g)
invisible(dev.off())