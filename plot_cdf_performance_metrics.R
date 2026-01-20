library(ggplot2)
library(reshape2)
library(patchwork)

filenames <- c("results_simulations_all.csv", 
               "results_simulations_without_date_without_rnn.csv",
               "results_simulations_all_with_rnn.csv",
                  "results_simulations_without_date_with_rnn.csv" 
                  )
titles <- c("With time DNN50", 
            "Without time DNN50",
            "with time RNN50",
            "Without time RNN50"
            )
plots <- list()

for (i in 1:length(filenames)){
  filename <- c(filenames[i])
  title <- c(titles[i])
  
  
  df <- read.csv(filename)
  
  # only 100 rows
  df_100 <- df[1:100, ]
  
  # melt dataframe for long format
  df_long <- melt(df_100)
  
  # CDFs
  p <- ggplot(df_long, aes(x = value, color = variable)) +
    stat_ecdf(geom = "step", size = 1) +
    labs(
      title = paste("Performance metrics", title, sep="-"),
      x = "Value",
      y = "CDF",
      color = "Metric"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "bottom"
    )
  plots[[i]] <- p
  #print(p)
}

combined_plot <- (plots[[1]] | plots[[2]]) /
  (plots[[3]] | plots[[4]])


print(combined_plot)