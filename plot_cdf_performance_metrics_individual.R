library(ggplot2)
library(reshape2)
library(patchwork)

filenames <- c(#"results_simulations_all.csv", 
               #"results_simulations_without_date_without_rnn.csv",
               "results_simulations_all_with_rnn.csv",#200
               "results_simulations_without_date_with_rnn.csv" #200
)
titles <- c(#"With time DNN50", 
            #"Without time DNN50",
            "with time",#200
            "Without time"#200
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
    scale_color_manual(
      values = c(
        "blue",   # primera métrica
        "orange", # segunda métrica
        "green",  # tercera métrica
        "red"     # cuarta métrica
      )
    ) +
    labs(
      title = "",#paste("Performance metrics", title, sep="-"),
      x = "Value",
      y = "CDF",
      color = "Metric"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16),
      axis.title.x = element_text(size = 16),  # tamaño explícito del label del eje X
      axis.title.y = element_text(size = 16),  # tamaño explícito del label del eje Y
      axis.text.x  = element_text(size = 14),  # tamaño de los números del eje X
      axis.text.y  = element_text(size = 14),  # tamaño de los números del eje Y
      legend.position = "bottom",
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 13)
    )
  
  plots[[i]] <- p
  print(p)
}
