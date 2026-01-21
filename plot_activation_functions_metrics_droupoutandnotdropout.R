library(tools)
library(ggplot2)

dir_A <- "~/Documentos/implementations/data_for_t/droupout_0_allactivations"
dir_B <- "~/Documentos/implementations/data_for_t/droupout_02_allactivations"
read_last_row <- function(dir_path, param_name) {
  
  files <- list.files(
    dir_path,
    pattern = "^results_simulations_all_.*\\.csv$",
    full.names = TRUE
  )
  
  out <- data.frame()
  
  for (f in files) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    last <- df[nrow(df), ]
    
    activation <- gsub("results_simulations_all_|\\.csv", "", basename(f))
    activation <- toupper(activation)
    
    out <- rbind(
      out,
      data.frame(
        activation = activation,
        param = param_name,
        precision = last$precision,
        recall    = last$recall,
        accuracy  = last$accuracy,
        f1        = last$f1
      )
    )
  }
  
  out
}



data_A <- read_last_row(dir_A, "Without dropout")
data_B <- read_last_row(dir_B, "Dropout=0.2")
data_all <- rbind(data_A, data_B)

metrics <- c("precision", "recall", "accuracy", "f1")

data_long <- reshape(
  data_all,
  varying = metrics,
  v.names = "value",
  timevar = "metric",
  times = metrics,
  direction = "long"
)



for (m in metrics) {
  
  df_plot <- subset(data_long, metric == m)
  
  p <- ggplot(df_plot, aes(x = activation, y = value, fill = param)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
    labs(
      title = paste("Comparación", toupper(m)),
      x = "Activation Function",
      y = m,
      fill = "Parámetro"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
}