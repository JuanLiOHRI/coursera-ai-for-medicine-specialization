load_data <- function() 
{
  library(dplyr)
  df <- read.csv("../pbc.csv", header = T)
  df <- select(df,-id) %>% 
    filter(status != 1) %>% 
    mutate(status = status / 2.0,
           time   = time / 365.0,
           trt    = trt - 1,
           sex    = ifelse(!is.na(sex), ifelse(sex == 'f', 0.0, 1.0), NA)) %>% 
    tidyr::drop_na()
  
  return(df)
}