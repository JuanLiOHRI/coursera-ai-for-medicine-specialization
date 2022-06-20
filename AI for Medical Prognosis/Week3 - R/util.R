load_data <- function() 
{
  df <- read.csv("../lymphoma.csv", header = T)
  names(df)[3] <- "Event"
  return(df)
}