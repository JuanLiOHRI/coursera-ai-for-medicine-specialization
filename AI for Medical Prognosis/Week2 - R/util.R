load_data <- function(threshold) 
{
  # Read in the whole dataset
  X <- read.csv("NHANESI_subset_X.csv", header = T)
  y <- read.csv("NHANESI_subset_y.csv", header = T)
  
  df <- bind_cols(X, y)
  names(df)[ncol(df)] <- "time"
  
  df <- df %>% 
    mutate(death = ifelse(time < 0, 0, 1),
           time = abs(time)) %>% 
    tidyr::drop_na()
  
  mask <- df$time > threshold | df$death == 1
  df <- df[mask,] %>% 
    mutate(y = time < threshold)
  ind <- 1:nrow(df)
  df <- df %>% mutate(ID = ind)
  
  set.seed(10)
  dev <- df %>% slice_sample(n = nrow(df) * 0.8)
  test <- df[!(df$ID %in% dev$ID),]
  
  feature_y <- 'Systolic BP'
  frac      <- 0.7
  
  # add an index column
  drop_rows <- sample_frac(dev, size = frac, replace = FALSE, weight = Age)
  dev[dev$ID %in% drop_rows$ID, "Systolic.BP"] <- NA
  
  X_dev  <- dev[,1:18]
  y_dev  <- dev$y
  X_test <- test[,1:18]
  y_test <- test$y
  
  return(list(X_dev = X_dev, X_test = X_test, y_dev = y_dev, y_test = y_test))
}

# cindex
cindex <- function(y_true, scores) {
  # Input:
  #   y_true (np.array): a 1-D array of true binary outcomes (values of zero or one)
  #       0: patient does not get the disease
  #       1: patient does get the disease
  #   scores (np.array): a 1-D array of corresponding risk scores output by the model
  # 
  #   Output:
  #   c_index (float): (concordant pairs + 0.5*ties) / number of permissible pairs
  
  n <- length(y_true)
  stopifnot(length(scores) == n)
  
  concordant <- 0
  permissible <- 0
  ties <- 0
  
  ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###  
  # use two nested for loops to go through all unique pairs of patients
  for (i in 1 : (n-1)) {
    for (j in (i+1) : n) #choose the range of j so that j>i
    {
      # Check if the pair is permissible (the patient outcomes are different)
      if (y_true[i] != y_true[j]){
        # Count the pair if it's permissible
        permissible = permissible + 1
        
        # For permissible pairs, check if they are concordant or are ties
        # check for ties in the score
        if (scores[i] == scores[j]) {
          # count the tie
          ties = ties + 1
          
          # if it's a tie, we don't need to check patient outcomes, 
          # continue to the top of the for loop.
          next     
        }
        
        # case 1: patient i doesn't get the disease, patient j does
        if (y_true[i] == 0 & y_true[j] == 1){
          # Check if patient i has a lower risk score than patient j
          if (scores[i] < scores[j]){
            # count the concordant pair
            concordant = concordant + 1
            # Otherwise if patient i has a higher risk score, it's not a concordant pair.
            # Already checked for ties earlier
          }
        }
        
        # case 2: patient i gets the disease, patient j does not
        if (y_true[i] == 1 & y_true[j] == 0){
          # Check if patient i has a higher risk score than patient j
          if (scores[i] > scores[j]){
            # count the concordant pair
            concordant = concordant + 1
            # Otherwise if patient i has a lower risk score, it's not a concordant pair.
            # Already checked for ties earlier
          }
        }
      }
    }
  }
  
  # calculate the c-index using the count of permissible pairs, concordant pairs, and tied pairs.
  c_index <- (concordant + 0.5 * ties) / permissible
  ### END CODE HERE ###
  return(c_index)
}