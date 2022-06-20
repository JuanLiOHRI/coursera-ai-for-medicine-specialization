Week 3 Survival Estimates that Vary with Time - using R
================
Juan Li (based on Courera matetials)
06/20/2022

-   <a href="#1-import-packages" id="toc-1-import-packages">1. Import
    Packages</a>
-   <a href="#2-load-the-dataset" id="toc-2-load-the-dataset">2. Load the
    Dataset</a>
-   <a href="#3-censored-data" id="toc-3-censored-data">3. Censored Data</a>
    -   <a href="#exercise-1---frac_censored"
        id="toc-exercise-1---frac_censored">Exercise 1 - frac_censored</a>
-   <a href="#4-survival-estimates" id="toc-4-survival-estimates">4.
    Survival Estimates</a>
    -   <a href="#exercise-2---naive_estimator"
        id="toc-exercise-2---naive_estimator">Exercise 2 - naive_estimator</a>
    -   <a href="#exercise-3---homemadekm"
        id="toc-exercise-3---homemadekm">Exercise 3 - HomemadeKM</a>
-   <a href="#5-subgroup-analysis" id="toc-5-subgroup-analysis">5. Subgroup
    Analysis</a>
    -   <a href="#51-bonus-log-rank-test" id="toc-51-bonus-log-rank-test">5.1
        Bonus: Log-Rank Test</a>
-   <a href="#congratulations" id="toc-congratulations">Congratulations!</a>

Welcome to the third assignment of Course 2. In this assignment, we’ll
use Python to build some of the statistical models we learned this past
week to analyze surivival estimates for a dataset of lymphoma patients.
We’ll also evaluate these models and interpret their outputs. Along the
way, you will be learning about the following:

-   Censored Data

-   Kaplan-Meier Estimates

-   Subgroup Analysis

# 1. Import Packages

We’ll first import all the packages that we need for this assignment.

-   lifelines is an open-source library for data analysis. **R:
    survival**
-   numpy is the fundamental package for scientific computing in python.
    **R: dplyr**
-   pandas is what we’ll use to manipulate our data. **R: dplyr**
-   matplotlib is a plotting library. **R: ggplot2**

``` r
library(survival)
library(survminer)
library(dplyr)
library(ggplot2)

source('../util.R', echo=TRUE)
# 
# > load_data <- function() {
# +     df <- read.csv("../lymphoma.csv", header = T)
# +     names(df)[3] <- "Event"
# +     return(df)
# + }
```

# 2. Load the Dataset

Run the next cell to load the lymphoma data set.

``` r
data <- load_data()
```

As always, you first look over your data.

``` r
dim(data)
# [1] 80  3
head(data)
#   Stage.group Time Event
# 1           1    6     1
# 2           1   19     1
# 3           1   32     1
# 4           1   42     1
# 5           1   42     1
# 6           1   43     0
```

The column `Time` states how long the patient lived before they died or
were censored.

The column `Event` says whether a death was observed or not. `Event` is
1 if the event is observed (i.e. the patient died) and 0 if data was
censored.

Censorship here means that the observation has ended without any
observed event. For example, let a patient be in a hospital for 100 days
at most. If a patient dies after only 44 days, their event will be
recorded as `Time` = 44 and `Event` = 1. If a patient walks out after
100 days and dies 3 days later (103 days total), this event is not
observed in our process and the corresponding row has `Time` = 100 and
`Event` = 0. If a patient survives for 25 years after being admitted,
their data for are still `Time` = 100 and `Event` = 0.

# 3. Censored Data

We can plot a histogram of the survival times to see in general how long
cases survived before censorship or events.

``` r
ggplot(data, aes(Time)) +
  geom_histogram(bins = 10)+
  xlab("Observation time before death or censorship (days)")+
  ylab("Frequency (number of patients)")
```

<img src="C2W3_A1_Survival-Estimates-that-Vary-with-Time_files/figure-gfm/unnamed-chunk-5-1.png" width="//textwidth" />

## Exercise 1 - frac_censored

In the next cell, write a function to compute the fraction ( ∈\[0,1\] )
of observations which were censored.

**Hints**

-   Summing up the `Event` column will give you the number of
    observations where censorship has NOT occurred.

``` r
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
frac_censored <- function(df) {
  # Return percent of observations which were censored.
  #   
  #   Args:
  #       df (dataframe): dataframe which contains column 'Event' which is 
  #                       1 if an event occurred (death)
  #                       0 if the event did not occur (censored)
  #   Returns:
  #       frac_censored (float): fraction of cases which were censored. 
  
  result <- 0.0
  
  ### START CODE HERE ###
    
  censored_count <- sum(df$Event == 0)
  result         <- censored_count / nrow(df)
  
  ### END CODE HERE ###
  
  return(result)
}
```

``` r
### do not edit this code cell
frac <- frac_censored(data)
print(paste("Observations which were censored:", frac))
# [1] "Observations which were censored: 0.325"
```

**Expected Output:**

Observations which were censored: 0.325 All tests passed.

Run the next cell to see the distributions of survival times for
censored and uncensored examples.

``` r
df_censored   <- data %>% filter(Event == 0)
df_uncensored <- data %>% filter(Event == 1)

ggplot(df_censored, aes(Time))+
  geom_histogram(bins = 10)+
  labs(title = "Censored",
       x = "Time (days)",
       y = "Frequency")
```

<img src="C2W3_A1_Survival-Estimates-that-Vary-with-Time_files/figure-gfm/unnamed-chunk-8-1.png" width="//textwidth" />

``` r

ggplot(df_uncensored, aes(Time))+
  geom_histogram(bins = 10)+
  labs(title = "Uncensored",
       x = "Time (days)",
       y = "Frequency")
```

<img src="C2W3_A1_Survival-Estimates-that-Vary-with-Time_files/figure-gfm/unnamed-chunk-8-2.png" width="//textwidth" />

# 4. Survival Estimates

We’ll now try to estimate the survival function:

![S(t) = P(T \> t)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;S%28t%29%20%3D%20P%28T%20%3E%20t%29 "S(t) = P(T > t)")

To illustrate the strengths of Kaplan Meier, we’ll start with a naive
estimator of the above survival function. To estimate this quantity,
we’ll divide the number of people who we know lived past time 𝑡 by the
number of people who were not censored before
![t](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t "t").

Formally, let
![i = 1,...,n](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i%20%3D%201%2C...%2Cn "i = 1,...,n")
be the cases, and let
![T_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;T_i "T_i")
be the time when
![i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i "i")
was censored or an event happened. Let
![e_i = 1](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;e_i%20%3D%201 "e_i = 1")
if an event was observed for
![i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;i "i")
and 0 otherwise. Then let
![X_i = \\{i: T_i \> t\\}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;X_i%20%3D%20%5C%7Bi%3A%20T_i%20%3E%20t%5C%7D "X_i = \{i: T_i > t\}"),
and let
![M_t = \\{i: e_i = 1\\ or\\ T_i \> t\\}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;M_t%20%3D%20%5C%7Bi%3A%20e_i%20%3D%201%5C%20or%5C%20T_i%20%3E%20t%5C%7D "M_t = \{i: e_i = 1\ or\ T_i > t\}").
The estimator you will compute will be:

![\hat{S}(t) = \frac{\|X_t\|}{\|M_t\|}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Chat%7BS%7D%28t%29%20%3D%20%5Cfrac%7B%7CX_t%7C%7D%7B%7CM_t%7C%7D "\hat{S}(t) = \frac{|X_t|}{|M_t|}")

## Exercise 2 - naive_estimator

Write a function to compute this estimate for arbitrary *t* in the cell
below.

``` r
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
naive_estimator <- function(t, df) {
  # Return naive estimate for S(t), the probability
  #   of surviving past time t. Given by number
  #   of cases who survived past time t divided by the
  #   number of cases who weren't censored before time t.
  #   
  #   Args:
  #       t (int): query time
  #       df (dataframe): survival data. Has a Time column,
  #                       which says how long until that case
  #                       experienced an event or was censored,
  #                       and an Event column, which is 1 if an event
  #                       was observed and 0 otherwise.
  #   Returns:
  #       S_t (float): estimator for survival function evaluated at t.
  
  S_t <- 0.0
  
  ### START CODE HERE ###
    
  X <- sum(df$Time > t)
  
  M <- sum((df$Time > t) | (df$Event == 0))
  
  S_t = X / M
  
  ### END CODE HERE ###
  
  return(S_t)
}
```

``` r
### do not edit this code cell
df1  <- data.frame(Time = c(5, 10, 15),
                  Event = c(0, 1, 0))

print(paste("Test Case 1: S(3)", naive_estimator(3, df1)))
# [1] "Test Case 1: S(3) 1"
print(paste("Test Case 2: S(12)", naive_estimator(12, df1)))
# [1] "Test Case 2: S(12) 0.5"
print(paste("Test Case 3: S(20)", naive_estimator(20, df1)))
# [1] "Test Case 3: S(20) 0"

df2  <- data.frame(Time = c(5, 5, 10),
                  Event = c(0, 1, 0))

print(paste("Test Case 4: S(5)", naive_estimator(5, df2)))
# [1] "Test Case 4: S(5) 0.5"
```

**Expected Output:**

Test Case 1: S(3) Output: 1.0

Test Case 2: S(12) Output: 0.5

Test Case 3: S(20) Output: 0.0

Test case 4: S(5) Output: 0.5

All tests passed.

In the next cell, we will plot the naive estimator using the real data
up to the maximum time in the dataset.

``` r
max_time <- max(data$Time, na.rm = TRUE)
x <- 0:(max_time+1)
y <- rep(0, length(x))

for (i in 1:length(x))
{
  y[i] <- naive_estimator(x[i], data)
}

ggplot(data.frame(x = x, y = y), aes(x, y))+
  geom_path()+
  labs(title = "Naive Survival Estimate",
       x = "Time",
       y = "Estimated cumulative survival rate")
```

<img src="C2W3_A1_Survival-Estimates-that-Vary-with-Time_files/figure-gfm/unnamed-chunk-11-1.png" width="//textwidth" />

## Exercise 3 - HomemadeKM

Next let’s compare this with the Kaplan Meier estimate. In the cell
below, write a function that computes the Kaplan Meier estimate of
![S(t)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;S%28t%29 "S(t)")
at every distinct time in the dataset.

Recall the Kaplan-Meier estimate:

![S(t) = \prod\_{t_i \<= t}(1-\frac{d_i}{n_i})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;S%28t%29%20%3D%20%5Cprod_%7Bt_i%20%3C%3D%20t%7D%281-%5Cfrac%7Bd_i%7D%7Bn_i%7D%29 "S(t) = \prod_{t_i <= t}(1-\frac{d_i}{n_i})")

where
![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")
are the events observed in the dataset,
![d_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;d_i "d_i")
is the number of deaths at time
![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")
and
![n_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n_i "n_i")
is the number of people who we know have survived up to time
![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i").

``` r
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
HomemadeKM <- function(df) {
  #   Return KM estimate evaluated at every distinct
  #   time (event or censored) recorded in the dataset.
  #   Event times and probabilities should begin with
  #   time 0 and probability 1.
  #   
  #   Example:
  #   
  #   input: 
  #   
  #        Time  Censor
  #   0     5       0
  #   1    10       1
  #   2    15       0
  #   
  #   correct output: 
  #   
  #   event_times: [0, 5, 10, 15]
  #   S: [1.0, 1.0, 0.5, 0.5]
  #   
  #   Args:
  #       df (dataframe): dataframe which has columns for Time
  #                         and Event, defined as usual.
  #                         
  #   Returns:
  #       event_times (list of ints): array of unique event times
  #                                     (begins with 0).
  #       S (list of floats): array of survival probabilites, so that
  #                           S[i] = P(T > event_times[i]). This 
  #                           begins with 1.0 (since no one dies at time
  #                           0).
  
  # individuals are considered to have survival probability 1
  # at time 0
  event_times = c(0)
  p = 1.0
  S = c(p)
  
  ### START CODE HERE ###
    
  # get collection of unique observed event times
  observed_event_times <- df$Time
  
  # sort event times
  observed_event_times <- sort(observed_event_times)
  
  # iterate through event times
  for (i in 1:length(observed_event_times))
  {
    t <- observed_event_times[i]
    # compute n_t, number of people who survive to time t
    n_t <- nrow(df %>% filter(Time >= t))
    
    # compute d_t, number of people who die at time t
    d_t = nrow(df %>% filter(Event == 1 & Time == t))
    
    # update P
    p = p * (1 - d_t / n_t)
    
    # update S and event_times
    event_times <- c(event_times, t)
    S           <- c(S, p)
  }
  
  ### END CODE HERE ###
  
  return(list(event_times = event_times, S = S))
}
```

``` r
### do not edit this code cell
df1  <- data.frame(Time = c(5, 10, 15),
                  Event = c(0, 1, 0))
res1 <- HomemadeKM(df1)
print(paste("Test Case 1 Event times: ", list(res1$event_times), ", Survival Probabilities: ", list(res1$S), sep = ""))
# [1] "Test Case 1 Event times: c(0, 5, 10, 15), Survival Probabilities: c(1, 1, 0.5, 0.5)"

df2  <- data.frame(Time = c(2, 15, 12, 10, 20),
                  Event = c(0, 0, 1, 1, 1))
res2 <- HomemadeKM(df2)
print(paste("Test Case 2 Event times: ", list(res2$event_times), ", Survival Probabilities: ", list(res2$S), sep = ""))
# [1] "Test Case 2 Event times: c(0, 2, 10, 12, 15, 20), Survival Probabilities: c(1, 1, 0.75, 0.5, 0.5, 0)"
```

**Expected Output:**

Test Case 1 Event times: \[0, 5, 10, 15\], Survival Probabilities:
\[1.0, 1.0, 0.5, 0.5\] Test Case 2 Event times: \[0, 2, 10, 12, 15,
20\], Survival Probabilities: \[1.0, 1.0, 0.75, 0.5, 0.5, 0.0\]

All tests passed.

Now let’s plot the two against each other on the data to see the
difference.

``` r
max_time <- max(data$Time, na.rm = TRUE)
x <- 0:(max_time+1)
y <- rep(0, length(x))

for (i in 1:length(x))
{
  y[i] <- naive_estimator(x[i], data)
}

df_plt <- data.frame(x = x, 
                     y = y,
                     method = "Naive")

res <- HomemadeKM(data)
df_plt2 <- data.frame(x = res$event_times,
                      y = res$S,
                      method = "Kaplan-Meier")

df_plt <- bind_rows(df_plt, df_plt2)

ggplot(df_plt, aes(x, y, color = method))+
  geom_path()+
  labs(x = "Time",
       y = "Survival probability estimate")
```

<img src="C2W3_A1_Survival-Estimates-that-Vary-with-Time_files/figure-gfm/unnamed-chunk-14-1.png" width="//textwidth" />

**Question**

What differences do you observe between the naive estimator and
Kaplan-Meier estimator? Do any of our earlier explorations of the
dataset help to explain these differences?

# 5. Subgroup Analysis

We see that along with Time and Censor, we have a column called
`Stage_group`. - A value of 1 in this column denotes a patient with
stage III cancer - A value of 2 denotes stage IV.

We want to compare the survival functions of these two groups.

This time we’ll use the `KaplanMeierFitter` class from `lifelines`. Run
the next cell to fit and plot the Kaplan Meier curves for each group.
**R: survival::survfit**

``` r
fit <- survfit(Surv(Time, Event) ~ Stage.group, data = data)
ggsurvplot(fit, data = data,
           pval = TRUE, conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           surv.median.line = "hv" # Specify median survival
           )
```

<img src="C2W3_A1_Survival-Estimates-that-Vary-with-Time_files/figure-gfm/unnamed-chunk-15-1.png" width="//textwidth" />

Let’s compare the survival functions at 90, 180, 270, and 360 days

``` r
survivals <- data.frame(time = c(90, 180, 270, 360))
cfit <- rms::cph(Surv(Time, Event) ~ Stage.group, data = data, surv=TRUE)
rms::survest(cfit, newdata = expand.grid('Stage.group'=c(1,2)), times=survivals$time)$surv
# Warning in survest.cph(cfit, newdata = expand.grid(Stage.group = c(1, 2)), : S.E. and confidence intervals are approximate except at predictor means.
# Use cph(...,x=TRUE,y=TRUE) (and don't use linear.predictors=) for better estimates.
#          90       180       270 360
# 1 0.7273430 0.6145282 0.5365432  NA
# 2 0.4349544 0.2799175 0.1962942  NA
```

``` r
survivals
#   time
# 1   90
# 2  180
# 3  270
# 4  360
```

This makes clear the difference in survival between the Stage III and IV
cancer groups in the dataset.

## 5.1 Bonus: Log-Rank Test

To say whether there is a statistical difference between the survival
curves we can run the log-rank test. This test tells us the probability
that we could observe this data if the two curves were the same. The
derivation of the log-rank test is somewhat complicated, but luckily
lifelines has a simple function to compute it.

Run the next cell to compute a p-value using
`lifelines.statistics.logrank_test` **R::survival::surv_diff**.

``` r
surv_diff <- survdiff(Surv(Time, Event) ~ Stage.group, data = data)
surv_diff
# Call:
# survdiff(formula = Surv(Time, Event) ~ Stage.group, data = data)
# 
#                N Observed Expected (O-E)^2/E (O-E)^2/V
# Stage.group=1 19        8     16.7      4.52      6.71
# Stage.group=2 61       46     37.3      2.02      6.71
# 
#  Chisq= 6.7  on 1 degrees of freedom, p= 0.01
```

If everything is correct, you should see a p value of less than 0.05,
which indicates that the difference in the curves is indeed
statistically significant.

# Congratulations!

You’ve completed the third assignment of Course 2. You’ve learned about
the Kaplan Meier estimator, a fundamental non-parametric estimator in
survival analysis. Next week we’ll learn how to take into account
patient covariates in our survival estimates!
