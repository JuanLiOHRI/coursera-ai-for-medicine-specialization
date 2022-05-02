Week 1 Labs 1: Create a Linear Model - using R
================
Juan Li (based on python code on Github)
04/26/2022

## Linear model

We'll practice using stats::lm for linear regression. You will do something similar in this week's assignment (but with a logistic regression model).

[stats::glm](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/glm)

In R, you don't need to load the `stats` package and create an object to run `lm` and `glm`, so I will skip below two steps.

> First, import `LinearRegression`, which is a Python 'class'.
>
> Next, use the class to create an object of type LinearRegression.

Generate some data, note here I have downloaded the raw data files "X\_data.csv" and "y\_data.csv" from Github and saved them in the same folder as the R markdown file.

The features in `X` are:

-   Age: (years)
-   Systolic\_BP: Systolic blood pressure (mmHg)
-   Diastolic\_BP: Diastolic blood pressure (mmHg)
-   Cholesterol: (mg/DL)

The labels in `y` indicate whether the patient has a disease (diabetic retinopathy).

-   y = 1 : patient has retinopathy.
-   y = 0 : patient does not have retinopathy.

``` r
# Read in the whole dataset
Xdata <- read.csv("X_data.csv", header = T)
ydata <- read.csv("y_data.csv", header = T)
```

In R, it is more common to have features `X` and labels `y` in the same dataframe. And package 'dplyr' is often used for data manipulation.

``` r
# load package for manipulation 
library(dplyr)

# Generate a dataframe of features (X) and labels (y) by binding columns.
data_raw <- bind_cols(Xdata, ydata)
```

Random select 100 samples for later excercise.

``` r
data <- sample_n(data_raw, 100)
```

Explore the data by viewing the features and the labels

``` r
# View the features (columns 1:4). 
head(data[-5])
#        Age Systolic_BP Diastolic_BP Cholesterol
# 1 69.78935    90.11174     99.91958    94.44828
# 2 50.29677   114.90316     85.77021   101.63432
# 3 55.14415    99.23833    107.00037   115.29534
# 4 47.16580   101.31800     83.05721    91.51310
# 5 68.13002   111.64319     96.03498    93.17993
# 6 63.61284    80.26997     68.55473    84.08305
```

For the first histogram, I will show a base version and a ggplot2 version.

``` r
# load ggplot2
library(ggplot2)

# Plot a histogram of the Age feature 
# base version
hist(data$Age) 
```

<img src="C2W1_L1_Create-a-linear-model_files/figure-markdown_github/unnamed-chunk-6-1.png" width="//textwidth" />

``` r

# ggplot2 version: it may not be identical to the base version since they have different bin boundries.
ggplot(data, aes(Age)) + 
  geom_histogram(binwidth = 5)
```

<img src="C2W1_L1_Create-a-linear-model_files/figure-markdown_github/unnamed-chunk-6-2.png" width="//textwidth" />

I will only show the base version for below plots.

``` r
# Plot a histogram of the systolic blood pressure feature 
hist(data$Systolic_BP) 
```

<img src="C2W1_L1_Create-a-linear-model_files/figure-markdown_github/unnamed-chunk-7-1.png" width="//textwidth" />

``` r
# Plot a histogram of the diastolic blood pressure feature
hist(data$Diastolic_BP) 
```

<img src="C2W1_L1_Create-a-linear-model_files/figure-markdown_github/unnamed-chunk-8-1.png" width="//textwidth" />

``` r
# Plot a histogram of the cholesterol feature
hist(data$Cholesterol) 
```

<img src="C2W1_L1_Create-a-linear-model_files/figure-markdown_github/unnamed-chunk-9-1.png" width="//textwidth" />

Also take a look at the labels

``` r
# View a few values of the labels
head(data$y) 
# [1] 1 1 1 0 1 0
```

``` r
# Plot a histogram of the labels
hist(data$y) 
```

<img src="C2W1_L1_Create-a-linear-model_files/figure-markdown_github/unnamed-chunk-11-1.png" width="//textwidth" />

Fit the `lm` model using 'data'. To "fit" the model is another way of saying that we are training the model on the data.

``` r
model <- lm(y ~ Age + Systolic_BP + Diastolic_BP + Cholesterol, data = data)

# Since you are using all features in data (except for data$y), 
# you can also write the code as below (not run):

# model <- lm(y ~ ., data = data)
```

-   View the coefficients of the trained model.
-   The coefficients are the 'weights' or *β*s associated with each feature.
-   You'll use the coefficients for making predictions.
    $$\\hat{y} = \\beta\_1 x\_1 + \\beta\_2 x\_2 + ... + \\beta\_N x\_N$$

``` r
# View the coefficients of the model
model$coefficients
#  (Intercept)          Age  Systolic_BP Diastolic_BP  Cholesterol 
# -3.532647711  0.023967846  0.017326633  0.006482392  0.003041516

# View the summary of the model
summary(model)
# 
# Call:
# lm(formula = y ~ Age + Systolic_BP + Diastolic_BP + Cholesterol, 
#     data = data)
# 
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -0.80770 -0.32158  0.03853  0.31020  0.64471 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -3.532648   0.594858  -5.939 4.69e-08 ***
# Age           0.023968   0.004072   5.885 5.95e-08 ***
# Systolic_BP   0.017327   0.004533   3.822 0.000236 ***
# Diastolic_BP  0.006482   0.004937   1.313 0.192331    
# Cholesterol   0.003042   0.004431   0.686 0.494088    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# Residual standard error: 0.3979 on 95 degrees of freedom
# Multiple R-squared:  0.3783,  Adjusted R-squared:  0.3522 
# F-statistic: 14.45 on 4 and 95 DF,  p-value: 2.965e-09
```

In the assignment, you will do something similar, but using a logistic regression, so that the output of the prediction will be bounded between 0 and 1.

## This is the end of this practice section.

Please continue on with the lecture videos!
