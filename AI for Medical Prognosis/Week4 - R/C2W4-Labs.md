Week 4 lecture notebook - using R
================
Juan Li (based on Coursera materials)
06/21/2022

-   <a href="#one-hot-encode-categorical-variables"
    id="toc-one-hot-encode-categorical-variables">One-hot encode categorical
    variables</a>
    -   <a href="#import-packages" id="toc-import-packages">Import Packages</a>
    -   <a href="#which-features-are-categorical"
        id="toc-which-features-are-categorical">Which Features are
        Categorical?</a>
    -   <a href="#which-categorical-variables-to-one-hot-encode"
        id="toc-which-categorical-variables-to-one-hot-encode">Which Categorical
        Variables to One-Hot Encode?</a>
    -   <a href="#multi-collinearity-of-one-hot-encoded-features"
        id="toc-multi-collinearity-of-one-hot-encoded-features">Multi-collinearity
        of One-Hot Encoded Features</a>
    -   <a href="#make-the-numbers-decimals"
        id="toc-make-the-numbers-decimals">Make the Numbers Decimals</a>
    -   <a href="#hazard-function" id="toc-hazard-function">Hazard Function</a>
    -   <a href="#import-packages-1" id="toc-import-packages-1">Import
        Packages</a>
-   <a href="#permissible-pairs-with-censoring-and-time"
    id="toc-permissible-pairs-with-censoring-and-time">Permissible Pairs
    with Censoring and Time</a>
    -   <a href="#import-package" id="toc-import-package">Import Package</a>
    -   <a href="#when-at-least-one-patient-is-not-censored"
        id="toc-when-at-least-one-patient-is-not-censored">When At Least One
        Patient is Not Censored</a>
-   <a href="#if-neither-patient-was-censored"
    id="toc-if-neither-patient-was-censored">If Neither Patient was
    Censored:</a>
    -   <a href="#when-one-patient-is-censored"
        id="toc-when-one-patient-is-censored">When One Patient is Censored:</a>
-   <a href="#this-is-the-end-of-labs-of-week-4"
    id="toc-this-is-the-end-of-labs-of-week-4">This is the end of Labs of
    week 4.</a>

# One-hot encode categorical variables

Welcome to the first lab of the week!

## Import Packages

``` r
library(dplyr)
library(caret)
```

## Which Features are Categorical?

``` r
df <- data.frame(ascites = c(0,1,0,1),
                 edema = c(0.5,0,1,0.5),
                 stage = c(3,4,3,4),
                 cholesterol = c(200.5,180.2,190.5,210.3))
df
#   ascites edema stage cholesterol
# 1       0   0.5     3       200.5
# 2       1   0.0     4       180.2
# 3       0   1.0     3       190.5
# 4       1   0.5     4       210.3
```

In this small sample dataset, ‘ascites’, ‘edema’, and ‘stage’ are
categorical variables

-   ascites: value is either 0 or 1

-   edema: value is either 0, 0.5 or 1

-   stage: is either 3 or 4

‘cholesterol’ is a continuous variable, since it can be any decimal
value greater than zero.

## Which Categorical Variables to One-Hot Encode?

Of the categorical variables, which one should be one-hot encoded
(turned into dummy variables)?

-   ascites: is already 0 or 1, so there is not a need to one-hot encode
    it.

    -   We could one-hot encode ascites, but it is not necessary when
        there are just two possible values that are 0 or 1.
    -   When values are 0 or 1, 1 means a disease is present, and 0
        means normal (no disease).

-   edema: Edema is swelling in any part of the body. This data set’s
    ‘edema’ feature has 3 categories, so we will want to one-hot encode
    it so that there is one feature column for each of the three
    possible values.

    -   0: No edema
    -   0.5: Patient has edema, but did not receive diuretic therapy
        (which is used to treat edema)
    -   1: Patient has edeam, despite also receiving diuretic therapy
        (so the condition may be more severe).

-   stage: has values of 3 and 4. We will want to one-hot encode these
    because they are not values of 0 or 1.

    -   the “stage” of cancer is either 0, 1,2,3 or 4.
    -   Stage 0 means there is no cancer.
    -   Stage 1 is cancer that is limited to a small area of the body,
        also known as “early stage cancer”
    -   Stage 2 is cancer that has spread to nearby tissues
    -   stage 3 is cancer that has spread to nearby tissues, but more so
        than stage 2
    -   stage 4 is cancer that has spread to distant parts of the body,
        also known as “metastatic cancer”.
    -   We could convert stage 3 to 0 and stage 4 to 1 for the sake of
        training a model. This would may be confusing for anyone
        reviewing our code and data. We will one-hot encode the ‘stage’.
        -You’ll actually see that we end up with 0 representing stage 3
        and 1 representing stage 4 (see the next section).

## Multi-collinearity of One-Hot Encoded Features

Let’s see what happens when we one-hot encode the ‘stage’ feature.

We’ll use pandas.get_dummies. **R: caret::dummyVars**

``` r
df <- df %>% mutate(stage = factor(stage))
dmy <- dummyVars(" ~ .", data=df)
df_stage <- data.frame(predict(dmy, newdata = df))
df_stage %>% select(stage.3, stage.4)
#   stage.3 stage.4
# 1       1       0
# 2       0       1
# 3       1       0
# 4       0       1
```

What do you notice about the ‘stage_3’ and ‘stage_4’ features?

Given that stage 3 and stage 4 are the only possible values for stage,
If you know that patient 0 (row 0) has stage_3 set to 1, what can you
say about that same patient’s value for the stage_4 feature?

-   When stage_3 is 1, then stage_4 must be 0

-   When stage_3 is 0, then stage_4 must be 1

This means that one of the feature columns is actually redundant. We
should drop one of these features to avoid multicollinearity (where one
feature can predict another feature).

``` r
df_stage
#   ascites edema stage.3 stage.4 cholesterol
# 1       0   0.5       1       0       200.5
# 2       1   0.0       0       1       180.2
# 3       0   1.0       1       0       190.5
# 4       1   0.5       0       1       210.3
```

``` r
df_stage_drop_first <- select(df_stage, -stage.3)
df_stage_drop_first
#   ascites edema stage.4 cholesterol
# 1       0   0.5       0       200.5
# 2       1   0.0       1       180.2
# 3       0   1.0       0       190.5
# 4       1   0.5       1       210.3
```

Note, there’s actually a parameter of pandas.get_dummies() that lets you
drop the first one-hot encoded column. You’ll practice doing this in
this week’s assignment!

## Make the Numbers Decimals

We can cast the one-hot encoded values as floats by setting the data
type to numpy.float64.

-   This is helpful if we are feeding data into a model, where the model
    expects a certain data type (such as a 64-bit float, 32-bit float
    etc.)

``` r
df_stage <- data.frame(predict(dmy, newdata = df))
df_stage$stage.4
# [1] 0 1 0 1
```

``` r
df_stage <- df_stage %>% mutate(stage.4 = as.numeric(stage.4))
df_stage$stage.4
# [1] 0 1 0 1
```

## Hazard Function

In this week’s lab you’ll learn about implementing `Hazard Function`.

## Import Packages

``` r
library(dplyr)
```

Let’s say we fit the hazard function

![\lambda(t,x) = \lambda_0(t)e^{\theta^TX_i}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Clambda%28t%2Cx%29%20%3D%20%5Clambda_0%28t%29e%5E%7B%5Ctheta%5ETX_i%7D "\lambda(t,x) = \lambda_0(t)e^{\theta^TX_i}")

So that we have the coefficients
![\theta](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctheta "\theta")
for the features in
![X_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;X_i "X_i")

If you have a new patient, let’s predict their hazard
![\lambda(t,x)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Clambda%28t%2Cx%29 "\lambda(t,x)")

``` r
lambda_0 <- 1
coef     <- c(0.5,2)
coef
# [1] 0.5 2.0
```

``` r
X <- data.frame(age = c(20,30,40),
                cholesterol = c(180,220,170))
X
#   age cholesterol
# 1  20         180
# 2  30         220
# 3  40         170
```

-   First, let’s multiply the coefficients to the features.

-   Check the shapes of the coefficients and the features to decide
    which one to transpose

``` r
length(coef)
# [1] 2
```

``` r
dim(X)
# [1] 3 2
```

It looks like the coefficient is a 1D array, so transposing it won’t do
anything.

-   We can transpose the X so that we’re multiplying a (2,) array by a
    (2,3) dataframe.

So the formula looks more like this (transpose
![X_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;X_i "X_i")
instead of
![\theta](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctheta "\theta")

![\lambda(t,x) = \lambda_0(t)e^{\theta^TX_i}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Clambda%28t%2Cx%29%20%3D%20%5Clambda_0%28t%29e%5E%7B%5Ctheta%5ETX_i%7D "\lambda(t,x) = \lambda_0(t)e^{\theta^TX_i}")

 - Let’s multiply
![\theta X_i^T](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;%5Ctheta%20X_i%5ET "\theta X_i^T")

``` r
coef %*% t(X)
#      [,1] [,2] [,3]
# [1,]  370  455  360
```

Calculate the hazard for the three patients (there are 3 rows in X)

``` r
lambdas <- lambda_0 * exp(coef %*% t(X))
patients_df <- X %>% mutate(hazards = t(lambdas))
patients_df
#   age cholesterol       hazards
# 1  20         180 4.886054e+160
# 2  30         220 4.017809e+197
# 3  40         170 2.218265e+156
```

# Permissible Pairs with Censoring and Time

Welcome to the last practice lab of this week and of this course!

## Import Package

``` r
library(dplyr)
```

``` r
df <- data.frame(time = c(2,4,2,4,2,4,2,4),
                 event = c(1,1,1,1,0,1,1,0),
                 risk_score = c(20,40,40,20,20,40,40,20))
df
#   time event risk_score
# 1    2     1         20
# 2    4     1         40
# 3    2     1         40
# 4    4     1         20
# 5    2     0         20
# 6    4     1         40
# 7    2     1         40
# 8    4     0         20
```

We made this data sample so that you can compare pairs of patients
visually.

## When At Least One Patient is Not Censored

-   A pair may be permissible if at least one patient is not censored.

-   If both pairs of patients are censored, then they are definitely not
    a permissible pair.

``` r
df[1:2,]
#   time event risk_score
# 1    2     1         20
# 2    4     1         40
```

``` r
if (df$event[1] == 1 | df$event[2] == 1) {
  print("May be a permissible pair: 1 and 2")
} else 
{
  print("Definitely not permissible pair: 1 and 2")
}
# [1] "May be a permissible pair: 1 and 2"
```

``` r
df[c(5,8),]
#   time event risk_score
# 5    2     0         20
# 8    4     0         20
```

``` r
if (df$event[5] == 1 | df$event[8] == 1) {
  print("May be a permissible pair: 5 and 8")
} else 
{
  print("Definitely not permissible pair: 5 and 8")
}
# [1] "Definitely not permissible pair: 5 and 8"
```

# If Neither Patient was Censored:

-   If both patients had an event (neither one was censored). This is
    definitely a permissible pair.

``` r
df[1:2,]
#   time event risk_score
# 1    2     1         20
# 2    4     1         40
```

``` r
if (df$event[1] == 1 & df$event[2] == 1) {
  print("Definitely a permissible pair: 1 and 2")
} else 
{
  print("May be a permissible pair: 1 and 2")
}
# [1] "Definitely a permissible pair: 1 and 2"
```

## When One Patient is Censored:

-   If we know that one patient was censored and one had an event, then
    we can check if censored patient’s time is at least as great as the
    uncensored patient’s time. If so, it’s a permissible pair as well

``` r
df[7:8,]
#   time event risk_score
# 7    2     1         40
# 8    4     0         20
```

``` r
if (df$time[8]  >= df$time[7]) {
  print("Permissible pair: Censored patient 8 lasted at least as long as uncensored patient 7")
} else 
{
  print("Not a permisible pair")
}
# [1] "Permissible pair: Censored patient 8 lasted at least as long as uncensored patient 7"
```

``` r
df[5:6,]
#   time event risk_score
# 5    2     0         20
# 6    4     1         40
```

``` r
if (df$time[5]  >= df$time[6]) {
  print("Permissible pair")
} else 
{
  print("Not a permisible pair: patient 5 was censored before patient 6 had their event")
}
# [1] "Not a permisible pair: patient 5 was censored before patient 6 had their event"
```

# This is the end of Labs of week 4.
