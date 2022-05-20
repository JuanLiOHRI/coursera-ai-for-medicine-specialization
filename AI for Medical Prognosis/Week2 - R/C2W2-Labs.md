Week 2 lecture notebook - using R
================
Juan Li (based on python code on Github)
05/11/2022

-   <a href="#missing-values" id="toc-missing-values">Missing values</a>
    -   <a href="#check-if-each-value-is-missing"
        id="toc-check-if-each-value-is-missing">Check if each value is
        missing</a>
    -   <a href="#check-if-any-values-in-a-row-are-true"
        id="toc-check-if-any-values-in-a-row-are-true">Check if any values in a
        row are true</a>
    -   <a href="#sum-booleans" id="toc-sum-booleans">Sum booleans</a>
-   <a href="#decision-tree-classifier"
    id="toc-decision-tree-classifier">Decision Tree Classifier</a>
    -   <a href="#set-tree-parameters" id="toc-set-tree-parameters">Set tree
        parameters</a>
    -   <a href="#set-parameters-using-a-dictionary"
        id="toc-set-parameters-using-a-dictionary">Set parameters using a
        dictionary</a>
-   <a href="#apply-a-mask" id="toc-apply-a-mask">Apply a mask</a>
    -   <a href="#combining-comparison-operators"
        id="toc-combining-comparison-operators">Combining comparison
        operators</a>
-   <a href="#imputation" id="toc-imputation">Imputation</a>
    -   <a href="#mean-imputation" id="toc-mean-imputation">Mean imputation</a>
    -   <a href="#regression-imputation"
        id="toc-regression-imputation">Regression Imputation</a>

# Missing values

``` r
df <- data.frame(feature_1 = c(0.1, NA, NA, 0.4),
                 feature_2 = c(1.1, 2.2, NA, NA))
df
#   feature_1 feature_2
# 1       0.1       1.1
# 2        NA       2.2
# 3        NA        NA
# 4       0.4        NA
```

## Check if each value is missing

``` r
is.na(df)
#      feature_1 feature_2
# [1,]     FALSE     FALSE
# [2,]      TRUE     FALSE
# [3,]      TRUE      TRUE
# [4,]     FALSE      TRUE
```

## Check if any values in a row are true

``` r
df_booleans <- data.frame(col_1 = c(TRUE, TRUE, FALSE),
                          col_2 = c(TRUE, FALSE, FALSE))
df_booleans
#   col_1 col_2
# 1  TRUE  TRUE
# 2  TRUE FALSE
# 3 FALSE FALSE
```

-   $\color{green}{\text{In Python:}}$ If we use
    `pandas.DataFrame.any()`, it checks if at least one value in a
    column is `True`, and if so, returns `True`.
-   If all rows are `False`, then it returns `False` for that column
-   $\color{green}{\text{In R:}}$ we will use `lapply` to run `any()` on
    each column.

``` r
lapply(df_booleans, any)
# $col_1
# [1] TRUE
# 
# $col_2
# [1] TRUE
```

-   $\color{green}{\text{In Python:}}$ Setting the axis to `1` checks if
    any item in a row is `True`, and if so, returns true.
-   Similarily only when all values in a row are `False`, the function
    returns `False`.
-   $\color{green}{\text{In R:}}$ we will use `apply` to run `any()` on
    each row.

``` r
apply(df_booleans, 1, any)
# [1]  TRUE  TRUE FALSE
```

## Sum booleans

``` r
series_booleans <- c(TRUE, TRUE, FALSE)
series_booleans
# [1]  TRUE  TRUE FALSE
```

-   When applying `sum` to a series (or list) of booleans, the `sum`
    function treats `True` as `1` and `False` as zero.

``` r
sum(series_booleans)
# [1] 2
```

You will make use of these functions in this week’s assignment!

**This is the end of this practice section.**

Please continue on with the lecture videos!

# Decision Tree Classifier

``` r
df <- data.frame(X = c(0, 1, 2, 3),
                 y = c(0, 0, 1, 1))
```

``` r
df$X 
# [1] 0 1 2 3
```

``` r
df$y 
# [1] 0 0 1 1
```

``` r
library(rpart)
dt <- rpart(y ~ X, data=df, method = 'class')
```

## Set tree parameters

``` r
dt <- rpart(y ~ X, data=df, method = 'class', maxdepth=10, minsplit=2)
```

## Set parameters using a dictionary

Not applied to R, so skip.

**This is the end of this practice section.**

Please continue on with the lecture videos!

# Apply a mask

Use a ‘mask’ to filter data of a dataframe

``` r
df <- data.frame(feature_1 = c(0:4))
df
#   feature_1
# 1         0
# 2         1
# 3         2
# 4         3
# 5         4
```

``` r
mask <- df$feature_1 >= 3
mask
# [1] FALSE FALSE FALSE  TRUE  TRUE
```

``` r
df[mask,]
# [1] 3 4

# Or the dplyr version
require(dplyr)
df %>% filter(mask)
#   feature_1
# 1         3
# 2         4
```

## Combining comparison operators

You’ll want to be careful when combining more than one comparison
operator, to avoid errors.

-   $\color{green}{\text{In Python:}}$ Using the `and` operator on a
    series will result in a ValueError
-   $\color{green}{\text{In R:}}$ There is no `and` operator.

``` r
df$feature_1 >= 2
# [1] FALSE FALSE  TRUE  TRUE  TRUE
```

``` r
df$feature_1 <= 3
# [1]  TRUE  TRUE  TRUE  TRUE FALSE
```

``` r
df$feature_1 >= 2 & df$feature_1 <= 3
# [1] FALSE FALSE  TRUE  TRUE FALSE
```

**This is the end of this practice section.**

Please continue on with the lecture videos!

# Imputation

‘mice’ is the package that is often used for data imputation in R.

``` r
df <- data.frame(feature_1 = c(0:10),
                 feature_2 = c(0,NA,20,30,40,50,60,70,80,NA,100))
df
#    feature_1 feature_2
# 1          0         0
# 2          1        NA
# 3          2        20
# 4          3        30
# 5          4        40
# 6          5        50
# 7          6        60
# 8          7        70
# 9          8        80
# 10         9        NA
# 11        10       100
```

## Mean imputation

``` r
nparray_imputed_mean <- df
nparray_imputed_mean$feature_2[is.na(nparray_imputed_mean$feature_2)] <- mean(nparray_imputed_mean$feature_2, na.rm = TRUE)
nparray_imputed_mean
#    feature_1 feature_2
# 1          0         0
# 2          1        50
# 3          2        20
# 4          3        30
# 5          4        40
# 6          5        50
# 7          6        60
# 8          7        70
# 9          8        80
# 10         9        50
# 11        10       100
```

Notice how the missing values are replaced with 50 in both cases.

## Regression Imputation

``` r
nparray_imputed_reg <- df
model <- lm(feature_2~feature_1, data = nparray_imputed_reg)
imp   <- predict(model, newdata=nparray_imputed_reg %>% filter(is.na(feature_2)) %>% select(feature_1))
nparray_imputed_reg$feature_2[is.na(nparray_imputed_reg$feature_2)] <- imp
nparray_imputed_reg
#    feature_1 feature_2
# 1          0         0
# 2          1        10
# 3          2        20
# 4          3        30
# 5          4        40
# 6          5        50
# 7          6        60
# 8          7        70
# 9          8        80
# 10         9        90
# 11        10       100
```

Notice how the filled in values are replaced with `10` and `90` when
using regression imputation. The imputation assumed a linear
relationship between feature 1 and feature 2.

You can also use `impute_lm` from the `simputation` package.

``` r
imp <- simputation::impute_lm(df, feature_2~feature_1)
imp
#    feature_1 feature_2
# 1          0         0
# 2          1        10
# 3          2        20
# 4          3        30
# 5          4        40
# 6          5        50
# 7          6        60
# 8          7        70
# 9          8        80
# 10         9        90
# 11        10       100
```

**This is the end of this practice section.**

Please continue on with the lecture videos!
