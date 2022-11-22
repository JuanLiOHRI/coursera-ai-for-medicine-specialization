Week 3 lecture notebook - using R
================
Juan Li (based on Coursera materials)
06/20/2022

-   <a href="#count-patients" id="toc-count-patients">Count Patients</a>
    -   <a href="#import-packages" id="toc-import-packages">Import Packages</a>
    -   <a href="#count-number-of-censored-patients"
        id="toc-count-number-of-censored-patients">Count Number of Censored
        Patients</a>
    -   <a href="#count-number-of-patients-who-definitely-survived-past-time-t"
        id="toc-count-number-of-patients-who-definitely-survived-past-time-t">Count
        Number of Patients Who Definitely Survived Past Time <em>t</em></a>
    -   <a href="#count-number-of-patients-who-may-have-survived-past-time-t"
        id="toc-count-number-of-patients-who-may-have-survived-past-time-t">Count
        Number of Patients Who May Have Survived Past Time <em>t</em></a>
    -   <a href="#count-number-of-patients-who-were-not-censored-before-time-t"
        id="toc-count-number-of-patients-who-were-not-censored-before-time-t">Count
        Number of Patients Who were Not Censored Before Time <em>t</em></a>
-   <a href="#kaplan-meier" id="toc-kaplan-meier">Kaplan-Meier</a>
    -   <a href="#import-packages-1" id="toc-import-packages-1">Import
        Packages</a>
-   <a href="#find-those-who-survived-up-to-time-t_i"
    id="toc-find-those-who-survived-up-to-time-t_i">Find Those Who Survived
    Up to Time <img style="vertical-align:middle"
    src="https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&amp;space;%5Cbg_white&amp;space;%5Ctextstyle%20t_i"
    alt="t_i" title="t_i" class="math inline" /></a>
    -   <a href="#find-those-who-died-at-time-t_i"
        id="toc-find-those-who-died-at-time-t_i">Find Those Who Died at Time
        <img style="vertical-align:middle"
        src="https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&amp;space;%5Cbg_white&amp;space;%5Ctextstyle%20t_i"
        alt="t_i" title="t_i" class="math inline" /></a>

# Count Patients

Welcome to the first practice lab of this week!

## Import Packages

``` r
library(dplyr)
```

We’ll work with data where:

-   Time: days after a disease is diagnosed and the patient either dies
    or left the hospital’s supervision.
-   Event:
    -   1 if the patient died
    -   0 if the patient was not observed to die beyond the given ‘Time’
        (their data is censored)

Notice that these are the same numbers that you see in the lecture video
about estimating survival.

``` r
df <- data.frame(Time = c(10,8,60,20,12,30,15),
                 Event = c(1,0,1,1,0,1,0))
df
#   Time Event
# 1   10     1
# 2    8     0
# 3   60     1
# 4   20     1
# 5   12     0
# 6   30     1
# 7   15     0
```

## Count Number of Censored Patients

``` r
df$Event == 0
# [1] FALSE  TRUE FALSE FALSE  TRUE FALSE  TRUE
```

Patient 1, 4 and 6 were censored.

-   Count how many patient records were censored

When we sum a series of booleans, `True` is treated as 1 and `False` is
treated as 0.

``` r
sum(df$Event == 0)
# [1] 3
```

## Count Number of Patients Who Definitely Survived Past Time *t*

This assumes that any patient who was censored died at the time of being
censored (**died immediately**).

If a patient survived past time `t`:

-   Their Time of event should be greater than `t`.

-   Notice that they can have an `Event` of either 1 or 0. What matters
    is their `Time` value.

``` r
t <- 25
df$Time > t
# [1] FALSE FALSE  TRUE FALSE FALSE  TRUE FALSE
```

``` r
sum(df$Time > t)
# [1] 2
```

## Count Number of Patients Who May Have Survived Past Time *t*

This assumes that censored patients **never die**.

-   The patient is censored at any time and we assume that they live
    forever.

-   The patient died (`Event` is 1) but after time `t`

``` r
t <- 25
df$Time > t | df$Event == 0
# [1] FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE
```

``` r
sum(df$Time > t | df$Event == 0)
# [1] 5
```

## Count Number of Patients Who were Not Censored Before Time *t*

If patient was not censored before time `t`:

-   They either had an event (death) before `t`, at `t`, or after `t`
    (any time)

-   Or, their `Time` occurs after time `t` (they may have either died or
    been censored at a later time after `t`)

``` r
t <- 25
df$Event == 1 | df$Time > t
# [1]  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE
```

``` r
sum(df$Event == 1 | df$Time > t)
# [1] 4
```

# Kaplan-Meier

The Kaplan Meier estimate of survival probability is:

![S(t) = \prod\_{t_i \<= t}(1-\frac{d_i}{n_i})](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;S%28t%29%20%3D%20%5Cprod_%7Bt_i%20%3C%3D%20t%7D%281-%5Cfrac%7Bd_i%7D%7Bn_i%7D%29 "S(t) = \prod_{t_i <= t}(1-\frac{d_i}{n_i})")

-   ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")
    are the events observed in the dataset

-   ![d_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;d_i "d_i")
    is the number of deaths at time
    ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")

-   ![n_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n_i "n_i")
    is the number of people who we know have survived up to time
    ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i").

## Import Packages

``` r
library(dplyr)
```

``` r
df <- data.frame(Time = c(3,3,2,2),
                 Event = c(0,1,0,1))
df
#   Time Event
# 1    3     0
# 2    3     1
# 3    2     0
# 4    2     1
```

# Find Those Who Survived Up to Time ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")

If they survived up to time
![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i"),

-   Their Time is either greater than
    ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")

-   Or, their Time can be equal to
    ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")

``` r
t_i <- 2
df$Time >= t_i
# [1] TRUE TRUE TRUE TRUE
```

You can use this to help you calculate
![n_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;n_i "n_i")

## Find Those Who Died at Time ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")

-   If they died at
    ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i"):

-   Their Event value is 1.

-   Also, their Time should be equal to
    ![t_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;t_i "t_i")

``` r
t_i <- 2
df$Event == 1 & df$Time == t_i
# [1] FALSE FALSE FALSE  TRUE
```

You can use this to help you calculate
![d_i](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D&space;%5Cbg_white&space;d_i "d_i")

You’ll implement Kaplan Meier in this week’s assignment!
