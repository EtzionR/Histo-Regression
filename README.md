# Histo-Regression
A Statistical Machine Learning method for producing predictions. This Algorithm can use as a **Regressor** also as **Classifier**.

## Overview
While the training process, the algorithm split the data to **K Cells**, by using floor-division. For each cell, the algorithm use the function defined by the user, to calculate the choosen value of the cell. 

Now, when we want to get prediction for new data, the algorithm find the proper cell for each row. So we can use the choosen value of the cell as the prediction. as follow:

<img src="https://latex.codecogs.com/svg.image?&space;\widehat{Y_{i}}=\begin{cases}T_{y\in&space;C_{1}}(Y_{train})&,X_{i}\in&space;C_{1}\\T_{y\in&space;C_{2}}(Y_{train})&,X_{i}\in&space;C_{2}\\...&,...\\T_{y\in&space;C_{j}}(Y_{train})&,X_{i}\in&space;C_{j}\\...&,...\\T_{y\in&space;C_{k}}(Y_{train})&,X_{i}\in&space;C_{k}\end{cases}&space;" title=" \widehat{Y}=\begin{cases}T_{y\in C_{1}}(Y_{train})&,X_{i}\in C_{1}\\T_{y\in C_{2}}(Y_{train})&,X_{i}\in C_{2}\\...&,...\\T_{y\in C_{j}}(Y_{train})&,X_{i}\in C_{j}\\...&,...\\T_{y\in C_{k}}(Y_{train})&,X_{i}\in C_{k}\end{cases} " />

When:

**Xi** = the features we want to get prediction for them (test\validation),

**K** = the number of cells, 

**Cj** = the choosen cell (when j between 1 to K),

**T(Y)** = the defined function.

The user can choose any function he want to calculate the value of each cell. The default is **Mode** function (that used for classification):

<img src="https://latex.codecogs.com/svg.image?T(Y)=\begin{cases}Mean(Y)&&space;\\Median(Y)&&space;\\Mode(Y)&&space;\\?&&space;\end{cases}&space;" title="T(Y)=\begin{cases}Mean(Y)& \\Median(Y)& \\Mode(Y)& \\?& \end{cases} " />

As follows, the algorithm defines a fixed result for each range of values (cell), as can be seen in an example based on a one-dimensional case:

![1d](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/simplae_case.png)

As mentioned, the division into different cells is based on **floor division**, using of a value selected by the user ("**division**"). Using different values will lead to different results and a different quality of predication. Some inputs will result in **underfitting** results, while others will may cause **overfitting**. As can be seen in the following example:

![over_under_fitting](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/division.gif)

As we can see, we get the best results when division=1. when the value close to 10, it cause to underfitting, and when it close to 0, we can see overfitting of the curve.


