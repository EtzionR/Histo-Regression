# Histo-Regression
A Statistical Machine Learning method for producing predictions. This Algorithm can use as a **Regressor** also as **Classifier**.

## Overview
While the training process, the algorithm split the data to **K Cells**, by using floor-division. For each cell, the algorithm use the function defined by the user, to calculate the choosen value of the cell. 

Now, when we want to get prediction for new data, the algorithm find the proper cell for each row. So we can use the choosen value of the cell as the prediction. as follow:

<img src="https://latex.codecogs.com/svg.image?&space;\widehat{Y_{i}}=\begin{cases}T_{y\in&space;C_{1}}(Y_{train})&,X_{i}\in&space;C_{1}\\T_{y\in&space;C_{2}}(Y_{train})&,X_{i}\in&space;C_{2}\\...&,...\\T_{y\in&space;C_{j}}(Y_{train})&,X_{i}\in&space;C_{j}\\...&,...\\T_{y\in&space;C_{k}}(Y_{train})&,X_{i}\in&space;C_{k}\end{cases}&space;" title=" \widehat{Y}=\begin{cases}T_{y\in C_{1}}(Y_{train})&,X_{i}\in C_{1}\\T_{y\in C_{2}}(Y_{train})&,X_{i}\in C_{2}\\...&,...\\T_{y\in C_{j}}(Y_{train})&,X_{i}\in C_{j}\\...&,...\\T_{y\in C_{k}}(Y_{train})&,X_{i}\in C_{k}\end{cases} " />

When:

| Variable  | explanation                                                        |
| --------- | ------------------------------------------------------------------ |
|  **Xi**   | the features we want to get prediction for them (test\validation)  |
| **K**     | the number of cells                                                |
| **Cj**    | the choosen cell (when j between 1 to K)                           |
| **T(Y)**  | the defined function                                               |


The user can choose any function he want to calculate the value of each cell. The default is **Mode** function (that used for classification):

<img src="https://latex.codecogs.com/svg.image?T(Y)=\begin{cases}Mean(Y)&&space;\\Median(Y)&&space;\\Mode(Y)&&space;\\?&&space;\end{cases}&space;" title="T(Y)=\begin{cases}Mean(Y)& \\Median(Y)& \\Mode(Y)& \\?& \end{cases} " />

As follows, the algorithm defines a fixed result for each range of values (cell), as can be seen in an example based on a one-dimensional case:

![1d](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/linear_case.png)

As mentioned, the division into different cells is based on **floor division**, using of a value selected by the user ("**division**"). Using different values will lead to different results and a different quality of predication. Some inputs will result in **underfitting** results, while others will may cause **overfitting**. As can be seen in the following example:

![over_under_fitting](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/division.gif)

As we can see, we get the best results when division=1. when the value close to 10, it cause to underfitting, and when it close to 0, we can see overfitting of the curve.

In the **multi-dimensional** case, the division value can be set as a **vector**, so that each column in the dataset will have a **unique and customized** division value. The ability to give each column a unique value allows us to improve the resulting prediction. Because differents columns may have different scale, this aloow us to create model that fits to each column. In the following example, we use a [**coord.csv**](https://github.com/EtzionR/Histo-Regression/blob/main/examples/coord.csv) dataset, that displays 50,000 elevation points in space, as a function of their coordinates, which are presented as A & B features. Based on this data, using cross validation, we will perform a quality test for three prediction methods: 
- **Polynomial** regression, 
- **Historeg** based on **1** division value (division = 0.008), 
- **Historeg** based on **2** division values (division = [.009,.007]).

We will check the results by comparing the RMSE:

![a&b](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/a_b.png)

As we can see, we get the best results for the Historeg that used 2 division values!


