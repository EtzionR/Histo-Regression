# Histo-Regression
A Statistical Machine Learning method for producing predictions. This Algorithm can use as a **Regressor** also as **Classifier**.

## Overview
While the training process, the algorithm split the data to **K Cells**, by using floor-division. For each cell, the algorithm use the function defined by the user, to calculate the choosen value of the cell. 

Now, when we want to get prediction for new data, the algorithm find the proper cell for each row. So we can use the choosen value of the cell as the prediction. as follow:

<img src="https://latex.codecogs.com/svg.image?&space;\widehat{Y_{i}}=\begin{cases}T_{y\in&space;C_{1}}(Y)&,X\in&space;C_{1}\\T_{y\in&space;C_{2}}(Y)&,X\in&space;C_{2}\\...&,...\\T_{y\in&space;C_{j}}(Y)&,X\in&space;C_{j}\\...&,...\\T_{y\in&space;C_{k}}(Y)&,X\in&space;C_{k}\end{cases}&space;" title=" \widehat{Y}=\begin{cases}T_{y\in C_{1}}(Y)&,X\in C_{1}\\T_{y\in C_{2}}(Y)&,X\in C_{2}\\...&,...\\T_{y\in C_{j}}(Y)&,X\in C_{j}\\...&,...\\T_{y\in C_{k}}(Y)&,X\in C_{k}\end{cases} " />

When:

**K** = the number of cells, 

**Cj** = the choosen cell,

**T(Y)** = the defined function.

The user can choose any function he want to calculate the value of each cell. The default is **Mode** function (that used for classification):

<img src="https://latex.codecogs.com/svg.image?T(Y)=\begin{cases}Mean(Y)&&space;\\Median(Y)&&space;\\Mode(Y)&&space;\\?&&space;\end{cases}&space;" title="T(Y)=\begin{cases}Mean(Y)& \\Median(Y)& \\Mode(Y)& \\?& \end{cases} " />

