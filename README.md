# Histo-Regression
A Statistical Machine Learning method for producing predictions. This [**historeg.py**](https://github.com/EtzionR/Histo-Regression/blob/main/historeg.py) Algorithm can use as a **Regressor** also as **Classifier**.

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

Because we using floor-division, we just neeed to add the Y values to each cell, and calculate the cell value with our selected function. So, basically we only run once on the whole X table, and then only twice on the Y vector: once to add each Yi in its cell and again to apply the selected function on each cell. Therefore, basically the runtime of the algorithm is **O(n∙m)**, where m is the number of columns in X table and n is the number of rows (In case of using median function, the runtime changes slightly, and becomes O(max(m∙n, n∙logn)). The prediction process also requires a similar runtime. Therefore, this algorithm  allows us to get a prediction in short time!

As follows, the algorithm defines a fixed result for each range of values (cell), as can be seen in an example based on a one-dimensional case:

![1d](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/linear_case.png)

This prediction algorithm could even handle with complex function (that build from different function in every range):

![mixed](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/mixed_f.png)

As mentioned, the division into different cells is based on **floor division**, using of a value selected by the user ("**division**"). Using different values will lead to different results and a different quality of predication. Some inputs will result in **underfitting** results, while others will may cause **overfitting**. As can be seen in the following example:

![over_under_fitting](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/division.gif)

As we can see, we get the best results when division=1. when the value close to 10, it cause to underfitting, and when it close to 0, we can see overfitting of the curve.

In the **multi-dimensional** case, the division value can be set as a **vector**, so that each column in the dataset will have a **unique and customized** division value. The ability to give each column a unique value allows us to improve the resulting prediction. Because differents columns may have different scale, this aloow us to create model that fits to each column. In the following example, we use a [**coord.csv**](https://github.com/EtzionR/Histo-Regression/blob/main/examples/coord.csv) dataset, that displays **50,000** elevation points in space, as a function of their coordinates, which are presented as A & B features. Based on this data, using **cross validation**, we will perform a quality test for three prediction methods: 
- **Polynomial** regression, 
- **Historeg** based on **1** division value (division = 0.008), 
- **Historeg** based on **2** division values (division = [.009,.007]).

We will check the results by comparing the RMSE:

![a&b](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/a_b.png)

As we can see, we get the best results for the Historeg that used 2 division values!

As mentioned, the [historeg.py](https://github.com/EtzionR/Histo-Regression/blob/main/historeg.py) algorithm also allows classification on datasets. To apply classification using this algorithm, the user just need to choose the right function for classifiacation. In this case, we can use the default **Mode** function. In this example, we classify the [**class.csv**](https://github.com/EtzionR/Histo-Regression/blob/main/examples/class.csv) file. The file includes **+11,000** rows, each with 10 features. The variable we are trying to predict is the category to which each of the records belongs - **"cls"** - 4 categories in total. To do this, we will define division value = 0.2 on all fields, calculate the accuracy using **cross validation** and show the results as confusion matrix:

![confusion](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/confusion.png)

As we can see, we get pretty high accuracy **+95%**!

In addition, we can see that we got a slightly different confusion matrix than usual, which also includes another prediction column of category marked as **"-1"**. This column was created as a result of the default **empty value** of the algorithm. This value defiend for a situation when the given records not match to any cell. To prevent error, the algorithm return empty value. In our case, it can be seen that all the records were successfully associated with one of the given cells and that 0% of the observations were classified under category -1. 

At the same time, we can also see cases where the empty value is actually required. To illustrate this, we will artificially sample one-dimensional data from a simple linear function, but remove certain range from our data. Because we trying to use the algorithm as regressor, we defined empty=0. Now, let's look which prediction the algorithm returns:

![extra](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/extra.png)

As we can see, we get wrong prediction in the empty range. That teach as that the algorithm requires diverse training data in order to get the best result. In addition, the algorithm is **not** intended for extrapolation and can only give predictions about the data range on which it is trained.

When we comparing the [historeg.py](https://github.com/EtzionR/Histo-Regression/blob/main/historeg.py) algorithm to other methods, it seems to get quite good results. In the [multi-dimensional example](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/a_b.png) we have saw, we even found that it returns better results than the polynomial regression. At the same time, there seem to be times when other algorithms return better results, as can be seen in the following example:

![complex](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/complex.png)

So, as we see, this algorithm may be useful for our needs, but should examining it against other methods.

Another important aspect of the algorithm, is that in addition to the values of the predicate cells, its also calculate the **standard deviation** of each cell. Using this information, it is possible to examine how match the data of each cell diverse, and thus gain an idea about the quality of the value in each cell. To illustrate this, we can sample simple linear function, when the **noise** to each observation, gets bigger with x. Now, we can check the standard deviation of each cell in the output prediction to this function:

![std](https://github.com/EtzionR/Histo-Regression/blob/main/pictures/std.png)

As we can see, the standard deviation also get bigger with the x values!

## Libraries
the code required those libraries:

- **pandas**

- **numpy**


## Application
An application of the code is attached to this page under the name: 

[**implementation.py**](https://github.com/EtzionR/Histo-Regression/blob/main/implementation.py)

the examples outputs are also attached here.


## Example for using the code
To use this code, you just need to import it as follows:
``` sh
# import
from historeg import Historeg
import numpy as np

# define variables
n = 1000
x_train = np.random.uniform(0,10,n)
y_train = (x_train*.5) + 5 + np.random.normal(0,1.1,n)

x_test = np.random.uniform(0,10,100)
division = 1
empty = 0

# application
y_pred = Historeg(division,f=np.mean, empty=empty).fit(x_train,y_train).predict(x_test)
```

When the variables displayed are:

**division:** number or ary, defined the value for the floor division of the training data

**f:** the required function for calculate the value of each cell (default: mode function)

**empty:** the value for record out of range of the cells  (default: -1)

## License
MIT © [Etzion Harari](https://github.com/EtzionData)
