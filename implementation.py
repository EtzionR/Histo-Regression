# import
from historeg import Historeg
import pandas as pd
import numpy as np

# define variables
n = 1000
x_train = np.random.uniform(0,10,n)
y_train = (x_train*.5) + 5 + np.random.normal(0,1.1,n)

x_test = np.random.uniform(0,10,100)
y_test = (x_test*.5) + 5
division = 1
empty = 0

# application
y_pred = Historeg(division,f=np.mean, empty=empty).fit(x_train,y_train).predict(x_test)
print(f'RMSE = {round(np.sqrt(np.mean(np.power(y_pred-y_test,2))),2)}')