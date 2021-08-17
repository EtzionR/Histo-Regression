# Create by Etzion Harari
# https://github.com/EtzionR

# import libraries
import pandas as pd
import numpy as np

# defined variables
VECTOR = {type(np.array([])) ,type([]) ,type(())}
PDDF   = type(pd.DataFrame([]))

# defined useful functions
set_divisor = lambda divisor, m=1: np.array(divisor) if type(divisor) in VECTOR else np.array([divisor ] *m)
iterdf      = lambda df: df.values if type(df )==PDDF else df


def mode(ary):
    """
    mode function
    find the object with the biggest number of appears in the given ary
    :param ary: input ary (some iterable object)
    :return: the number with the largest appearence number
    """
    dct = {}
    for value in ary:
        if value in dct:
            dct[value]+=1
        else:
            dct[value] =1
    return max(dct, key=lambda k: dct[k])


class Historeg:
    """
    Histogram Regression Object

    Calculate the Prediction for given values,
    by using predictive cells and the training data
    """
    def __init__(self, divisor, f=mode, empty=-1):
        """
        the initilaize function of the object
        :param divisor: the value for floor division the input data
        :param f: the function for calculate the value of each cell
        :param empty: empty value for records out of cells range
        """
        self.cells = {}
        self.std = {}
        self.divisor = divisor
        self.f = f
        self.empty = empty

    def fit(self, x, y):
        """
        fitting function for training data
        the function also calculate the std for each cell
        :param x: x train
        :param y: y train
        """
        self.n = x.shape[0]
        self.m = x.shape[1] if len(x.shape) > 1 else 1
        self.divisor = set_divisor(self.divisor, m=self.m)
        floored = iterdf(x // self.divisor)
        table, y = {}, [*y]

        for i in range(self.n):
            key = (*floored[i],) if self.m > 1 else floored[i]
            if key in table:
                table[key] += [y[i]]
            else:
                table[key] = [y[i]]
        self.cells = {k: self.f(table[k]) for k in table}
        self.std   = {k: np.std(table[k]) for k in table}

        return self

    def predict(self, x):
        """
        predict function to calculate the value of input data,
        by using the predictive cells that create from the training data
        :param x: records the required prediction
        :return: prediction
        """
        x = x if type(x) == PDDF else pd.DataFrame(x)
        if self.m > 1:
            return np.array([self.cells[(*row,)] if (*row,) in self.cells else self.empty
                             for row in x.values // self.divisor])
        else:
            return np.array([self.cells[row] if row in self.cells else self.empty
                             for row in x.iloc[:, 0] // self.divisor])

# MIT © Etzion Harari
