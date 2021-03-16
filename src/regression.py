import numpy as np
import pandas as pd
import sympy
from sympy import lambdify


def process_fn(fn_string, symbols):
    """Preprocess and convert fn_string to a function with sympy.lambdify().

    Args: 
        fn_string (str)
        symbols (list or str)
    """
    fn_string = fn_string.replace('^', '**')
    fn = lambdify(sympy.symbols(symbols), fn_string)
    return fn
    
    
def evaluate_fn(df, fn_string, symbols):
    """Wrapper for process_fn() and test_fn().

    Args:
        df (pandas.DataFrame)
        fn_string (str)
        symbols (list or str)
    """
    fn = process_fn(fn_string, symbols)
    rmse, y_pred, y_true, c = test_fn(df, fn)
    return rmse, y_pred, y_true, c


def test_fn(df, fn):
    """Fit a coefficient to fn and evaluate RMSE with df.

    Args:
        df (pandas.DataFrame)
        fn: function from process_fn() or sympy.lambdify())
    """
    
    y_pred = []
    y_true = []
    
    for key in df.index:
        y_t, * inputs = df.loc[key]
        y_true.append(y_t)
        y_p = fn(*inputs)
        y_pred.append(y_p)
    
    # linear regression without intercept
    c = np.mean(y_true) / np.mean(y_pred)
    y_pred = np.multiply(y_pred, c)
    
    rmse = np.sqrt(np.mean(np.subtract(y_pred, y_true)**2))
    return rmse, y_pred, y_true, c
    

def loocv(df, fn_string, symbols):
    """Leave-one-out cross-validation with process_fn() and test_fn().

    Args:
        df (pandas.DataFrame)
        fn_string (str)
        symbols (list or str)
    """
    fn = process_fn(fn_string, symbols)
    y_pred = []
    y_true = df[df.columns[0]].values
    n_samples = len(y_true)
    sample_names = df.index
    
    for i in range(n_samples):
        test_name = sample_names[i]
        cv_subset = np.delete(np.arange(n_samples), i)
        cv_subset = [sample_names[j] for j in cv_subset]
        _, _, _, c = test_fn(df.loc[cv_subset], fn)
        _, *inputs = df.loc[test_name]
        y_pred.append(fn(*inputs) * c)
    rmse = np.sqrt(np.mean(np.subtract(y_pred, y_true)**2))
    return rmse, y_pred, y_true



