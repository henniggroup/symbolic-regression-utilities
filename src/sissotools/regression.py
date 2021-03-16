import numpy as np
import pandas as pd
import sympy
from sympy import lambdify


def process_fn(fn_string, symbols):
    """Preprocess and convert fn_string to a function with sympy.lambdify().

    Args: 
        fn_string (str): expression of interest.
        symbols (list or str): symbol(s) expected in expression.
    """
    fn_string = fn_string.replace('^', '**')
    fn = lambdify([sympy.symbols(symbols)], fn_string, 'numpy')
    return fn


def lsq_coefficients(x, y, fit_intercept=False):
    """Convenience function for least-squares fit of slope and intercept."""
    if fit_intercept:
        x = np.vstack([x, np.ones(len(x))]).T
    else:
        x = np.vstack([x, np.zeros(len(x))]).T
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), y)


def evaluate_model(fn_string, df, features,
                   coefficients=None,
                   target=None,
                   fit_intercept=False):
    """
    Pipeline function for evaluating an expression with data.

    Args:
        fn_string: expression of interest.
        df: pandas.DataFrame of interest with feature columns and, if fitting
            coefficients, a target columns.
        features: column keys corresponding to features that are expected to
             be present in fn_string. Only the intersection of keys in
             features and df.columns are used.
        coefficients: tuple of (slope, intercept). Fit with df if unspecified.
        target: DataFrame column key for fitting. If unspecified, defaults
            to the first column in the DataFrame.
        fit_intercept: (default is false).

    Returns:
        predictions: vector of predicted values using inputs from df.
        coefficients: passed or fit coefficients.
    """
    features = list(set(df.columns).intersection(features))
    array = df[features].to_numpy()
    func = process_fn(fn_string, features)
    n_samples = len(df)
    predictions = func(array.T)
    if coefficients is None:
        if target is None:
            target = df.columns[0]
        target_values = df[target]
        coefficients = lsq_coefficients(predictions, target_values,
                                        fit_intercept=fit_intercept)
        slope, intercept = coefficients
    else:
        slope, intercept = coefficients
    predictions = np.add(np.multiply(predictions, slope), intercept)
    return predictions, coefficients


def fit_evalute_fn(df, fn_string, symbols):
    """Legacy wrapper for process_fn() and test_fn().

    Args:
        df (pandas.DataFrame)
        fn_string (str)
        symbols (list or str)
    """
    fn = process_fn(fn_string, symbols)

    rmse, y_pred, y_true, c = test_fn(df, fn)
    return rmse, y_pred, y_true, c


def test_fn(df, fn):
    """Legacy function for fitting one multiplicative coefficient to fn
        and evaluating RMSE with df.

    Args:
        df (pandas.DataFrame)
        fn: function from process_fn() or sympy.lambdify())
    """

    y_pred = []
    y_true = []

    for key in df.index:
        y_t, *inputs = df.loc[key]
        y_true.append(y_t)
        y_p = fn(*inputs)
        y_pred.append(y_p)

    # linear regression without intercept
    c = np.mean(y_true) / np.mean(y_pred)
    y_pred = np.multiply(y_pred, c)

    rmse = np.sqrt(np.mean(np.subtract(y_pred, y_true) ** 2))
    return rmse, y_pred, y_true, c


def loocv(df, fn_string, symbols):
    """Legacy pipeline for leave-one-out cross-validation with
        process_fn() and test_fn().

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
    rmse = np.sqrt(np.mean(np.subtract(y_pred, y_true) ** 2))
    return rmse, y_pred, y_true



