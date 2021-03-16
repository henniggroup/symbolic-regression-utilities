import os
import re

import numpy as np
import pandas as pd

import sympy
from sympy.parsing.sympy_parser import parse_expr

from sympy.calculus.singularities import is_monotonic
from sympy.calculus.singularities import is_increasing
from sympy.calculus.singularities import is_decreasing
from sympy.calculus.singularities import singularities

from sympy.sets.sets import Interval
from sympy import oo


def check_constraints(model,
                      constraints,
                      intervals,
                      characteristic_vals=None,
                      verbose=0):
    """WIP function for Tc project.
    UseS Sympy to check for singularities and other limits."""
    intervals = dict(intervals)
    feature_set = list(constraints.keys())
    if characteristic_vals is None:
        characteristic_vals = {feature: i/10 for i, feature 
                               in enumerate(feature_set)}
    for feature in feature_set:
        if feature not in intervals.keys():
            intervals[feature] = sympy.Reals
            continue
        interval_min, interval_max = intervals[feature]
        if interval_min == "-oo" or interval_min == -np.inf:
            interval_min = -oo
        elif interval_max == "oo" or interval_max == np.inf:
            interval_max = oo
        interval = Interval(interval_min, interval_max)
        intervals[feature] = interval

    symbol_dict = {k: v for k, v
                   in zip(feature_set,
                          sympy.symbols(feature_set,
                                        positive=True,
                                        finite=True,
                                        infinite=False))}
    expr = parse_expr(model.replace('^', '**'), local_dict=symbol_dict)

    passed = True
    checks = {k: {} for k in constraints.keys()}
    for feature, symbol in symbol_dict.items():
        symbol_set = list(symbol_dict.values())
        variable = symbol_set.pop(symbol_set.index(symbol))
        interval = intervals[feature]

        univariate_expr = expr.subs([(symbol, characteristic_vals[str(symbol)])
                                     for symbol in symbol_set])
        if verbose > 1:
            print(univariate_expr)

        if constraints[feature].get('increasing', None) is not None:
            try:
                increasing = is_increasing(univariate_expr,
                                           interval=interval)
            except TypeError:
                increasing = False
            if increasing is None:  # bug?
                increasing = False
            checks[feature]['increasing'] = increasing
            if increasing != constraints[feature]['increasing']:
                passed = False

        if constraints[feature].get('decreasing', None) is not None:
            try:
                decreasing = is_decreasing(univariate_expr,
                                           interval=interval)
            except TypeError:
                decreasing = False
            if decreasing is None:  # bug?
                decreasing = False
            checks[feature]['decreasing'] = decreasing
            if decreasing != constraints[feature]['decreasing']:
                passed = False

        if constraints[feature].get('monotonic', None) is not None:
            try:
                monotonic = is_monotonic(univariate_expr,
                                         interval=interval)
            except TypeError:
                monotonic = False
            checks[feature]['monotonic'] = monotonic
            if monotonic != constraints[feature]['monotonic']:
                passed = False

        if constraints[feature].get('singularities', None) is not None:
            try:
                singularity_set = singularities(expr, variable,
                                                domain=interval)
            except TypeError:
                singularity_set = sympy.EmptySet
            checks[feature]['singularities'] = singularity_set
            # has_singularities = singularity_set is not sympy.EmptySet
            if singularity_set != constraints[feature]['singularities']:
                passed = False

        if constraints[feature].get('zero limit', None) is not None:
            try:
                zero_limit = sympy.limit(expr, variable, 0)
            except TypeError:
                zero_limit = None
            checks[feature]['zero limit'] = zero_limit
            if zero_limit != constraints[feature]['zero limit']:
                passed = False
    if verbose == 0:
        return passed
    else:
        return checks, passed
