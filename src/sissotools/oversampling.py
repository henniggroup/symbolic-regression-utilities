import numpy as np
import pandas as pd


def oversampling_concat(df_list, target_size=None, exact=False,
                        weight_key='weight', verbose=True, copy=True):
    """
    Concatenate DataFrames with oversampling, where smaller DataFrames
    are repeated until they are equal in size to the largest DataFrame.
    Residual sizes are handled by "exact"
    e.g. given three DataFrames with lengths (50, 27, 8):
        exact = true yields oversampled lengths (50, 27+23, 48+2)
        exact = false yields oversampled lengths (50, 54, 48)

    Args:
        df_list (list): list of pd.DataFrame to concatenate and oversample.
        target_size (int): desired total size, used to determine number of
            repeats of largest size. When exact=False, specifying and/or
             increasing this value reduces the discrepancy between weights.
        exact (bool): handle residual fractions exactly by resampling randomly
            or approximately by rounding.
        weight_key (str): column name for weight assignment.
        verbose (bool): print initial and oversampled sizes.
        copy (bool): see pandas.concat.

    Returns:
        df_combined (pd.DataFrame): oversampled, joined DataFrame.
    """
    n_set = [len(df) for df in df_list]
    if verbose:
        print('Original:', n_set)
    n_max = np.max(n_set)
    n_total = n_max * len(n_set)
    if target_size is None:
        target_size = n_total
    size_factor = max(1, np.floor(target_size / n_total))  # multiplicative factor before rounding

    weights = []
    for df in df_list:
        n = len(df)
        raw_factor = n_max / n * size_factor
        if exact:
            factor = max(1, np.floor(raw_factor))
            weight_vector = np.repeat(factor, n)
            n_residual = int(np.round(n * (raw_factor - factor)))
            additional = np.random.choice(np.arange(len(weight_vector)),
                                          n_residual,
                                          replace=False)
            weight_vector[additional] += 1
        else:
            factor = max(1, np.round(raw_factor))
            weight_vector = np.repeat(factor, n)
        weights.append(weight_vector)
    df_combined = pd.concat(df_list, copy=copy)
    df_combined[weight_key] = np.concatenate(weights)
    if verbose:
        print('Oversampled:', [np.sum(w) for w in weights])
    return df_combined


def apply_oversampling(df, weight_key):
    """
    Apply oversampling, e.g. from oversampling_concat, by repeating
    samples according to weight given by weight_key column.

    Args:
        df (pd.DataFrame): data.
        weight_key (str): column of repeats (integer numbers) per sample.

    Returns:
        df_expanded (pd.DataFrame): data with oversampled entries.
    """
    indices = np.arange(len(df))
    repeats = df[weight_key]
    iloc_vector = np.repeat(indices, repeats)
    df_expanded = df.iloc[iloc_vector]
    return df_expanded
