import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.parsing.sympy_parser import parse_expr

from matplotlib.ticker import MultipleLocator


def prettify_fn(fn_string, lsubs=None, simplify=False):
    """
    Prepare function for Latex with optional simplification and substitutions.

    Args:
        fn_string (str): expression of interest.
        lsubs (dictionary): dictionary of substitutions, e.g. variable aliases
            to latex symbols.
        simplify (bool, optional): call sympy.simplify(); default is False.

    Returns:
        latex (str): processed expression.
    """
    fn_string = fn_string.replace('^', '**')
    expr = parse_expr(fn_string)
    if simplify:
        expr = sympy.simplify(expr)
    latex = sympy.latex(expr)
    if lsubs is not None:
        for target, sub in lsubs.items():
            latex = latex.replace(target, sub)
    return latex


def plot_distributions(df, columns,
                       n_bins=50, round_factor=0.5,
                       fontsize=8,
                       label_substitutions=None,
                       n_row=None, n_col=None,
                       ax=None, dpi=200):
    if n_row is None:
        if n_col is None:
            n_row = 2
            n_col = int(np.ceil(len(columns) / n_row))
        else:
            n_row = int(np.ceil(len(columns) / n_col))
    elif n_col is None:
        n_col = int(np.ceil(len(columns) / n_row))
    figsize = (1.5 * n_col, 1.5 * n_row)

    if label_substitutions is None:
        label_substitutions = {}

    if ax is None:
        fig, ax = plt.subplots(n_row, n_col, figsize=figsize, dpi=dpi)
        fig_tuple = (fig, ax)
    else:
        fig_tuple = (None, None)
        # try:
        #     fig = ax.get_figure()
        # except AttributeError:
        #     fig = ax[0].get_figure()
    ax = ax.flatten()

    for i, feature in enumerate(columns):
        v = df[feature]
        lims, tick_factor = round_lims(v, round_factor=round_factor)
        ax[i].xaxis.set_minor_locator(MultipleLocator(tick_factor * 2))
        bin_edges = np.linspace(lims[0], lims[1], n_bins + 1)
        histogram, bin_edges = np.histogram(v, bin_edges)
        width = bin_edges[1] - bin_edges[0]
        ax[i].bar(bin_edges[1:], histogram, width=width, linewidth=1)
        ax[i].set_xlim(lims)
        label = label_substitutions.get(feature, feature)
        ax[i].set_xlabel(label, fontsize=fontsize)
        ax[i].set_ylabel("Frequency", fontsize=fontsize)
        ax[i].tick_params(axis='both', labelsize=fontsize)
    for i in range(len(columns), n_col * n_row):
        ax[i].axis('off')

    return fig_tuple


def round_lims(values, round_factor=0.5):
    """
    Identify rounded minimum and maximum based on appropriate power of 10
        and round_factor.

        round_place = 10 ** ceil( log10((max-min))-1 )
        Minimum = (floor(min / round_place / round_factor)
                   * round_place * round_factor)
        Maximum = (ceil(max / round_place / round_factor)
                   * round_place * round_factor)

        E.g. [10, 39, 43] yields (10, 50) with round_factor = 1 (nearest 10)
             [10, 39, 43] yields (0, 100) with round_factor = 10 (nearest 100)
             [10, 39, 43] yields (0, 45) with round_factor = 0.5 (nearest 5)
    Args:
        values (np.ndarray, list): vector of values of interest.
        round_factor (float): Multiplicative factor for rounding power
            (Default = 0.5).

    Returns:
        lims: tuple of (rounded minimum, rounded maximum)

    """
    min_val = np.min(values)
    max_val = np.max(values)

    round_place = 10 ** np.ceil(np.log10(np.ptp([min_val, max_val])) - 1)
    rounded_min = (np.floor(min_val / round_place / round_factor)
                   * round_place * round_factor)
    rounded_max = (np.ceil(max_val / round_place / round_factor)
                   * round_place * round_factor)
    lims = (rounded_min, rounded_max)
    tick_factor = round_place * round_factor

    assert min_val >= lims[0]
    assert max_val <= lims[1]
    return lims, tick_factor


def pretty_scatter(references, predictions,
                   ax=None, loglog=False, lims=None, lim_factor=0.5,
                   figsize=(3.5, 3.5), dpi=200,
                   metrics=True, text_size=8,
                   units=None, labels=True, label_size=8,
                   **scatter_kwargs):
    """
    Scatter plot of predictions vs. references, colored by density.

    Args:
        predictions (list, np.ndarray): Vector of X-axis values.
        references (list, np.ndarray): Vector of Y-axis values.
        subset_threshold (float): Maximum points for
        ax: Optional handle for existing matplotlib axis object
        **scatter_kwargs: Optional keyword arguments for scatter function.

    Returns:
        fig & ax: matplotlib figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig_tuple = (fig, ax)
    else:
        fig_tuple = (None, None)
    if 's' not in scatter_kwargs.keys():
        scatter_kwargs['s'] = 1
    x = np.array(references)
    y = np.array(predictions)
    xy_stack = np.vstack([x, y])
    assert len(x) == len(y), "Dimension mismatch."
    # Compute RMSE
    residuals = np.subtract(y, x)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_over = np.max(residuals)
    max_under = np.min(residuals)
    ax.scatter(x, y,  **scatter_kwargs)
    # Axis scale and limits
    ax.axis('square')
    if loglog is True:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if lims is None:
            lims = ax.get_xlim()
    else:
        if lims is None:
            lims, tick_factor = round_lims(np.concatenate([x, y]),
                                           round_factor=lim_factor)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, color='lightgray', linestyle='--', linewidth=0.5)
    # Error Metrics
    if metrics is True:
        error_text = 'RMSE = {0:.3f}'.format(rmse)
        error_text += '\nMAE = {0:.3f}'.format(mae)
        res_text = 'Max Res. = {0:.3f}'.format(max_over)
        res_text += '\nMin Res. = {0:.3f}'.format(max_under)
        ax.text(0.02, 0.98, error_text,
                ha='left', va='top',
                fontsize=text_size,
                transform=ax.transAxes)
        ax.text(0.98, 0.02, res_text,
                ha='right', va='bottom',
                fontsize=text_size,
                transform=ax.transAxes)
    # Axis Labels
    if labels is True:
        if isinstance(units, str):
            unit_string = " " + units
            if all([c not in unit_string for c in ['[', ']', '(', ')']]):
                unit_string = ' [{}]'.format(units)
        else:
            unit_string = ""
        ax.set_ylabel('Predicted' + unit_string, fontsize=label_size)
        ax.set_xlabel('Reference' + unit_string, fontsize=label_size)
    ax.tick_params(axis='both', labelsize=label_size)
    return fig_tuple
