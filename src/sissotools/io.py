import os
import re
import numpy as np
import pandas as pd

BASIC_DEFAULT = dict(ptype=1,
                     ntask=1,
                     nsample=None,
                     task_weighting=1,
                     desc_dim=1,
                     restart=False)
SIS_DEFAULT = dict(nsf=None,
                   rung=3,
                   maxcomplexity=10,
                   opset="'(+)(-)(*)(/)(exp)(exp-)(^-1)(^2)(^3)(sqrt)(log)'",
                   dimclass=None,
                   maxfval_lb="1e-6",
                   maxfval_ub="1e6",
                   subs_sis=10000)
SO_DEFAULT = dict(method='L0',
                  fit_intercept=False,
                  L1L0_size4L0=1,
                  metric='RMSE',
                  isconvex="(1, 1)",
                  width=0.001,
                  nm_output=10000,
                  L1_weighted=False)


def create_sisso_input(parameters, filename=None):
    """
    Generate SISSO input file with defaults.

    See https://github.com/rouyang2017/SISSO/tree/master/input_template

    Args:
        parameters (dict): Parameters for SISSO.
            At minimum, nsample, nsf, and dimclass must be specified.

        filename (str): Path for writing file (optional).

    Returns (if filename==None):
        input_text (string)
    """
    # TODO: Remove regression/classification-specific keywords
    #       based on ptype (harmlessly ignored by SISSO).
    separator = "!{0:>>78}".format("\n")
    input_text = ""
    for default_dict in [BASIC_DEFAULT, SIS_DEFAULT, SO_DEFAULT]:
        input_text += separator
        for parameter, default_value in default_dict.items():
            value = parameters.get(parameter, default_value)
            if value is True:
                value = ".true."
            elif value is False:
                value = ".false."
            elif value is None:
                raise ValueError("Need to specify {}".format(parameter))
            input_text += "{} = {}\n".format(parameter, value)
    if filename is not None:
        with open(filename, "w") as f:
            f.write(input_text)
    else:
        return input_text


def create_tsv(df, filename=None):
    """Convert pandas DataFrame to text-based table for SISSO.

    Args:
        df (pandas DataFrame): Dataframe of samples and features.
        filename: Path for writing file (optional).

    Returns (if filename==None):
        table (string)
    """
    table = df.to_string()
    lines = table.splitlines()
    index_name = lines.pop(1).strip()
    lines[0] = index_name + lines[0][len(index_name):]
    table = '\n'.join(lines)
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(table)
    else:
        return table


def read_tsv(filename):
    """Read text-based table into pandas DataFrame.

    Args:
        filename (str): Path for reading file.

    Returns:
        df (pandas DataFrame): DataFrame of samples and features.
    """
    df = pd.read_csv(filename, delim_whitespace=True, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def read_features(path):
    """Parse SISSO features (e.g. ./feature_space/Uspace.name)

    Args:
        path (str): Path for reading file.

    Returns:
        feature_df (pandas DataFrame): 1-indexed DataFrame with columns:
            "Feature" (str): expression generated by SISSO.
            "Correlation" (float): feature correlation, [0,1].
    """
    columns = ["Feature", "Correlation"]
    regex_str = '([^\s]+)(?:\s+corr=\s+)([^\n]+)'
    with open(path, 'r') as f:
        text = f.read()
    feature_tuples = re.compile(regex_str).findall(text)
    index = np.arange(len(feature_tuples)) + 1
    # features and models are 1-indexed
    feature_df = pd.DataFrame(index=index, 
                              data=feature_tuples, 
                              columns=columns)
    feature_df = feature_df.dropna()
    feature_df['Correlation'] = pd.to_numeric(feature_df['Correlation'])
    return feature_df


def read_models(path):
    """Parse SISSO models (e.g. ./models/top9999_001d)

    Args:
        path (str): Path for reading file.

    Returns:
        feature_df (pandas DataFrame): 1-indexed DataFrame with columns:
            "rmse" (float): root-mean-square error
            "mae" (float): mean-absolute error
            "feature id" (int): 1-indexed feature indices
    """
    columns = ['model index', 'rmse', 'mae', '_', 'feature id']

    def clean_func(string):
        return int(string.replace(')', ''))
    model_df = pd.read_csv(path,
                           delim_whitespace=True, 
                           index_col=0,
                           header=0,
                           names=columns)
    del model_df['_']  # TODO: better handling of this SISSO formatting
    model_df = model_df.dropna()
    model_df['feature id'] = model_df['feature id'].apply(clean_func)
    model_df = model_df.apply(pd.to_numeric)
    return model_df


def read_coefficients(path):
    """Parse SISSO model coefficients (e.g. ./models/top9999_001d_coeff)

    Args:
        path (str): Path for reading file.

    Returns:
        feature_df (pandas DataFrame): 1-indexed DataFrame with columns:
            "intercept" (float)
            "slope" (float)
    """
    columns = ['model index', 'intercept', 'slope']
    coeff_df = pd.read_csv(path, 
                           delim_whitespace=True, 
                           index_col=0,
                           header=0, 
                           names=columns)
    coeff_df = coeff_df.dropna()
    coeff_df = coeff_df.apply(pd.to_numeric)
    return coeff_df


def parse_sisso_run(path,
                    dim="001",
                    re_pattern="top([^_]+)_([0-9]+)d"):
    """
    Pipeline function for parsing SISSO outputs from directory.

    Args:
        path (str): Path to run directory.
        dim (str): identifier for SISSO dimensionality, e.g. "001" for S=1.
        re_pattern (str): regular expression for SISSO files yielding
            (identifier, dim), e.g. top0100_001d becomes ("0100", "001")

    Returns:
        df_features, df_models, df_coef, df_val
    """
    feature_fn = 'feature_space/Uspace.name'
    feature_fpath = os.path.join(path, feature_fn)
    models_dir = os.path.join(path, 'models')
    model_files = "\n".join(os.listdir(models_dir))
    patterns = set(re.compile(re_pattern).findall(model_files))
    assert any([s[1] == dim for s in patterns]), \
        "Desired dimension is not present: \"{}\"".format(dim)
    identifier = sorted([s[0] for s in patterns],
                        key=lambda x: x.replace('*', '9')).pop()
    models_fn = 'top{}_{}d'.format(identifier, dim)
    models_fpath = os.path.join(models_dir, models_fn)
    coef_fn = 'top{}_{}d_coeff'.format(identifier, dim)
    coef_fpath = os.path.join(models_dir, coef_fn)

    df_features = read_features(feature_fpath)
    df_models = read_models(models_fpath)
    df_coef = read_coefficients(coef_fpath)

    fname_val = os.path.join(path, 'predict.dat')
    if os.path.isfile(fname_val):
        df_val = read_tsv(fname_val)
    else:
        df_val = None
    return df_features, df_models, df_coef, df_val