import unittest
import os
import sissotools
from sissotools.dimensional_analysis import check_dimensionality
from sissotools.io import create_tsv
from sissotools.io import create_sisso_input
from sissotools.io import read_tsv
from sissotools.io import read_features
from sissotools.io import read_models
from sissotools.io import read_coefficients

root = os.path.dirname(os.path.dirname(os.path.dirname(sissotools.__file__)))
DATA_PATH = os.path.join(root, "datasets/AllenDynes")


class TestDimensionalAnalysis(unittest.TestCase):
    def setUp(self):
        self.feature_units = {'wlog': 'K', 
                              'lambd': 'dimensionless',
                              'mu': 'dimensionless'}

    def test_conventional(self):
        model = '(((lambd)^3*(wlog*lambd))/(sqrt(mu)+(lambd)^3))'
        units = check_dimensionality(model, self.feature_units)
        assert units == ('[temperature]', 1)
        
    def test_sqrt(self):
        model = '((exp(-mu)+exp(-lambd))*((lambd-mu)*sqrt(wlog)))'
        units = check_dimensionality(model, self.feature_units)
        assert units == ('[temperature]', 0.5)
        
    def test_cbrt(self):
        model = '((exp(-mu)+exp(-lambd))*((lambd-mu)*cbrt(wlog)))'
        units = check_dimensionality(model, self.feature_units)
        assert units == ('[temperature]', 1/3)
        
    def test_dimension_mismatch(self):
        model = '(log((lambd/wlog))*(log(wlog)*(lambd-mu)))'
        units = check_dimensionality(model, self.feature_units)
        assert units == ('Dimension Mismatch', 0)
        
    def test_zero_division(self):
        model = '(((mu+lambd)+(mu/lambd))/((mu)^2-(mu)^2))'
        units = check_dimensionality(model, self.feature_units)
        assert units == ('Zero Division', 0)


class TestParsing(unittest.TestCase):
    def test_create_input(self):
        text = create_sisso_input(dict(nsample=1,
                                       nsf=1,
                                       dimclass="(1:1)"))
        assert len(text) == 621

    def test_read_write_traindat(self):
        fname = os.path.join(DATA_PATH, "train.dat")
        df = read_tsv(fname)
        assert list(df.columns) == ["t_c", "wlog", "lambd", "mu"]
        text = create_tsv(df)
        assert len(text) == 1379

    def test_read_features(self):
        fname = os.path.join(DATA_PATH, "feature_space/Uspace.name")
        df = read_features(fname)
        assert list(df.columns) == ['Feature', 'Correlation']

    def test_read_models(self):
        fname = os.path.join(DATA_PATH, "models/top9999_001d")
        df = read_models(fname)
        assert list(df.columns) == ['rmse', 'mae', 'feature id']

    def test_read_coef(self):
        fname = os.path.join(DATA_PATH, "models/top9999_001d_coeff")
        df = read_coefficients(fname)
        assert list(df.columns) == ['intercept', 'slope']


if __name__ == '__main__':
    unittest.main()
