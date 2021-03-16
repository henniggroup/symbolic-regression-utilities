import unittest
from dimensional_analysis import check_dimensionality
from parsing import create_train_dat
from parsing import create_sisso_input
from parsing import read_train_dat
from parsing import read_features
from parsing import read_models
from parsing import read_coefficients


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
        assert len(text) == 599

    def test_read_write_traindat(self):
        df = read_train_dat("./datasets/AllenDynes/train.dat")
        assert list(df.columns) == ["t_c", "wlog", "lambd", "mu"]
        text = create_train_dat(df)
        assert len(text) == 1379

    def test_read_features(self):
        df = read_features("./datasets/AllenDynes/feature_space/Uspace.name")
        assert list(df.columns) == ['Feature', 'Correlation']

    def test_read_models(self):
        df = read_models("./datasets/AllenDynes/models/top9999_001d")
        assert list(df.columns) == ['rmse', 'mae', 'feature id']

    def test_read_coef(self):
        filename = "./datasets/AllenDynes/models/top9999_001d_coeff"
        df = read_coefficients(filename)
        assert list(df.columns) == ['intercept', 'slope']


if __name__ == '__main__':
    unittest.main()
