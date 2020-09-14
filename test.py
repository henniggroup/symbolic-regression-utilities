import unittest
from dimensional_analysis import check_dimensionality


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
        
if __name__ == '__main__':
    unittest.main()
