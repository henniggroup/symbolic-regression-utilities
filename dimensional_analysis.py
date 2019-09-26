import numpy as np
import re
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()


def check_dimensionality(equation, units_reference):
    """Returns unit (string) / dimension (float/int) of "equation" using Pint.
    
    Args:
        equation (str)
        units_reference (dict): map of symbols (str) to units (str)

    Returns:
        tuple of unit (str), degree (int or float)
    """
    equation = equation.replace('^', '**')
    units_map = {k: ureg[v or ""] for k, v in units_reference.items()}
    units_map.update({'sqrt': np.sqrt, 'cbrt': np.cbrt})
    
    try:
        #  isolated from global and local variables
        y = eval(equation, {'__builtins__':None}, units_map)

        if isinstance(y, float):
            return 'dimensionless', 0
        elif y.dimensionless:
            return 'dimensionless', 0
        else:
            
            unit, degree = list(y.units.dimensionality.items()).pop()
            if np.isclose(np.round(degree), degree):
                return unit, int(degree)
            else:
                return unit, degree
    except DimensionalityError:
        # Unit mismatch
        return None, None
    except ZeroDivisionError:
        return None, None
    except TypeError:
        return None, None
