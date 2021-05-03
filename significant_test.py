"""
Adapted from code at https://github.com/ariaaay/NeuralTaskonomy
"""

import numpy as np
from statsmodel.stats.multitest import fdrcorrection

def compute_adjust_p(pvalues):
    pvalues = np.array(correlation_with_p)[:, 1]
    adjusted_p = fdrcorrection(pvalues)[1]
    return 