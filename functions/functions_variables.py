import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

def get_variables_paraschiv_2021():
    # Paraschiv, Florentina, Markus Schmid, and Ranik Raaen Wahlstrøm. 2021. 
    # “Bankruptcy Prediction of Privately Held SMEs Using Feature Selection Methods.”
    # Working Paper, Norwegian University of Science and Technology and University of St. Gallen. 
    # https://doi.org/10.2139/ssrn.3911490
    variables = [
        '(current liabilities - short-term liquidity) / total assets',
        'accounts payable / total assets',
        'dummy; one if paid-in equity is less than total equity',
        'dummy; one if total liability exceeds total assets',
        'interest expenses / total assets',
        'inventory / current assets',
        'log(age in years)',
        'net income / total assets',
        'public taxes payable / total assets',
        'short-term liquidity / current assets',
    ]
    return variables

def get_variables_altman_and_sabato_2007():
    # Altman, Edward I., and Gabriele Sabato. 2007. 
    # “Modelling Credit Risk for SMEs: Evidence from the U.S. Market.” 
    # Abacus 43 (3): 332–57. 
    # https://doi.org/10.1111/j.1467-6281.2007.00234.x
    variables = [
        'current liabilities / total equity',
        'EBITDA / interest expense',
        'EBITDA / total assets',
        'retained earnings / total assets',
        'short-term liquidity / total assets',
    ]
    return variables

def get_variables_altman_1968():
    # Altman, Edward I. 1968. 
    # “Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy.”
    # The Journal of Finance 23 (4): 589–609. 
    # https://doi.org/10.2307/2978933
    variables = [
        'EBIT / total assets',
        'retained earnings / total assets',
        'sales / total assets',
        'total equity / total liabilities',
        'working capital / total assets',
    ]
    return variables



