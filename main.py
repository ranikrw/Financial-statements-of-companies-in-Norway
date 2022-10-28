import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

from tqdm import tqdm # for-loop progress bar

import statsmodels.api as sm
from sklearn import metrics

import time # For timing the execution of code

import os

# Importing files with functions
# These are found in the folder "functions"
import sys
sys.path.insert(1, 'functions')
from functions import *
from function_remove_unused_columns import *
from functions_variables import *

##################################################################
##  Define accounting data to include
##################################################################
# Loading all data takes up a lot of memory, making the computer
# run very slowly. Therefore, the code loads only the accounting
# posts you define in this list:

list_of_accounting_columns_to_include = [
    'Bankinnskudd, kontanter og lignende',
    'Sum bankinnskudd, kontanter og lignende',
    'Skyldige offentlige avgifter',
    'Leverandoergjeld',
    'Sum kortsiktig gjeld',
    'Sum inntekter',
    'Sum innskutt egenkapital',
    'Sum egenkapital',
    'SUM EIENDELER',
    'Avskrivning paa varige driftsmidler og immaterielle eiendeler',
    'Ordinaert resultat foer skattekostnad',
    'Ordinaert resultat etter skattekostnad',
    'Sum gjeld',
    'Aarsresultat',
    'Sum finanskostnader',
    'Annen renteinntekt',
    'Utbytte',
    'Sum opptjent egenkapital',
    'Gjeld til kredittinstitusjoner',
    'Salgsinntekt',
    'Loennskostnad',
    'Sum varer',
    'Kundefordringer',
    'Sum omloepsmidler',
    'Nedskrivning av varige driftsmidler og immaterielle eiendeler',
]

# If you still encounter memory problems, try moving filters from
# the section "Filtering" inside the for-loop in the section "Load data"

##################################################################
##  Starts timing of code
##################################################################
time_start = time.time()

##################################################################
##  Load data                                                   ##
##################################################################
# Define folder_name to the path to the folder containing the data
# The folder should contain one file per accounting year, and nothing more
folder_name = '../../../Research/datasett_aarsregnskaper/data4/'

print('-----------------------------------------')
print('Loading data:')
print('-----------------------------------------')
files = os.listdir(folder_name)
for current_file in files:
    file_year = int(current_file[0:4])

    # Loading one year file
    data_loaded = remove_unused_columns(pd.read_csv(folder_name+current_file,sep=';',low_memory=False),list_of_accounting_columns_to_include)

    # Adding all data together into data
    if current_file == files[0]:
        data = pd.DataFrame(columns=data_loaded.columns)
    data = pd.concat([data,data_loaded])
    print('Imported for accounting year {}'.format(file_year))

# Reset index 
data = data.reset_index(drop=True)

# Checking that all financial statements are unique
unique_orgnr = data.groupby(['orgnr','regnaar']).size().reset_index()
temp = unique_orgnr[0].unique()
if len(temp)==1:
    print('All orgnr unique')
else:
    print('ERROR: not all orgnr unique')

# Considering only accountint year 2020 or earlier
data = data[data.regnaar<=2020]
data = data.reset_index(drop=True) # Reset index

##################################################################
##  Filtering
##################################################################
#  Performing data filtering

# Considering only AS, that is, only
# private (i.e., not listed) limited liability companies
data = data[data['orgform']=='AS']
data = data.reset_index(drop=True) # Reset index

# Excluding all financial statements with total assets below NOK 500,000
# For many rows, accounting variables have no value. No value for accounting
# variables actually means that it is zero. Thus, we use .fillna(0).
data = data[data['SUM EIENDELER'].fillna(0)>=5e5]
data = data.reset_index(drop=True) # Reset index 

# Considering only SMEs (definition: https://ec.europa.eu/growth/smes/sme-definition_en)
ind = (data['sum_omsetning_EUR'].fillna(0)<=50e6) | (data['sum_eiendeler_EUR'].fillna(0)<=43e6)
data = data[ind]
data = data.reset_index(drop=True) # Reset index

# Excluding some industries
data = data[data['naeringskoder_level_1']!='L'] # Real estate activities
data = data[data['naeringskoder_level_1']!='K'] # Financial and insurance activities
data = data[data['naeringskoder_level_1']!='D'] # Electricity and gas supply
data = data[data['naeringskoder_level_1']!='E'] # Water supply, sewerage, waste
data = data[data['naeringskoder_level_1']!='MISSING'] # Missing
data = data[data['naeringskoder_level_1']!='0'] # companies for investment and holding purposes only
data = data[data['naeringskoder_level_1']!='O'] # Public sector
data = data.reset_index(drop=True) # Reset index 

##################################################################
##  Making accounting variables
##################################################################
# In the following code, we first prepare objects for accounting posts.
# Secondly, we construct variable sets of three bankruptcy prediction models

# For the accounting data, missing values means that it is zero. 
# We therefore use .fillna(0) at the end for the accounting variables

# Usually 'Bankinnskudd, kontanter og lignende' catches everything, but 
# sometimes 'Sum bankinnskudd, kontanter og lignende' needs to be used instead
string1 = 'Bankinnskudd, kontanter og lignende'
string2 = 'Sum bankinnskudd, kontanter og lignende'
bankinnskudd_kontanter_og_lignende = pd.Series([None]*data.shape[0])
for i in tqdm(range(data.shape[0])):
    if pd.isnull(data[string1].iloc[i])==False:
        bankinnskudd_kontanter_og_lignende[i] = data[string1].iloc[i]
    elif pd.isnull(data[string2].iloc[i])==False:
        bankinnskudd_kontanter_og_lignende[i] = data[string2].iloc[i]
    else: # if both is 'None'
        bankinnskudd_kontanter_og_lignende[i] = np.double(0)
        
skyldige_offentlige_avgifter            = data['Skyldige offentlige avgifter'].fillna(0)
leverandorgjeld                         = data['Leverandoergjeld'].fillna(0)
sum_kortsiktig_gjeld                    = data['Sum kortsiktig gjeld'].fillna(0)
sum_inntekter                           = data['Sum inntekter'].fillna(0)
sum_innskutt_egenkapital                = data['Sum innskutt egenkapital'].fillna(0)
sum_egenkapital                         = data['Sum egenkapital'].fillna(0)
sum_eiendeler                           = data['SUM EIENDELER'].fillna(0)
avskrivninger                           = data['Avskrivning paa varige driftsmidler og immaterielle eiendeler'].fillna(0)
ordinaert_resultat_foer_skattekostnad   = data['Ordinaert resultat foer skattekostnad'].fillna(0)
ordinaert_resultat_etter_skattekostnad  = data['Ordinaert resultat etter skattekostnad'].fillna(0)
sum_gjeld                               = data['Sum gjeld'].fillna(0)
arsresultat                             = data['Aarsresultat'].fillna(0)
annen_rentekostnad                      = data['Sum finanskostnader'].fillna(0)
annen_renteinntekt                      = data['Annen renteinntekt'].fillna(0)
utbytte                                 = data['Utbytte'].fillna(0)
opptjent_egenkapital                    = data['Sum opptjent egenkapital'].fillna(0)
gjeld_til_kredittinstitusjoner          = data['Gjeld til kredittinstitusjoner'].fillna(0)
salgsinntekt                            = data['Salgsinntekt'].fillna(0)
lonnskostnad                            = data['Loennskostnad'].fillna(0)
sum_varer                               = data['Sum varer'].fillna(0)
kundefordringer                         = data['Kundefordringer'].fillna(0)
sum_omlopsmidler                        = data['Sum omloepsmidler'].fillna(0)
nedskrivninger                          = data['Nedskrivning av varige driftsmidler og immaterielle eiendeler'].fillna(0)

EBIT    = ordinaert_resultat_foer_skattekostnad + annen_rentekostnad - annen_renteinntekt
EBITDA  = EBIT + avskrivninger + nedskrivninger

############################################################
## Creating variables of Paraschiv et al. (2021)
############################################################
# Paraschiv, Florentina, Markus Schmid, and Ranik Raaen Wahlstrøm. 2021. 
# “Bankruptcy Prediction of Privately Held SMEs Using Feature Selection Methods.”
# Working Paper, Norwegian University of Science and Technology and University of St. Gallen. 
# https://doi.org/10.2139/ssrn.3911490

data['accounts payable / total assets'] = make_ratio(leverandorgjeld,sum_eiendeler)
data['dummy; one if total liability exceeds total assets'] = (sum_gjeld > sum_eiendeler).astype(int)

numerator = sum_kortsiktig_gjeld-bankinnskudd_kontanter_og_lignende
data['(current liabilities - short-term liquidity) / total assets'] = make_ratio(numerator,sum_eiendeler)

data['net income / total assets'] = make_ratio(arsresultat,sum_eiendeler)
data['public taxes payable / total assets'] = make_ratio(skyldige_offentlige_avgifter,sum_eiendeler)
data['interest expenses / total assets'] = make_ratio(annen_rentekostnad,sum_eiendeler)
data['dummy; one if paid-in equity is less than total equity'] = (sum_innskutt_egenkapital < sum_egenkapital).astype(int)

temp = data['age_in_days'].copy()
ind = temp<0
if np.sum(ind)!=0:
    print('For {} observations, age is negative. Setting these to age zero.'.format(np.sum(ind)))
    temp[ind] = 0
data['log(age in years)'] = np.log((temp+1)/365)

data['inventory / current assets'] = make_ratio(sum_varer,sum_omlopsmidler)
data['short-term liquidity / current assets'] = make_ratio(bankinnskudd_kontanter_og_lignende,sum_omlopsmidler)

############################################################
## Creating variables of Altman and Sabato (2007)
############################################################
# Altman, Edward I., and Gabriele Sabato. 2007. 
# “Modelling Credit Risk for SMEs: Evidence from the U.S. Market.” 
# Abacus 43 (3): 332–57. 
# https://doi.org/10.1111/j.1467-6281.2007.00234.x

data['current liabilities / total equity'] = make_ratio(sum_kortsiktig_gjeld,sum_egenkapital)
data['EBITDA / interest expense'] = make_ratio(EBITDA,annen_rentekostnad)
data['EBITDA / total assets'] = make_ratio(EBITDA,sum_eiendeler)
data['retained earnings / total assets'] = make_ratio(opptjent_egenkapital,sum_eiendeler)
data['short-term liquidity / total assets'] = make_ratio(bankinnskudd_kontanter_og_lignende,sum_eiendeler)

############################################################
## Creating variables of Altman (1968)
############################################################
# Altman, Edward I. 1968. 
# “Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy.”
# The Journal of Finance 23 (4): 589–609. 
# https://doi.org/10.2307/2978933

data['EBIT / total assets'] = make_ratio(EBIT,sum_eiendeler)

# This variable is not created here as it is created above for 
# the Altman and Sabato (2007) model
# data['retained earnings / total assets'] = make_ratio(opptjent_egenkapital,sum_eiendeler)

data['sales / total assets'] = make_ratio(salgsinntekt,sum_eiendeler)
data['total equity / total liabilities'] = make_ratio(sum_egenkapital,sum_gjeld)

numerator = sum_omlopsmidler - sum_kortsiktig_gjeld
data['working capital / total assets'] = make_ratio(numerator,sum_eiendeler)


############################################################
## Winzorise ratio variables
############################################################
# The code below winsorize all ratio variables between the 1st and 99th percentiles
# This is to reduce outliers (see Paraschiv et al. (2021))

interval_winsorizing_ratios = [0.01,0.99] # In numbers, so 0.01 = restricting at 1%

# Defining variables that shall be winsorized
ratio_variables_to_winsorize = get_variables_paraschiv_2021()
ratio_variables_to_winsorize = ratio_variables_to_winsorize + get_variables_altman_1968()
ratio_variables_to_winsorize = ratio_variables_to_winsorize + get_variables_altman_and_sabato_2007()
ratio_variables_to_winsorize = list(np.unique(ratio_variables_to_winsorize)) # Making sure all are unique

# Removing variables that are not ratios, as only ratios should be winsorized
ratio_variables_to_winsorize.remove('dummy; one if total liability exceeds total assets')
ratio_variables_to_winsorize.remove('dummy; one if paid-in equity is less than total equity')
ratio_variables_to_winsorize.remove('log(age in years)')

# Winsorizing, per accounting year
# (Before winsorizing, inf and -inf are set to maximum and minimum, respectively, values)
data_winsorized = pd.DataFrame(columns=data.columns)
for regnaar in tqdm(data['regnaar'].unique()):
    data_regnaar = data[data['regnaar']==regnaar].copy()
    for var in ratio_variables_to_winsorize:
        ratio = data_regnaar[var]
        
        # Setting inf and -inf to maximum and minimum, respectively, values
        ratio = ratio.replace(np.inf,np.max(ratio[ratio != np.inf]))
        ratio = ratio.replace(-np.inf,np.max(ratio[ratio != -np.inf]))

        lower = ratio.quantile(interval_winsorizing_ratios[0])
        upper = ratio.quantile(interval_winsorizing_ratios[1])
        
        # Winsorizing
        data_regnaar[var] = ratio.clip(lower=lower, upper=upper)
    data_winsorized = pd.concat([data_winsorized,data_regnaar],axis=0)

# Controlling data
if data.shape[0]!=data_winsorized.shape[0]:
    print('ERROR: not same num rows after winsorizing')
if (data.shape[0])!=data_winsorized.shape[0]:
    print('ERROR: not right num cols after winsorizing')
if np.sum(np.sum(pd.isnull(data_winsorized[ratio_variables_to_winsorize])))!=0:
    print('ERROR: some ratio values are still missing/NULL')

# Controlling data
if data_winsorized.shape[0]!=data.shape[0]:
    print('ERROR when winsorizing')
if data_winsorized.shape[1]!=data.shape[1]:
    print('ERROR when winsorizing')
if np.sum(data_winsorized['orgnr'])!=np.sum(data['orgnr']):
    print('ERROR when winsorizing')
if np.sum(data_winsorized['regnaar'])!=np.sum(data['regnaar']):
    print('ERROR when winsorizing')
if np.sum(data_winsorized['log(age in years)'])!=np.sum(data['log(age in years)']):
    print('ERROR when winsorizing')

data = data_winsorized.copy()

del data_winsorized # Deleting the object to save computer memory


##################################################################
## Bankruptcy prediction
##################################################################
# In this final code, we exemplify bankruptcy prediction. The model
# for bankrutpcy prediction is evaluated out-of-sample with all financial
# statements for the accounting years 2017, 2018, 2019, and 2020, respectively, 
# and trained and evaluated in-sample with all financial statements
# from the three previous years

# Defining test years
test_years = [
    2017,
    2018,
    2019,
    2020,
]

# Defining input variables for the bankruptcy prediction model
# Uncomment the others if you insteadwant to use these
input_variables = get_variables_paraschiv_2021()
# input_variables = get_variables_altman_and_sabato_2007()
# input_variables = get_variables_altman_1968()

# The response variable determining bankruptcy
response_variable = ['bankrupt_fs']

# Making index for result_fold
index_for_results = []
for name in input_variables:
    index_for_results.append(name)
index_for_results.append('intercept')
index_for_results.append('AUC on training set')
index_for_results.append('AUC on test set')
index_for_results.append('Pseudo R-squared')
index_for_results.append('Number observations test set')
index_for_results.append('Number bankrupt in test set')

# Empty data frame for saving results
results_table = pd.DataFrame(index=index_for_results)

# Empty data frame for saving the full data set with predictions, 
# in case you want to use this furter
data_with_predictions = pd.DataFrame(columns=data.columns)

for test_year in tqdm(test_years):

    # Making training and test data
    ind = (data['regnaar']<test_year) & (data['regnaar']>=(test_year-3))
    data_train   = data[ind][input_variables+response_variable]
    data_test    = data[ind][input_variables+response_variable]

    # Shuffeling rows
    # Good practice to avoid bias in results due to the order of observations
    data_train   = data_train.sample(frac=1,random_state=0).reset_index(drop=True)
    data_test    = data_test.sample(frac=1,random_state=0).reset_index(drop=True)

    # Dividing between X (independent variables) and y (dependent variable)
    X_train, y_train, X_test, y_test = prepare_training_and_test_data(data_train,data_test,input_variables,response_variable)

    # Adding intercept
    X_train = sm.add_constant(X_train)
    X_test  = sm.add_constant(X_test)

    # Maximum iterations and numerical method for training 
    # the logistic regression models
    maxiter = 1e6
    method = 'bfgs'

    # Defining and training the logistic regression model
    model = sm.Logit(y_train, X_train)
    model = model.fit(maxiter=maxiter,method=method)
    
    # Printing statistics:
    # print(model.summary())

    # Making predictions
    y_hat_train  = model.predict(X_train)
    y_hat_test   = model.predict(X_test)

    # Making AUC on training data
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train,y_hat_train)
    auc_train = metrics.roc_auc_score(y_train,y_hat_train)

    # Making AUC on test data
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test,y_hat_test)
    auc_test = metrics.roc_auc_score(y_test,y_hat_test)

    # Making result_fold
    result_fold = pd.Series([None]*len(index_for_results), index=index_for_results, name=test_year)
    result_fold['intercept'] = model.params[0]
    for var in input_variables:
        result_fold[var] = model.params[var]
    result_fold['AUC on training set'] = auc_train
    result_fold['AUC on test set'] = auc_test
    result_fold['Pseudo R-squared'] = model.prsquared
    result_fold['Number observations test set'] = y_test.shape[0]
    result_fold['Number bankrupt in test set'] = np.sum(y_test[response_variable]).iloc[0]

    # model.params, used above, gives the coefficient estimates of the 
    # logistic regression model. Note that the z-scores ("t-values") 
    # for the coefficients are found with model.tvalues

    # Adding result_fold to table with all results
    results_table = results_table.assign(temp_name = result_fold)
    results_table.rename(columns={'temp_name':str(result_fold.name)}, inplace=True)

    # Making data with predictions
    data_test['prediction'] = y_hat_test
    data_with_predictions = pd.concat([data_with_predictions,data_test])
    data_with_predictions = data_with_predictions.reset_index(drop=True) # Reset index 

# Saving results to excel:
filename = 'results_of_bankruptcy_prediction_example.xlsx'
results_table.to_excel(filename)
print('Results saved to '+filename)

# data_with_predictions is the original data set, but only for the years
# of which you have created test sets, and with predictions for each
# observation. You may want to save this data, or work further on it

print('Elapset time: {} minutes'.format(np.round((time.time()-time_start)/60,2)))
# For your comparison, the execution of this code on my computer
# took 8.04 minutes