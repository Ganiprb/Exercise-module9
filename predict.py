# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 10:58:17 2021

@author: Gani
"""
  
import pickle
import numpy as np
import pandas as pd

"""
1. handling values null
	- Untuk person_emp_length dan loan_int_rate
	  karena pada data training memilik null maka untuk data input akan dimasukkan ke dalam ketegori binning sendiri meggunakan WOE
	- Untuk kolo numeric lainnya
	  akan direplace dengan 0

2. Outlier handling dilakukan dengan binning WOE 

additional notes:
Mas maaf untuk notebook dr awal saya sudah terlanjur untuk convert jadi script karena lebih mudah runningnya
"""

def preprocess_input(input_data):
    """ preprocess input data """
    input_data = pd.DataFrame.from_dict(input_data, orient='index').T
    target = 'loan_status'
    continuous_features =['person_age', 'person_income','person_emp_length','loan_amnt',
                          'loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
    nominal_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
    ordinal_cat_features = ['loan_grade'] 
    categorical_features=nominal_features
    
    #handling null
    #for i has null in training data
    for i in ['person_emp_length','loan_int_rate']:
        if  i=='person_emp_length':
            input_data.loc[(input_data[i] <= 1), i] =  0.3222
            input_data.loc[(input_data[i] > 1) & (input_data[i] <= 4), i] = 0.0417
            input_data.loc[(input_data[i] > 4), i] = -0.2532
            input_data.loc[(input_data[i].isnull()), i]=0.5048
        if i=='loan_int_rate':
            input_data.loc[(input_data[i] <= 12), i] =  -0.626
            input_data.loc[(input_data[i] > 12) & (input_data[i] <= 15), i] = 0.285
            input_data.loc[(input_data[i] > 15), i] = 1.5995
            input_data.loc[(input_data[i].isnull()), i]=-0.0691
    # input_data[[i]].isnull().any().values[0]==True &
    input_data[['person_age', 'person_income','loan_amnt',
        'loan_percent_income','cb_person_cred_hist_length']] = input_data[['person_age', 'person_income',
                                                                   'loan_amnt','loan_percent_income',
                                                                   'cb_person_cred_hist_length']].fillna(value=0)
    #numeric
    #woe
    for i in ['person_age', 'person_income','loan_amnt',
        'loan_percent_income','cb_person_cred_hist_length']:
        if  i=='person_age':
            input_data.loc[(input_data[i] <= 22), i] =  0.224
            input_data.loc[(input_data[i] > 22) & (input_data[i] <= 25), i] = -0.002
            input_data.loc[(input_data[i] > 25) & (input_data[i] <= 30), i] = -0.0574
            input_data.loc[(input_data[i] > 30), i]= -0.0789
        if  i=='person_income':
            input_data.loc[(input_data[i] <= 18000), i] =  2.704
            input_data.loc[(input_data[i] > 18000) & (input_data[i] <= 30000), i] = 0.7703
            input_data.loc[(input_data[i] > 30000) & (input_data[i] <= 50000), i] = 0.2095
            input_data.loc[(input_data[i] > 50000), i]= -0.5309
        if  i=='loan_amnt':
            input_data.loc[(input_data[i] <= 6000), i] =  -0.1574
            input_data.loc[(input_data[i] > 6000) & (input_data[i] <= 13000), i] = -0.1524
            input_data.loc[(input_data[i] > 13000) & (input_data[i] <= 15000), i] = 0.2447
            input_data.loc[(input_data[i] > 15000), i]= 0.5374
        if  i=='loan_percent_income':
            input_data.loc[(input_data[i] <= 0.1), i] =  -0.7425
            input_data.loc[(input_data[i] > 0.1) & (input_data[i] <= 0.25), i] = -0.3711
            input_data.loc[(input_data[i] > 0.25) & (input_data[i] <= 0.4), i] =  1.2113
            input_data.loc[(input_data[i] > 0.4), i]= 2.3283
        if  i=='cb_person_cred_hist_length':
            input_data.loc[(input_data[i] <= 2.0), i] =  0.1039
            input_data.loc[(input_data[i] > 2.0) & (input_data[i] <= 3.0), i] = 0.0291
            input_data.loc[(input_data[i] > 3.0) & (input_data[i] <= 6.0), i] = -0.0076
            input_data.loc[(input_data[i] > 6.0), i]= -0.0681
    
    input_data['loan_grade_label_encode'] = np.where(input_data['loan_grade'].isin(['A']), 1, 
                      np.where(input_data['loan_grade'].isin(['B']), 2,
                    np.where(input_data['loan_grade'].isin(['C']), 3,
                    np.where(input_data['loan_grade'].isin(['D']), 4,
                    np.where(input_data['loan_grade'].isin(['E']), 5,
                    np.where(input_data['loan_grade'].isin(['F']), 6,7))))))
           
    input_data['person_home_ownership_OTHER']=np.where(input_data['person_home_ownership'].isin(['OTHER']), 
                                                       1,0)
    input_data['person_home_ownership_OWN']=np.where(input_data['person_home_ownership'].isin(['OWN']), 
                                                       1,0)
    input_data['person_home_ownership_RENT']=np.where(input_data['person_home_ownership'].isin(['RENT']), 
                                                       1,0)
    input_data['loan_intent_EDUCATION']=np.where(input_data['loan_intent'].isin(['EDUCATION']), 
                                                       1,0)
    input_data['loan_intent_HOMEIMPROVEMENT']=np.where(input_data['loan_intent'].isin(['HOMEIMPROVEMENT']), 
                                                       1,0)
    input_data['loan_intent_MEDICAL']=np.where(input_data['loan_intent'].isin(['MEDICAL']), 
                                                       1,0)
    input_data['loan_intent_PERSONAL']=np.where(input_data['loan_intent'].isin(['PERSONAL']), 
                                                       1,0)
    input_data['loan_intent_VENTURE']=np.where(input_data['loan_intent'].isin(['VENTURE']), 
                                                       1,0)
    input_data['cb_person_default_on_file_Y']=np.where(input_data['cb_person_default_on_file'].isin(['Y']), 
                                                       1,0)
    input_data['loan_grade_label_encode']=np.where(input_data['cb_person_default_on_file'].isin(['Y']), 
                                                       1,0)
    input_data = input_data.rename({'person_age': 'person_age_WOE', 
                                    'person_income': 'person_income_WOE',
                                    'person_emp_length': 'person_emp_length_WOE', 
                                    'loan_amnt': 'loan_amnt_WOE',
                                    'loan_int_rate': 'loan_int_rate_WOE', 
                                    'loan_percent_income': 'loan_percent_income_WOE', 
                                    'cb_person_cred_hist_length': 'cb_person_cred_hist_length_WOE'}, axis=1)
    
    input_data=input_data[['person_home_ownership_OTHER', 'person_home_ownership_OWN',
           'person_home_ownership_RENT', 'loan_intent_EDUCATION',
           'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
           'loan_intent_PERSONAL', 'loan_intent_VENTURE',
           'cb_person_default_on_file_Y', 'loan_grade_label_encode',
           'person_age_WOE', 'person_income_WOE', 'person_emp_length_WOE',
           'loan_amnt_WOE', 'loan_int_rate_WOE', 'loan_percent_income_WOE',
           'cb_person_cred_hist_length_WOE']]
    
    return input_data


def make_predictions(input_data):
    """ function to make final prediction using pipeline """

    with open('M-LR-exercise.pkl', 'rb') as f:
        model = pickle.load(f)
    input_data = preprocess_input(input_data).T.replace({
        None: np.nan,
        "null": np.nan,
        "": np.nan
        })
    input_data=input_data.T
    input_data=input_data[['person_age_WOE', 'person_home_ownership_OTHER',
       'cb_person_cred_hist_length_WOE', 'person_emp_length_WOE',
       'loan_amnt_WOE', 'loan_int_rate_WOE', 'loan_intent_MEDICAL',
       'cb_person_default_on_file_Y']]

    # model prediction
    result = model.predict_proba(input_data)[:, 1]
    return result 

# df=pd.read_csv('credit_risk_dataset.csv')
# df_sample=df.sample(1).drop('loan_status',axis=1)
# df_sample=df_sample.to_dict('records')[0]
# input_dict=df_sample.copy()

