#!/usr/bin/env python
# coding: utf-8

# # Analysis Chalenge

# ## Import Module

# In[166]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from math import log, ceil 
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

#Import csv
df=pd.read_csv('credit_risk_dataset.csv')

#check duplicates
df[df.duplicated()].sort_values('person_age')

#drop duplicates
data=df.drop_duplicates().copy()

# ### Metadata definition
target = 'loan_status'

continuous_features =['person_age', 'person_income','person_emp_length','loan_amnt',
                      'loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
nominal_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
discrete_features = []
ordinal_cat_features = ['loan_grade'] 
ordinal_num_features = []

categorical_features=ordinal_cat_features+ordinal_num_features+nominal_features

# ### Handling Nan Columns
# List Columns with NaN
list_column_nan_data = data[data.columns[data.isnull().any()]].isnull().sum() * 100 / data.shape[0]

# No need for drop columns
# ### Filling Nan values
list_fill_na = list_column_nan_data[list_column_nan_data<75].index.tolist()

# #### Continous feature
list_fill_na_continuous = list(set(continuous_features).intersection(set(list_fill_na)))

# n_cols = 4
# n_rows = ceil(len(list_fill_na_continuous)/n_cols)
# counter = 1

# fig = plt.figure(figsize=(20,5))
# for col in list_fill_na_continuous:
#     plt.subplot(n_rows, n_cols, counter)
#     plt.xlabel(col)
#     g = plt.hist(data[col], bins=20)
    
#     counter += 1

# plt.show();

# for i in list_fill_na_continuous:
#     data[f'{i}_nan'] = data[i].fillna(data[i].median())


# ### Dummy variable
# #### Nominal
data = pd.concat([data,pd.get_dummies(data[nominal_features],
                               drop_first=True)], axis=1).reset_index(drop=True)

# ##### Using labeling
# #### Ordinal
data['loan_grade_label_encode'] = np.where(data['loan_grade'].isin(['A']), 1, 
                  np.where(data['loan_grade'].isin(['B']), 2,
                np.where(data['loan_grade'].isin(['C']), 3,
                np.where(data['loan_grade'].isin(['D']), 4,
                np.where(data['loan_grade'].isin(['E']), 5,
                np.where(data['loan_grade'].isin(['F']), 6,7))))))

# #### Continuous
# ### WOE
# for i in continuous_features:
#     g = sns.FacetGrid(data[data[target].notnull()], col=target)
#     g.map(plt.hist, i, bins=20)

feature = 'person_age'
binning=[-float("inf"),22, 25,30, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[0.224, -0.002, -0.0574,-0.0789])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

feature = 'person_income'
binning=[-float("inf"),18000,30000,50000, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[2.704, 0.77, 0.209,-0.53])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

feature = 'person_emp_length'
binning=[-float("inf"),1,4, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[0.3222,0.0417,-0.2532])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0.504)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

feature = 'loan_amnt'
binning=[-float("inf"),6000, 13000,15000, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[-0.1574,-0.1524,0.2447,0.5374])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

feature = 'loan_int_rate'
binning=[-float("inf"),12, 15, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[-0.626,0.285,1.599])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', -0.069)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

feature = 'loan_percent_income'
binning=[-float("inf"),0.1, 0.25,0.4, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[-0.742,-0.371,1.211,2.328])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

feature = 'cb_person_cred_hist_length'
binning=[-float("inf"),2, 3,6, float("inf")]
bin_feature = pd.cut(data[feature], bins=binning).values.add_categories('Nan').fillna('Nan')
data_woe_iv = (pd.crosstab(bin_feature,data[target],normalize='columns')
             .assign(woe=lambda datax: np.log(datax[1] / datax[0]))
             .assign(iv=lambda datax: np.sum(datax['woe']*
                                           (datax[1]-datax[0]))))
data[f'{feature}_WOE'] = pd.cut(data[feature], bins=binning, labels=[-0.1039,0.029,-0.00763,-0.068])
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].values.add_categories('Nan').fillna('Nan') 
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].replace('Nan', 0)
data[f'{feature}_WOE'] = data[f'{feature}_WOE'].astype(float)

#Splitt
def sssplit(df,label,random_state=42,test_size=0.3):
    X = df.drop(label, axis = 1)
    y = df[label]
    sss= StratifiedShuffleSplit(n_splits = 5, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X,y):
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    #gabung dataframe
    dfsmtr=pd.concat([original_Xtrain,original_ytrain],axis=1)
    dfsmts=pd.concat([original_Xtest,original_ytest],axis=1)
    return dfsmtr,dfsmts
train,test_hold=sssplit(data,target,test_size=0.2)
hold,test=sssplit(test_hold,target,test_size=0.25)

# ## Feature selection
# ### Define set features
#define woe
train=train[['person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE',
       'cb_person_default_on_file_Y', 'loan_grade_label_encode',
       'person_age_WOE', 'person_income_WOE', 'person_emp_length_WOE',
       'loan_amnt_WOE', 'loan_int_rate_WOE', 'loan_percent_income_WOE',
       'cb_person_cred_hist_length_WOE','loan_status']]
test=test[['person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE',
       'cb_person_default_on_file_Y', 'loan_grade_label_encode',
       'person_age_WOE', 'person_income_WOE', 'person_emp_length_WOE',
       'loan_amnt_WOE', 'loan_int_rate_WOE', 'loan_percent_income_WOE',
       'cb_person_cred_hist_length_WOE','loan_status']]
hold=hold[['person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE',
       'cb_person_default_on_file_Y', 'loan_grade_label_encode',
       'person_age_WOE', 'person_income_WOE', 'person_emp_length_WOE',
       'loan_amnt_WOE', 'loan_int_rate_WOE', 'loan_percent_income_WOE',
       'cb_person_cred_hist_length_WOE','loan_status']]


# ### L1 regularization
selector = SelectFromModel(estimator=LogisticRegression(penalty='l1', C=1, solver='liblinear', max_iter=50000))
selector.fit(train.drop(target,axis=1), train[target])
feature_importance_df = pd.DataFrame(
    {
        'feature': train.drop(target,axis=1).columns,
        'importance': abs(selector.estimator_.coef_)[0],
        'selected': selector.get_support()
    }
)
select_from_model = feature_importance_df[feature_importance_df['selected']].sort_values(by='importance', ascending=False)
select_from_model=select_from_model['feature'][(select_from_model['importance']>0.1)&
                                               (select_from_model['importance']<0.5)].values.tolist()
train=train[select_from_model+[target]]
test=test[select_from_model+[target]]
hold=hold[select_from_model+[target]]

# ## Modelling
# ### Target distribution
# fig = plt.figure(figsize=(8,4))
# plt.xlabel('Survived Distribution')
# plt.hist(data[target]);



def print_evaluate(true, predicted):
    auc = metrics.roc_auc_score(true, predicted)
    accuracy = metrics.accuracy_score(true, predicted)
    print('Accuracy:', accuracy)
    print('AUC:', auc)
    print('__________________________________')
    
def evaluate(true, predicted):
    auc = metrics.roc_auc_score(true, predicted)
    accuracy = metrics.accuracy_score(true, predicted)
    return accuracy, auc

# ### Logres
def logres(data_train,data_test,label,modtype):
    param_dictLR = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear'],
    # 'classifier__random_state' : [42],
    'classifier__class_weight': [{0:1000,1:100},{0:1000,1:10}, {0:1000,1:1.0}, 
     {0:500,1:1.0}, {0:400,1:1.0}, {0:300,1:1.0}, {0:200,1:1.0}, 
     {0:150,1:1.0}, {0:100,1:1.0}, {0:99,1:1.0}, {0:10,1:1.0}, 
     {0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100}, 
     {0:0.001,1:1.0}, {0:0.005,1:1.0}, {0:1.0,1:1.0}, 
     {0:1.0,1:0.1}, {0:10,1:0.1}, {0:100,1:0.1}, 
     {0:10,1:0.01}, {0:1.0,1:0.01}, {0:1.0,1:0.001}, {0:1.0,1:0.005}, 
     {0:1.0,1:10}, {0:1.0,1:99}, {0:1.0,1:100}, {0:1.0,1:150}, 
     {0:1.0,1:200}, {0:1.0,1:300},{0:1.0,1:400},{0:1.0,1:500}, 
     {0:1.0,1:1000}, {0:10,1:1000},{0:100,1:1000},'balanced',None]
    }]
    scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall':make_scorer(recall_score),
           'f1-score':make_scorer(f1_score),
           'roc-auc':make_scorer(roc_auc_score)}
    pipe = Pipeline([('classifier' , LogisticRegression())])
    clf = RandomizedSearchCV(pipe, param_distributions = param_dictLR, scoring=scoring, refit='accuracy', cv = 10, n_jobs=-1)
    best_clf = clf.fit(data_train.drop(label,axis=1), data_train[label])
    probs = best_clf.predict_proba(data_test.drop(label,axis=1))
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(data_test[label], preds)
    roc_auc = metrics.auc(fpr, tpr)

    #ROC CURVE
    plt.figure()
    plt.plot(fpr, tpr, label='RandomizedSearchCV (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic '+modtype)
    plt.legend(loc="lower right")
    plt.show()
    
    classes = best_clf.predict(data_test.drop(label,axis=1))
    #EVALUATION
    classes = best_clf.predict(data_test.drop(label,axis=1))
    accuracy = metrics.accuracy_score(data_test[label],classes)
#    balanced_accuracy = metrics.balanced_accuracy_score(data_test[label],classes)
    precision = metrics.precision_score(data_test[label],classes)
#    average_precision = metrics.average_precision_score(data_test[label],classes)
    f1_score_var = metrics.f1_score(data_test[label],classes)
    recall = metrics.recall_score(data_test[label],classes)
    print(metrics.classification_report(data_test[label],classes))

    #PR CURVE
    data_test_predprob = best_clf.predict_proba(data_test.drop(label,axis=1))[:,1]    
    lr_precision, lr_recall, _ = precision_recall_curve(data_test[label], data_test_predprob)
    lr_auc =auc(lr_recall, lr_precision)
    print(modtype+': Precision-Recall auc=%.3f' % (lr_auc))
    no_skill = len(data_test[label][data_test[label]==1]) / len(data_test[label])
    plt.figure()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label=modtype)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve '+modtype)
    plt.legend()
    
    df_cr = pd.DataFrame(metrics.classification_report(data_test[label],classes,output_dict=True)).transpose()
    label_predict = (data_test_predprob >= 0.5).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(data_test[label], label_predict).ravel()
    log=pd.DataFrame({'Model':modtype,
                 'accuracy':accuracy,
#                 'balanced_accuracy':balanced_accuracy,
#                 'average_precision':average_precision,
                 'precision':precision,
                 'recall':recall,
                'f1_score':f1_score_var,
                 'roc_auc':roc_auc,
                 'tn':tn,
                 'fp':fp,
                 'fn':fn,
                 'tp':tp},index=[0])
    
    return best_clf
logres_model=logres(train,test,target,'logres')

print('for logres')
test_pred = logres_model.predict(test.drop(target,axis=1))
train_pred = logres_model.predict(train.drop(target,axis=1))
hold_pred = logres_model.predict(hold.drop(target,axis=1))

print('Test set evaluation:\n_____________________________________')
print_evaluate(test[target], test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(train[target], train_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(hold[target], hold_pred)

index = pd.MultiIndex.from_product([['Train', 'Valid','Holdout'], ['Accuracy', 'AUC']])
result_logreg_opt = pd.DataFrame([pd.DataFrame({'Train' : list(evaluate(train[target], train_pred)),
             'Valid' : list(evaluate(test[target], test_pred)),
            'Holdout' : list(evaluate(hold[target], hold_pred))}).unstack().values], columns=index)
result_logreg_opt.insert(loc=0, column='Model', value = 'Log Reg')

import pickle 

MODELNAME = 'M-LR-exercise.pkl'
# PREPROCESSNAME = 'FE-IMP-WOE-RS-1.0.1.pkl'

# final_model = rndm_s.best_estimator_.named_steps['model']
# final_pipe = rndm_s.best_estimator_.named_steps['preprocess']

with open(MODELNAME, 'wb') as f: # save model
    pickle.dump(logres_model, f)

# with open(PREPROCESSNAME, 'wb') as f: # save pipeline
#     pickle.dump(final_pipe, f)

# # calculate importances based on coefficients.
# importances = abs(logres_model.best_estimator_.named_steps['classifier'].coef_[0])
# importances = 100.0 * (importances / importances.max())
# # sort 
# indices = np.argsort(importances)[::-1]

# # Rearrange feature names so they match the sorted feature importances
# names = [train.columns[i] for i in indices]

# # visualize
# plt.figure(figsize = (12, 5))
# sns.set_style("whitegrid")
# chart = sns.barplot(x = names, y = importances[indices])
# plt.xticks(
#     rotation=45, 
#     horizontalalignment='right',
#     fontweight='light'  
# )
# plt.title('Logistic regression. Feature importance')
# plt.tight_layout()

