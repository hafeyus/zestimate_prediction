import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import re
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# gradient boosting machines
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (cross_val_score, train_test_split)
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool, cv

# hyperparameter tuning
from bayes_opt import BayesianOptimization

# to print all the outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# functions
def reduce_mem_usage(df, verbose=True):
    '''
    reduce memory usage. This function is taken from FabienDaniel from another Kaggle competition
    '''
    df.fillna(np.nan, inplace=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.\
                      format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_prop(properties):
    # modify properties dataset, handle in null values
    prop = properties.copy()

    # drop columns of which 99% of values are null
    drop_thresh = prop.shape[0] * 0.01
    prop.dropna(thresh=drop_thresh, axis=1, inplace=True)

    # drop columns if the number of unique values is one
    unique_mask = (prop.nunique() == 1)
    prop.drop(prop.columns[unique_mask], axis=1, inplace=True)
    
    # propertycountylandusecode column has mix of float and str
    prop['propertycountylandusecode'] = prop['propertycountylandusecode'].astype('str')

    # fill in null values
    '''
    Zillow team has built model to predict the sale price of the property and the target they give us is the error
    between their predicted value and the actual value. Our goal is to predict the error they have made, so I believe
    it is better to fill in the null with values that deviate from the actual values instead of filling in median just
    to mimic the situation Zillow team faced when they are doing prediction.
    '''
    for col in prop.columns:
        if prop[col].dtype == 'object':
            prop[col].fillna('NULL', inplace=True)
        else:
            prop[col].fillna(-999, inplace=True)
            
    # put create feature function here if needed
            
    return prop

def get_train(train_set):
    
    train = train_set.copy()
    
    train['month'] = train['transactiondate'].dt.month
    train['month_since_2016'] = (train['transactiondate'].dt.year - 2016) * 12 + train['month']
    train['quarter'] = (train['transactiondate'].dt.month - 1) // 3 + 1
    train.drop('transactiondate', axis=1, inplace=True)
    
    return train

def get_cat_feats(dataset):
    '''
    get categorical feature columns of dataset
    '''
    cat_thresh = 1000    # categorical features if number of unique values under cat_thresh
    cat_feats = ['propertyzoningdesc']
    for col in dataset.columns:
        not_count = (re.search('.*(sqft|cnt|nbr|number|since|year).*', col) is None)
        if not_count and dataset[col].nunique() <= cat_thresh:
            cat_feats.append(col)
    return cat_feats

def convert_cat_columns(dataset, cat_feats):
    '''
    convert categorical feature column data type to string as required by catboost
    '''
    for col in cat_feats:
        if dataset[col].dtype == 'object':
            dataset[col] = dataset[col].astype('str')
        else:
            dataset[col] = dataset[col].astype('int')
        
    return dataset

def bayes_param_opt(X, y, cat_feats, init_points=15, n_iter=25, n_folds=5, random_seed=1, \
                        output_process=False):
    
    # prepare dataset
    dtrain = Pool(data=X, label=y, cat_features= cat_feats)
    
    # define objective function to minimize
    def cat_cv(bagging_fraction, max_depth, lambda_l2, \
                 num_iterations, learning_rate):
        
        params = {'loss_function':'MAE', 'random_seed':1, 'eval_metric':'MAE',
                 'use_best_model':True, 'early_stopping_rounds':20, 'bootstrap_type':'Bernoulli'}
        params['n_estimators'] = int(num_iterations)
        params['learning_rate'] = max(learning_rate, 0)
        params['subsample'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['l2_leaf_reg'] = max(lambda_l2, 0)
#         lgbr = LGBMRegressor(params)
#         cv_result = cross_val_score(estimator=lgbr, X=X, y=y, cv=n_folds, scoring='neg_mean_absolute_error')
        cv_result = cv(dtrain=dtrain, params=params, nfold=n_folds, seed=1, shuffle=True)
        return - cv_result['test-MAE-mean'].min()
    
    optimizer = BayesianOptimization(cat_cv, {
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 15),
                                            'lambda_l2': (0, 5),
                                            'num_iterations':(100, 2000),
                                            'learning_rate':(0.025,0.1),
                                            }, random_state=0)
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    return optimizer.max


if __name__ == '__main__':
	# loading dataset
	train_16 = pd.read_csv('./data/train_2016_v2.csv', parse_dates=['transactiondate'])
	train_17 = pd.read_csv('./data/train_2017.csv', parse_dates=['transactiondate'])
	properties_16 = pd.read_csv('./data/properties_2016.csv')
	properties_17 = pd.read_csv('./data/properties_2017.csv')

	# prepare training set
	prop_16 = get_prop(properties_16)
	prop_17 = get_prop(properties_17)
	cat_feats = get_cat_feats(prop_16)

	transac_16 = get_train(train_16)
	df_16 = transac_16.merge(prop_16, how='left', on='parcelid')

	transac_17 = get_train(train_17)
	df_17 = transac_17.merge(prop_17, how='left', on='parcelid')

	train = pd.concat([df_16, df_17], axis=0)

	train = convert_cat_columns(train, cat_feats)

	# get train feature columns
	train_feats = [col for col in train if col not in {'parcelid', 'logerror'}]

	# prepare x and y
	x = train[train_feats]
	y = train['logerror']

	# prepare training pool
	dtrain = Pool(data=x, label=y, cat_features=cat_feats_16)

	# training model
	regressor = CatBoostRegressor(subsample=0.8885796692189906,
								  l2_leaf_reg=4.469523358625079,
								  learning_rate=0.07802234255790369,
								  max_depth=5,
								  iterations=1887,
								  loss_function='MAE', 
								  random_seed=1,
								  eval_metric='MAE',
								  bootstrap_type='Bernoulli')

	regressor.fit(X=dtrain)  

	# predict 2016 result
	test_16 = prop_16.copy()
	test_16.insert(0, 'month', 0)
	test_16.insert(1, 'month_since_2016', 0)
	test_16.insert(2, 'quarter', 4)
	test_16 = convert_cat_columns(test_16, cat_feats)
	test_month = [(10, 201610), (11, 201611), (12, 201612)]
	result_16 = pd.DataFrame()
	result_16['parcelid'] = prop_16['parcelid']
	for month, col in test_month:
	    test_16['month'] = month
	    test_16['month_since_2016'] = month
	    test_pool = Pool(data=test_16[train_feats], cat_features=cat_feats)
	    temp = regressor.predict(test_pool)
	    result_16[col] = temp

    # predict 2017 result
	test_17 = prop_17.copy()
	test_17.insert(0, 'month', 0)
	test_17.insert(1, 'month_since_2016', 0)
	test_17.insert(2, 'quarter', 4)
	test_17 = convert_cat_columns(test_17, cat_feats)
	test_month = [(10, 201710), (11, 201711), (12, 201712)]
	result_17 = pd.DataFrame()
	result_17['parcelid'] = prop_17['parcelid']
	for month, col in test_month:
	    test_17['month'] = month
	    test_17['month_since_2016'] = month + 12
	    test_pool = Pool(data=test_17[train_feats], cat_features=cat_feats)
	    temp = regressor.predict(test_pool)
	    result_17[col] = temp

    # prepare submission file
    submission = result_16.merge(result_17, on='parcelid', how='left')
	submission.iloc[:,1:] = submission.iloc[:, 1:].round(4)
	submission.to_csv('submission_1617.csv', index=False)

	# private leaderboard 0.07510. 186th place / 3775 teams. Top 4.9%