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

def create_features(dataset):
    
    '''
    create more features for properties table
    '''
    
    #years since the property is built
    dataset['property_life'] = 2018 - dataset['yearbuilt']

    #error in calculation of the finished living area of home
    dataset['living_error'] = dataset['calculatedfinishedsquarefeet']/dataset['finishedsquarefeet12']

    #proportion of living area
    dataset['living_lot_ratio'] = dataset['calculatedfinishedsquarefeet']/dataset['lotsizesquarefeet']
    dataset['living_totoal_ratio'] = dataset['finishedsquarefeet12']/dataset['finishedsquarefeet15']

    #Amout of extra space
    dataset['lot_extra'] = dataset['lotsizesquarefeet'] - dataset['calculatedfinishedsquarefeet'] 
    dataset['extra_space'] = dataset['finishedsquarefeet15'] - dataset['finishedsquarefeet12'] 

    #Total number of rooms
    dataset['total_rooms'] = dataset['bathroomcnt'] + dataset['bedroomcnt']

    #Ratio of the built structure value to land area
    dataset['structure_land_val_ratio'] = dataset['structuretaxvaluedollarcnt']/dataset['landtaxvaluedollarcnt']

    #Ratio of tax of property over parcel
    dataset['inverse_tax_rate'] = dataset['taxvaluedollarcnt']/dataset['taxamount']

    #Length of time since unpaid taxes
    dataset['past_due_tax_years'] = 2018 - dataset['taxdelinquencyyear']
    
    #Number of properties in the zip
    zip_count = dataset['regionidzip'].value_counts().to_dict()
    dataset['zip_cnt'] = dataset['regionidzip'].map(zip_count)

    #Number of properties in the city
    city_count = dataset['regionidcity'].value_counts().to_dict()
    dataset['city_cnt'] = dataset['regionidcity'].map(city_count)

    #Number of properties in the city
    region_count = dataset['regionidcounty'].value_counts().to_dict()
    dataset['county_cnt'] = dataset['regionidcounty'].map(region_count)
    
    #polnomials of the variable
    dataset["structuretaxvaluedollarcnt_2"] = dataset["structuretaxvaluedollarcnt"] ** 2
    dataset["structuretaxvaluedollarcnt_3"] = dataset["structuretaxvaluedollarcnt"] ** 3

    #Average structuretaxvaluedollarcnt by city
    group = dataset.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    dataset['avg_structuretaxvaluedollarcnt'] = dataset['regionidcity'].map(group)

    #Deviation away from average
    dataset['dev_structuretaxvaluedollarcnt'] = \
    abs((dataset['structuretaxvaluedollarcnt'] - \
         dataset['avg_structuretaxvaluedollarcnt']))/dataset['avg_structuretaxvaluedollarcnt']
    
    #area per unit if there are multiple units
    dataset['area_per_unit'] = dataset['finishedsquarefeet15'] / dataset['unitcnt']
    
    #living area per floor
    dataset['living_per_floor'] = dataset['finishedsquarefeet12'] / dataset['numberofstories']
    
    return dataset

def get_cat_feats(dataset):
    '''
    get categorical feature columns of dataset
    '''
    cat_thresh = 1000    # categorical features if number of unique values under cat_thresh
    cat_feats = ['propertyzoningdesc']
    for col in dataset.columns:
        not_count = (re.search('.*(sqft|cnt|nbr|number|since|year|life|total).*', col) is None)
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


# feature selection class
# the idea is from Olivier: https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
class FeatureSelection(object):
    
    '''
    FeatureSelection uses target permutation method to compare the features importances for actual train set and 
    null importance features for fake train set (training data unchanged, but target shuffled). Features of which the
    actual importance value does not fall too close with the fake importance value are considered valid features.
    '''
    
    def __init__(self, df, categorical_feats):
        self.df = df
        self.cat_feats = categorical_feats
        self.cat_feats_set = set(categorical_feats)
        self.target = df['logerror']
        self.train_features = [col for col in df.columns if col not in {'parcelid', 'logerror'}]
        self.scores = None
        
    def get_feature_importances(self, shuffle, seed=1):
        
        data=self.df
        categorical_feats=self.cat_feats

        y = self.target.copy()
        train_features = self.train_features

        # Shuffle target if required
        if shuffle:
            y = data['logerror'].copy().sample(frac=1.0)

        # Fit LightGBM 
        dtrain = Pool(data=data[train_features], label=y, cat_features=categorical_feats)
        regressor = CatBoostRegressor(subsample=0.8885796692189906,
                                      l2_leaf_reg=4.469523358625079,
                                      learning_rate=0.07802234255790369,
                                      max_depth=5,
                                      iterations=300,
                                      loss_function='MAE', 
                                      random_seed=1,
                                      eval_metric='MAE',
                                      bootstrap_type='Bernoulli')

        regressor.fit(X=dtrain)    

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance"] = regressor.feature_importances_

        return imp_df
    
    def get_imp(self, fake):
        
        '''
        get importance dataframe. if fake is True, will return the null importance df based on permuted labels.
        '''
        
        imp_df = pd.DataFrame()
        nb_runs = 100
        
        for i in range(nb_runs):
            # Get current run importances
            temp_df = self.get_feature_importances(shuffle=fake)
            temp_df['run'] = i + 1 
            # Concat the latest importances with the old ones
            imp_df = pd.concat([imp_df, temp_df], axis=0)
            
        return imp_df
    
    def run_imp(self):
        self.act_imp_df = self.get_imp(fake=False)
        self.null_imp_df = self.get_imp(fake=True)
        
    def display_distributions(self,  feature_):
        
        '''
        plot the distribution of feature importance of feature_ based on actual feature importance and
        null feature importance. For feature that is informative, the actual importance value should be much larger
        than the null impotance value.
        '''
        # load feature importance df
        actual_imp_df_=self.act_imp_df 
        null_imp_df_=self.null_imp_df
        
        plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 1)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance'].values, bins=30, \
                    label='Null importances')
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance'].mean(), 
                   ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Feature Importance of %s' % feature_.upper(), fontweight='bold')
#         plt.xlabel('Null Importance Distribution for %s ' % feature_.upper())
        
    
    def get_scores(self):
        
        actual_imp_df_= self.act_imp_df 
        null_imp_df_ = self.null_imp_df
        correlation_scores = []
        
        for _f in actual_imp_df_['feature'].unique():
            f_null_imps = null_imp_df_.loc[null_imp_df_['feature'] == _f, 'importance'].values
            f_act_imps = actual_imp_df_.loc[actual_imp_df_['feature'] == _f, 'importance'].values
            score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            
            correlation_scores.append((_f, score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'score'])
        self.scores = corr_scores_df.copy()
        
    def cv_score(self, train_features, cat_feats, df=None, target=None):
        
        if not df:
            df=self.df
            target=self.target
            
        dtrain = Pool(data=df[train_features], label=target, cat_features=cat_feats)
        
        opt_cat_params = {
              'subsample': 0.8885796692189906,
              'l2_leaf_reg': 4.469523358625079,
              'learning_rate': 0.07802234255790369,
              'max_depth': 5,
              'iterations': 2000,    # set to large number, but trees will stop growing by overfitting detector
              'loss_function':'MAE', 
              'random_seed':1,
              'eval_metric':'MAE',
              'use_best_model':True, 
              'early_stopping_rounds':50, 
              'bootstrap_type':'Bernoulli'
                          }
    
    
        # Fit the model
        scores = cv(dtrain=dtrain, params=opt_cat_params, nfold=5, seed=1, shuffle=True)
        # Return the last mean / std values 
        return scores['test-MAE-mean'].min()
    
    def choose_threshold(self):
        
        imp_scores = self.scores
        
        threshold_score = []
        features = {}
        for threshold in np.arange(0,100,10):
            good_feats = imp_scores.loc[imp_scores['score'] > threshold, 'feature']
            good_cat_feats = [_f for _f in good_feats if _f in self.cat_feats_set]
            
            feature_dict = {'good_feats':good_feats, 'good_cat_feats':good_cat_feats}
            features[threshold] = feature_dict
            
            print('Results for threshold %3d' % threshold)
            results = self.cv_score(train_features=good_feats, cat_feats=good_cat_feats)
            print('score: {}'.format(results))

            threshold_score.append([threshold, results])
            
        threshold_df = pd.DataFrame(data=threshold_score, columns=['threshold', 'MAE-mean'])
        threshold_df.sort_values(by='MAE-mean', ascending=True, inplace=True)
        
        return threshold_df, features


if __name__ == '__main__':
	# loading dataset
	train_16 = pd.read_csv('./data/train_2016_v2.csv', parse_dates=['transactiondate'])
	train_17 = pd.read_csv('./data/train_2017.csv', parse_dates=['transactiondate'])
	properties_16 = pd.read_csv('./data/properties_2016.csv')
	properties_17 = pd.read_csv('./data/properties_2017.csv')

	# prepare training set
	prop_16 = get_prop(properties_16)
	prop_16 = create_features(prop_16)
	prop_17 = get_prop(properties_17)
	prop_17 = create_features(prop_17)

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


	# feature selection
	ctb_fs = FeatureSelection(train, cat_feats)
	ctb_fs.run_imp()
	ctb_fs.act_imp_df.to_csv('act_imp_df.csv', index=False)
	ctb_fs.null_imp_df.to_csv('null_imp_df.csv', index=False)
	ctb_fs.get_scores()
	ctb_fs.choose_threshold()

	# from the feature selection result, filtering the feature with threshold=30 yileds better cv score
	score_df = ctb_fs.scores
	good_feats = score_df.loc[score_df.score > 30, 'feature']
	good_cat_feats = [col for col in good_feats if col in cat_feats]
	x = train[good_feats]
	y = train['logerror']


	# train model
	dtrain = Pool(data=x, label=y, cat_features=good_cat_feats)
	regressor = CatBoostRegressor(subsample=0.8885796692189906,
	  l2_leaf_reg=4.469523358625079,
	  learning_rate=0.07802234255790369,
	  max_depth=5,
	  iterations=1400,
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

	submission = result_16.merge(result_17, on='parcelid', how='left')
	submission.iloc[:,1:] = submission.iloc[:, 1:].round(4)
	submission.to_csv('submission_fs.csv', index=False)

	# private leaderboard 0.07508. 155th place / 3775 teams. Top 4.1%