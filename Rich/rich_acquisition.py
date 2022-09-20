
import pandas as pd
import env
import os
import math 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
from itertools import combinations ,product
from sympy import symbols
from numpy import arange ,percentile

import scipy.stats as stats
warnings.filterwarnings('ignore')

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
def get_zillow_data():






    '''
    aquire the zillow data utilizing the query defined earlier in this wrangle file.
    will read in cached data from any present "zillow.csv" present in the current directory.
    first-read data will be saved as "zillow.csv" following query.

    parameters: none

    '''



    query = '''
    select prop.parcelid
        , pred.logerror
        , bathroomcnt
        , bedroomcnt
        , calculatedfinishedsquarefeet
        , fips
        , latitude
        , longitude
        , lotsizesquarefeet
        , regionidcity
        , regionidcounty
        , regionidzip
        , yearbuilt
        , structuretaxvaluedollarcnt
        , taxvaluedollarcnt
        , landtaxvaluedollarcnt
        , taxamount
    from properties_2017 prop
    inner join predictions_2017 pred on prop.parcelid = pred.parcelid
    where propertylandusetypeid = 261;
    '''
    if os.path.exists('zillow_alt.csv'):
        df = pd.read_csv('zillow_alt.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(query, url)
        df.to_csv('zillow_alt.csv', index=False)
    return df


def get_counties(df):
    '''
    This function will create dummy variables out of the original fips column. 
    And return a dataframe with all of the original columns except regionidcounty.
    We will keep fips column for data validation after making changes. 
    New columns added will be 'LA', 'Orange', and 'Ventura' which are boolean 
    The fips ids are renamed to be the name of the county each represents. 
    '''
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    # drop regionidcounty and fips columns
    df_dummies = df_dummies.drop(columns = ['regionidcounty'])
    return df_dummies





def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet


    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )


    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df




def remove_outliers(df):
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) & 
               (df.regionidzip < 100000) & 
               (df.bathroomcnt > 0) & 
               (df.bedroomcnt > 0) & 
               (df.acres < 20) &
               (df.calculatedfinishedsquarefeet < 10000) & 
               (df.taxrate < 10)
              )]





def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions




def county_train_test_split(counttydfs,target_var='logerror'):
    '''
    this is the main function that splits our data into countyies calls the other functions like spilt etc, it deals with all of our feature enginnering, data cleaning, etc
    it returns a list of lists. eg partitionslist[0]->df_la,  partitionslist[1]->df_orange,partitionslist[2]->df_ventura
    then partitionslist[0][0:6]-> [la_train, la_X_train, la_X_validate, la_X_test, la_y_train, la_y_validate, la_y_test]
    the other counties follow the same pattern






    '''
    partitionslist=[]
    [partitionslist.append(split(county, target_var))for county in counttydfs]
    return partitionslist


def partitionedZillowbyCounty():
    df=get_zillow_data()
    df.head().T
    df.shape
    df.isna().sum().sum()
    df.dropna(inplace=True)
    df.isna().sum().sum()
    
    df = get_counties(df)
    df.head().T
    df = create_features(df)
    df.head().T
    # for col in df.columns:
    #     sns.boxplot(df[col])
    #     plt.title(col)
    #     plt.show()
    
    df = remove_outliers(df)

    # Now that the most extreme outliers have been removed, let's look at the summary statistics of each numeric field. 
   
    df.dropna(inplace=True)
    print('We dropped {} rows'.format(52442-df.shape[0]))
     



    df_la = df[df.LA == 1].drop(columns = ['parcelid', 'bedroomcnt', 'taxamount', 'taxvaluedollarcnt', 'fips', 
                                           'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'yearbuilt', 
                                           'lotsizesquarefeet', 'regionidcity', 'regionidzip', 
                                           'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                                           'LA', 'Ventura', 'Orange']) 



    df_orange = df[df.Orange == 1].drop(columns = ['parcelid', 'bedroomcnt', 'taxamount', 'taxvaluedollarcnt', 'fips', 
                                           'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'yearbuilt', 
                                           'lotsizesquarefeet', 'regionidcity', 'regionidzip', 
                                           'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                                           'LA', 'Ventura', 'Orange']) 



    df_ventura = df[df.Ventura == 1].drop(columns = ['parcelid', 'bedroomcnt', 'taxamount', 'taxvaluedollarcnt', 'fips', 
                                           'structure_dollar_per_sqft', 'land_dollar_per_sqft', 'yearbuilt', 
                                           'lotsizesquarefeet', 'regionidcity', 'regionidzip', 
                                           'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                                           'LA', 'Ventura', 'Orange'])          
    # partitions = split(df, target_var='logerror')
    counttydfs=[df_la,df_orange,df_ventura]
    partitionslist=county_train_test_split(counttydfs,target_var='logerror')
    for i in partitionslist[0:4]:
        i[0]=logerrorbins(i[0])
    return partitionslist    






def logerrorbins(df):
    df['logerror_bins'] = pd.cut(df.logerror, [-5, -.2, -.05, .05, .2, 4])
    df.logerror_bins.value_counts() 
    # sns.pairplot(data = df, hue = 'logerror_bins', 
    #          x_vars = ['logerror', 'structure_dollar_sqft_bin', 'lot_dollar_sqft_bin', 'taxrate', 
    #                    'bath_bed_ratio'],
    #          y_vars = ['logerror', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'acres', 'age'])
    # print(df.logerror_bins.value_counts())
    
    return df
