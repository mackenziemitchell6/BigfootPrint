#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:57:30 2019

@authors: mackenziemitchell & jonbebi
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import pandas as pd
import scipy.stats as sci
import numpy as np
import seaborn as sns
import pickle
import functions
#Setting styles for graphs
sns.set(style='whitegrid',palette='bright', color_codes=True)
sns.set_context('talk')

#Importing csv file containing data
df=pd.read_csv('NFA2018.csv')
#Dropping unneeded columns in data
df = df.drop(['ISO alpha-3 code', 'UN_region', 'UN_subregion', 'total'], axis=1)
#Renaming GDP column to make it easier to call
df.rename(columns={'Percapita GDP (2010 USD)':'gdp'},inplace=True)
#Drop zeros in the following columns:
df= df[df['forest_land'] != 0.000000]
df= df[df['built_up_land'] != 0.000000]
df= df[df['carbon'] != 0.000000e+00]
#Looking only at the record type EF consumption per capita
df=df[df['record']=='EFConsPerCap']
#Getting rid of some outliers
df=df[df['carbon']<=15]
#Dropping all NaN values
df.dropna(inplace=True)
df.isna().sum()
df.head()

#Feature engineering
df['excess_carbon']=df['carbon']-df['forest_land']
#Removing some outliers
df=df[df['excess_carbon']>=-2.5]
df['percapbuilt']=df['built_up_land']/df['population']
df['landtypeInter']=df['crop_land']*df['grazing_land']*df['fishing_ground']*df['built_up_land']
df['fishcrop']=df['fishing_ground']*df['crop_land']
df['gdppop']=df['gdp']*df['population']
df['excess_carbonsq']=(df['excess_carbon'])**2
df['gdplog']=np.log(df['gdp'])
df['percapbuiltlog']=np.log(df['percapbuilt'])
df['invexcarbon']=1/df['excess_carbon']
df['percaplandtypes']=df['landtypeInter']/df['population']
df['all']=df['gdp']*df['population']*df['crop_land']*df['grazing_land']*df['built_up_land']*df['fishing_ground']

#create categorical, use categorical, continuous, AND reaction btwn them
df['fish']=np.where(df['fishing_ground']>=0.1, 1, 0)
df['fishinter']=df['fish']*df['fishing_ground']
df['graze']=np.where(df['grazing_land']>=0.1, 1, 0)
df['grazeinter']=df['graze']*df['grazing_land']

#Saving the cleaned dataframe into a pickle file
with open('df.pickle','wb') as f:
    pickle.dump(df,f,pickle.HIGHEST_PROTOCOL)
    
#Plotting all of the distributions of the target and predictor variables
fig, axes = plt.subplots(4,4,figsize=(30,30))
sns.despine(left=True)
sns.distplot(df['year'],ax=axes[0,0])
sns.distplot(df['crop_land'],ax=axes[0,1])
sns.distplot(df['built_up_land'],ax=axes[0,2])
sns.distplot(df['grazing_land'],ax=axes[0,3])
sns.distplot(df['forest_land'],ax=axes[1,0])
sns.distplot(df['fishing_ground'],ax=axes[1,1])
sns.distplot(df['gdp'],ax=axes[1,2])
sns.distplot(df['excess_carbon'],ax=axes[1,3])
sns.distplot(df['population'],ax=axes[2,0])
sns.distplot(df['percapbuilt'],ax=axes[2,1])
sns.distplot(df['landtypeInter'],ax=axes[2,2])
sns.distplot(df['carbon'],ax=axes[2,3])
sns.distplot(df['fishcrop'],ax=axes[3,0])
#Saving the plots of the distributions
fig.savefig('Distributions.png',bbox_inches='tight')



