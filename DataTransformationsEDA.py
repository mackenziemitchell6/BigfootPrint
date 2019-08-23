#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:53:20 2019

@authors: mackenziemitchell & jonbebi
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf
import scipy.stats as stats
import pickle
import warnings
import functions
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 300)
sns.set(style='whitegrid',palette='bright', context='talk', color_codes=True)

#Loading in the dataframe
with open('df.pickle','rb') as file:
    df=pickle.load(file)
df.head()

#Looking at the distributions of carbon vs. excess carbon (carbon-forest_land)
fig1, axes = plt.subplots(1,2,figsize=(30,15),sharex=True)
sns.despine(left=True)
sns.distplot(df['carbon'],bins=30,axlabel='Carbon Distribution',ax=axes[0],color='darkgreen')
sns.distplot(df['excess_carbon'],bins=30,axlabel='Excess Carbon Distribution (Carbon-Forest_land)',ax=axes[1],color='deeppink')
fig1.savefig('TargetDistplots.png', bbox_inches='tight')

#Looking at scatter plots of some of the features vs. excess carbon(target)
features=['year','crop_land','grazing_land','fishing_ground','built_up_land','gdp','population','fishcrop','percapbuilt']
n = 4
row_groups= [features[i:i+n] for i in range(0, len(features), n) ]
x=1
for i in row_groups:
    a=sns.pairplot(df, y_vars='excess_carbon',x_vars=i,kind='reg',height=5)
    a.savefig('Scatterplots{}.png'.format(x), bbox_inches='tight')
    x+=1
    
#Putting excess carbon on mixmax scale in order to try log transform
d=df['excess_carbon']
dff=pd.DataFrame(d,columns=['excess_carbon'])
dff.head()
float_array=dff['excess_carbon'].values.astype(float)
float_array.reshape(1,-1)
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(float_array.reshape(-1,1))
scaled_df = pd.DataFrame(scaled_df, columns=['excess_carbon'])
df['excess_carbon_minmax']=scaled_df

corr=df.corr()

#Printing correlation matrix for data, see functions.py
CorrMtx(corr, dropDuplicates = True).savefig('CorrelationMatrix', bbox_inches='tight')

#Looking at partial regression plots
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)
fig.savefig('PartialRegressions.png', bbox_inches='tight')

#Looking at relationship between gdp and excess carbon
sns.regplot(x='gdp',y='excess_carbon',data=df,x_estimator=np.mean,lowess=True)

#Looking at correlation between crop land and population
stats.pearsonr(df['crop_land'],df['population'])