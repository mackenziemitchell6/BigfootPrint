#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:59:12 2019

@authors: mackenziemitchell & jonbebi
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import statsmodels.stats as sta
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import warnings
from functions import checkresiduals
from functions import CorrMtx
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 300)
sns.set(style='whitegrid',palette='bright', color_codes=True)
sns.set_context('talk')

#Loading in dataframe to work with
with open('df.pickle','rb') as file:
    df=pickle.load(file)
df.head()
#Tested to see if this might give different results when grouped, it did not
dfi=df.groupby(['country','year']).mean()
dfi.head()

#Beginning to try different models including different predictor variables:
mod=smf.ols('excess_carbon~crop_land+gdp+fishing_ground+percapbuilt+landtypeInter',data=df).fit()
mod.summary()
#Saving best model summary in variable so as to not lose track when trying different models
bestModel=mod.summary()
# Saving the summary of our best model
pickle.dump(bestModel, open('ModelSummary(BestFit)', 'wb'))
bestModel

#REMEMBER TO RUN CHECKRESIDUALS ONE LAST TIME AFTER FINDING BEST MODEL TO ENSURE 
#SAVING THE RESIDUAL PLOTS FOR THAT MODEL
checkresiduals(df,'excess_carbon',mod)

#ensuring the pickle worked
loaded_model = pickle.load(open('ModelSummary(BestFit)', 'rb'))
result = loaded_model
result

#Trying to put excess_carbon on MinMax scale in order to do log transform
#Because we were not able to do log transform as excess_carbon had negative numbers
#Log(negative number)=DOES NOT EXIST
d=df['excess_carbon']
dff=pd.DataFrame(d,columns=['excess_carbon'])
dff.head()
float_array=dff['excess_carbon'].values.astype(float)
# float_array.reshape(1,-1)
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(float_array.reshape(-1,1))
scaled_df = pd.DataFrame(scaled_df, columns=['excess_carbon'])
df['excess_carbon_minmax']=scaled_df

#More feature engineering for interaction variables to try to make residuals normal
df['log_ex_carbon_scaled']=np.log(df['excess_carbon_minmax'])
df['logcrop']=np.log(df['crop_land'])
df['logpercapbuilt']=np.log(df['percapbuilt'])
df['logbuilt']=np.log(df['built_up_land'])
df['gdplog']=np.log(df['gdp'])
df['logpop']=np.log(df['population'])
df['expcarbon']=np.exp(df['excess_carbon_minmax'])
df['ex_carbon_transformed']=stats.boxcox(df['excess_carbon_minmax'])[0]
df['croppop']=df['crop_land']*df['population']
df['cropgdp']=df['crop_land']*df['gdp']
df['builtgdp']=df['built_up_land']*df['gdp']
df['fishinggdp']=df['fishing_ground']*df['gdp']
df['grazinggdp']=df['grazing_land']*df['gdp']
df['percapcrop']=df['crop_land']/df['population']
df['percapfish']=df['fishing_ground']/df['population']
df['percapgrazing']=df['grazing_land']/df['population']

model=smf.ols('excess_carbon~cropgdp+grazinggdp+fishinggdp+builtgdp',data=df).fit()
model.summary()

checkresiduals(df , 'excess_carbon' , model)

#Trying transformed target variable, R-Sq not even 2%... very bad
model=smf.ols('log_ex_carbon_scaled~crop_land+gdp+fishing_ground+percapbuilt',data=df).fit()
model.summary()
#Since we have 5000+ observations, it is not critical for our residuals to be normal
#With mean centered at 0 and constant variance
#Due to this, and the fact that our transformed model does not make things better,
#We will accept the prior model (code earlier) as our "best fit" and conclude
#that s different type of modeling approach may be better to model this data
#rather than a linear regression

#Checking to see if limiting the years to recent years will make our results better
#It did not
df1=df[df['year']>1990]
df1.head()
model2=smf.ols('excess_carbon~crop_land+gdp+fishing_ground+percapbuilt+landtypeInter',data=df1).fit()
model2.summary()