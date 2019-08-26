# MultipleLinearRegression

By Mackenzie Mitchell and Jon Bebi

**Overview**</n>
In this project, we analyze the different variables that affect a country's excess carbon emissions. Using data obtained from the National Footprint Accounts, we attempted to create a multiple linear regression model using Python's statsmodels OLS feature that could predict a country's excess carbon and help to determine the steps a given country could take in order to reduce their excess carbon emissions.

![TargetDistplots](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/TargetDistplots.png)

Please find the raw dataset here: https://www.kaggle.com/footprintnetwork/national-footprint-accounts-2018
Note that most of the variables used in this multiple linear regression are measures in global hectares. The observations are quantified depending on the number of global hectares of that landtype either required to support consumption/production or that are supported by the biological productivity. This unit is a unit of land normalized by biological productivity across landtype, country, and year. Using this unit of measurement, we are able to compare the usage of different regions, times, and land types on the same scale. 

![HectareVisualization](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/hectare.png)

**Exploratory Data Analysis EDA**
We explored and cleaned the data set using pandas which contains over 15,000 data points per year for 196 countries over more than 50 years. Slicing the data to only contain the observations that were recorded in terms of Ecological Footprint (or EFConsPerCap,) and using the columns country, year, crop_land, grazing_land, forrest_land, fishing_ground, built_up_land, carbon, gdp, and population, we begun our analysis with almost 5500 observations.

We chose specifically to look at Ecological Footprint of Consumption Per Capita because it encompasses the Ecological Footprint of Production as well as the difference between the Ecological Footprint of Imports and Exports. This way, we ensure that we are taking the country's full Ecological Footprint into account. 

**Transformations and Feature Engineering**
Our target variable was initally just carbon. However, upon exploring the data, it became clear that excess carbon was a more useful variable that would provide more valuable information.
    excess_carbon= carbon - forest_land
Carbon is initally calculated as the global hectares of world-average forest required to sequester carbon emissions and forest_land is initally calculated as the global hectares of forest land avaiable for sequestration among other things. Thus, the engineered target variable, excess_carbon, represents the carbon that a country produces that is not sequestered by their forest land. 

We decided to engineer a feature called per_cap_built which encapsulates the built_up_land on a per person level, as we thought that the built up land may increase as population increased.
Initally, our best fit model included the predictors crop_land + gdp + fishing_ground + per_cap_built. This model explained 67.9% of the variation in excess_carbon, however, the residual plots contained a strong correlation, with and upwards sloping residual plot. 

This prompted our thoughts that perhaps all the land type variables experienced multicollinearity. If a country increases their built_up_land, maybe they took away from their forest or crop land in order to make room for the new built land. For the reason, we engineered yet another feature that modeled the interation between crop_land, built_up_land, fishing_ground, and grazing_land. Forrest_land could not be included has this is a part of the target variable. 

![CorrelationMatrix](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/CorrelationMatrix.png)

After trying our model with the land type interaction variable (landtypeInter), we found that our best fit multivariate linear regression model for this data contained the predictor variables crop_land, gdp, fishing_ground, per_cap_built, and landtypeInter. This model explained 68.7% of the variation in excess carbon. 

![BestModelSummary](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/ModelSummary(BestFit))

Unfortunately, however, the residuals still showed a strong relationship. While the variance of the residuals looked to be constant, the mean was not centered at zero. In order to attempt a log transformation, the target variable had to be scaled on a MinMaxScale due to the negative values. Using the log of the scaled excess carbon as the target variable produces an R-sqaured value of less than 10%. No matter the transformations nor the interaction variables attempted, we were unable to completely successfuly fit the data to a multiple linear regression.


![ResidualPlots](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/ResidualPlots(BestFit).png)

In the future, we would like to try to fit this data to a time series or a ridge regression model. The scatter plots of some of the features displayed an L-shape. Upon research, it was revealed that an L-shaped scatter plot represents sudden changes in the relationship between two time series (Steve Haroz, Robert Kosara, and Steven L. Franconeri). Again, through research, it was also found that Ridge Regression can be helpful to model data that experience strong multicollinearity (NCSS).
![ScatterPlots1](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/Scatterplots1.png)
![ScatterPlots2](https://github.com/mackenziemitchell6/BigfootPrint/blob/mack-wip/Visualizations/Scatterplots2.png)

