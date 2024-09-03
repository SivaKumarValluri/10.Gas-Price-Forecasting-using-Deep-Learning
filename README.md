# CS547--Deep-Learning-Project
Team Name
Oglesby

Team Members
Siva Kumar Valluri |
Pallav Ranjan |
Dimitrios Bralios |
Pratik Sinha |


## Problem Statement ##
Due to the recent economic and supply-chain turmoil gasoline prices across the world have been on the rise. This has impacted the price of every commodity, and contributed to inflation which in turn feeds back to further impact gas prices. Thus, the ability to predict future gasoline prices would be crucial to prepare and device measures to ameliorate the situation. We have identified that gasoline prices are primarily driven by classical economic theory of supply and demand. The supply of gasoline to the US is primarily from domestic crude oil wells and imports from other nations. The US also maintains a strategic petroleum reserve to handle the disruptions in the crude oil supply chain. Demand of gasoline is primarily reflected by domestic consumption. Other than these, inflation limits the purchasing power thus affecting both production/import and consumtion of gasoline. It is usually measured as Consumer Price Index (CPI) by the Federal Bank of the US. Further, changes in taxation at federal and state level will also impact the prices, however it remains constant over a long duration. Finally, we will also use historical trends in gasoline prices to capture any missing features. 

### To summarize, we are building a Recurrent Neural Network (RNN) to predict the weekly gasoline price based on the following features: ###

    - Weekly crude oil import (Supply)
    - Strategic Petroleum Reserve (Supply)
    - Domestic Production (Supply)
    - Domestic Consumption of Gas (Demand)
    - Domestic Inflation as CPI
    - Historical Price of Gasoline


## License ##
Copyright 2022 University of Illinois Team Oglesby

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


## Data Preprocessing and Initial Analysis ##
This code performs the following steps for predicting weekly gasoline prices:

### 1. Library Imports

    - Data Handling and Visualization: numpy, pandas, matplotlib, scipy
    
    - Machine Learning: tensorflow, sklearn
    
    - Additional Utilities: datetime, time, random

### 2. Plotting Parameters

    Sets default plot parameters using matplotlib to adjust font size and line width for better visualization.

### 3. Data Loading Function (getfile)

    - Purpose: Loads data from Google Drive links or a local version if available.
    
    = How: Uses pandas.read_pickle to read binary pickle files.
    
### 4. Data Import 
    - CPI: Consumer Price Index data, a measure of inflation. 
    
    - Consumption: Domestic gasoline consumption.  
    
    - Price: Historical gasoline prices.
    
    - Imports: Weekly crude oil import data.
    
    - Reserve: Strategic Petroleum Reserve data.
    
    - Production: Domestic gasoline production data.
    
    - Each dataset is loaded, processed (e.g., renaming columns, converting dates), and prepared for alignment.

### 5. Data Alignment and Truncation 

    - Aligns all features to the date range of the price data.
    
    - Truncates earlier data to match the date range of the price data.
    
### 6. Data Matching

    - Matches each price data point with the closest earlier data points for each feature.
    
    - Creates a combined dataset where each row contains price and corresponding feature values.

### 8. Normalization and Visualization

    - Normalization: Standardizes features by subtracting the mean and dividing by the standard deviation. 
    
    - Visualization: Plots the normalized dataset to visualize trends and seasonality.
    
### 9. Data Splitting

    - Splits the dataset into training (70%), validation (20%), and test (10%) sets.
    
    - Drops the date column for model training and validation.
    
### 10. Linear Regression Baseline

    - Initial Model: Trains a linear regression model using features excluding historical prices.
    
    - Augmented Model: Adds the previous week’s price as a feature, improving the model's performance.
    
    - Visualization: Plots the predicted vs. actual values for training, validation, and test sets.
    
### 11. Feature Importance Analysis
    - Pearson Correlation: Examines correlations between features and the target variable.
    
    - ANOVA: Uses statistical tests to rank feature importance.
    
    - Principal Component Analysis (PCA): Reduces dimensionality and identifies key features based on variance explained.
    
    - Recursive Feature Elimination (RFE): Identifies the most important features by recursively removing the least important ones.
    
### Key Findings
    - Best Features: CPI is identified as the most significant feature affecting gasoline prices, with other features like Reserve and Consumption also important.
    
    - Model Improvement: Including historical prices in the linear regression model improves predictions.
    
    - Feature Analysis: Various methods confirm that CPI, Production, and Reserve are crucial features for predicting gasoline prices.
    
    - Overall, the code demonstrates a structured approach to data preprocessing, model training, and feature analysis to predict gasoline prices effectively.
