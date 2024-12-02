# Data Analysis and Forecasting of Natural Disasters

This project performs Exploratory Data Analysis (EDA) as well as train and implement 4 machine learning models that could predict future natural disasters based on a synthetic data.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Data Description](#data-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Implementation](#model-implementation)
- [Observations and Conclusions](#observations-and-conclusions)
- [Summary and Suggestions](#summary-and-suggestions)

## Introduction

Our projects aim is to showcase what we have learned over the duration of our course. This includes performing EDA on the selected data, data preparation and feature engineering, training and implementing machine learning models, and presenting our findings in the form of visuals and recommendations. A more detailed explanation could be found in our [project description file](Final_Project_Description.docx). For this project, we used the dataset from ["Forecasting Disaster Management in 2024"](https://www.kaggle.com/datasets/umeradnaan/prediction-of-disaster-management-in-2024/data).

## Features

- Includes the [description file](Final_Project_Description.docx) with an Executive Summary, Problem Statement, Research Objectives, a written explanation of our EDA and a written explanation of our Data Preparation and Feature Engineering.
- Performs EDA on the dataset, providing clear visuals.
- Performs Data Preparation and Feature Engineering.
- Trains and implements (insert model names here) models.
- Evaluates the models performance using (insert evaluation metrics here).

## Data Description

- **Disaster_ID:** A special number assigned to every calamity.
- **Disaster_Type:** Category (e.g., Flood, Fire, Earthquake).
- **Location:** The nation where the catastrphe happened.
- **Magnitude:** The disaster's intensity (scale of 1.0 to 10.0).
- **Date:** The event's timestamp.
- **Fatalities:** The total number of people killed by the calamity.
- **Economic_Loss($):** Damage to finances expressed in US dollars.

## Exploratory Data Analysis

Going into our EDA, it is important to note that the dataset we work with is synthetically generated. The data does not represent real-life catastrophes and the features might not follow a trend.

### Check for missing values in the dataset.

```
Disaster_Type       0
Location            0
Magnitude           0
Date                0
Fatalities          0
Economic_Loss($)    0
dtype: int64
```

Checking for the missing values, we see that there are no missing values for any of the features. This means that we don't have to worry about missing value handling.

### Check unique values for the categorical features.

```
Disaster_Type: 5 unique values
Location: 6 unique values
Date: 10000 unique values
```

In order to better understand these categorical features, we can look into what these unique values are. Printing all of the `.unique()` values for Disaster_Type and Location returns the following categories:

- Disaster_Type:
```
Wildfire
Hurricane
Tornado
Flood
Earthquake
```

- Location
```
Brazil
Indonesia
China
India
USA
Japan
```

Since the Date feature has 10000 unique values, we can invastigate some of the values through our `disaster.head()` and find out that Date values change at an hourly rate starting from 01/01/2024.

### Investigate the distribution of the target variable

For our project, we wanted to see how our features would change the economic loss of the respective countries, and decided to choose that feature as our target.

First, we wanted to look at the density distribution of our target variable:

![Density Distribution](images/econ_loss_dens_dist.png)

As we can see, the distribution of the target variable does not follow a normal distribution, or any trend for that matter. It seems to be randomly distributed.

Next, we generate a scatterplot to investigate a potential connection between the magnitude of the disaster and our target variable, and compare the results based on the type of the disaster:

![Magnitude vs Economic Loss by Disaster Type](images/mag_vs_econ_loss_distype.png)

Similar to our previous observations, we can see that the economic loss does not seem to be affected by the magnitude or the type of the disaster.

### Investigate correlation between features

We wanted to look into potential correlations between our features. For this, we generated a scatterplot between the disaster magnitude and number of fatalities, based on the type of the disaster:

![Magnitude vs Fatalities by Disaster Type](images/mag_vs_fatal_distype.png)

Once again, the scatter plot does not show any clear correlation between the magnitude of the disaster and the number of casualties.

To take a more general approach into finding correlations between features and the target variable, we generated a correlation matrix using the quantitative features:

![Correlation Matrix](images/cor_matrix.png)

The correlation matrix does not show any significant correlation between different features.

### Time series analysis

One other data exploration we performed was on the number of disasters per month. This was in order to catch any potential seasonal trends.

![Disasters per Month](images/dist_tim_ser.png)

While the time series shows a drastic decline for Feb-2025, this is likely due to the synthetic data generation process that resulted in the hourly disaster generation not lasting until the end of the month of February. We recommend the removal of these data points for the sake of avoiding unintentional skewing of the dataset.

## Feature Engineering

For feature engineering, we created two new features. One newly generated feature is "Loss Per Fatality", which is generated by dividing the economic loss by the number of fatalities. We believe that this feature would better represent any potential correlation between economic loss and the loss of human lives.

Second feature we generated is the log of the economic loss. We hoped that investigating this value could better show any variations in the economic loss while handling the outliers that could potentially skew the data.

In addition to the two features we generated, we have also encoded the categorical data. This would assign numbers to disaster type and location features, allowing us to better investigate these features and any correlation to the target variable.

## Model Implementation



## Observations and Conclusions



## Summary and Suggestions

