# Predicting Popularity of Airbnb Listings (Summary)

## Introduction
In this project, I forecast the number of reviews per month for Airbnb listings, using this metric as a measure of their popularity. The predictive ML model aims to help Airbnb and hosts estimate the popularity of future listings, guiding the creation of more attractive listings.

## Exploring the Data
The 2019 New York Airbnb Listings dataset, with 48,895 listings and 16 features, was used. Ten features were identified as highly predictive, focusing on neighborhood, geographical coordinates, room types, and numerical attributes like price and availability.

## Data Preprocessing and Feature Engineering
Price distribution was addressed by binning prices into quartiles. Listings with missing 'reviews_per_month' were removed to maintain model reliability. Median and mode imputation was used for other missing values, with no scaling performed as the model is insensitive to feature scale.

## Model Optimization
A comparative analysis of models was conducted, with the LightGBM model showing the highest R^2 score of 0.3791, indicating it can explain about 37.91% of the variance in reviews per month. The model's robustness is highlighted by a low standard deviation across validation sets. Optimal hyperparameters were determined through randomized search.

## Understanding Limitations
The potential predictive power of listing descriptions was not utilized, and Randomized Search was chosen over Grid Search for hyperparameter tuning due to computational limits. The inclusion of 'noisy features' and their impact on predictive accuracy was also discussed.

## Figures
- **Figure 1:** Frequency Distribution of Airbnb Listing Prices in New York City.
- **Figure 2:** Best CV R^2 Score with Standard Deviation.
- **Figure 3:** Mean Training vs. Validation R^2 scores for LightGBM Model.
- **Figure 4:** SHAP Values for features.

## Contact
For more information, contact Thiago Amin at thiagoamin2021@gmail.com.
