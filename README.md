# Predicting Popularity of Airbnb Listings (Summary)

## Introduction
In this latest project, I aim to forecast the number of reviews per month for Airbnb listings, using it as a measure of their popularity. This predictive machine learning (ML) model would allow companies like Airbnb to estimate the popularity of future listings before they're even posted. This insight could be invaluable for hosts, guiding them in creating more attractive listings. 

## Exploring the Data
To train and test my model, I used a 2019 New York Airbnb Listings dataset. The dataset contains 48,895 listings (observations), each with 16 features (e.g., Price, Minimum Nights, Reviews Per Month) that provided information about each listing. The target variable for the model is 'reviews_per_month', chosen because it reflects the frequency of reviews, a good indicator of a listing's popularity. 

Out of the 15 available features, 10 were identified as being particularly predictive and beneficial for the model. The aforementioned 10 features, encompassing neighborhood details, geographical coordinates, room types, and several numerical attributes like price and availability, were selected for their strong predictive potential in the model. The remaining features, such as “host_id” and “host_name,” were dropped as they are unique identifiers or names that don't inherently influence the listing's popularity. 

## Data Preprocessing and Feature Engineering
The histogram provided (Figure 1) illustrates a heavily right-skewed distribution of prices, with a vast majority of listings concentrated at the lower end of the price spectrum and a few listings extending to the high price range. 

**Figure 1:** Frequency Distribution of Airbnb Listing Prices in New York City.
<img width="600" alt="image" src="https://github.com/thiagoamin/Airbnb_LightGBM/assets/122248078/9e2e3dc1-5887-4783-9e96-6841f2f985ef">

To improve model robustness, we've binned prices into quartiles: 0-25th percentile (low-end), 25-50th (moderate), 50-75th (above-average), and 75-100th (high-end). This categorization helps the tree-based model to discern broader pricing trends rather than individual variations, enhancing predictability and generalizability.

When refining the dataset for my predictive model, a decision was made to remove entries with NA (not available) values in the 'reviews_per_month' target feature. This decision was grounded in the principle of not violating the 'golden rule.' In this context, attempting to impute the 'reviews_per_month' data could have skewed my model, leading to unreliable predictions. 

To handle other missing values in our dataset, we used median values for imputation in numerical features and the most frequent values for categorical ones. This approach was tailored to preserve data integrity without skewing the original distributions. Additionally, scaling was not performed as our chosen tree-based model (Refer to the subsequent paragraph for further details) is not sensitive to feature scale, making this step redundant for our analysis.

## Model Optimization
Below is a comparative graph of all the models I considered, showcasing their best cross-validation (CV) scores, standard deviations, and best training scores. Each of the best scores were found using randomized search in these models’ key hyper-parameters:

**Figure 2:** Best CV R^2 Score with Standard Deviation.
<img width="600" alt="Screenshot 2024-02-22 at 12 49 01 AM" src="https://github.com/thiagoamin/Airbnb_LightGBM/assets/122248078/2e6c5970-7c09-4869-8b68-6dc9423715d3">

Based on Figure 2, it’s evident that my LightGBM model yielded the highest R^2 score and also maintained a reasonable standard deviation, indicating consistent performance. The chosen model achieved an R^2 score of 0.3791, indicating it can explain about 37.91% of the variance in the Reviews Per Month feature. The standard deviation of the R^2 score, at 0.0098, demonstrates a consistent performance across different validation sets. 

Such consistency is particularly crucial in the context of our original problem—predicting the popularity of listings. With this model, Airbnb could somewhat effectively estimate the future success of listings, guiding hosts in creating offerings that are likely to be more appealing to potential guests.

The best performing hyperparameters were a learning rate of 0.1, a max depth of 7, 100 trees, and 120 leaves. This specific combination seemed to offer the most effective balance between complexity and predictive power during the grid search process. Figure 3 includes a red data point indicating the chosen model, which is the one closest to the diagonal line representing a perfect balance between the training (seen data) and validation (unseen data) R^2 scores. This shows the model has good generalization to unseen data without overfitting or underfitting. 

**Figure 3:** Comparison of Mean Training vs. Validation R^2 scores for LightGBM Model using random combinations of hyper-parameter values
![Picture1](https://github.com/thiagoamin/Airbnb_LightGBM/assets/122248078/67213583-3d7e-4195-b008-6215c7b1fc15)


## Limitations
In the process of feature selection for the model, one variable that was dropped is the 'description' of the listings. On the surface, the 'description' seems to hold little predictive power since no two listings are likely to have identical descriptions. Yet, if processed using natural language processing (NLP) techniques, 'description' could potentially reveal patterns and keywords indicative of a listing's popularity. 

Another limitation of my model was the choice to use Randomized Search instead of Grid Search to fine-tune the hyperparameters of the LightGBM model, primarily due to the lack of computational resources. Grid Search could have methodically worked through multiple combinations of parameters and determined the optimal settings with potentially greater precision. For a sophisticated model like LightGBM, which can be sensitive to hyperparameter tuning, Grid Search offers the benefit of a more exhaustive approach, ensuring that the best possible parameters are not overlooked. 

## Contact
For more information, contact Thiago Amin at thiagoamin2021@gmail.com.
