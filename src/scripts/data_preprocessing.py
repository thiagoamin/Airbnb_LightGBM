import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:

    def __init__(self):
        self.data_path = '../data/AB_NYC_2019.csv'

    def preprocess_data(self):
        df = pd.read_csv(self.data_path)
        train_df, test_df = train_test_split(df, test_size=0.6, random_state=123)

        # Define features to drop
        drop_columns = ['id', 'name', 'host_id', 'host_name', 'last_review']

        # Splitting the dataset into train and test
        X_train = train_df.drop(columns=['reviews_per_month', 'number_of_reviews'] + drop_columns)
        Y_train = train_df['reviews_per_month']
        X_test = test_df.drop(columns=['reviews_per_month', 'number_of_reviews'] + drop_columns)
        Y_test = test_df['reviews_per_month']

        # Define numeric and categorical transformers
        numeric_features = ['latitude', 'longitude', 'price', 'minimum_nights', 'calculated_host_listings_count', 'availability_365']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_features = ['neighbourhood_group', 'neighbourhood', 'room_type']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Filtering non-NA target values
        non_na_indices = Y_train.notna()
        X_train_non_na = X_train[non_na_indices]
        Y_train_non_na = Y_train[non_na_indices]
        print("This is x-test")
        for col in X_test.columns:
            print(col);

        # Create a map for easy retrieval
        data_map = {
            'preprocessor': preprocessor,
            'train data (x)': X_train_non_na,
            'train data (y)': Y_train_non_na,
            'test_data (x)': X_test,
            'test_data (y)': Y_test
        }

        return data_map
