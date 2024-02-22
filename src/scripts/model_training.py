from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
import pandas as pd

class ModelTrainer:
    def __init__(self, preprocessor, X_train, Y_train, cv_folds, random_state=123):
        self.preprocessor = preprocessor
        self.X_train = X_train
        self.Y_train = Y_train
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.results_df = None

    def randomized_lgbm_search(self, n_start, n_end, depth_start, depth_end, leaves_start, leaves_end, num_iter):
        pipeline_lgbm = Pipeline([
            ('preprocessor', self.preprocessor),
            ('lgbm', LGBMRegressor(verbose=-1, force_row_wise=True))
        ])

        param_distributions = {
            'lgbm__n_estimators': randint(n_start, n_end),
            'lgbm__max_depth': randint(depth_start, depth_end),
            'lgbm__num_leaves': randint(leaves_start, leaves_end),
            'lgbm__learning_rate': uniform(0.01, 0.2)
        }

        random_search = RandomizedSearchCV(
            pipeline_lgbm, param_distributions=param_distributions,
            n_iter=num_iter, cv=self.cv_folds, scoring='r2', n_jobs=-1, random_state=self.random_state,
            return_train_score=True
        )

        random_search.fit(self.X_train, self.Y_train)

        best_pipeline = random_search.best_estimator_
        self.best_model = best_pipeline.named_steps['lgbm']
        self.best_score = random_search.best_score_
        self.best_params = random_search.best_params_
        std_dev = random_search.cv_results_['std_test_score'][random_search.best_index_]

        print(f"Best R^2 Score for LightGBM: {self.best_score:.4f}")
        print(f"Standard Deviation of R^2 Score for LightGBM: {std_dev:.4f}")
        print(f"Best Hyperparameters for LightGBM: {self.best_params}")

        self.results_df = pd.DataFrame({
            'Number of Trees': random_search.cv_results_['param_lgbm__n_estimators'],
            'Max Depth': random_search.cv_results_['param_lgbm__max_depth'],
            'Num Leaves': random_search.cv_results_['param_lgbm__num_leaves'],
            'Mean Validation R^2 Score': random_search.cv_results_['mean_test_score'],
            'Mean Training R^2 Score': random_search.cv_results_['mean_train_score'],
            'Standard Deviation': random_search.cv_results_['std_test_score']
        })

        print(self.results_df)

    def get_best_model(self):
        return self.best_model
