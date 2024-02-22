from scripts.data_preprocessing import DataPreprocessor
from scripts.model_training import ModelTrainer
from joblib import dump, load


def main():
    data_preprocessor = DataPreprocessor()  # path relative to scripts/data_preprocessing
    data_map = data_preprocessor.preprocess_data()

    model_trainer = ModelTrainer(data_map.get('preprocessor'),
                                 data_map.get('train data (x)'),
                                 data_map.get('train data (y)'),
                                 3
                                 )
    model_trainer_results = model_trainer.randomized_lgbm_search(10, 200, 1,
                                                      10, 32, 1024, 5)
    dump(model_trainer_results, '../models/lightgbm')


if __name__ == "__main__":
    main()
