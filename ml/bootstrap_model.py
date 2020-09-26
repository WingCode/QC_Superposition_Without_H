import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class BootstrapModel:
    model_param_1 = None
    model_param_2 = None

    def __init__(self, model_base_path=None):
        if model_base_path:
            self.model_param_1 = joblib.load(model_base_path + "/model_param_1.pkl")
            self.model_param_2 = joblib.load(model_base_path + "/model_param_2.pkl")

    def train_model(self, df, target_column_name):
        X = df.drop(columns=["final_param_1", "final_param_2"])
        y = df[[target_column_name]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': ['mse'],
            'num_leaves': 1000,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        gbm = lgb.LGBMRegressor(**params)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mse', early_stopping_rounds=1000)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
        print('The mse of prediction is:', mean_squared_error(y_test, y_pred))
        return gbm

    def train_bootstrap_models(self, loc_data_shots_file, loc_model_base_path):
        # df = pd.read_csv("./data_shots_1000_costTolerance_0.01_learningRate_1.csv")
        df = pd.read_csv(loc_data_shots_file)
        self.model_param_1 = self.train_model(df, "final_param_1")
        self.model_param_2 = self.train_model(df, "final_param_2")
        self.save_model(self.model_param_1, loc_model_base_path + "/model_param_1.pkl")
        self.save_model(self.model_param_2, loc_model_base_path + "/model_param_2.pkl")

    def save_model(self, model, path):
        joblib.dump(model, path)

    def predict_params(self, params):
        bootstrap_param_1 = self.model_param_1.predict([params])
        bootstrap_param_2 = self.model_param_2.predict([params])
        return [bootstrap_param_1, bootstrap_param_2]
