import gc
from logging import getLogger
from typing import Tuple

import gokart
import pandas as pd
import numpy as np
from lightgbm import Booster
from sklearn import metrics
import lightgbm as lgb

logger = getLogger(__name__)


class TrainLGBM(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    feature_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.feature_task

    def output(self):
        model = self.make_target(relative_file_path='model/lgb.pkl')
        feature_importance = self.make_target(relative_file_path='model/feature_importance.csv')
        return dict(model=model, feature_importance=feature_importance)

    def run(self):
        data = self.load()
        model, feature_importance = self._run(data)
        self.dump(model, 'model')
        self.dump(feature_importance, 'feature_importance')

    @staticmethod
    def _run(data: pd.DataFrame) -> Tuple[Booster, pd.DataFrame]:

        # define list of features
        # TODO: delete these lines
        # features = ["wday", "month", "year",
        #             "event_name_1", "event_type_1",
        #             "snap_CA", "snap_TX", "snap_WI",
        #             "sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7",
        #             "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60",
        #             "rolling_mean_t90", "rolling_mean_t180", "rolling_std_t7", "rolling_std_t30",
        #             "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        # x_train = data['x_train']
        # y_train = data['y_train']
        # x_val = data['x_val']
        # y_val = data['y_val']

        train_set = lgb.Dataset(data['x_train'].drop(['id', 'd'], axis=1), data['y_train'])
        val_set = lgb.Dataset(data['x_val'].drop(['id', 'd'], axis=1), data['y_val'])

        # del x_train, y_train

        params = {'boosting_type': 'gbdt', 'metric': 'rmse', 'objective': 'poisson', 'n_jobs': -1, 'seed': 20,
                  'learning_rate': 0.075, 'bagging_fraction': 0.66, 'bagging_freq': 1, 'colsample_bytree': 0.77,
                  'num_leaves': 63, 'lambda_l2': 0.1}
        num_boost_round = 5
        model = lgb.train(params, train_set, num_boost_round=num_boost_round, early_stopping_rounds=400,
                          valid_sets=[train_set, val_set], verbose_eval=100)
        # TODOl: final score
        # val_pred = model.predict(x_val[features])
        # val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
        # print(f'Our val rmse score is {val_score}')

        # TODO: feature importance
        # feature_importance = pd.DataFrame(
        #     dict(name=features, imp=model.feature_importance(importance_type='gain'))).sort_values(by='imp', ascending=False)
        feature_importance = pd.DataFrame()

        return model, feature_importance
