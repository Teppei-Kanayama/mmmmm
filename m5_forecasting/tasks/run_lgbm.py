import gc
from logging import getLogger

import gokart
import pandas as pd
import numpy as np
from lightgbm import Booster
from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt

from m5_forecasting.data.utils import reduce_mem_usage

logger = getLogger(__name__)


class TrainLGBM(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    feature_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.feature_task

    def output(self):
        return self.make_target(relative_file_path='model/lgb.pkl')

    def run(self):
        data = self.load_data_frame()
        model = self._run(data)
        self.dump(model)

    @staticmethod
    def _run(data: pd.DataFrame) -> Booster:

        # define list of features
        features = ["wday", "month", "year",
                    "event_name_1", "event_type_1",
                    "snap_CA", "snap_TX", "snap_WI",
                    "sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7",
                    "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60",
                    "rolling_mean_t90", "rolling_mean_t180", "rolling_std_t7", "rolling_std_t30",
                    "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        # going to evaluate with the last 28 days
        x_train = data[data['d'] < 1914 - 28]
        y_train = x_train['demand']
        x_val = data[(data['d'] < 1914) & (data['d'] >= 1914 - 28)]
        y_val = x_val['demand']
        del data
        gc.collect()

        # define random hyperparammeters
        params = {'boosting_type': 'gbdt', 'metric': 'rmse', 'objective': 'poisson', 'n_jobs': -1, 'seed': 20,
                  'learning_rate': 0.075, 'bagging_fraction': 0.66, 'bagging_freq': 1, 'colsample_bytree': 0.77, 'num_leaves': 63,
                  'lambda_l2': 0.1}

        train_set = lgb.Dataset(x_train[features], y_train)
        val_set = lgb.Dataset(x_val[features], y_val)

        del x_train, y_train

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=400,
                          valid_sets=[train_set, val_set], verbose_eval=100)
        val_pred = model.predict(x_val[features])
        val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
        print(f'Our val rmse score is {val_score}')
        return model
