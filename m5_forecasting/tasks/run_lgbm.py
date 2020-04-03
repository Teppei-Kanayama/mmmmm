import gc
from logging import getLogger

import gokart
import pandas as pd
import numpy as np
from sklearn import metrics
# import lightgbm as lgb

from m5_forecasting.data.utils import reduce_mem_usage

logger = getLogger(__name__)


class RunLGBM(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    feature_task = gokart.TaskInstanceParameter()
    raw_data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return dict(preprocessed_data=self.feature_task, raw_data=self.raw_data_task)

    def run(self):
        data = self.load_data_frame('preprocessed_data')
        submission = self.load('raw_data')['sample_submission']
        output = self._run(data, submission)
        self.dump(output)

    @classmethod
    def _run(cls, data: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
        data = cls._transform(data)
        data = cls._simple_fe(data)
        data = reduce_mem_usage(data)
        test = cls._run_lgb(data)
        output = cls._predict(test, submission)
        return output

    @staticmethod
    def _run_lgb(data: pd.DataFrame) -> pd.DataFrame:
        # define list of features
        features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek',
                    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
                    'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7',
                    'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90',
                    'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365',
                    'rolling_price_std_t7', 'rolling_price_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']

        # going to evaluate with the last 28 days
        x_train = data[data['date'] <= '2016-03-27']
        y_train = x_train['demand']
        x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
        y_val = x_val['demand']
        test = data[(data['date'] > '2016-04-24')]
        del data
        gc.collect()

        # define random hyperparammeters
        params = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'objective': 'regression',
            'n_jobs': -1,
            'seed': 236,
            'learning_rate': 0.1,
            'bagging_fraction': 0.75,
            'bagging_freq': 10,
            'colsample_bytree': 0.75}

        train_set = lgb.Dataset(x_train[features], y_train)
        val_set = lgb.Dataset(x_val[features], y_val)

        del x_train, y_train

        model = lgb.train(params, train_set, num_boost_round=2500, early_stopping_rounds=50,
                          valid_sets=[train_set, val_set], verbose_eval=100)
        val_pred = model.predict(x_val[features])
        val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
        print(f'Our val rmse score is {val_score}')
        y_pred = model.predict(test[features])
        test['demand'] = y_pred
        return test

    @staticmethod
    def predict(test: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
        predictions = test[['id', 'date', 'demand']]
        predictions = pd.pivot(predictions, index='id', columns='date', values='demand').reset_index()
        predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

        evaluation_rows = [row for row in submission['id'] if 'evaluation' in row]
        evaluation = submission[submission['id'].isin(evaluation_rows)]

        validation = submission[['id']].merge(predictions, on='id')
        final = pd.concat([validation, evaluation])
        return final
