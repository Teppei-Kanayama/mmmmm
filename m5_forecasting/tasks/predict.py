import gc
from logging import getLogger

import gokart
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
# import lightgbm as lgb

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.data.preprocess import PreprocessInputData
from m5_forecasting.data.utils import reduce_mem_usage

logger = getLogger(__name__)


class Predict(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        return self.make_target('submission.csv', use_unique_id=False)

    def requires(self):
        return dict(preprocessed_data=PreprocessInputData(), raw_data=LoadInputData())

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
    def _transform(data: pd.DataFrame) -> pd.DataFrame:
        nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for feature in nan_features:
            data[feature].fillna('unknown', inplace=True)

        cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2',
               'event_type_2']
        for feature in cat:
            encoder = preprocessing.LabelEncoder()
            data[feature] = encoder.fit_transform(data[feature])
        return data

    @staticmethod
    def _simple_fe(data: pd.DataFrame) -> pd.DataFrame:
        # rolling demand features
        data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
        data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
        data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
        data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
        data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
        data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
        data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
        data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
        data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
        data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
        data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())

        # price features
        data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
        data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
        data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(
            lambda x: x.shift(1).rolling(365).max())
        data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (
        data['rolling_price_max_t365'])
        data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
        data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
        data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace=True, axis=1)

        # time features
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['week'] = data['date'].dt.week
        data['day'] = data['date'].dt.day
        data['dayofweek'] = data['date'].dt.dayofweek
        return data

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
