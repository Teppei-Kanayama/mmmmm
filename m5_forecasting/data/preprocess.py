from logging import getLogger

import pandas as pd
import gc
import gokart
from sklearn import preprocessing

from m5_forecasting.data.utils import reduce_mem_usage

logger = getLogger(__name__)


class MergeInputData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'
    data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.data_task

    def run(self):
        calendar = self.load_data_frame('calendar')
        sales_train_validation = self.load_data_frame('sales_train_validation')
        sample_submission = self.load_data_frame('sample_submission')
        sell_prices = self.load_data_frame('sell_prices')
        output = self._run(calendar, sell_prices, sales_train_validation, sample_submission)
        self.dump(output)

    @classmethod
    def _run(cls, calendar: pd.DataFrame, sell_prices: pd.DataFrame, sales_train_validation: pd.DataFrame,
             submission: pd.DataFrame, nrows=55000000) -> pd.DataFrame:
        # melt sales data, get it ready for training
        sales_train_validation = pd.melt(sales_train_validation,
                                         id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                                         var_name='day', value_name='demand')
        print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0],
                                                                                sales_train_validation.shape[1]))
        sales_train_validation = reduce_mem_usage(sales_train_validation)

        # seperate test dataframes
        test1_rows = [row for row in submission['id'] if 'validation' in row]
        test2_rows = [row for row in submission['id'] if 'evaluation' in row]
        test1 = submission[submission['id'].isin(test1_rows)]
        test2 = submission[submission['id'].isin(test2_rows)]

        # change column names
        test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921',
                         'd_1922',
                         'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931',
                         'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940',
                         'd_1941']
        test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949',
                         'd_1950',
                         'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959',
                         'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968',
                         'd_1969']

        # get product table
        product = sales_train_validation[
            ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

        # merge with product table
        test2['id'] = test2['id'].str.replace('_evaluation', '_validation')
        test1 = test1.merge(product, how='left', on='id')
        test2 = test2.merge(product, how='left', on='id')
        test2['id'] = test2['id'].str.replace('_validation', '_evaluation')

        #
        test1 = pd.melt(test1, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                        var_name='day',
                        value_name='demand')
        test2 = pd.melt(test2, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                        var_name='day',
                        value_name='demand')

        sales_train_validation['part'] = 'train'
        test1['part'] = 'test1'
        test2['part'] = 'test2'

        data = pd.concat([sales_train_validation, test1, test2], axis=0)

        del sales_train_validation, test1, test2

        # get only a sample for fst training
        data = data.loc[nrows:]

        # drop some calendar features
        calendar.drop(['weekday', 'wday', 'month', 'year'], inplace=True, axis=1)

        # delete test2 for now
        data = data[data['part'] != 'test2']

        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
        data = pd.merge(data, calendar, how='left', left_on=['day'], right_on=['d'])
        data.drop(['d', 'day'], inplace=True, axis=1)
        # get the sell price data (this feature should be very important)
        data = data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
        gc.collect()

        return data


class LabelEncode(gokart.TaskOnKart):
    data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.data_task

    def run(self):
        data = self.load_data_frame()
        output = self._run(data)
        self.dump(output)

    @staticmethod
    def _run(data: pd.DataFrame) -> pd.DataFrame:
        nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for feature in nan_features:
            data[feature].fillna('unknown', inplace=True)

        cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2',
               'event_type_2']
        for feature in cat:
            encoder = preprocessing.LabelEncoder()
            data[feature] = encoder.fit_transform(data[feature])
        return data


class FeatureEngineering(gokart.TaskOnKart):
    data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.data_task

    def run(self):
        data = self.load_data_frame()
        output = self._run(data)
        self.dump(output)

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
