from logging import getLogger

import luigi
import pandas as pd
import gokart
import numpy as np

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.utils.pandas_utils import cross_join

logger = getLogger(__name__)


class PreprocessSales(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    from_date: int = luigi.IntParameter()
    to_date: int = luigi.IntParameter()
    is_small: bool = luigi.BoolParameter()

    def requires(self):
        return LoadInputData(filename='sales_train_evaluation.csv')

    def run(self):
        required_columns = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'} | set([f'd_{d}' for d in range(1, 1942)])
        data = self.load_data_frame(required_columns=required_columns)
        output = self._run(data, self.from_date, self.to_date, self.is_small)
        self.dump(output)

    @staticmethod
    def _run(df: pd.DataFrame, from_date: int, to_date: int, is_small: bool) -> pd.DataFrame:
        if is_small:
            df = df.iloc[:3]
        df = df.drop(["d_" + str(i + 1) for i in range(from_date - 1)], axis=1)
        df['id'] = df['id'].str.replace('_evaluation', '')
        # df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1942 + i) for i in range(28)])
        df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name='d', value_name='demand')
        df['d'] = df['d'].str[2:].astype('int64')
        df = df[df['d'] < to_date]
        return df


class MekeSalesFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    sales_data_task = gokart.TaskInstanceParameter()  # 真のsalesデータ
    predicted_sales_data_task = gokart.TaskInstanceParameter()  # 予測されたsalesデータ
    make_feature_to_date: int = luigi.IntParameter(default=None)

    def requires(self):
        return dict(sales=self.sales_data_task, predicted_sales=self.predicted_sales_data_task)

    def run(self):
        sales = self.load_data_frame('sales')
        predicted_sales = self.load_data_frame('predicted_sales')

        sales = pd.concat([sales, predicted_sales]).reset_index(drop=True)
        output = self._run(sales, self.make_feature_to_date)
        self.dump(output)

    @classmethod
    def _run(cls, df: pd.DataFrame, make_feature_to_date: int) -> pd.DataFrame:
        original_columns = df.columns
        future_sales_df = cls._make_future_sales(df, from_date=df['d'].max() + 1, to_date=make_feature_to_date)
        df = pd.concat([df, future_sales_df])
        df = cls._make_feature(df)
        to_float32 = list(set(df.columns) - set(original_columns))
        df[to_float32] = df[to_float32].astype("float32")
        return df

    @classmethod
    def _make_feature(cls, df):
        # grouped rolling mean
        # lags = [7, 28]
        # wins = [7, 28]
        # levels = ['item_id', 'store_id']
        # for lag in lags:
        #     for win in wins:
        #         for level in levels:
        #             df[f'grouped_lag_{level}_lag{lag}_win{win}'] = cls._calculate_grouped_rolling_mean(df, level, lag, win)

        # lag
        # lags = [28 + 7 * i for i in range(5)]
        lags = [7, 28]
        for lag in lags:
            df[f'lag{lag}'] = cls._calculate_lag(df, lag)

        # rolling mean
        # lags = [28 * i for i in range(1, 14)]
        # wins = [28]
        lags = [7, 28]
        wins = [7, 28]
        for lag in lags:
            for win in wins:
                df[f'rolling_mean_lag{lag}_win{win}'] = cls._calculate_rolling_mean(df, lag, win)
                # df[f'rolling_std_lag{lag}_win{win}'] = cls._calculate_rolling_std(df, lag, win)

        # shorter lag
        # df['lag7'] = cls._calculate_lag(df, 7)
        # df['lag_short_mean'] = cls._calculate_short_lag(df, stat='mean')
        # df['lag_short_median'] = cls._calculate_short_lag(df, stat='median')
        # df['lag_short_std'] = cls._calculate_short_lag(df, stat='std')
        # df[f'rolling_mean_lag7_win28'] = cls._calculate_rolling_mean(df, 7, 28)
        # df[f'rolling_mean_lag7_win7'] = cls._calculate_rolling_mean(df, 7, 7)

        # longer rolling mean
        df['rolling_mean_t60'] = cls._calculate_rolling_mean(df, 28, 60)
        df['rolling_mean_t90'] = cls._calculate_rolling_mean(df, 28, 90)
        df['rolling_mean_t180'] = cls._calculate_rolling_mean(df, 28, 180)

        # sold out
        win = 60
        df[f'sold_out_{win}'] = cls._calculate_sold_out(df, win)

        return df

    @staticmethod
    def _make_future_sales(df: pd.DataFrame, from_date: int, to_date: int) -> pd.DataFrame:
        base_df = df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
        d_df = pd.DataFrame(dict(d=np.arange(from_date, to_date)))
        future_sales_df = cross_join(base_df, d_df)
        future_sales_df['demand'] = None
        return future_sales_df

    @classmethod
    def _calculate_short_lag(cls, df: pd.DataFrame, stat: str) -> pd.Series:
        tmp = pd.DataFrame()
        tmp['s1'] = cls._calculate_lag(df, 7)
        tmp['s2'] = cls._calculate_lag(df, 14)
        tmp['s3'] = cls._calculate_lag(df, 21)
        tmp['s4'] = cls._calculate_lag(df, 28)
        if stat == 'mean':
            return tmp.mean(axis=1)
        if stat == 'median':
            return tmp.median(axis=1)
        if stat == 'std':
            return tmp.std(axis=1)
        raise Exception('stat is invalid!')

    @staticmethod
    def _calculate_lag(df: pd.DataFrame, lag: int, target_column: str = 'demand') -> pd.Series:
        return df.groupby(['id'])[target_column].transform(lambda x: x.shift(lag))

    @staticmethod
    def _calculate_rolling_mean(df: pd.DataFrame, lag: int, win: int, target_column: str = 'demand') -> pd.Series:
        return df.groupby(['id'])[target_column].transform(lambda x: x.shift(lag).rolling(win).mean())

    @staticmethod
    def _calculate_rolling_std(df: pd.DataFrame, lag: int, win: int, target_column: str = 'demand') -> pd.Series:
        return df.groupby(['id'])[target_column].transform(lambda x: x.shift(lag).rolling(win).std())

    @classmethod
    def _calculate_grouped_lag(cls, df: pd.DataFrame, level: str, lag: int) -> pd.Series:
        df = df.copy()
        df['grouped_demand'] = df.groupby(['d', level])['demand'].transform('median')
        return cls._calculate_lag(df, lag, target_column='grouped_demand')

    @classmethod
    def _calculate_grouped_rolling_mean(cls, df: pd.DataFrame, level: str, lag: int, win: int) -> pd.Series:
        df = df.copy()
        df['grouped_demand'] = df.groupby(['d', level])['demand'].transform('median')
        return cls._calculate_rolling_mean(df, lag, win, target_column='grouped_demand')

    @staticmethod
    def _calculate_sold_out(df: pd.DataFrame, win: int) -> pd.Series:
        df = df.copy()
        df['rolling_sum_backward'] = df.groupby('id')['demand'].transform(lambda x: x.rolling(win).sum())
        df['rolling_sum_forward'] = df.groupby('id')['demand'].transform(lambda x: x.rolling(win).sum().shift(1 - win))
        return ((df['rolling_sum_backward'] == 0) | (df['rolling_sum_forward'] == 0)).astype(int)
