from logging import getLogger

import luigi
import pandas as pd
import gokart

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class PreprocessSales(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    drop_old_data_days: int = luigi.IntParameter(default=None)
    is_small: bool = luigi.BoolParameter()

    def requires(self):
        return LoadInputData(filename='sales_train_validation.csv')

    def run(self):
        required_columns = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'} | set([f'd_{d}' for d in range(1, 1914)])
        data = self.load_data_frame(required_columns=required_columns)
        output = self._run(data, self.drop_old_data_days, self.is_small)
        self.dump(output)

    @staticmethod
    def _run(df: pd.DataFrame, drop_old_data_days: int, is_small: bool) -> pd.DataFrame:
        if is_small:
            df = df.iloc[:3]
        else:
            df = df.drop(["d_" + str(i + 1) for i in range(drop_old_data_days - 1)], axis=1)
        df['id'] = df['id'].str.replace('_validation', '')
        df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])

        df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name='d',
                     value_name='demand')
        df['d'] = df['d'].str[2:].astype('int64')
        return df


class MekeSalesFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    # 特徴量作成のための元データ（真の値）
    sales_data_task = gokart.TaskInstanceParameter()

    # 特徴量作成のための元データ（予測値を真の値と見なして補う）
    predicted_sales_data_task = gokart.TaskInstanceParameter()

    # 特徴量を作る必要がある対象期間
    # Noneの場合は全ての期間に対して特徴量を作る
    from_date: int = luigi.IntParameter(default=None)
    to_date: int = luigi.IntParameter(default=None)

    def requires(self):
        return dict(sales=self.sales_data_task, predicted_sales=self.predicted_sales_data_task)

    def run(self):
        sales = self.load_data_frame('sales')
        predicted_sales = self.load_data_frame('predicted_sales')
        if not predicted_sales.empty:
            sales.loc[sales[(sales['id'].isin(predicted_sales['id']))
                            & (sales['d'].isin(predicted_sales['d']))].index, 'demand'] \
                = predicted_sales['demand'].values
        output = self._run(sales, self.from_date, self.to_date)
        self.dump(output)

    @classmethod
    def _run(cls, df, from_date, to_date):
        # from_date - bufferからto_dateまでに限定して特徴量を作ることで計算量を削減する
        # 過去のsalesデータを使って特徴量を作るためにbufferを用意している
        original_columns = df.columns
        buffer = 28 + 180  # TODO: decide buffer automatically
        if from_date and to_date:
            df = df[(from_date - buffer <= df['d']) & (df['d'] < to_date)]

        # grouped lag
        # lags = [7, 14]
        # levels = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        # for lag in lags:
        #     for level in levels:
        #         df[f'grouped_lag_{level}_lag{lag}'] = cls._calculate_grouped_lag(df, level, lag)

        # lag
        # lags = [i for i in range(28, 28 + 15)]
        # for lag in lags:
        #     df[f'lag{lag}'] = cls._calculate_lag(df, lag)

        # rolling mean
        lags = [7, 14]
        wins = [7, 14, 30, 60]
        for lag in lags:
            for win in wins:
                df[f'rolling_mean_lag{lag}_win{win}'] = cls._calculate_rolling_mean(df, lag, win)

                # TODO: std? It is useless so far.
                # df[f'rolling_std_lag{lag}_win{win}'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag).rolling(win).std())

        # longer rolling mean
        df['rolling_mean_t90'] = cls._calculate_rolling_mean(df, 28, 90)
        df['rolling_mean_t180'] = cls._calculate_rolling_mean(df, 28, 180)

        # TODO: shorter lag? NOT good so far.
        # df['rolling_mean_lag1_win13'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(1).rolling(13).mean())

        # 新たに作成した特徴量を全てfloat32に変換
        to_float32 = list(set(df.columns) - set(original_columns))
        df[to_float32] = df[to_float32].astype("float32")

        # sold out
        win = 60
        df[f'sold_out_{win}'] = cls._calculate_sold_out(df, win)

        # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings
        df = df[(df.d >= 1914) | (pd.notna(df.rolling_mean_t180))]
        return df

    @staticmethod
    def _calculate_lag(df: pd.DataFrame, lag: int, target_column: str = 'demand') -> pd.Series:
        return df.groupby(['id'])[target_column].transform(lambda x: x.shift(lag))

    @staticmethod
    def _calculate_rolling_mean(df: pd.DataFrame, lag: int, win: int) -> pd.Series:
        return df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag).rolling(win).mean())

    @classmethod
    def _calculate_grouped_lag(cls, df: pd.DataFrame, level: str, lag: int) -> pd.Series:
        df = df.copy()
        df['grouped_demand'] = df.groupby(['d', level])['demand'].transform('sum')
        return cls._calculate_lag(df, lag, target_column='grouped_demand')

    @staticmethod
    def _calculate_sold_out(df: pd.DataFrame, win: int) -> pd.Series:
        df = df.copy()
        df['rolling_sum_backward'] = df.groupby('id')['demand'].transform(lambda x: x.rolling(win).sum())
        df['rolling_sum_forward'] = df.groupby('id')['demand'].transform(lambda x: x.rolling(win).sum().shift(1 - win))
        return ((df['rolling_sum_backward'] == 0) | (df['rolling_sum_forward'] == 0)).astype(int)
