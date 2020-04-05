from logging import getLogger

import pandas as pd
import gokart

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class PreprocessSales(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return LoadInputData(filename='sales_train_validation.csv')

    def run(self):
        required_columns = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'} | set([f'd_{d}' for d in range(1, 1914)])
        data = self.load_data_frame(required_columns=required_columns)
        output = self._run(data)
        self.dump(output)

    @classmethod
    def _run(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls._reshape_sales(df, 1000)
        df = cls._prep_sales(df)
        return df

    @staticmethod
    def _reshape_sales(df, drop_d=None):
        if drop_d is not None:
            df = df.drop(["d_" + str(i + 1) for i in range(drop_d - 1)], axis=1)  # 1日目からdrop_d日目までを削除する（古すぎるから？）
        df['id'] = df['id'].str.replace('_validation', '')

        # validation, evaluation 期間も足す（値はNaNが入る）
        df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])

        # もともとは (unique id)行だったが、 (unique id * 時系列)行に変換する。1商品・1日ごとに1行
        df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], var_name='d',
                     value_name='demand')

        df['d'] = df['d'].str[2:].astype('int64')
        return df

    @staticmethod
    def _prep_sales(df):
        df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
        df['rolling_mean_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
        df['rolling_mean_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
        df['rolling_mean_t60'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
        df['rolling_mean_t90'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
        df['rolling_mean_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
        df['rolling_std_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
        df['rolling_std_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())

        to_float32 = ['lag_t28', 'rolling_mean_t7', 'rolling_mean_t30', 'rolling_mean_t60', 'rolling_mean_t90',
                      'rolling_mean_t180', 'rolling_std_t7', 'rolling_std_t30']
        df[to_float32] = df[to_float32].astype("float32")

        # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings
        df = df[(df.d >= 1914) | (pd.notna(df.rolling_mean_t180))]
        return df
