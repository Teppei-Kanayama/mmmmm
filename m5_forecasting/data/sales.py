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

    sales_data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.sales_data_task

    def run(self):
        data = self.load_data_frame()
        output = self._run(data)
        self.dump(output)

    @staticmethod
    def _run(df):
        # 28日空いているのは、最大で28日前までのデータしか無いため。

        df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))  # 28日前

        df['rolling_mean_t7'] = df.groupby(['id'])['demand'].transform(
            lambda x: x.shift(28).rolling(7).mean())  # 28日前から28+7日前までの平均
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
