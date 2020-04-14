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
    predicted_sales_data_task = gokart.TaskInstanceParameter()
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

    @staticmethod
    def _run(df, from_date, to_date):
        buffer = 28 + 180  # TODO: decide buffer automatically
        if from_date and to_date:
            df = df[(from_date - buffer <= df['d']) & (df['d'] < to_date)]

        to_float32 = []

        df['rolling_mean_t60'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
        df['rolling_mean_t90'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
        df['rolling_mean_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
        to_float32.append('rolling_mean_t60')
        to_float32.append('rolling_mean_t90')
        to_float32.append('rolling_mean_t180')

        lags = [7, 28]
        wins = [7, 28]
        for lag in lags:
            column = f'lag{lag}'
            df[column] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag))
            to_float32.append(column)
            for win in wins:
                column = f'rolling_mean_lag{lag}_win{win}'
                df[column] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag).rolling(win).mean())
                to_float32.append(column)

                # TODO: std? It is useless so far.
                # column = f'rolling_std_lag{lag}_win{win}'
                # df[column] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag).rolling(win).std())
                # to_float32.append(column)

        # TODO: shorter lag? NOT good so far.
        # df['rolling_mean_lag1_win13'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(1).rolling(13).mean())
        # to_float32.append('rolling_mean_lag1_win13')

        df[to_float32] = df[to_float32].astype("float32")

        # TODO: must be removed?
        # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings
        # df = df[(df.d >= 1914) | (pd.notna(df.rolling_mean_t180))]
        return df
