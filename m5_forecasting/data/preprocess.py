from logging import getLogger

import pandas as pd
import gc
import gokart
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class PreprocessCalendar(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return LoadInputData(filename='calendar.csv')

    def run(self):
        data = self.load_data_frame()
        output = self._run(data)
        self.dump(output)

    @staticmethod
    def _run(df: pd.DataFrame) -> pd.DataFrame:
        # dateはd, weekdayはwdayと同じ情報なので落とす。event_name_2, event_type_2はなぜ使わない？
        df = df.drop(["date", "weekday", "event_name_2", "event_type_2"], axis=1)
        df = df.assign(d=df['d'].str[2:].astype(int))  # 'd_100' などの日付を100に変換する
        to_ordinal = ["event_name_1", "event_type_1"]
        df[to_ordinal] = df[to_ordinal].fillna("1")  # なんでもいいから埋める
        df[to_ordinal] = OrdinalEncoder(dtype="int").fit_transform(df[to_ordinal]) + 1  # 'ValentinesDay'などの文字列を数字に対応させる
        to_int8 = ["wday", "month", "snap_CA", "snap_TX", "snap_WI"] + to_ordinal
        df[to_int8] = df[to_int8].astype("int8")  # int64は無駄なのでint8に落とす
        return df  # columns: {'wm_yr_wk', 'wday', 'month', 'year', 'd', 'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI'}


class PreprocessSellingPrice(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return LoadInputData(filename='sell_prices.csv')

    def run(self):
        data = self.load_data_frame(required_columns={'store_id', 'item_id', 'wm_yr_wk', 'sell_price'})
        output = self._run(data)
        self.dump(output)

    @classmethod
    def _run(cls, df: pd.DataFrame) -> pd.DataFrame:
        gr = df.groupby(["store_id", "item_id"])["sell_price"]
        df["sell_price_rel_diff"] = gr.pct_change()  # なんか集約して新しい特徴量を作っている
        df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())  # なんか集約して新しい特徴量を作っている
        df["sell_price_roll_sd7"] = cls._zapsmall(gr.transform(lambda x: x.rolling(7).std()))  # なんか集約して新しい特徴量を作っている
        to_float32 = ["sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7"]
        df[to_float32] = df[to_float32].astype("float32")  # float64は無駄なのでfloat32に落とす
        return df  # original columns + 'sell_price_rel_diff', 'sell_price_cumrel', 'sell_price_roll_sd7'

    @staticmethod
    def _zapsmall(z: pd.DataFrame, tol=1e-6) -> pd.DataFrame:
        z[abs(z) < tol] = 0  # 小さすぎるデータは切り捨てる
        return z


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


class MergeData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    calendar_data_task = gokart.TaskInstanceParameter()
    selling_price_data_task = gokart.TaskInstanceParameter()
    sales_data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return dict(calendar=self.calendar_data_task, selling_price=self.selling_price_data_task,
                    sales=self.sales_data_task)

    def run(self):
        calendar = self.load_data_frame('calendar')
        selling_price = self.load_data_frame('selling_price')
        sales = self.load_data_frame('sales')
        output = self._run(calendar, selling_price, sales)
        self.dump(output)

    @staticmethod
    def _run(calendar, selling_price, sales):
        sales = sales.merge(calendar, how="left", on="d")
        gc.collect()

        sales = sales.merge(selling_price, how="left", on=["store_id", "item_id", "wm_yr_wk"])
        sales.drop(["wm_yr_wk"], axis=1, inplace=True)
        gc.collect()
        del selling_price

        return sales


class MakeFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    merged_data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.merged_data_task

    def run(self):
        data = self.load_data_frame()
        self.dump(self._run(data))

    @staticmethod
    def _run(data):
        for i, v in tqdm(enumerate(["item_id", "dept_id", "store_id", "cat_id", "state_id"])):
            data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]]).astype("int16") + 1
        gc.collect()
        return data
