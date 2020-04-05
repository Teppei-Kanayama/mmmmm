from logging import getLogger

import pandas as pd
import gokart
from sklearn.preprocessing import OrdinalEncoder

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
