from logging import getLogger

import pandas as pd
import gokart
from sklearn.preprocessing import OrdinalEncoder

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)

# original columns

# wm_yr_wk: 週のIDで、全データを通して異なる11101~11621の通し番号。
# wday: 曜日ID。日曜日が0
# month: 1から12まで
# year: 2011~2016
# d: データ開始日から終了日までの通し番号。1~1969

# artificial columns

# week_of_year: 1年間における第何週か。1~52くらい。
# quarter: 日付の四半期。1~4。
# day: 日にち。1~31。


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
        df = df.assign(d=df['d'].str[2:].astype(int))

        # 'event_name_1', 'event_type_1'を label encodeする
        to_ordinal = ["event_name_1", "event_type_1"]
        df[to_ordinal] = df[to_ordinal].fillna("1")
        df[to_ordinal] = OrdinalEncoder(dtype="int").fit_transform(df[to_ordinal]) + 1

        # イベントの前後を表す特徴量 -> 今のところ意味なし
        # for i in range(7):
        #     df[f'event_name_1_{i}'] = df['event_name_1'].shift(i)
        #     df[f'event_name_1_-{i}'] = df['event_name_1'].shift(-i)

        # 日付に関する特徴量を追加する
        df['date'] = pd.to_datetime(df['date'])
        df['week_of_year'] = df['date'].dt.weekofyear
        df['quarter'] = df['date'].dt.quarter
        df['day'] = df['date'].dt.day

        to_int8 = ["wday", "month", "snap_CA", "snap_TX", "snap_WI", 'week_of_year', 'quarter', 'day'] + to_ordinal
        df[to_int8] = df[to_int8].astype("int8")

        # dateはd, weekdayはwdayと同じ情報なので落とす。
        # event_name_2, event_type_2はほぼ空なので使わない。
        df = df.drop(["date", "weekday", "event_name_2", "event_type_2"], axis=1)
        return df

