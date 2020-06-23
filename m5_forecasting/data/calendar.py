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

        event_columns = ["event_name_1", "event_type_1"]
        df[event_columns] = df[event_columns].fillna('')

        # NBA feature
        NBA_DAYS = list(range(123, 136)) + list(range(501, 511)) + list(range(860, 875)) + list(range(1224, 1235)) \
                   + list(range(1588, 1601)) + list(range(1952, 1970))
        nba_df = pd.DataFrame(dict(d=NBA_DAYS, nba=1))
        df = pd.merge(df, nba_df, how='left').fillna(0)
        df.loc[df['event_name_1'].str.contains('NBA'), 'event_type_1'] = ''
        df.loc[df['event_name_1'].str.contains('NBA'), 'event_name_1'] = ''

        # Ramadan feature
        RAMADAN_DAYS = list(range(185, 185+31)) + list(range(589, 589+31)) + list(range(893, 893+31)) \
                       + list(range(1248, 1248+31)) + list(range(1602, 1602+31)) + list(range(1957, 1957+31))
        ramadan_df = pd.DataFrame(dict(d=RAMADAN_DAYS, ramadan=1))
        df = pd.merge(df, ramadan_df, how='left').fillna(0)

        # merge event1 and evnet2
        df.loc[df['d'] == 1234, 'event_name_1'] = df.loc[df['d'] == 1234, 'event_name_2']
        df.loc[df['d'] == 1234, 'event_type_1'] = df.loc[df['d'] == 1234, 'event_type_2']
        df.loc[df['d'] == 1969, 'event_name_1'] = df.loc[df['d'] == 1969, 'event_name_2']
        df.loc[df['d'] == 1969, 'event_type_1'] = df.loc[df['d'] == 1969, 'event_type_2']

        df[event_columns] = OrdinalEncoder(dtype="int").fit_transform(df[event_columns]) + 1

        # 日付に関する特徴量を追加する
        df['date'] = pd.to_datetime(df['date'])
        df['week_of_year'] = df['date'].dt.weekofyear
        df['quarter'] = df['date'].dt.quarter
        df['day'] = df['date'].dt.day

        to_int8 = ["wday", "month", "snap_CA", "snap_TX", "snap_WI", 'week_of_year', 'quarter', 'day', 'nba', 'ramadan'] + event_columns
        df[to_int8] = df[to_int8].astype("int8")

        # dateはd, weekdayはwdayと同じ情報なので落とす。
        df = df.drop(["date", "weekday", "event_name_2", "event_type_2"], axis=1)
        return df
