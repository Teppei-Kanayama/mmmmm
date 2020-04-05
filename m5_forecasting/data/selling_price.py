from logging import getLogger

import pandas as pd
import gokart

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


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
