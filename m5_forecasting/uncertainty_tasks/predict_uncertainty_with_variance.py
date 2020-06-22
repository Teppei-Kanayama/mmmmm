from typing import List

import gokart
import pandas as pd

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.uncertainty_tasks.calculate_variance import CalculateVariance
from m5_forecasting.uncertainty_tasks.constant_values import *
from m5_forecasting.utils.pandas_utils import get_uncertainty_ids


class PredictUncertaintyWithVariance(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        accuracy_task = LoadInputData(filename='kkiller_first_public_notebook_under050_v5.csv')
        sales_data_task = LoadInputData(filename='sales_train_validation.csv')
        variance_task = CalculateVariance()
        return dict(accuracy=accuracy_task, sales=sales_data_task, variance=variance_task)

    def run(self):
        accuracy = self.load_data_frame('accuracy')
        sales = self.load_data_frame('sales')
        variance = self.load_data_frame('variance', required_columns={'id', 'percentile_diff'})
        output = self._run(accuracy, sales, variance)
        self.dump(output)

    @classmethod
    def _run(cls, accuracy: pd.DataFrame, sales: pd.DataFrame, variance: pd.DataFrame) -> pd.DataFrame:
        sub = accuracy.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on="id")
        sub["_all_"] = "Total"
        df_list = [cls._calculate_uncertaity(sub, variance, levels) for levels in LEVELS]
        df = pd.concat(df_list, axis=0, sort=False).reset_index(drop=True)
        df = cls._float_to_int(df)
        return df

    @staticmethod
    def _calculate_uncertaity(point_prediction, variance, levels: List):
        df = point_prediction.groupby(levels)[COLS].sum()
        q = np.repeat(PERCENTILES, len(df))
        df = pd.concat([df] * 9, axis=0, sort=False).reset_index()
        df['percentile'] = q
        df['id'] = get_uncertainty_ids(df, levels)
        df = df[['id'] + COLS]
        df = pd.merge(df, variance)
        df.loc[:, COLS] = df[COLS].values + (df[['percentile_diff'] * len(COLS)]).values  # main
        df.loc[:, COLS] = df[COLS].clip(0)
        return df.drop('percentile_diff', axis=1)

    @staticmethod
    def _float_to_int(df: pd.DataFrame) -> pd.DataFrame:
        for u in PERCENTILE_LOWERS:
            df.loc[df['id'].str.contains(str(u)), COLS] = np.floor(df.loc[df['id'].str.contains(str(u)), COLS])
        for u in PERCENTILE_UPPERS:
            df.loc[df['id'].str.contains(str(u)), COLS] = np.ceil(df.loc[df['id'].str.contains(str(u)), COLS])
        return df
