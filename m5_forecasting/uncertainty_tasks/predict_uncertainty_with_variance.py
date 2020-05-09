from typing import List, Dict, Union

import gokart
import luigi
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.utils.file_processors import RoughCsvFileProcessor

PERCENTILES = np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])

LEVELS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "_all_"]
COUPLES = [["state_id", "item_id"], ["state_id", "dept_id"], ["store_id", "dept_id"], ["state_id", "cat_id"],
           ["store_id", "cat_id"]]
COLS = [f"F{i}" for i in range(1, 29)]


def cross_join(df1, df2):
    df1['key'] = 0
    df2['key'] = 0
    return df1.merge(df2, how='outer').drop('key', axis=1)


class CalculateVariance(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return LoadInputData(filename='sales_train_validation.csv')

    def run(self):
        sales = self.load_data_frame()
        sales["_all_"] = "Total"
        level_list = [["item_id"], ["dept_id"], ["cat_id"], ["store_id"], ["state_id"], ["_all_"],
                      ["state_id", "item_id"], ["state_id", "dept_id"], ["store_id", "dept_id"],
                      ["state_id", "cat_id"], ["store_id", "cat_id"]]
        training_columns = [f'd_{i}' for i in range(1913 - 365 + 1, 1913 + 1)]

        variance_list = []
        for level in level_list:
            agg_sales = sales[level + training_columns].groupby(level).sum()
            sales_variance = agg_sales.var(axis=1).reset_index().rename(columns={0: 'variance'})
            sales_variance['sigma'] = np.sqrt(sales_variance['variance'])

            percentile_df = pd.DataFrame(dict(percentile=PERCENTILES))
            percentile_df['n_sigma'] = percentile_df['percentile'].apply(norm.ppf)

            sales_variance = cross_join(sales_variance, percentile_df)
            sales_variance['percentile_diff'] = sales_variance['sigma'] * sales_variance['n_sigma']

            if len(level) > 1:
                sales_variance["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1, lev2, q in
                            zip(sales_variance[level[0]].values, sales_variance[level[1]].values, sales_variance['percentile'].values)]
            else:
                sales_variance["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(sales_variance[level[0]].values, sales_variance['percentile'].values)]

            variance_list.append(sales_variance)

        variance_df = pd.concat(variance_list)
        self.dump(variance_df[['id', 'percentile_diff']])


class PredictUncertaintyWithVariance(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        accuracy_task = LoadInputData(filename='kkiller_first_public_notebook_under050_v5.csv')
        # accuracy_task = LoadInputData(filename='submission_1499b9c5b60efee9f8358927876a8d26.csv')
        sales_data_task = LoadInputData(filename='sales_train_validation.csv')
        variance_task = CalculateVariance()
        return dict(accuracy=accuracy_task, sales=sales_data_task, variance=variance_task)

    def run(self):
        accuracy = self.load_data_frame('accuracy')
        sales = self.load_data_frame('sales')
        variance = self.load_data_frame('variance')
        output = self._run(accuracy, sales, variance)
        self.dump(output)

    @classmethod
    def _run(cls, accuracy, sales, variance):
        sub = accuracy.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on="id")
        sub["_all_"] = "Total"
        df_list = [cls._calculate_uncertaity(sub, variance, levels) for levels in (COUPLES + LEVELS)]
        df = pd.concat(df_list, axis=0, sort=False).reset_index(drop=True)
        return df

    @staticmethod
    def _calculate_uncertaity(point_prediction, variance, levels):
        df = point_prediction.groupby(levels)[COLS].sum()
        q = np.repeat(PERCENTILES, len(df))
        df = pd.concat([df] * 9, axis=0, sort=False).reset_index()
        if type(levels) == list:
            df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1, lev2, q in
                        zip(df[levels[0]].values, df[levels[1]].values, q)]
        elif levels != "id":
            df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in zip(df[levels].values, q)]
        else:
            df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in zip(df[levels].values, q)]
        df = df[["id"] + COLS]

        df = pd.merge(df, variance)
        df.loc[:, COLS] = df[COLS].values + (df[['percentile_diff'] * len(COLS)]).values
        df.loc[:, COLS] = df[COLS].clip(0)
        return df.drop('percentile_diff', axis=1)


# DATA_SIZE=small python main.py m5-forecasting.PredictUncertainty --interval=7 --local-scheduler
# python main.py m5-forecasting.PredictUncertaintyWithVariance --interval=7 --local-scheduler
