from functools import reduce

import gokart
import luigi
import pandas as pd
from scipy.stats import poisson

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.uncertainty_tasks.constant_values import *
from m5_forecasting.utils.pandas_utils import get_uncertainty_ids


# 条件1: 予測値（の集約値）が mu_upper_bound である
# 条件2: 50 percentileでない

class PredictUncertaintyWithPoisson(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    mu_upper_bound: int = luigi.IntParameter(default=20)

    def requires(self):
        accuracy_task = LoadInputData(filename='kkiller_first_public_notebook_under050_v5.csv')
        sales_data_task = LoadInputData(filename='sales_train_validation.csv')
        # accuracy_task = LoadInputData(filename='submission_1499b9c5b60efee9f8358927876a8d26.csv')
        return dict(accuracy=accuracy_task, sales=sales_data_task)

    def run(self):
        accuracy = self.load_data_frame('accuracy')
        sales = self.load_data_frame('sales')
        output = self._run(accuracy, sales, self.mu_upper_bound)
        self.dump(output)

    @classmethod
    def _run(cls, accuracy: pd.DataFrame, sales: pd.DataFrame, mu_upper_bound: int) -> pd.DataFrame:
        accuracy = accuracy.merge(sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]], on="id")
        accuracy["_all_"] = "Total"
        poisson_distribution_df = cls._get_poisson_distribution_df(mu_upper_bound=mu_upper_bound)

        df_list = [cls._calculate_uncertainty(accuracy, poisson_distribution_df, levels) for levels in LEVELS]
        df = pd.concat(df_list)
        df = df[~df['id'].str.contains('0.500')]  # 条件2
        return df

    @staticmethod
    def _calculate_uncertainty(point_prediction, poisson_distribution_df, levels) -> pd.DataFrame:
        df = point_prediction.groupby(levels)[COLS].sum()
        q = np.repeat(PERCENTILES, len(df))
        df = pd.concat([df] * 9, axis=0, sort=False).reset_index()
        df['percentile'] = q
        df.loc[:, COLS] = df.loc[:, COLS].round()
        df['id'] = get_uncertainty_ids(df, levels)

        df_list = []
        for col in COLS:
            df_one_day = pd.merge(df[['id', col, 'percentile']], poisson_distribution_df, left_on=[col, 'percentile'],
                                  right_on=['mu', 'percentile'])
            df_one_day = df_one_day[['id', 'percentile', 'value']].rename(columns={'value': col})
            df_list.append(df_one_day)
        df_final = reduce(lambda x, y: pd.merge(x, y), df_list)
        df_final = df_final.drop('percentile', axis=1)
        return df_final

    @staticmethod
    def _get_poisson_distribution_df(mu_upper_bound: int) -> pd.DataFrame:

        def f(mu, size=400):
            xs = np.arange(size)
            cmfs = np.array([poisson.cdf(x, mu) for x in xs])
            d_lower = {percentile: np.max(xs[cmfs < percentile]) if len(xs[cmfs < percentile]) else 0 for percentile in
                       PERCENTILE_LOWERS}
            d_upper = {percentile: np.min(xs[cmfs > percentile]) for percentile in PERCENTILE_UPPERS}
            d = {**d_lower, 0.5: mu, **d_upper}

            if mu == 0:
                return pd.DataFrame(dict(percentile=list(d.keys()), value=[0, 0, 0, 0, 0, 1, 1, 1, 1]))

            return pd.DataFrame(dict(percentile=list(d.keys()), value=list(d.values())))

        l = []
        for i in range(mu_upper_bound):  # 条件1
            df = f(i)
            df['mu'] = i
            l.append(df)
        return pd.concat(l)
