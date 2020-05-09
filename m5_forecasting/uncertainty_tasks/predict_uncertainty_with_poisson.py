from functools import reduce

import gokart
import pandas as pd
from scipy.stats import poisson

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.uncertainty_tasks.constant_values import *
from m5_forecasting.utils.pandas_utils import cross_join


class PredictUncertaintyWithPoisson(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        accuracy_task = LoadInputData(filename='kkiller_first_public_notebook_under050_v5.csv')
        # accuracy_task = LoadInputData(filename='submission_1499b9c5b60efee9f8358927876a8d26.csv')
        return dict(accuracy=accuracy_task)

    def run(self):
        accuracy = self.load_data_frame('accuracy')
        output = self._run(accuracy)
        self.dump(output)

    @classmethod
    def _run(cls, accuracy: pd.DataFrame) -> pd.DataFrame:
        accuracy = accuracy[accuracy['id'].str.contains('validation')]
        poisson_distribution_df = cls._get_poisson_distribution_df()
        df_point = cross_join(accuracy, poisson_distribution_df)
        df_point.loc[:, COLS] = df_point.loc[:, COLS].round()

        df_list = []
        for col in COLS:
            df = pd.merge(df_point[['id', col, 'percentile']], poisson_distribution_df, left_on=[col, 'percentile'], right_on=['mu', 'percentile'])
            df = df[['id', 'percentile', 'value']].rename(columns={'value': col})
            df_list.append(df)
        df_final = reduce(lambda x, y: pd.merge(x, y), df_list)
        df_final['id'] = [f"{lev.replace('_validation', '')}_{q: .3f}_validation" for lev, q in zip(df_final['id'].values, df_final['percentile'].values)]
        df_final = df_final.drop('percentile', axis=1)
        df_final = df_final[~df_final['id'].str.contains('0.500')]
        return df_final

    @staticmethod
    def _get_poisson_distribution_df() -> pd.DataFrame:

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
        for i in range(160):
            df = f(i)
            df['mu'] = i
            l.append(df)
        return pd.concat(l)
