from typing import List

import gokart
import luigi
import pandas as pd
from scipy.stats import norm

from m5_forecasting.data.sales import PreprocessSales
from m5_forecasting.pointwise_tasks.submit import Load
from m5_forecasting.uncertainty_tasks.constant_values import *
from m5_forecasting.utils.pandas_utils import cross_join, get_uncertainty_ids


class CalculateVariance(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    variance_from_date: int = luigi.IntParameter()
    variance_to_date: int = luigi.IntParameter()
    interval: int = luigi.IntParameter()

    def requires(self):
        ground_truth_task = PreprocessSales(to_date=self.valiance_to_date, is_small=self.is_small)
        prediction_load_tasks = [Load(from_date=t, to_date=t + self.interval)
                                 for t in range(self.variance_from_date, self.variance_to_date, self.interval)]
        return dict(ground_truth=ground_truth_task, predict=prediction_load_tasks)

    def run(self):
        ground_truth = self.load_data_frame('ground_truth')
        prediction = pd.concat(self.load('predict'))
        output = self._run(ground_truth, prediction)
        self.dump(output)

    @classmethod
    def _run(cls, ground_truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
        ground_truth["_all_"] = "Total"
        prediction["_all_"] = "Total"
        variance_list = [cls._calculate_variance(level, ground_truth, prediction) for level in LEVELS]
        variance = pd.concat(variance_list)
        return variance

    @staticmethod
    def _calculate_variance(level: List, ground_truth: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
        ground_truth_agg = ground_truth.groupby(level + ['d'], as_index=False).agg({'demand': 'sum'})
        prediction_agg = prediction.groupby(level + ['d'], as_index=False).agg({'demand': 'sum'})
        df = pd.merge(prediction_agg, ground_truth_agg, on=level + ['d'], suffixes=['_pred', '_gt'])
        df['demand_diff'] = df['demand_gt'] - df['demand_pred']
        agg_df = df.groupby(level, as_index=False).agg({'demand_diff': ['mean', 'var']})
        agg_df = pd.DataFrame(agg_df.to_records())
        agg_df['sigma'] = np.sqrt(agg_df["('demand_diff', 'var')"])
        percentile_df = pd.DataFrame(dict(percentile=PERCENTILES))
        percentile_df['n_sigma'] = percentile_df['percentile'].apply(norm.ppf)
        variance_df = cross_join(agg_df, percentile_df)
        variance_df['percentile_diff'] = variance_df['sigma'] * variance_df['n_sigma']
        for lev in level:
            variance_df = variance_df.rename(columns={f"('{lev}', '')": lev})
        variance_df['id'] = get_uncertainty_ids(variance_df, level)
        return variance_df[['id', 'percentile_diff']]
