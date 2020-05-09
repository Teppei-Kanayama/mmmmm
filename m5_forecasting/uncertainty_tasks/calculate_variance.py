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
        ground_truth_task = PreprocessSales(is_small=self.is_small)
        prediction_load_tasks = [Load(from_date=t, to_date=t + self.interval)
                                 for t in range(self.variance_from_date, self.variance_to_date, self.interval)]
        return dict(ground_truth=ground_truth_task, predict=prediction_load_tasks)

    def run(self):
        ground_truth = self.load_data_frame('ground_truth')
        prediction = pd.concat(self.load('predict'))

        df = pd.merge(prediction, ground_truth, on=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd'], suffixes=['_pred', '_gt'])
        df['diff'] = df['demand_gt'] - df['demand_pred']
        df["_all_"] = "Total"

        # TODO: DRY
        level_list = [["id"], ["item_id"], ["dept_id"], ["cat_id"], ["store_id"], ["state_id"], ["_all_"],
                      ["state_id", "item_id"], ["state_id", "dept_id"], ["store_id", "dept_id"], ["state_id", "cat_id"],
                      ["store_id", "cat_id"]]

        variance_list = []
        for level in level_list:
            agg_df = df.groupby(level, as_index=False).agg({'diff': 'var'}).rename(columns={'diff': 'variance'})
            agg_df['sigma'] = np.sqrt(agg_df['variance'])
            percentile_df = pd.DataFrame(dict(percentile=PERCENTILES))
            percentile_df['n_sigma'] = percentile_df['percentile'].apply(norm.ppf)
            variance_df = cross_join(agg_df, percentile_df)
            variance_df['percentile_diff'] = variance_df['sigma'] * variance_df['n_sigma']
            variance_df['id'] = get_uncertainty_ids(variance_df, level)

            # TODO: DRY
            # if len(level) > 1:
            #     variance_df["id"] = [f"{lev1}_{lev2}_{q:.3f}_validation" for lev1, lev2, q in
            #                             zip(variance_df[level[0]].values, variance_df[level[1]].values,
            #                                 variance_df['percentile'].values)]
            # elif level[0] == "id":
            #     variance_df["id"] = [f"{lev.replace('_validation', '')}_{q:.3f}_validation" for lev, q in
            #                          zip(variance_df['id'].values, variance_df['percentile'])]
            # else:
            #     variance_df["id"] = [f"{lev}_X_{q:.3f}_validation" for lev, q in
            #                          zip(variance_df[level[0]].values, variance_df['percentile'].values)]
            variance_list.append(variance_df)

        variance = pd.concat(variance_list)
        self.dump(variance[['id', 'percentile_diff']])
