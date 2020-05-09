import gokart
import luigi
import pandas as pd
from scipy.stats import norm

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.pointwise_tasks.predict import PredictPointwise
from m5_forecasting.pointwise_tasks.submit import Load
from m5_forecasting.uncertainty_tasks.constant_values import *
from m5_forecasting.utils.pandas_utils import cross_join

# 予測誤差の分散を計算する
# 複数のlevelでgroupbyする
# 複数のpercentileに対する
# level, percentileの情報をidに込める
# 出力： id, percentile


class CalculateVariance(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    variance_from_date: int = luigi.IntParameter()
    variance_to_date: int = luigi.IntParameter()
    interval: int = luigi.IntParameter()

    def requires(self):
        ground_truth_task = LoadInputData(filename='sales_train_validation.csv')
        prediction_load_tasks = [Load(from_date=t, to_date=t + self.interval)
                                 for t in range(self.variance_from_date, self.variance_to_date, self.interval)]
        return dict(ground_truth=ground_truth_task, predict=prediction_load_tasks)

    def run(self):
        ground_truth = self.load_data_frame('ground_truth')
        prediction = pd.concat(self.load('predict'))

        import pdb; pdb.set_trace()
