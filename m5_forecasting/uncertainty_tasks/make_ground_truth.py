import gokart
import luigi
import numpy as np
import pandas as pd
import scipy.stats as stats

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.pointwise_tasks.submit import SubmitPointwise


class MakeGroundTruth(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()
    interval: int = luigi.IntParameter()

    def requires(self):
        pointwise_prediction_task = SubmitPointwise(is_small=self.is_small, interval=self.interval)
        ground_truth_task = LoadInputData(filename='sales_train_validation.csv')
        return dict(pointwise_prediction=pointwise_prediction_task, ground_truth=ground_truth_task)

    def run(self):
        pointwise_prediction = self.load_data_frame('pointwise_prediction')
        ground_truth = self.load_data_frame('ground_truth')
        output = self._run(pointwise_prediction, ground_truth)
        self.dump(output)

    @staticmethod
    def _run(pointwise_prediction, ground_truth):
        import pdb; pdb.set_trace()
