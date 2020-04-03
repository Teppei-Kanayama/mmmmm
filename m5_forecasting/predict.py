from logging import getLogger

import gokart

from m5_forecasting.data.preprocess import MergeInputData, LabelEncode, FeatureEngineering
from m5_forecasting.tasks.run_lgbm import RunLGBM

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class Predict(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        return self.make_target('submission.csv', use_unique_id=False)

    def requires(self):
        input_data_task = LoadInputData()
        merged_data_task = MergeInputData(data_task=input_data_task)
        label_encode_task = LabelEncode(data_task=merged_data_task)
        feature_task = FeatureEngineering(data_task=label_encode_task)
        run_model_task = RunLGBM(feature_task=feature_task, raw_data_task=input_data_task)
        return run_model_task

    def run(self):
        output = self.load_data_frame()
        self.dump(output)

# python main.py m5-forecasting.Predict --local-scheduler
