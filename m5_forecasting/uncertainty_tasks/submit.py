import gokart
import pandas as pd

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.uncertainty_tasks.predict_uncertainty_with_variance import PredictUncertaintyWithVariance
from m5_forecasting.utils.file_processors import RoughCsvFileProcessor


class SubmitUncertainty(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        return self.make_target('submission_uncertainty.csv')

    def requires(self):
        uncertainty_validation_task = PredictUncertaintyWithVariance(phase='validation')
        uncertainty_evaluation_task = PredictUncertaintyWithVariance(phase='evaluation')
        sample_submission_task = LoadInputData(filename='sample_submission_uncertainty.csv')
        return dict(uncertainty_validation=uncertainty_validation_task,
                    uncertainty_evaluation=uncertainty_evaluation_task,
                    sample=sample_submission_task)

    def run(self):
        score_validation = self.load_data_frame('uncertainty_validation')
        score_evaluation = self.load_data_frame('uncertainty_evaluation')
        sample = self.load_data_frame('sample')
        output = self._make_submission(score_validation, score_evaluation)
        assert output.shape == sample.shape
        self.dump(output)

    @staticmethod
    def _make_submission(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df1, df2])
        df = df.reset_index(drop=True)
        return df


 # python main.py m5-forecasting.SubmitUncertainty --local-scheduler