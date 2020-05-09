import gokart
import pandas as pd

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.uncertainty_tasks.predict_uncertainty import PredictUncertainty
from m5_forecasting.uncertainty_tasks.predict_uncertainty_with_poisson import PredictUncertaintyWithPoisson
from m5_forecasting.uncertainty_tasks.predict_uncertainty_with_variance import PredictUncertaintyWithVariance
from m5_forecasting.utils.file_processors import RoughCsvFileProcessor


class SubmitUncertainty(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        return self.make_target('submission_uncertainty.csv', processor=RoughCsvFileProcessor())

    def requires(self):
        uncertainty_with_variance_task = PredictUncertaintyWithVariance()
        uncertainty_baseline_task = PredictUncertainty()
        uncertainty_poisson_task = PredictUncertaintyWithPoisson()
        sample_submission_task = LoadInputData(filename='sample_submission_uncertaity.csv')
        return dict(uncertainty_with_variance=uncertainty_with_variance_task, uncertainty_poisson=uncertainty_poisson_task, uncertainty_baseline=uncertainty_baseline_task, sample=sample_submission_task)

    def run(self):
        score_v = self.load_data_frame('uncertainty_with_variance')
        score_b = self.load_data_frame('uncertainty_baseline')
        score_p = self.load_data_frame('uncertainty_poisson')
        sample = self.load_data_frame('sample')
        output = self._run(score_v, score_p, score_b)
        assert output.shape == sample.shape
        self.dump(output)

    @classmethod
    def _run(cls, score_v, score_p, score_b):
        score_p_v = pd.concat([score_p, score_v])
        score = pd.concat([score_p_v, score_b[~score_b['id'].isin(score_p_v['id'])]])
        df = cls._make_submission(score)
        return df

    @staticmethod
    def _make_submission(df: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([df, df], axis=0, sort=False)
        df.reset_index(drop=True, inplace=True)
        df.loc[df.index >= len(df.index) // 2, "id"] = df.loc[df.index >= len(df.index) // 2, "id"].str.replace(
            "_validation$", "_evaluation")
        return df


 # python main.py m5-forecasting.SubmitUncertainty --local-scheduler