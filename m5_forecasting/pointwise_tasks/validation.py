import gokart
import luigi
import pandas as pd
from scipy.sparse import csr_matrix

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.data.sales import PreprocessSales
from m5_forecasting.metric.wrmsse import WRMSSECalculator
from m5_forecasting.pointwise_tasks.submit import Load


class ValidatePointwise(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    validate_from_date: int = luigi.IntParameter()
    validate_to_date: int = luigi.IntParameter()
    interval: int = luigi.IntParameter()

    def output(self):
        return self.make_target(f'validation/validation_{self.validate_from_date}_{self.validate_to_date}.csv')

    def requires(self):
        ground_truth_task = PreprocessSales(is_small=self.is_small)
        prediction_load_tasks = [Load(from_date=t, to_date=t + self.interval) for t in
                                 range(self.validate_from_date, self.validate_to_date, self.interval)]
        sample_submission_task = LoadInputData(filename='sample_submission.csv')
        sales_task = LoadInputData(filename='sales_train_validation.csv')
        rmsse_weight_task = LoadInputData(filename='weights_validation.csv')
        return dict(ground_truth=ground_truth_task, predict=prediction_load_tasks, sample_submission=sample_submission_task,
                    sales=sales_task, rmsse_weight=rmsse_weight_task)

    def run(self):
        ground_truth = self.load_data_frame('ground_truth')
        prediction = pd.concat(self.load('predict'))
        sample_submission = self.load_data_frame('sample_submission')
        sales = self.load_data_frame('sales')
        rmsse_weight = self.load_data_frame('rmsse_weight')
        roll_matrix = pd.read_pickle('resources/input/roll_mat_df.pkl')
        output = self._run(ground_truth, prediction, sample_submission, sales, rmsse_weight, roll_matrix,
                           self.validate_from_date, self.validate_to_date)
        self.dump(output)

    @classmethod
    def _run(cls, ground_truth: pd.DataFrame, prediction: pd.DataFrame, sample_submission, sales, rmsse_weight, roll_matrix,
             validate_from_date: int, validate_to_date: int) -> pd.DataFrame:
        ground_truth = ground_truth[ground_truth['d'].between(validate_from_date, validate_to_date - 1)]
        pivot_prediction = cls._make_pivot(prediction)
        pivot_ground_truth = cls._make_pivot(ground_truth)
        roll_mat_csr = csr_matrix(roll_matrix.values)
        calculator = WRMSSECalculator(weight=rmsse_weight, roll_mat_csr=roll_mat_csr, sample_submission=sample_submission, sales=sales)
        score, score_df = calculator.calculate_scores(pivot_prediction, pivot_ground_truth)
        print(score)
        return score_df

    @staticmethod
    def _make_pivot(df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(id=df['id'] + "_" + "validation", V="V" + df['d'].astype(str))
        return df.pivot(index="id", columns="V", values="demand").reset_index()


 # python main.py m5-forecasting.SubmitUncertainty --local-scheduler
