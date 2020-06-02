from typing import Tuple

import gokart
import luigi
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from m5_forecasting.data.load import LoadInputData
from m5_forecasting.data.sales import PreprocessSales
from m5_forecasting.metric.wrmsse import WRMSSECalculator
from m5_forecasting.pointwise_tasks.submit import Load


class CalculateRollMatrix(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return LoadInputData(filename='sales_train_evaluation.csv')

    def run(self):
        sales = self.load_data_frame()

        # List of categories combinations for aggregations as defined in docs:
        dummies_list = [sales.state_id, sales.store_id, sales.cat_id, sales.dept_id,
                        sales.state_id + '_' + sales.cat_id, sales.state_id + '_' + sales.dept_id,
                        sales.store_id + '_' + sales.cat_id, sales.store_id + '_' + sales.dept_id, sales.item_id,
                        sales.state_id + '_' + sales.item_id, sales.id]

        ## First element Level_0 aggregation 'all_sales':
        dummies_df_list = [pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8), index=sales.index, columns=['all']).T]

        # List of dummy dataframes:
        for i, cats in enumerate(dummies_list):
            dummies_df_list += [pd.get_dummies(cats, drop_first=False, dtype=np.int8).T]

        # Concat dummy dataframes in one go:
        ## Level is constructed for free.
        roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)),
                                names=['level', 'id'])

        self.dump(roll_mat_df)


class ValidatePointwise(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    validate_from_date: int = luigi.IntParameter()
    validate_to_date: int = luigi.IntParameter()
    interval: int = luigi.IntParameter()

    def output(self):
        return dict(score=self.make_target(f'validation/score_{self.validate_from_date}_{self.validate_to_date}.csv'),
                    score_df=self.make_target(f'validation/score_df_{self.validate_from_date}_{self.validate_to_date}.csv'))

    def requires(self):
        ground_truth_task = PreprocessSales(is_small=self.is_small)
        prediction_load_tasks = [Load(from_date=t, to_date=t + self.interval) for t in
                                 range(self.validate_from_date, self.validate_to_date, self.interval)]
        sample_submission_task = LoadInputData(filename='sample_submission.csv')
        sales_task = LoadInputData(filename='sales_train_evaluation.csv')
        rmsse_weight_task = LoadInputData(filename='weights_validation.csv')
        roll_matrix_task = CalculateRollMatrix()
        return dict(ground_truth=ground_truth_task, predict=prediction_load_tasks, sample_submission=sample_submission_task,
                    sales=sales_task, rmsse_weight=rmsse_weight_task, roll_matrix=roll_matrix_task)

    def run(self):
        ground_truth = self.load_data_frame('ground_truth')
        prediction = pd.concat(self.load('predict'))
        sample_submission = self.load_data_frame('sample_submission')
        sales = self.load_data_frame('sales')
        rmsse_weight = self.load_data_frame('rmsse_weight')
        roll_matrix = self.load('roll_matrix')
        score, score_df = self._run(ground_truth, prediction, sample_submission, sales, rmsse_weight, roll_matrix,
                                    self.validate_from_date, self.validate_to_date)
        self.dump(score, 'score')
        self.dump(score_df, 'score_df')

    @classmethod
    def _run(cls, ground_truth: pd.DataFrame, prediction: pd.DataFrame, sample_submission, sales, rmsse_weight, roll_matrix,
             validate_from_date: int, validate_to_date: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ground_truth = ground_truth[ground_truth['d'].between(validate_from_date, validate_to_date - 1)]
        pivot_prediction = cls._make_pivot(prediction)
        pivot_ground_truth = cls._make_pivot(ground_truth)
        roll_mat_csr = csr_matrix(roll_matrix.values)
        calculator = WRMSSECalculator(weight=rmsse_weight, roll_mat_csr=roll_mat_csr, sample_submission=sample_submission, sales=sales)
        score, score_df = calculator.calculate_scores(pivot_prediction, pivot_ground_truth)
        score = pd.DataFrame(dict(score=[score]))
        return score, score_df

    @staticmethod
    def _make_pivot(df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(id=df['id'] + "_" + "evaluation", V="V" + df['d'].astype(str))
        return df.pivot(index="id", columns="V", values="demand").reset_index()
