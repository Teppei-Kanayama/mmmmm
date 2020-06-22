import os
from logging import getLogger

import gokart
import luigi
import pandas as pd
import numpy as np

from m5_forecasting.data.load import LoadInputData


logger = getLogger(__name__)


VALIDATION_START_DATE = 1914
EVALUATION_START_DATE = 1942
DURATION = 28


class Load(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    from_date: int = luigi.IntParameter()
    to_date: int = luigi.IntParameter()

    def output(self):
        prediction_file_directory = 'predict'
        return self.make_target(os.path.join(prediction_file_directory, f'predict_{self.from_date}_{self.to_date}.csv'),
                                use_unique_id=False)


class SubmitPointwise(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()
    interval: int = luigi.IntParameter()

    def output(self):
        return self.make_target('submission.csv')

    def requires(self):
        public_prediction_load_tasks = [Load(from_date=t, to_date=t+self.interval) for t in range(1914, 1942, self.interval)]
        private_prediction_load_tasks = [Load(from_date=t, to_date=t+self.interval) for t in range(1942, 1970, self.interval)]
        sample_submission_data_task = LoadInputData(filename='sample_submission.csv')
        return dict(public_prediction=public_prediction_load_tasks,
                    private_prediction=private_prediction_load_tasks,
                    submission=sample_submission_data_task)

    def run(self):
        public_df = pd.concat(self.load('public_prediction'))
        private_df = pd.concat(self.load('private_prediction'))
        sample_submission = self.load_data_frame('submission')
        test = public_df.assign(id=public_df.id + "_" + np.where(public_df.d < EVALUATION_START_DATE, "validation", "evaluation"),
                                F="F" + (public_df.d - VALIDATION_START_DATE + 1 - DURATION * (public_df.d >= EVALUATION_START_DATE)).astype("str"))
        submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]
        submission = pd.merge(sample_submission[['id']], submission, on=['id'], how='left').fillna(-1)
        self.dump(submission)


# python main.py m5-forecasting.Submit --local-scheduler
