import os
from logging import getLogger

import gokart
import luigi
import pandas as pd

from m5_forecasting.data.load import LoadInputData


logger = getLogger(__name__)

START_DATES = dict(validation=1914, evaluation=1942)


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

        public_submission = self._process_submission(public_df, 'validation', sample_submission)
        private_submission = self._process_submission(private_df, 'evaluation', sample_submission)
        submission = pd.concat([public_submission, private_submission]).reset_index(drop=True)

        logger.info(f'shape: {submission.shape}')
        logger.info(f'shape: {sample_submission.shape}')
        assert set(submission['id'].unique()) == set(sample_submission['id'].unique()), f'ID is invalid!'
        assert set(submission.columns) == set(sample_submission.columns), 'columns is invalid!'

        self.dump(submission)

    @staticmethod
    def _process_submission(df, term, sample_submission):
        df['id'] = df['id'] + "_" + term
        df['F'] = 'F' + (df['d'] - START_DATES[term] + 1).astype('str')
        submission = df.pivot(index="id", columns="F", values="demand").reset_index()
        return submission[sample_submission.columns]
