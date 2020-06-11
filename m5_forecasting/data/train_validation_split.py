from logging import getLogger
from typing import Tuple

import gokart
import luigi
import pandas as pd

logger = getLogger(__name__)


class TrainValidationSplit(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    data_task = gokart.TaskInstanceParameter()

    validation_days: int = luigi.IntParameter()
    train_to_date: int = luigi.IntParameter()

    def requires(self):
        return self.data_task

    def run(self):
        data = self.load_data_frame()
        x_train, y_train = self._run(data, self.validation_days, self.train_to_date)
        output = dict(x_train=x_train, y_train=y_train)
        self.dump(output)

    @staticmethod
    def _run(data: pd.DataFrame, validation_days: int, train_to_date: int) -> Tuple[pd.DataFrame, ...]:
        data['target'] = data['demand'] * data['sell_price']
        data = data.dropna(subset={'target'})

        y_train = data['target']
        x_train = data.drop({'demand', 'target'}, axis=1)
        # y_val = pd.DataFrame(columns={'target'})['target']
        # x_val = pd.DataFrame(columns=x_train.columns)
        # return x_train, y_train, x_val, y_val
        return x_train, y_train
