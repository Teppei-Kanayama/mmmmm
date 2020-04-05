from logging import getLogger

import gokart
import pandas as pd

logger = getLogger(__name__)


class TrainValidationSplit(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.data_task

    def run(self):
        data = self.load_data_frame()
        x_train, y_train, x_val, y_val = self._run(data)
        output = dict(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        self.dump(output)

    @staticmethod
    def _run(data: pd.DataFrame, validation_days: int = 28):
        x_train = data[data['d'] < 1914 - validation_days]
        y_train = x_train['demand']
        x_val = data[(data['d'] < 1914) & (data['d'] >= 1914 - validation_days)]
        y_val = x_val['demand']
        return x_train.drop('demand', axis=1), y_train, x_val.drop('demand', axis=1), y_val
