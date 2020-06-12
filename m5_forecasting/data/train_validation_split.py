from logging import getLogger
from typing import Tuple

import gokart
import luigi
import pandas as pd

logger = getLogger(__name__)


# TODO: そもそもこの関数いらない説（validationがないので）
class TrainValidationSplit(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    data_task = gokart.TaskInstanceParameter()
    train_to_date: int = luigi.IntParameter()

    def requires(self):
        return self.data_task

    def run(self):
        data = self.load_data_frame()
        x_train, y_train = self._run(data, self.train_to_date)
        output = dict(x_train=x_train, y_train=y_train)
        self.dump(output)

    @staticmethod
    def _run(data: pd.DataFrame, train_to_date: int) -> Tuple[pd.DataFrame, ...]:
        data = data[data['d'] < train_to_date]  # TODO: move!
        data = data.dropna(subset={'sell_price'})  # TODO: move!
        y_train = data['demand']
        x_train = data.drop({'demand'}, axis=1)
        return x_train, y_train
