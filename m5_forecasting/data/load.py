import os
from logging import getLogger

import gokart
import luigi

logger = getLogger(__name__)


class LoadInputData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    filename: str = luigi.Parameter()

    def output(self):
        input_file_directory = 'input'
        # calendar = self.make_target('input/calendar.csv', use_unique_id=False)
        # sales_train_validation = self.make_target('input/sales_train_validation.csv', use_unique_id=False)
        # sample_submission = self.make_target('input/sample_submission.csv', use_unique_id=False)
        # sell_prices = self.make_target('input/sell_prices.csv', use_unique_id=False)
        # return dict(calendar=calendar, sales_train_validation=sales_train_validation,
        #             sample_submission=sample_submission, sell_prices=sell_prices)
        return self.make_target(os.path.join(input_file_directory, self.filename), use_unique_id=False)
