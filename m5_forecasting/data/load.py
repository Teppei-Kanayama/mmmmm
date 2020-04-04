from logging import getLogger

import gokart

logger = getLogger(__name__)


class LoadInputData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        calendar = self.make_target('input/calendar.csv', use_unique_id=False)
        sales_train_validation = self.make_target('input/sales_train_validation.csv', use_unique_id=False)
        sample_submission = self.make_target('input/sample_submission.csv', use_unique_id=False)
        sell_prices = self.make_target('input/sell_prices.csv', use_unique_id=False)
        return dict(calendar=calendar, sales_train_validation=sales_train_validation,
                    sample_submission=sample_submission, sell_prices=sell_prices)
