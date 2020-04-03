from logging import getLogger

import gokart

from m5_forecasting.tasks.predict import Predict

logger = getLogger(__name__)


class Sample(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return Predict()

    def run(self):
        output = self.load_data_frame()
        import pdb; pdb.set_trace()


# python main.py m5-forecasting.Sample --local-scheduler