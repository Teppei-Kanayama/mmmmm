from logging import getLogger

import gokart

from m5_forecasting.data.preprocess import PreprocessInputData

logger = getLogger(__name__)


class Sample(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def requires(self):
        return PreprocessInputData()

    def run(self):
        data = self.load_data_frame()
        import pdb; pdb.set_trace()


# python main.py m5-forecasting.Sample --local-scheduler