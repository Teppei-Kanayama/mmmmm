from logging import getLogger

import gokart

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class Sample(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    # def output(self):
    #     return self.make_target('data/sample.pkl')
    #
    # def run(self):
    #     self.dump('sample output')

    def requires(self):
        return LoadInputData()

    def run(self):
        calender = self.load_data_frame('calender')
        import pdb; pdb.set_trace()
