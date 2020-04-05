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
        return self.make_target(os.path.join(input_file_directory, self.filename), use_unique_id=False)
