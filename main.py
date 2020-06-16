import os

import luigi
import numpy as np
import gokart

import m5_forecasting

if __name__ == '__main__':

    luigi.configuration.LuigiConfigParser.add_config_path('./conf/environment.ini')
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/default.ini')
    if os.environ.get('DATA_SIZE', '') == 'small':
        luigi.configuration.LuigiConfigParser.add_config_path('./conf/small.ini')
    np.random.seed(57)
    gokart.run()
