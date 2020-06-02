from logging import getLogger

import gc
import gokart
import luigi
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import pandas as pd

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class MergeData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    calendar_data_task = gokart.TaskInstanceParameter()
    selling_price_data_task = gokart.TaskInstanceParameter()
    sales_data_task = gokart.TaskInstanceParameter()

    def requires(self):
        return dict(calendar=self.calendar_data_task, selling_price=self.selling_price_data_task,
                    sales=self.sales_data_task)

    def run(self):
        calendar = self.load_data_frame('calendar')
        selling_price = self.load_data_frame('selling_price')
        sales = self.load_data_frame('sales')
        output = self._run(calendar, selling_price, sales)
        self.dump(output)

    @staticmethod
    def _run(calendar, selling_price, sales):
        sales = sales.merge(calendar, how="left", on="d")
        gc.collect()

        sales = sales.merge(selling_price, how="left", on=["store_id", "item_id", "wm_yr_wk"])
        sales.drop(["wm_yr_wk"], axis=1, inplace=True)
        gc.collect()
        del selling_price

        return sales


class MakeFeature(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    merged_data_task = gokart.TaskInstanceParameter()
    is_small: bool = luigi.BoolParameter()

    def requires(self):
        feature1_task = LoadInputData(filename='grid_part_1.pkl')
        feature2_task = LoadInputData(filename='grid_part_2.pkl')
        feature3_task = LoadInputData(filename='grid_part_3.pkl')
        mean_encoding_feature_task = LoadInputData(filename='small_mean_encoding_feature.pkl')
        return dict(data=self.merged_data_task, feature1=feature1_task, feature2=feature2_task,
                    feature3=feature3_task, mean_encoding_feature=mean_encoding_feature_task)

    def run(self):
        data = self.load_data_frame('data')
        feature1 = self.load('feature1')
        feature2 = self.load('feature2')
        feature3 = self.load('feature3')
        mean_encoding_feature = self.load('mean_encoding_feature')
        output = self._run(data, feature1, feature2, feature3, mean_encoding_feature)
        self.dump(output)

    @classmethod
    def _run(cls, data, feature1, feature2, feature3, mean_encoding_feature):
        data = cls._label_encode(data)
        data = cls._merge_outside_feature(data, feature1, feature2, feature3, mean_encoding_feature)
        return data

    @staticmethod
    def _label_encode(data):
        for i, v in tqdm(enumerate(["item_id", "dept_id", "store_id", "cat_id", "state_id"])):
            data[v] = OrdinalEncoder(dtype="int").fit_transform(data[[v]]).astype("int16") + 1
        return data

    @staticmethod
    def _merge_outside_feature(data, feature1, feature2, feature3, mean_encoding_feature):
        logger.info('merge mean_encoding_feature ...')
        mean_encoding_feature['id'] = mean_encoding_feature['id'].apply(lambda x: x.split('_evaluation')[0])
        data = pd.merge(data, mean_encoding_feature, on=['id', 'd'], how='left')
        del mean_encoding_feature

        logger.info('merge feature1 ...')
        feature1['id'] = feature1['id'].apply(lambda x: x.split('_evaluation')[0])
        feature1_columns = ['id', 'd', 'release']
        data = pd.merge(data, feature1[feature1_columns], on=['id', 'd'], how='left')
        del feature1

        logger.info('merge feature2 ...')
        feature2['id'] = feature2['id'].apply(lambda x: x.split('_evaluation')[0])
        feature2['d'] = feature2['d'].apply(lambda x: int(x.split('_')[1]))
        feature2_columns = ['id', 'd', 'price_max', 'price_min', 'price_std', 'price_mean', 'price_norm',
                            'price_nunique', 'item_nunique',
                            'price_momentum', 'price_momentum_m', 'price_momentum_y']
        data = pd.merge(data, feature2[feature2_columns], on=['id', 'd'], how='left')
        del feature2

        logger.info('merge feature3 ...')
        feature3['id'] = feature3['id'].apply(lambda x: x.split('_evaluation')[0])
        feature3['d'] = feature3['d'].apply(lambda x: int(x.split('_')[1]))
        feature3_columns = ['id', 'd', 'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']
        data = pd.merge(data, feature3[feature3_columns], on=['id', 'd'], how='left')
        del feature3

        return data
