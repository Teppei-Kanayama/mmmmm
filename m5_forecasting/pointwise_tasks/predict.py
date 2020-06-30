import os
from logging import getLogger
from typing import List

import gokart
import luigi
import pandas as pd

from m5_forecasting.data.calendar import PreprocessCalendar
from m5_forecasting.data.feature_engineering import MergeData, MakeFeature
from m5_forecasting.data.sales import PreprocessSales, MekeSalesFeature
from m5_forecasting.data.selling_price import PreprocessSellingPrice
from m5_forecasting.pointwise_tasks.train import TrainPointwiseModel

logger = getLogger(__name__)


class EmptySalesTask(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def run(self):
        self.dump(pd.DataFrame(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "demand"]))


class LoadPredictionData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    from_date: int = luigi.IntParameter()
    to_date: int = luigi.IntParameter()

    def output(self):
        prediction_file_directory = 'predict'
        return self.make_target(os.path.join(prediction_file_directory, f'predict_{self.from_date}_{self.to_date}.csv'),
                                use_unique_id=False)


class ConcatPredictionData(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    from_date: int = luigi.IntParameter()
    to_date: int = luigi.IntParameter()
    interval: int = luigi.IntParameter()

    def requires(self):
        load_tasks = [LoadPredictionData(from_date=t, to_date=(t+self.interval))
                      for t in range(self.from_date, self.to_date, self.interval)]
        if load_tasks:
            return load_tasks
        return [EmptySalesTask()]

    def run(self):
        self.dump(pd.concat(self.load()))


class PredictPointwise(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()

    train_to_date: int = luigi.IntParameter()  # 学習終了日（固定）
    prediction_start_date: int = luigi.IntParameter()  # 予測開始日（固定）
    predict_from_date: int = luigi.IntParameter()  # 分割後の予測開始日
    predict_to_date: int = luigi.IntParameter()  # 分割後の予測終了日
    interval: int = luigi.IntParameter()

    def output(self):
        return self.make_target(os.path.join('predict', f'predict_{self.predict_from_date}_{self.predict_to_date}.csv'),
                                use_unique_id=False)

    def requires(self):
        trained_model_task = TrainPointwiseModel(train_to_date=self.train_to_date)
        assert trained_model_task.input()['model'].exists(), "trained model doesn't exists!"

        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()
        original_sales_data_task = PreprocessSales(to_date=self.prediction_start_date, is_small=self.is_small)
        predicted_sales_data_task = ConcatPredictionData(from_date=self.prediction_start_date, to_date=self.predict_from_date, interval=self.interval)
        sales_feature_task = MekeSalesFeature(sales_data_task=original_sales_data_task,
                                              predicted_sales_data_task=predicted_sales_data_task,
                                              make_feature_from_date=self.predict_from_date,
                                              make_feature_to_date=self.predict_to_date)
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_feature_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)

        return dict(model=trained_model_task, sales=original_sales_data_task, feature=feature_task)

    def run(self):
        model = self.load('model')['model']
        feature_columns = self.load('model')['feature_columns']
        feature = self.load_data_frame('feature')
        sales = self.load_data_frame('sales')
        output = self._run(model, feature_columns, feature, sales, self.predict_from_date, self.predict_to_date)
        self.dump(output)

    @staticmethod
    def _run(model, feature_columns: List[str], feature: pd.DataFrame, sales, predict_from_date, predict_to_date) -> pd.DataFrame:
        test = feature[(predict_from_date <= feature['d']) & (feature['d'] < predict_to_date)]
        test['demand'] = model.predict(test[feature_columns])
        base_df = sales.drop_duplicates(subset={'id'}).drop(['d', 'demand'], axis=1)
        return pd.merge(base_df, test[['id', 'd', 'demand']])
