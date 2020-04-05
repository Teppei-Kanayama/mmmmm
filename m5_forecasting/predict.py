from logging import getLogger

import gokart
import pandas as pd
from lightgbm import Booster

from m5_forecasting.data.preprocess import PreprocessCalendar, PreprocessSellingPrice, PreprocessSales, MergeData, \
    MakeFeature
from m5_forecasting.data.utils import reduce_mem_usage
from m5_forecasting.tasks.run_lgbm import TrainLGBM

from m5_forecasting.data.load import LoadInputData

logger = getLogger(__name__)


class Predict(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    def output(self):
        return self.make_target('submission.csv', use_unique_id=False)

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = PreprocessSellingPrice()
        sales_data_task = PreprocessSales()
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task,
                                     sales_data_task=sales_data_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)
        model_task = TrainLGBM(feature_task=feature_task)
        sample_submission_data_task = LoadInputData(filename='sample_submission.csv')
        return dict(model=model_task, sample_submission=sample_submission_data_task, feature=feature_task)

    def run(self):
        model = self.load('model')
        sample_submission = self.load('sample_submission')
        feature = self.load_data_frame('feature')
        output = self._run(model, feature, sample_submission)
        self.dump(output)

    @staticmethod
    def _run(model: Booster, feature: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
        feature = reduce_mem_usage(feature)
        test = feature[(feature['d'] >= 1914)]

        # define list of features
        features = ["wday", "month", "year", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
                    "sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7", "lag_t28",
                    "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", "rolling_mean_t90", "rolling_mean_t180",
                    "rolling_std_t7", "rolling_std_t30", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

        df = pd.DataFrame(dict(name=features, imp=model.feature_importance(importance_type='gain'))).sort_values(by='imp', ascending=False)
        df.to_csv('resources/feature_importance.csv', index=False)

        y_pred = model.predict(test[features])
        test['demand'] = y_pred

        predictions = test[['id', 'd', 'demand']]
        predictions = pd.pivot(predictions, index='id', columns='d', values='demand').reset_index()
        predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

        evaluation_rows = [row for row in submission['id'] if 'evaluation' in row]
        evaluation = submission[submission['id'].isin(evaluation_rows)]

        validation = submission[['id']].merge(predictions, on='id')
        final = pd.concat([validation, evaluation])
        return final

# python main.py m5-forecasting.Predict --local-scheduler
