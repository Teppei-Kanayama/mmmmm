from typing import Tuple, List

import gokart
import luigi
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from m5_forecasting.data.calendar import PreprocessCalendar
from m5_forecasting.data.feature_engineering import MergeData, MakeFeature
from m5_forecasting.data.load import LoadInputData
from m5_forecasting.data.sales import PreprocessSales
from m5_forecasting.data.selling_price import PreprocessSellingPrice


class TrainBinaryLGBM(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    feature_task = gokart.TaskInstanceParameter()
    target_term: Tuple = luigi.ListParameter(default=[1914 - 365, 1914 - 1])
    source_term: Tuple = luigi.ListParameter()
    test_frequency: int = luigi.IntParameter(default=6)

    def output(self):
        model = self.make_target(relative_file_path='adversarial_validation/lgb.pkl')
        feature_columns = self.make_target(relative_file_path='adversarial_validation/feature_columns.pkl')
        feature_importance = self.make_target(relative_file_path='adversarial_validation/feature_importance.csv')
        test_data = self.make_target(relative_file_path='adversarial_validation/test_data.pkl')
        return dict(model=model, feature_columns=feature_columns, feature_importance=feature_importance, test_data=test_data)

    def requires(self):
        return self.feature_task

    def run(self):
        feature = self.load_data_frame()
        feature_columns = [feature_column for feature_column in feature.columns if
                           feature_column not in ['id', 'd', 'year', 'week_of_year', 'day', 'sell_price']]  # sell_price?

        # target column
        target_feature = feature[feature['d'].between(*self.target_term)]
        target_feature['target'] = 1
        source_feature = feature[feature['d'].between(*self.source_term)]
        source_feature['target'] = 0
        data = pd.concat([target_feature, source_feature])

        # prepare data
        # 6日ごとにval, testデータをサンプルする
        # TODO: refactor
        validation_days = np.concatenate([np.arange(self.source_term[0], self.source_term[1], 6), np.arange(self.target_term[0], self.target_term[1], 6)])
        test_days = np.concatenate([np.arange(self.source_term[0] + 1, self.source_term[1], 6), np.arange(self.target_term[0] + 1, self.target_term[1], 6)])

        X_val = data[data['d'].isin(validation_days)][feature_columns].values
        X_test = data[data['d'].isin(test_days)][feature_columns].values
        X_train = data[~(data['d'].isin(validation_days) | data['d'].isin(test_days))][feature_columns].values

        y_val = data[data['d'].isin(validation_days)]['target'].values
        y_test = data[data['d'].isin(test_days)]['target'].values
        y_train = data[~(data['d'].isin(validation_days) | data['d'].isin(test_days))]['target'].values

        test_ids = data[data['d'].isin(test_days)]['id']

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val)
        valid_sets = [train_set, val_set]

        # TODO: change parameters!
        params = {"objective": "binary", "metric": "auc", "force_row_wise": True, "learning_rate": 0.075, "sub_row": 0.75,
                  "bagging_freq": 1, "lambda_l2": 0.1, 'verbosity': 1}
        num_boost_round = 1000
        early_stopping_rounds = 100

        model = lgb.train(params, train_set, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                          valid_sets=valid_sets, verbose_eval=100)
        feature_importance = pd.DataFrame(
            dict(name=feature_columns, imp=model.feature_importance(importance_type='gain'))).sort_values(by='imp',
                                                                                                          ascending=False)
        self.dump(model, 'model')
        self.dump(feature_columns, 'feature_columns')
        self.dump(feature_importance, 'feature_importance')
        self.dump(dict(X_test=X_test, y_test=y_test, test_ids=test_ids), 'test_data')


class AdversarialValidation(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    is_small: bool = luigi.BoolParameter()
    source_term_list: List = luigi.ListParameter(default=[[1914 - 365 * (i + 1), 1914 - 365 * i - 1] for i in range(1, 5)])

    def output(self):
        return self.make_target('adversarial_validation/result.csv')

    def requires(self):
        calendar_data_task = PreprocessCalendar()
        selling_price_data_task = LoadInputData(filename='sell_prices.csv')  # 特徴量計算後は PreprocessSellingPrice
        sales_data_task = PreprocessSales(is_small=self.is_small)  # 特徴量計算後は MekeSalesFeature
        merged_data_task = MergeData(calendar_data_task=calendar_data_task,
                                     selling_price_data_task=selling_price_data_task, sales_data_task=sales_data_task)
        feature_task = MakeFeature(merged_data_task=merged_data_task)
        train_tasks = [TrainBinaryLGBM(feature_task=feature_task, source_term=source_term) for source_term in self.source_term_list]
        return train_tasks

    def run(self):
        data = self.load()
        model_list = [d['model'] for d in data]
        test_data_list = [d['test_data'] for d in data]

        score_list = []
        for model, test_data, source_term in zip(model_list, test_data_list, self.source_term_list):
            y_hat_test = model.predict(test_data['X_test'])
            y_test = test_data['y_test']
            df = pd.DataFrame(dict(id=test_data['test_ids'], y_hat=y_hat_test, y=y_test))

            for target_id in df['id'].unique():
                target_df = df[df['id'] == target_id]
                target_y_hat = target_df['y_hat']
                target_y = target_df['y']
                fpr, tpr, thresholds = metrics.roc_curve(target_y, target_y_hat)
                score = metrics.auc(fpr, tpr)
                score_list.append(dict(start=source_term[0], end=source_term[1], id=target_id, score=score))

        score_df = pd.DataFrame(score_list)
        self.dump(score_df)


# DATA_SIZE=small python main.py m5-forecasting.AdversarialValidation --local-scheduler
