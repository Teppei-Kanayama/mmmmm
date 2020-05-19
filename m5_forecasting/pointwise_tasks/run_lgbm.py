from logging import getLogger
from typing import Tuple, List

import gokart
import luigi
import pandas as pd
from lightgbm import Booster
import lightgbm as lgb

logger = getLogger(__name__)


class TrainPointwiseLGBM(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    num_boost_round: int = luigi.IntParameter()
    early_stopping_rounds: int = luigi.IntParameter(default=None)

    feature_task = gokart.TaskInstanceParameter()

    def requires(self):
        return self.feature_task

    def output(self):
        model = self.make_target(relative_file_path='model/lgb.pkl')
        feature_columns = self.make_target(relative_file_path='model/feature_columns.pkl')
        feature_importance = self.make_target(relative_file_path='model/feature_importance.csv')
        return dict(model=model, feature_columns=feature_columns, feature_importance=feature_importance)

    def run(self):
        data = self.load()
        model, feature_columns, feature_importance = self._run(data, self.num_boost_round, self.early_stopping_rounds)
        self.dump(model, 'model')
        self.dump(feature_columns, 'feature_columns')
        self.dump(feature_importance, 'feature_importance')

    @staticmethod
    def _run(data: pd.DataFrame, num_boost_round: int, early_stopping_rounds: int) -> Tuple[Booster, List[str], pd.DataFrame]:
        feature_columns = [feature for feature in data['x_train'].columns if feature not in ['id', 'd']]
        print(f'feature columns: {feature_columns}')
        train_set = lgb.Dataset(data['x_train'][feature_columns], data['y_train'])
        val_set = lgb.Dataset(data['x_val'][feature_columns], data['y_val'])

        lgb_params = {'boosting_type': 'gbdt',   # 固定
                      'objective': 'tweedie',
                      'tweedie_variance_power': 1.1,   # TODO: CVで決める
                      'metric': 'rmse',  # 固定。なんでもいい
                      'subsample': 0.5,  # TODO: 重要, bagging_fractionと同じ。
                      'subsample_freq': 1,  # TODO: CVで決める bagging_freqと同じ。
                      'learning_rate': 0.1,  # あとで小さくする。
                      'num_leaves': 128,  # TODO: 超重要  # もともとは 2 ** 11 - 1 だったが2~3時間経っても終わらない
                      'min_data_in_leaf': 2 ** 12 - 1,  # TODO: 重要  # これがあるとsmallで動かなくなる
                      'feature_fraction': 0.5,  # TODO: 重要
                      'max_bin': 100,
                      'n_estimators': 1400,  # TODO: CVで決める。何回で学習をやめるか？ early stoppingを使わない場合はこれが重要になる。
                      }

        valid_sets = [train_set, val_set] if not data['x_val'].empty else None
        model = lgb.train(lgb_params, train_set, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                          valid_sets=valid_sets, verbose_eval=100)

        feature_importance = pd.DataFrame(
            dict(name=feature_columns, imp=model.feature_importance(importance_type='gain'))).sort_values(by='imp', ascending=False)
        return model, feature_columns, feature_importance
