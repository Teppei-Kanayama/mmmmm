from logging import getLogger
from typing import Tuple, List

import gokart
import luigi
import pandas as pd
from lightgbm import Booster
import lightgbm as lgb

logger = getLogger(__name__)


class TrainLGBM(gokart.TaskOnKart):
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
        import pdb; pdb.set_trace()
        feature_columns = [feature for feature in data['x_train'].columns if feature not in ['id', 'd']]
        print(f'feature columns: {feature_columns}')
        train_set = lgb.Dataset(data['x_train'][feature_columns], data['y_train'])
        val_set = lgb.Dataset(data['x_val'][feature_columns], data['y_val'])
        # params = {'boosting_type': 'gbdt', 'metric': 'rmse', 'objective': 'poisson', 'n_jobs': -1, 'seed': 20,
        #           'learning_rate': 0.075, 'bagging_fraction': 0.66, 'bagging_freq': 1, 'colsample_bytree': 0.77,
        #           'num_leaves': 63, 'lambda_l2': 0.1}

        # TODO: get best parameters using optuna.
        # params = {"objective": "poisson", "metric": "rmse", "force_row_wise": True, "learning_rate": 0.075,
        #           "sub_row": 0.75, "bagging_freq": 1, "lambda_l2": 0.1,
        #           'verbosity': 1, 'num_iterations': num_boost_round, 'num_leaves': 128}

        params = {"objective": "poisson", "metric": "rmse", "force_row_wise": True, "learning_rate": 0.075,
                  "sub_row": 0.75, "bagging_freq": 1, "lambda_l2": 0.1, 'verbosity': 1, 'num_iterations': 2500, }

        valid_sets = [train_set, val_set] if not data['x_val'].empty else None
        model = lgb.train(params, train_set, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                          valid_sets=valid_sets, verbose_eval=100)

        feature_importance = pd.DataFrame(
            dict(name=feature_columns, imp=model.feature_importance(importance_type='gain'))).sort_values(by='imp', ascending=False)
        return model, feature_columns, feature_importance
