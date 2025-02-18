from logging import getLogger

import gokart
import luigi
import pandas as pd


logger = getLogger(__name__)


class TrainPointwiseLGBM(gokart.TaskOnKart):
    task_namespace = 'm5-forecasting'

    feature_task = gokart.TaskInstanceParameter()

    max_depth = luigi.IntParameter()
    num_leaves = luigi.IntParameter()
    tweedie_variance_power = luigi.FloatParameter()
    min_child_weight = luigi.FloatParameter()
    feature_fraction = luigi.FloatParameter()
    subsample = luigi.FloatParameter()
    learning_rate = luigi.FloatParameter()
    n_estimators = luigi.IntParameter()

    def requires(self):
        return dict(feature=self.feature_task)

    def output(self):
        model = self.make_target(relative_file_path='model/lgb.pkl')
        feature_columns = self.make_target(relative_file_path='model/feature_columns.pkl')
        feature_importance = self.make_target(relative_file_path='model/feature_importance.csv')
        return dict(model=model, feature_columns=feature_columns, feature_importance=feature_importance)

    def run(self):
        data = self.load_data_frame('feature')
        model, feature_columns, feature_importance = self._run(data)
        self.dump(model, 'model')
        self.dump(feature_columns, 'feature_columns')
        self.dump(feature_importance, 'feature_importance')

    def _run(self, data: pd.DataFrame):
        y_train = data['demand']
        x_train = data.drop('demand', axis=1)

        feature_columns = [feature for feature in x_train.columns if feature not in ['id', 'd']]
        logger.info(f'feature columns: {feature_columns}')

        import lightgbm as lgb
        train_set = lgb.Dataset(x_train[feature_columns], y_train)

        # min_data_in_leaf = 2 ** 12 - 1 if x_train['id'].nunique() > 10 else None
        # lgb_params = {'boosting_type': 'gbdt',   # 固定
        #               'objective': 'tweedie',
        #               'tweedie_variance_power': 1.1,  # fix
        #               'metric': 'rmse',  # 固定。なんでもいい
        #               'subsample': 0.5,  # TODO: 重要, bagging_fractionと同じ。
        #               'subsample_freq': 1,  # TODO: CVで決める bagging_freqと同じ。
        #               'learning_rate': 0.03,  # あとで小さくする。 0.1 -> 0.03
        #               'num_leaves': 2 ** 11 - 1,  # fix
        #               'min_data_in_leaf': min_data_in_leaf,  # TODO: 重要
        #               'feature_fraction': 0.5,  # TODO: 重要
        #               'max_bin': 100,
        #               'n_estimators': 1400,  # TODO: CVで決める。early stoppingを使わない場合はこれが重要になる。 1400 -> 2500,
        #               'max_depth': 16  # fix
        #               }

        lgb_params = {'boosting_type': 'gbdt',  # 固定
                      'objective': 'tweedie',
                      'metric': 'rmse',

                      'max_depth': self.max_depth,
                      'num_leaves': self.num_leaves,
                      'tweedie_variance_power': self.tweedie_variance_power,
                      'min_child_weight': self.min_child_weight,
                      'feature_fraction': self.feature_fraction,
                      'subsample': self.subsample,
                      'learning_rate': self.learning_rate,
                      'n_estimators': self.n_estimators,
                      }

        model = lgb.train(lgb_params, train_set, valid_sets=None, verbose_eval=100)

        feature_importance = pd.DataFrame(
            dict(name=feature_columns, imp=model.feature_importance(importance_type='gain'))).sort_values(by='imp', ascending=False)
        return model, feature_columns, feature_importance
