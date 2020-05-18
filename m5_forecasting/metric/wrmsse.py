from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class WRMSSECalculator:
    def __init__(self, weight: pd.DataFrame, roll_mat_csr: csr_matrix, sample_submission: pd.DataFrame, sales: pd.DataFrame) -> None:
        self._roll_mat_csr = roll_mat_csr
        self._sample_submission = sample_submission
        self._sales = sales
        self._s = self._get_s()
        self._w = weight['Weight'].values
        self._w_df = weight
        self._sw = self._w/np.sqrt(self._s)

    def calculate_scores(self, y_hat: pd.DataFrame, y_gt: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        y_hat = self._sort_id(y_hat)
        y_gt = self._sort_id(y_gt)
        score_matrix = (np.square(self._rollup(y_hat.values - y_gt.values)) * np.square(self._w)[:, None]) / self._s[:, None]
        score_df = pd.DataFrame(score_matrix, columns=[f'V{i+1}' for i in range(score_matrix.shape[1])])
        score_df = pd.concat([self._w_df.drop('Weight', axis=1), score_df], axis=1)
        score_df['SW'] = self._sw
        score = np.sum(np.sqrt(np.mean(score_matrix, axis=1))) / 12
        return score, score_df

    # Fucntion to calculate S weights:
    def _get_s(self, drop_days=0):
        """
        drop_days: int, equals 0 by default, so S is calculated on all data.
                   If equals 28, last 28 days won't be used in calculating S.
        """
        # Rollup sales:
        d_name = ['d_' + str(i + 1) for i in range(1913 - drop_days)]
        sales_train_val = self._roll_mat_csr * self._sales[d_name].values

        no_sales = np.cumsum(sales_train_val, axis=1) == 0
        sales_train_val = np.where(no_sales, np.nan, sales_train_val)

        # Denominator of RMSSE / RMSSE
        weight1 = np.nanmean(np.diff(sales_train_val, axis=1) ** 2, axis=1)

        return weight1

    def _rollup(self, v: np.ndarray) -> np.ndarray:
        # 複数階層でのgroupbyを行列一発で解決する
        # v - np.array of size (30490 rows, n day columns)
        # v_rolledup - array of size (n, 42840)
        return self._roll_mat_csr * v  # (v.T*roll_mat_csr.T).T

    def _sort_id(self, df: pd.DataFrame) -> pd.DataFrame:
        assert 'id' in df.columns
        return pd.merge(self._sample_submission[['id']], df).drop('id', axis=1)


def main():
    file_pass = 'resources/input/'

    # sample submission
    ss = pd.read_csv(file_pass + 'sample_submission.csv')

    # sales
    sales = pd.read_csv(file_pass + 'sales_train_validation.csv')

    # Load weight and roll up matrix
    weight = pd.read_csv(file_pass + 'weights_validation.csv')
    roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df.pkl')
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df

    calculator = WRMSSECalculator(weight=weight, roll_mat_csr=roll_mat_csr, sample_submission=ss, sales=sales)

    # Predictions:
    sub = pd.read_csv(file_pass + 'emsembled.csv')
    sub = sub[sub.id.str.endswith('validation')]

    # Psudo Ground truth:
    ground_truth = pd.read_csv('resources/kernel/kkiller_first_public_notebook_under050_v5.csv')
    ground_truth = ground_truth[ground_truth.id.str.endswith('validation')]

    score, score_df = calculator.calculate_scores(sub, ground_truth)

    print(score)  # 0.196
    print(score_df)




if __name__ == '__main__':
    main()
