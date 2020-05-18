from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class WRMSSECalculator:
    def __init__(self, s: np.ndarray, w: np.ndarray, sw: np.ndarray, roll_mat_csr: csr_matrix) -> None:
        self._s = s
        self._w = w
        self._sw = sw,
        self._roll_mat_csr = roll_mat_csr

    def calculate_scores(self, y_hat: pd.DataFrame, y_gt: pd.DataFrame) -> float:
        # y_hat, y_gt: ID数x日数のDataFrame
        # y_hat.shape == y_gt.shape == (30490, 28)
        # IDの順序はsumple submissionと同じである必要あり
        return np.sum(
            np.sqrt(
                np.mean(
                    np.square(self._rollup(y_hat.values - y_gt.values))
                    , axis=1)) * self._sw) / 12

    def calculate_scores_with_matrix(self, y_hat: pd.DataFrame, y_gt: pd.DataFrame) -> Tuple[float, np.ndarray]:
        score_matrix = (np.square(self._rollup(y_hat.values - y_gt.values)) * np.square(self._w)[:, None]) / self._s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix, axis=1))) / 12
        return score, score_matrix

    def _rollup(self, v: np.ndarray) -> np.ndarray:
        # 複数階層でのgroupbyを行列一発で解決する
        # v - np.array of size (30490 rows, n day columns)
        # v_rolledup - array of size (n, 42840)
        return self._roll_mat_csr * v  # (v.T*roll_mat_csr.T).T


def main():
    file_pass = 'resources/input/'

    # Load S and W weights for WRMSSE calcualtions:
    sw_df = pd.read_pickle(file_pass + 'sw_df.pkl')
    S = sw_df.s.values
    W = sw_df.w.values
    SW = sw_df.sw.values

    # Load roll up matrix to calcualte aggreagates:
    roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df.pkl')
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df

    calculator = WRMSSECalculator(s=S, w=W, sw=SW, roll_mat_csr=roll_mat_csr)

    # Predictions:
    sub = pd.read_csv(file_pass + 'sample_submission_accuracy.csv')
    sales = pd.read_csv(file_pass + 'sales_train_validation.csv')
    sub = sub[sub.id.str.endswith('validation')]
    sub.drop(['id'], axis=1, inplace=True)

    DAYS_PRED = sub.shape[1]  # 28

    # Ground truth:
    dayCols = ["d_{}".format(i) for i in range(1914 - DAYS_PRED, 1914)]
    y_true = sales[dayCols]

    score = calculator.calculate_scores(sub, y_true)
    x, y = calculator.calculate_scores_with_matrix(sub, y_true)

    print(score, x, y)


if __name__ == '__main__':
    main()
