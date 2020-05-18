from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class WRMSSECalculator:
    def __init__(self, s: np.ndarray, w: np.ndarray, sw: np.ndarray, roll_mat_csr: csr_matrix, sample_submission: pd.DataFrame) -> None:
        self._s = s
        self._w = w
        self._sw = sw,
        self._roll_mat_csr = roll_mat_csr
        self._sample_submission = sample_submission

    def calculate_scores(self, y_hat: pd.DataFrame, y_gt: pd.DataFrame) -> float:
        assert 'id' in y_hat.columns
        assert 'id' in y_gt.columns

        # y_hat, y_gt: ID数x日数のDataFrame
        # y_hat.shape == y_gt.shape == (30490, 28+1)
        y_hat = pd.merge(self._sample_submission[['id']], y_hat).drop('id', axis=1)
        y_gt = pd.merge(self._sample_submission[['id']], y_gt).drop('id', axis=1)
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

    # sample submission
    ss = pd.read_csv(file_pass + 'sample_submission_accuracy.csv')

    # Predictions:
    sub = pd.read_csv(file_pass + 'emsembled.csv')
    sub = sub[sub.id.str.endswith('validation')]

    # Psudo Ground truth:
    ground_truth = pd.read_csv('resources/kernel/kkiller_first_public_notebook_under050_v5.csv')
    ground_truth = ground_truth[ground_truth.id.str.endswith('validation')]

    # Load S and W weights for WRMSSE calcualtions:
    sw_df = pd.read_pickle(file_pass + 'sw_df.pkl')
    S = sw_df.s.values
    W = sw_df.w.values
    SW = sw_df.sw.values

    # Load roll up matrix to calcualte aggreagates:
    roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df.pkl')
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df

    calculator = WRMSSECalculator(s=S, w=W, sw=SW, roll_mat_csr=roll_mat_csr, sample_submission=ss)

    score = calculator.calculate_scores(sub, ground_truth)
    # x, y = calculator.calculate_scores_with_matrix(sub, ground_truth)

    print(score)  # 0.196


if __name__ == '__main__':
    main()
