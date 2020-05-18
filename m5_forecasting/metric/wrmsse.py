import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Function to do quick rollups:
def rollup(v, roll_mat_csr):
    '''
    v - np.array of size (30490 rows, n day columns)
    v_rolledup - array of size (n, 42840)
    '''
    return roll_mat_csr * v  # (v.T*roll_mat_csr.T).T


# Function to calculate WRMSSE:
def wrmsse(preds, y_true, s, w, sw, roll_mat_csr, score_only=False):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''

    if score_only:
        return np.sum(
            np.sqrt(
                np.mean(
                    np.square(rollup(preds.values - y_true.values, roll_mat_csr))
                    , axis=1)) * sw) / 12  # <-used to be mistake here
    else:
        score_matrix = (np.square(rollup(preds.values - y_true.values, roll_mat_csr)) * np.square(w)[:, None]) / s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix, axis=1))) / 12  # <-used to be mistake here
        return score, score_matrix


def main():
    file_pass = 'resources/input/'

    # Load S and W weights for WRMSSE calcualtions:
    sw_df = pd.read_pickle(file_pass + 'sw_df.pkl')
    S = sw_df.s.values
    W = sw_df.w.values
    SW = sw_df.sw.values

    # Load roll up matrix to calcualte aggreagates:
    roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df.pkl')
    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df

    # Predictions:
    sub = pd.read_csv(file_pass + 'sample_submission_accuracy.csv')
    sales = pd.read_csv(file_pass + 'sales_train_validation.csv')
    sub = sub[sub.id.str.endswith('validation')]
    sub.drop(['id'], axis=1, inplace=True)

    DAYS_PRED = sub.shape[1]  # 28

    # Ground truth:
    dayCols = ["d_{}".format(i) for i in range(1914 - DAYS_PRED, 1914)]
    y_true = sales[dayCols]

    score = wrmsse(sub, y_true, S, W, SW, roll_mat_csr, score_only=True)


    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
