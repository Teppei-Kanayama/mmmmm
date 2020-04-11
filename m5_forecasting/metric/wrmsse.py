import numpy as np
import pandas as pd


weight_mat = np.c_[np.ones([NUM_ITEMS, 1]).astype(np.int8),  # level 1
                   pd.get_dummies(product.state_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.store_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(product.cat_id.astype(str),
                                                                                          drop_first=False).astype(
    'int8').values, pd.get_dummies(product.dept_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.state_id.astype(str) + product.cat_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.state_id.astype(str) + product.dept_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.store_id.astype(str) + product.cat_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.store_id.astype(str) + product.dept_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.item_id.astype(str), drop_first=False).astype('int8').values, pd.get_dummies(
    product.state_id.astype(str) + product.item_id.astype(str), drop_first=False).astype('int8').values, np.identity(
    NUM_ITEMS).astype(np.int8)  # item :level 12
].T

weight_mat_csr = csr_matrix(weight_mat)
del weight_mat;
gc.collect()


def weight_calc(data, product):
    # calculate the denominator of RMSSE, and calculate the weight base on sales amount

    sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
    d_name = ['d_' + str(i + 1) for i in range(1913)]

    sales_train_val = weight_mat_csr * sales_train_val[d_name].values

    # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
    df_tmp = ((sales_train_val > 0) * np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1)))

    start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1) - 1

    flag = np.dot(np.diag(1 / (start_no + 1)), np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1))) < 1

    sales_train_val = np.where(flag, np.nan, sales_train_val)

    # denominator of RMSSE / RMSSEの分母
    weight1 = np.nansum(np.diff(sales_train_val, axis=1) ** 2, axis=1) / (1913 - start_no)

    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp = df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp[product.id].values

    weight2 = weight_mat_csr * df_tmp

    weight2 = weight2 / np.sum(weight2)

    del sales_train_val
    gc.collect()

    return weight1, weight2


weight1, weight2 = weight_calc(data, product)


# TODO: classにする
# TODO: まずは定義を理解する
def wrmsse(preds, data):
    # this function is calculate for last 28 days to consider the non-zero demand period

    # actual obserbed values / 正解ラベル
    y_true = data.get_label()

    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED

    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T

    train = weight_mat_csr * np.c_[reshaped_preds, reshaped_true]

    score = np.sum(np.sqrt(np.mean(np.square(train[:, :num_col] - train[:, num_col:]), axis=1) / weight1) * weight2)

    return 'wrmsse', score, False
