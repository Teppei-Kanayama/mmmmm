from typing import List

import pandas as pd


def cross_join(df1, df2):
    df1['key'] = 0
    df2['key'] = 0
    return df1.merge(df2, how='outer').drop('key', axis=1)


def get_uncertainty_ids(df: pd.DataFrame, level: List[str], surfix: str) -> List[str]:
    assert 'percentile' in df.columns
    if len(level) > 1:
        return [f'{lev1}_{lev2}_{q:.3f}_{surfix}' for lev1, lev2, q in
                zip(df[level[0]].values, df[level[1]].values, df['percentile'].values)]
    elif level[0] == 'id':
        return [f"{lev}_{q:.3f}_{surfix}" for lev, q in zip(df['id'].values, df['percentile'])]
    return [f"{lev}_X_{q:.3f}_{surfix}" for lev, q in zip(df[level[0]].values, df['percentile'].values)]
