def cross_join(df1, df2):
    df1['key'] = 0
    df2['key'] = 0
    return df1.merge(df2, how='outer').drop('key', axis=1)
