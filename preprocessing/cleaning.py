def test_clean(df):
    features = df.drop(columns=["Species"])
    df_features_transformed = features * 2 + 1
    df[features.columns] = df_features_transformed
    return df