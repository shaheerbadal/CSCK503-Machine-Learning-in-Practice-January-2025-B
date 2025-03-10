import pandas as pd


def remove_outliers_by_iqr(df, features_to_check, threshold=3):

    outlier_mask = pd.Series([False] * len(df))

    for col in features_to_check:
        #Q1 (25th percentile)
        Q1 = df[col].quantile(0.25)
        #Q3 (75th percentile)
        Q3 = df[col].quantile(0.75)

        #IQR
        IQR = Q3 - Q1

        #lower and upper bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        #outlier mask for the column
        outlier_mask |= ((df[col] < lower_bound) | (df[col] > upper_bound))

    #outliers removal
    df = df[~outlier_mask]

    outliers_count = outlier_mask.sum()
    outliers_pct = (outliers_count / df.index.size * 100)
    print(f"Dropping {outliers_count} ({outliers_pct:.2f}%) as outliers.")

    return df