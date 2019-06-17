import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def func_df_describe_all(df):
    """
    function similar to describe() with missing value
    :param df: input dataframe
    :return: df_summary
    """
    df_summary = df.describe(include='all').T
    df_summary['miss_perc'] = (df.isnull().sum()/df.shape[0]*100).values
    return df_summary


def func_df_display_all(df, max_rows=1000, max_cols=1000):
    """
    function similar to display, but temporarily extend the max number of rows and columns
    :param df: dataframe
    :param max_rows:
    :param max_cols:
    :return: display the dataframe
    """
    import pandas as pd
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_cols):
        display(df)


def func_eda_hist_by_label_plot(df_X, Y, normal=False, figsize=(8,6)):
    """
    function to plot the normalized probability density distribution colored by label
    :param df_X: feature dataframe
    :param Y: label pandas series
    :param normal: density plot
    :param figsize: figure size control
    :return: display plot
    """

    y_uniques = Y.unique()  # unique labels, e.g. 0 or 1

    for col in df_X.columns:
        # filter nan vales
        mask = ~df_X[col].isnull()
        x = df_X.loc[mask, col]
        y = Y[mask]

        # plot
        fig = plt.figure(figsize=figsize)
        plt.hist([x[y == y_unique] for y_unique in y_uniques],
                 label=y_uniques,
                 weights=[np.ones(x[y == y_unique].count()) / x[y == y_unique].count() for y_unique in y_uniques])#density=True)
        plt.xlabel(col)
        plt.ylabel('Normalized Probability Distribution')
        plt.legend()
        plt.tight_layout()
        #plt.savefig(dir_png+'eda-hist-'+prefix+col+'.png')
        #plt.close(fig)
        plt.show()

    return


def func_box_plot(df, cols):
    """
    function to make box plot
    :param df: dataframe
    :param cols: columns to be plotted
    :return:
    """
    for col in cols:
        fig = plt.figure(figsize=(12,4))
        plt.subplot(211)
        plt.xlim(df[col].min(), df[col].max()*1.1)
        ax = df[col].plot(kind='kde')
        plt.subplot(212)
        plt.xlim(df[col].min(), df[col].max()*1.1)
        sns.boxplot(x=df[col])
        plt.tight_layout()
        plt.show()
    return


def func_IQR(df):
    """
    function to calculate interquantile range (IQR) and upper and lower bounds for outlier removal
    :param df: dataframe
    :return: dataframe of IQR
    """
    df_IQR = df.quantile([0.25, 0.75], axis=0).T
    df_IQR.columns = ['Q1', 'Q3']
    df_IQR['IQR'] = df_IQR['Q3'] - df_IQR['Q1']
    df_IQR['Lower_Bound'] = df_IQR['Q1'] - 1.5 * df_IQR['IQR']
    df_IQR['Upper_Bound'] = df_IQR['Q3'] + 1.5 * df_IQR['IQR']
    return df_IQR


def func_filter_outlier_iqr(df, df_IQR):
    """
    function to filter outlier based on IQR
    :param df: dataframe
    :param df_IQR: dataframe of IQR ranges
    :return: filtered dataframe with outliers replaced with NaN
    """
    # set variables outside the upper and lower bounds
    for index, row in df_IQR.iterrows():
        df.loc[(df[index] < row['Lower_Bound']) | (df[index] > row['Upper_Bound']), index] = np.nan
    return df
