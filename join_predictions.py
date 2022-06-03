import pandas as pd
import os



def join_external_dfs_from_one_interval(path: str, initializers_labels: list, train_start: str, dataset_name: str) -> pd.DataFrame:
    """
    Join DataFrames which where created (NNModel) by passing the model externally

    :param path: str, DataFrame dir
    :param initializers_labels: list
    :param train_start: str, date in format 'YYYY-MM-DD'
    :param dataset_name: str, name of dataset type i.e. 'validation' or 'test'
    :return: pd.DataFrame
    """

    file_name = 'Prediction_' + dataset_name + '_' + initializers_labels[0] + '_' + train_start + '.csv'
    file = os.path.join(path, file_name)
    df = pd.read_csv(file)

    for init_l in initializers_labels[1:]:

        file_name = 'Prediction_' + dataset_name + '_' + init_l + '_' + train_start + '.csv'
        file = os.path.join(path, file_name)
        df_part = pd.read_csv(file)
        df_part = df_part.loc[:, ['DateTime', file_name[:-4]]]

        df = df.merge(df_part, on='DateTime', how='left')

    return df



def join_dfs_from_all_intervals(nns_for_intervals: list, initializers_labels: list, train_starts: list, dataset_name: str) -> pd.DataFrame:
    """
    Join DataFrames from all intervals into one DataFrame

    :param nns_for_intervals: list, list of one-interval DataFrames
    :param initializers_labels: list
    :param train_starts: list, list of start dates in format 'YYYY-MM-DD'
    :param dataset_name: str, name of dataset type i.e. 'validation' or 'test'
    :return: pd.DataFrame
    """

    df_all = []

    for nn_no in range(len(nns_for_intervals)):
        df_all.append(nns_for_intervals[nn_no].iloc[:, :-len(initializers_labels)])

    df_all = pd.concat(df_all).drop_duplicates().reset_index(drop=True)

    for no_date in range(len(train_starts)):

        col_name = 'Prediction_' + dataset_name + '_'
        predictive_cols = nns_for_intervals[no_date].loc[:, nns_for_intervals[no_date].columns.str.startswith(col_name)].columns.to_list()
        predictive_cols.insert(0, 'DateTime')
        nns_one_interval = nns_for_intervals[no_date].loc[:, predictive_cols]

        df_all = df_all.merge(nns_one_interval, on='DateTime', how='left')

    return df_all