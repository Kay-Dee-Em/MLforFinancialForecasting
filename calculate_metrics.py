import os
import pandas as pd
from itertools import combinations
from multiprocessing import Pool 
from join_predictions import join_dfs_from_all_intervals, join_external_dfs_from_one_interval
from func_validation import calc_metrics_for_combinations
from func_test import determine_best_combination_for_all_intervals, determine_best_combination_for_one_by_one_interval, evaluate_strategies



def calc_combinations(combinations_list: list, train_starts: list, initializers_labels: list, path_type_list: list, combs_res_dir_name: str, calc_one_by_one_or_all: str, save_specific_combination: bool = False, specific_combination_params: tuple = None) -> None:
    """"
    Calculate combinations for main and side model based on one by one neural network or all intervals at once

    :param combinations_list: list
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param initializers_labels: list
    :param path_type_list: list of tuples -> (predictions_dir_name, model_type) e.g. [('PREDICTIONS', 'MAIN'), ('PREDICTIONS_SIDE', 'SIDE')]
    :param combs_res_dir_name: str, directory where files will be saved
    :param calc_one_by_one_or_all: str, calculate one by one neural network (one interval) or group and then calculate (all intervals at the same time) (options: 'ONE_BY_ONE', 'ALL_AT_ONCE')
    :param save_specific_combination: bool, default: False, save predictions for given combination
    :param specific_combination_params: tuple(combination, thold, share, gain_type, dataset_name, nn_type, nn_type, method, top_comb_share, gain)
    :return: None
    """
    
    if not os.path.isdir(combs_res_dir_name):
        os.mkdir(combs_res_dir_name)

    pool = Pool(os.cpu_count())

    datasets_name = ['validation', 'test'] if specific_combination_params is None else specific_combination_params[4]

    for nn_path_type in path_type_list:
        for dataset_name in datasets_name:

            if calc_one_by_one_or_all == 'ONE_BY_ONE':
                for date in train_starts:

                    if nn_path_type[1] == 'MAIN':
                        file_name = 'Prediction_' + dataset_name + '_VS_' + date + '.csv'
                        df_10nn = pd.read_csv(os.path.join(nn_path_type[0], file_name))

                    elif nn_path_type[1] == 'SIDE':
                        df_10nn = join_external_dfs_from_one_interval(nn_path_type[0], initializers_labels, date, dataset_name)
                    
                    pool.apply_async(calc_metrics_for_combinations, args=(df_10nn, combinations_list, initializers_labels, [date], dataset_name, nn_path_type[1], combs_res_dir_name, calc_one_by_one_or_all, save_specific_combination, specific_combination_params)).get()


            elif calc_one_by_one_or_all == 'ALL_AT_ONCE':

                nns_for_intervals = []
                for date in train_starts:
        
                    if nn_path_type[1] == 'MAIN':

                        file_name = 'Prediction_' + dataset_name + '_VS_' + date + '.csv'
                        df_10nn = pd.read_csv(os.path.join(nn_path_type[0], file_name))
                        nns_for_intervals.append(df_10nn)

                    elif nn_path_type[1] == 'SIDE':
                        df_10nn = join_external_dfs_from_one_interval(nn_path_type[0], initializers_labels, date, dataset_name)
                        nns_for_intervals.append(df_10nn)

                df_nns_all = join_dfs_from_all_intervals(nns_for_intervals, initializers_labels, train_starts, dataset_name)

                pool.apply_async(calc_metrics_for_combinations, args=(df_nns_all, combinations_list, initializers_labels, train_starts, dataset_name, nn_path_type[1], combs_res_dir_name, calc_one_by_one_or_all, save_specific_combination, specific_combination_params)).get()

    pool.close()



def determine_best_combination_and_evaluate(combinations_list: list, train_starts: list, nn_type: str, combs_res_dir_name: str) -> None:
    """"
    Determine the best combinations for main and side model and evaluate the best model on test datasets
    Compare different approaches of determining the best model

    :param combinations_list: list
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param nn_type: str, name of the approach type i.e. 'MAIN' (main model) or 'SIDE' (side model)
    :param combs_res_dir_name: str, directory where neural network approach's files are located and the results will be saved
    :return: None
    """

    max_combinations_results = []    
    comb_list = len(combinations_list)-1

    # Determine combinations for % largest gains (top_comb_share) 
    for top_comb_share in [0.1, 0.05, 0.01]: 
        best_one_by_one = determine_best_combination_for_one_by_one_interval(comb_list, train_starts, nn_type, top_comb_share, combs_res_dir_name)
        best_all_at_once = determine_best_combination_for_all_intervals(comb_list, nn_type, top_comb_share, combs_res_dir_name, best_one_by_one)
        evaluate_strategies(best_one_by_one, best_all_at_once, top_comb_share, train_starts, nn_type, combs_res_dir_name, max_combinations_results)


    df_max_combinations_results = pd.concat(max_combinations_results)
    df_max_combinations_results.to_csv(os.path.join(combs_res_dir_name, 'MAX_combinations_results.csv'), index=None)



def main() -> None:

    initializers_labels = ['GN', 'GU', 'HN', 'HU', 'LU', 'OR', 'RN', 'RU', 'TN', 'VS']
    train_starts = ['2001-04-02', '2001-07-02', '2001-10-01', '2002-01-02', '2002-04-01', '2002-07-01', '2002-10-01', '2003-01-02', '2003-04-01']
    combinations_list = sum([list(map(list, combinations(initializers_labels, i))) for i in range(len(initializers_labels) + 1)], [])

    nn_type = 'MAIN'
    predictions_dir_name_nn_type = [('PREDICTIONS', nn_type)]
    combinations_results_dir_name = predictions_dir_name_nn_type[0][0] + '_MAX_PREDICTIONS/'

    calc_combinations(combinations_list, train_starts, initializers_labels, predictions_dir_name_nn_type, combinations_results_dir_name, 'ALL_AT_ONCE')
    calc_combinations(combinations_list, train_starts, initializers_labels, predictions_dir_name_nn_type, combinations_results_dir_name, 'ONE_BY_ONE')
    determine_best_combination_and_evaluate(combinations_list, train_starts, nn_type, combinations_results_dir_name) 

    # Save predictions per day
    df = pd.read_csv(os.path.join(combinations_results_dir_name, 'MAX_combinations_results.csv'))
    df = df.loc[(df['which_best'] == 'BEST') & (df['which_strategy'].isin(['ALL_AT_ONCE', 'ONE_BY_ONE_TESTED_ON_ALL_AT_ONCE'])) & (df['which_approach'] == nn_type)]

    for combination in df['combination'].unique():

        thold = df.loc[df['combination'] == combination, 'treshold'].values[0]
        share = df.loc[df['combination'] == combination, 'share'].values[0] 
        method = df.loc[df['combination'] == combination, 'which_strategy'].values[0]
        
        if method == 'ONE_BY_ONE_TESTED_ON_ALL_AT_ONCE': method = 'OBOT'
        elif method == 'ALL_AT_ONCE': method - 'AAOM'
        elif method == 'ONE_BY_ONE': method = 'OBOM'
        elif method == 'ALL_AT_ONCE_TESTED_ON_ONE_BY_ONE': method = 'AAOT'        

        top_comb_share = df.loc[df['combination'] == combination, 'which_top_comb_share'].values[0]
        gain = df.loc[df['combination'] == combination, 'which_gain'].values[0]

        combination = combination.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '')
        combination = list(combination.split(','))

        params = (combination, thold, share, combinations_results_dir_name, ['test'], nn_type, method, top_comb_share, gain)
        calc_combinations(combinations_list, train_starts, initializers_labels, predictions_dir_name_nn_type, combinations_results_dir_name, 'ALL_AT_ONCE', True, params)



if __name__ == '__main__':

    main()

        


