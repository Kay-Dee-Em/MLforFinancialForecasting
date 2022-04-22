import os
import pandas as pd
from itertools import combinations
import numpy as np
import datetime
from multiprocessing import Pool 



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



def calc_metrics_for_combinations(df: pd.DataFrame, combinations_list: list, initializers_labels: list, train_starts: list, dataset_name: str, nn_type: str, dir_name: str, calc_one_by_one_or_all: str) -> None:
    """"
    Calculate metrics (gain and accuracy) for all possible models' combinations (1023 combinations for 10-item list)
    Metrics are calculated based on predictions and given tresholds (uncertainty intervals 0.5 +- thold: 0, 0.005, 0.01, 0.025, 0.05)
    Each combination is also considered based on share of positive (1 == LONG) predictions (share: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    For each combination the maximum gain and maximum accuracy are calculated
    Both maximum combinations and all combinations are saved as CSV file

    :param df: pd.DataFrame, data for which metrics will be determined
    :param combinations_list: list
    :param initializers_labels: list
    :param train_starts: list, list of start dates in format 'YYYY-MM-DD'
    :param dataset_name: str, name of dataset type i.e. 'validation' or 'test'
    :param nn_type: str, name of the approach type i.e. 'MAIN APPROACH' (main model) or 'SIDE APPROACH' (side model)
    :param dir_name: str, directory where files will be saved
    :param calc_one_by_one_or_all: str, calculate one by one neural network (one interval) or group and then calculate (all intervals at the same time) (options: 'ONE_BY_ONE', 'ALL_AT_ONCE')
    :return: None
    """
    
    base_col_name = 'Prediction_' + dataset_name + '_'
    df_base_columns = ['Change', 'Decision']

    comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options, comb_max_options = [], [], [], [], [], [], [], []

   
    if calc_one_by_one_or_all == 'ONE_BY_ONE': print('START:', nn_type.split(' ')[0], dataset_name, train_starts[0], datetime.datetime.now())
    elif calc_one_by_one_or_all == 'ALL_AT_ONCE': print('START:', nn_type.split(' ')[0], dataset_name, datetime.datetime.now())

    # For each combination of models
    for combination in combinations_list[1:]:
        
        # Create temporary DataFrame with combination columns (predictions) and base columns i.e. 'Change' and 'Decision'
        for combination_el in combination:
            for train_start in train_starts:
                column_in_comb = base_col_name + combination_el + '_' + train_start
                df_base_columns.append(column_in_comb)

        df_combination = df.loc[:, df_base_columns].copy()
        df_base_columns = ['Change', 'Decision']
        df_combination.fillna(-9999, inplace=True)

        # For each interval of uncertainty 
        for thold in [0, 0.005, 0.01, 0.025, 0.05]:
            calc_metrics_acc_thold(df_combination, combination, df_base_columns, initializers_labels, base_col_name, train_starts, thold,
                                   comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options)

        # Determine max values for combination, append to list comb_max_options
        max_acc = max(comb_accs_opt); max_gain = max(comb_gains_opt); max_avg_acc = max(comb_avg_accs_opt); max_avg_gain = max(comb_avg_gains_opt)

        max_option_row = [combination, tholds[comb_gains_opt.index(max_gain)], shares[comb_gains_opt.index(max_gain)], comb_accs_opt[comb_gains_opt.index(max_gain)], max_gain,
                                       tholds[comb_accs_opt.index(max_acc)], shares[comb_accs_opt.index(max_acc)], max_acc, comb_gains_opt[comb_accs_opt.index(max_acc)],
                                       tholds[comb_avg_gains_opt.index(max_avg_gain)], shares[comb_avg_gains_opt.index(max_avg_gain)], comb_avg_accs_opt[comb_avg_gains_opt.index(max_avg_gain)], max_avg_gain,
                                       tholds[comb_avg_accs_opt.index(max_avg_acc)], shares[comb_avg_accs_opt.index(max_avg_acc)], max_avg_acc, comb_avg_gains_opt[comb_avg_accs_opt.index(max_avg_acc)]]
        
        comb_max_options.append(max_option_row)
        comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares = [], [], [], [], [], []

    if calc_one_by_one_or_all == 'ONE_BY_ONE':
        
        file_name_all = nn_type.split(' ')[0] + '_All_' + dataset_name + '_' + train_starts[0] + '.csv'
        file_name_max = nn_type.split(' ')[0] + '_Max_' + dataset_name + '_' + train_starts[0] + '.csv'

    elif calc_one_by_one_or_all == 'ALL_AT_ONCE':

        file_name_all = nn_type.split(' ')[0] + '_All_' + dataset_name + '_intervals.csv'
        file_name_max = nn_type.split(' ')[0] + '_Max_' + dataset_name + '_intervals.csv'

    # Crete and save DataFrames for all and max values for each combination
    df_comb_options = pd.DataFrame(comb_options, columns=['combination', 'treshold', 'share', 'accuracy', 'gain', 'avg_accuracy', 'avg_gain'])
    df_comb_max_options = pd.DataFrame(comb_max_options, columns=['combination', 'MG_treshold', 'MG_share', 'MG_accuracy', 'MG_gain', 'MA_treshold', 'MA_share', 'MA_accuracy', 'MA_gain',
                                                                                 'AVG_MG_treshold', 'AVG_MG_share', 'AVG_MG_accuracy', 'AVG_MG_gain', 'AVG_MA_treshold', 'AVG_MA_share', 'AVG_MA_accuracy', 'AVG_MA_gain'])
    
    df_comb_options.to_csv(os.path.join(dir_name, file_name_all), index=None)
    df_comb_max_options.to_csv(os.path.join(dir_name, file_name_max), index=None)

    print('END:', datetime.datetime.now())
    


def calc_metrics_acc_thold(df_combination: pd.DataFrame, combination: list, df_base_columns: list, initializers_labels: list, base_col_name: str, train_starts: list, thold: float, 
                           comb_accs_opt: list, comb_gains_opt: list, comb_avg_accs_opt: list, comb_avg_gains_opt: list, tholds: list, shares: list, comb_options: list) -> None:
    """
    Auxiliary function for func calc_metrics_for_combinations (main description in calc_metrics_for_combinations)

    :param df_combination: pd.DataFrame, data for which metrics will be determined
    :param combination: list, list of initializers in combination
    :param df_base_columns: list, base columns i.e. 'Change' and 'Decision'
    :param initializers_labels: list
    :param base_col_name: str, include dataset type  i.e. 'validation' or 'test'
    :param train_starts: list, list of start dates in format 'YYYY-MM-DD'
    :param thold: float, threshold, uncertainty interval +- 0.5
    :param comb_accs_opt: list
    :param comb_gains_opt: list
    :param comb_avg_accs_opt: list
    :param comb_avg_gains_opt: list
    :param tholds: list
    :param shares: list
    :param comb_options: list
    :return: None
    """

    # Create temporary DataFrame
    # Calculate Average Prediction ('Prediction_avg) as an average of all predictive columns
    # For predictions between uncertainty interval (0.5 +- treshold ('thold')) assign np.nan
    # Round predictions according to math's rule (columns 'Predicition_avg')  
    df_comb_opt = df_combination.copy()
    pred_col_names = list(set(df_comb_opt.columns.to_list()).difference(set(df_base_columns)))
    df_comb_opt['Prediction_avg'] = np.nanmean(df_comb_opt[df_comb_opt.gt(-9999)][pred_col_names], axis=1) 
    df_comb_opt.loc[(df_comb_opt['Prediction_avg'] >= (0.5-thold)) & (df_comb_opt['Prediction_avg'] <= (0.5+thold)), 'Prediction_avg'] = np.nan
    df_comb_opt['Prediction_avg'] = round(df_comb_opt['Prediction_avg'])

    # For each column calculate predictions in the same way i.e. include uncertainty interval and round predictions (if applicable)
    for combination_el in combination:
        for train_start in train_starts:

            col_el_opt = base_col_name + combination_el + '_' + train_start
            df_comb_opt.loc[(df_comb_opt[col_el_opt] >= (0.5-thold)) & (df_comb_opt[col_el_opt] <= (0.5+thold)), col_el_opt] = np.nan
            df_comb_opt[col_el_opt] = round(df_comb_opt[col_el_opt])
    
    # Calculate number of predictive columns where prediction was determined and equals 1 ('Long Predictions')
    # Calculate number of predictive columns where decision was determined and equals 0 or 1  ('How many NNs')
    # Calculate number od predictive columns where decision was not determined and equals np.nan ('NaNs Predictions')
    # Create summary DataFrame and assign above summarizing columns (df_comb_smry) 

    decision_sum_cols = pd.Series(np.nansum(df_comb_opt[df_comb_opt.gt(-9999)][pred_col_names], axis=1))
    decision_count_cols = pd.Series(df_comb_opt[df_comb_opt.gt(-9999)][pred_col_names].count(axis=1))
    decision_count_nan_cols = df_comb_opt[df_comb_opt.ge(-9999)][pred_col_names].isnull().sum(axis=1)

    df_comb_smry = pd.DataFrame([df_comb_opt['Change'], df_comb_opt['Decision'], decision_sum_cols, decision_count_cols, decision_count_nan_cols],
                                    index=['Change', 'Decision', 'Long Predictions', 'How many NNs', 'NaNs Predictions'])

    # Calculate share of LONG predictions in all decisions made ('Share Long Predictions')
    # Assign to df_comb_smry column with average predictions ('Prediction_avg')
    df_comb_smry = df_comb_smry.T
    df_comb_smry['Share Long Predictions'] = df_comb_smry['Long Predictions'] / df_comb_smry['How many NNs']
    df_comb_smry['Average Prediction'] = df_comb_opt['Prediction_avg']

    # For each share of positive (1) predictions (LONG)
    for share in np.linspace(0, 1, len(initializers_labels)+1)[1:]:
        calc_metrics_acc_thold_and_share(df_comb_smry, combination, thold, round(share,3), comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options)



def calc_metrics_acc_thold_and_share(df_summary: pd.DataFrame, combination: list, thold: int, share: float,
                                     comb_accs_opt: list, comb_gains_opt: list, comb_avg_accs_opt: list, comb_avg_gains_opt: list, tholds: list, shares: list, comb_options: list) -> None:
    """
    Auxiliary function for calc_metrics_acc_thold and func calc_metrics_for_combinations (main description in calc_metrics_for_combinations)

    :param df_summary: pd.DataFrame, summary DataFrame for given DataFrame
    :param combination: list, list of initializers in combination
    :param thold: int, threshold, uncertainty interval +- 0.5
    :param share: float, threshold, share of positive predictions
    :param comb_accs_opt: list
    :param comb_gains_opt: list
    :param comb_avg_accs_opt: list
    :param comb_avg_gains_opt: list
    :param tholds: list
    :param shares: list
    :param comb_options: list
    :return: None
    """

    # Calculate final prediction ('Final Prediction')
    # If number of Nans columns (columns where decision was not determined in func calc_metrics_acc_thold) is greater or equals 0.5, assign np.nan
    # Else determine final prediction based on share value
    df_smry = df_summary.copy()
    df_smry['Final Prediction'] = 0
    df_smry.loc[(df_smry['NaNs Predictions']/(df_smry['NaNs Predictions']+df_smry['How many NNs'])) >= 0.5, 'Final Prediction'] = np.nan
    df_smry.loc[df_smry['Share Long Predictions'] >= share, 'Final Prediction'] = 1
    
    df_smry_wo_nans = df_smry.loc[df_smry['Final Prediction'].notnull()].copy()
    df_smry_w_nans = df_smry.loc[df_smry['Final Prediction'].isnull()].copy()

    # Calculate decision accuracy (only for rows without NaNs' values - df_smry_wo_nans) else (if all rows are null) assign np.nan
    # Calculate decision gain based on determined decision (DataFrame without NaNs - df_smry_wo_nans)
    # Add gain based on hold strategy - no action (the decision was not determined) as sum of column Change
    dec_accuracy = np.nan if len(df_smry_wo_nans) == 0 else (df_smry_wo_nans['Decision'] == df_smry_wo_nans['Final Prediction']).sum()/len(df_smry_wo_nans['Decision'])
    dec_gain = ((-abs(df_smry_wo_nans['Decision']-df_smry_wo_nans['Final Prediction']) + (df_smry_wo_nans['Decision'] == df_smry_wo_nans['Final Prediction'])) * abs(df_smry_wo_nans['Change'])).sum()
    dec_gain += df_smry_w_nans['Change'].sum()

    df_smry_wo_nans_avg = df_smry.loc[(df_smry['Final Prediction'].notnull()) & (df_smry['Average Prediction'].notnull())].copy()
    df_smry_w_nans_avg = df_smry.loc[(df_smry['Final Prediction'].isnull()) | (df_smry['Average Prediction'].isnull())].copy()

    # Repeat above calculations for predictions made using average
    # Calculate decision accuracy based on average predictions, decision gain based on average predictions and add gain based on average decision when no strategy was determined
    dec_accuracy_by_avg =  np.nan if len(df_smry_wo_nans_avg) == 0 else (df_smry_wo_nans_avg['Decision'] == df_smry_wo_nans_avg['Average Prediction']).sum()/len(df_smry_wo_nans_avg['Decision'])
    dec_gain_by_avg = ((-abs(df_smry_wo_nans_avg['Decision']-df_smry_wo_nans_avg['Average Prediction']) + (df_smry_wo_nans_avg['Decision'] == df_smry_wo_nans_avg['Average Prediction'])) * abs(df_smry_wo_nans_avg['Change'])).sum()
    dec_gain_by_avg += df_smry_w_nans_avg['Change'].sum()

    # Append results to lists
    comb_accs_opt.append(dec_accuracy); comb_gains_opt.append(dec_gain); comb_avg_accs_opt.append(dec_accuracy_by_avg); comb_avg_gains_opt.append(dec_gain_by_avg); tholds.append(thold); shares.append(share)

    option_row = [combination, thold, share, dec_accuracy, dec_gain, dec_accuracy_by_avg, dec_gain_by_avg]
    comb_options.append(option_row)



def calc_combinations(combinations_list: list, train_starts: list, initializers_labels: list, combs_res_dir_name: str, calc_one_by_one_or_all: str) -> None:
    """"
    Calculate combinations for main and side model based on one by one neural network or all intervals at once

    :param combinations_list: list
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param initializers_labels: list
    :param combs_res_dir_name: str, directory where files will be saved
    :param calc_one_by_one_or_all: str, calculate one by one neural network (one interval) or group and then calculate (all intervals at the same time) (options: 'ONE_BY_ONE', 'ALL_AT_ONCE')
    :return: None
    """
    
    if not os.path.isdir(combs_res_dir_name):
        os.mkdir(combs_res_dir_name)

    pool = Pool(os.cpu_count())

    for nn_path_type in list(zip(['PREDICTIONS', 'PREDICTIONS_SIDE'], ['MAIN APPROACH', 'SIDE APPROACH'])):
        for dataset_name in ['validation', 'test']:

            if calc_one_by_one_or_all == 'ONE_BY_ONE':
                for date in train_starts:

                    if nn_path_type[1] == 'MAIN APPROACH':
                        file_name = 'Prediction_' + dataset_name + '_VS_' + date + '.csv'
                        df_10nn = pd.read_csv(os.path.join(nn_path_type[0], file_name))

                    elif nn_path_type[1] == 'SIDE APPROACH':
                        df_10nn = join_external_dfs_from_one_interval(nn_path_type[0], initializers_labels, date, dataset_name)
                    
                    pool.apply_async(calc_metrics_for_combinations, args=(df_10nn, combinations_list, initializers_labels, [date], dataset_name, nn_path_type[1], combs_res_dir_name, calc_one_by_one_or_all)).get()


            elif calc_one_by_one_or_all == 'ALL_AT_ONCE':

                nns_for_intervals = []
                for date in train_starts:
        
                    if nn_path_type[1] == 'MAIN APPROACH':

                        file_name = 'Prediction_' + dataset_name + '_VS_' + date + '.csv'
                        df_10nn = pd.read_csv(os.path.join(nn_path_type[0], file_name))
                        nns_for_intervals.append(df_10nn)

                    elif nn_path_type[1] == 'SIDE APPROACH':
                        df_10nn = join_external_dfs_from_one_interval(nn_path_type[0], initializers_labels, date, dataset_name)
                        nns_for_intervals.append(df_10nn)

                df_nns_all = join_dfs_from_all_intervals(nns_for_intervals, initializers_labels, train_starts, dataset_name)

                pool.apply_async(calc_metrics_for_combinations, args=(df_nns_all, combinations_list, initializers_labels, train_starts, dataset_name, nn_path_type[1], combs_res_dir_name, calc_one_by_one_or_all)).get()

    pool.close()



def determine_best_combination_and_evaluate(combinations_list: list, train_starts: list, combs_res_dir_name: str) -> None:
    """"
    Determine the best combinations for main and side model and evaluate the best model on test datasets
    Compare different approaches of determining the best model

    :param combinations_list: list
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param combs_res_dir_name: str, directory where files are located
    :return: None
    """
    max_combinations_results = []
    # For each of neural network approach 
    for nn_type in ['MAIN APPROACH', 'SIDE APPROACH']:
        
        # Determine combinations for % largest gains (top_comb_share) 
        comb_list = len(combinations_list)-1
        
        for top_comb_share in [0.1, 0.05, 0.01]:
            best_one_by_one = determine_best_combination_for_one_by_one_interval(comb_list, train_starts, nn_type, top_comb_share, combs_res_dir_name)
            best_all_at_once = determine_best_combination_for_all_intervals(comb_list, nn_type, top_comb_share, combs_res_dir_name, best_one_by_one)
            evaluate_strategies(best_one_by_one, best_all_at_once, top_comb_share, train_starts, nn_type, combs_res_dir_name, max_combinations_results)


    df_max_combinations_results = pd.concat(max_combinations_results)
    df_max_combinations_results.to_csv(os.path.join(combs_res_dir_name, 'MAX_combinations_results.csv'), index=None)



def determine_best_combination_for_one_by_one_interval(comb_list: list, train_starts: list, nn_type: str, top_comb_share: float, combs_res_dir_name: str) -> tuple:
    """"
    Auxiliary function for determine_best_combination_and_evaluate (main description in determine_best_combination_and_evaluate)
    Determine the best combinations for main and side model

    :param comb_list: list
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param nn_type: str, name of the approach type i.e. 'MAIN APPROACH' (main model) or 'SIDE APPROACH' (side model)
    :param top_comb_share: float, share of largest gain 
    :param combs_res_dir_name: str, directory where files are located
    :return: tuple
    """

    val_result_maxs = []

    # Including all neural networks intervals (train_starts)
    val_max_score, val_avg_max_score = 0, 0
    for date in train_starts:
    
        file_name = nn_type.split(' ')[0] + '_Max_validation_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        
        # For each neural network determine
        # Whether the mean of 25% largest values is higher for an approach based on all columns or 
        # An approach treated as an average of predictions (func calc_metrics_for_combinations)
        val_max = df_result['MG_gain'].nlargest(int(np.floor(comb_list*0.25))).mean()
        val_avg_max = df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list*0.25))).mean()
        #print('MG',val_max ,'AVG',val_avg_max)

        if val_max > val_avg_max: val_max_score += 1 
        else: val_avg_max_score += 1

        # Determine combinations for n largest values (top_comb_share) for approach with higher mean (val_max vs val_avg_max)
        val_result_max = df_result.loc[df_result['MG_gain'].nlargest(int(np.floor(comb_list*top_comb_share))).index]['combination'] if val_max > val_avg_max \
                    else df_result.loc[df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list*top_comb_share))).index]['combination']

        # And append them (i.e. combinations to list val_result_maxs)
        val_result_maxs.extend(val_result_max)

    # Determine the combination with the highest number of occurrences
    # If more than one combination exists, choose the most numerous combination
    # With the highest mean of column which 25% largest values were chosen more often (val_max_score vs val_avg_max)
    df_top_combinations = pd.Series(val_result_maxs).value_counts()
    top_combinations = df_top_combinations.loc[df_top_combinations.values == max(df_top_combinations)].index.to_list()
    top_combination_len = len(max(top_combinations, key=len))
    top_combinations_names = [comb for comb in top_combinations if len(comb) == top_combination_len]

    which_gain = 'MG_gain' if val_max_score > val_avg_max_score else 'AVG_MG_gain'
    
    top_combinations_temp = []
    for date in train_starts:
    
        file_name = nn_type.split(' ')[0] + '_Max_validation_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        top_comb_temp = df_result.loc[df_result['combination'].isin(top_combinations_names)].sort_values(by=which_gain, ascending=False)[['combination', which_gain]]
        top_combinations_temp.append(top_comb_temp)

    df_top_combination = pd.concat(top_combinations_temp)
    top_combination = df_top_combination.groupby('combination').mean()[which_gain].nlargest(1000).index.to_list()[0]

    # Determine which treshold was the most common for chosen top combination
    tholds = {}
    tholds[0.0],  tholds[0.005], tholds[0.01], tholds[0.025], tholds[0.05] = 0, 0, 0, 0, 0

    shares = {}
    shares[0.1], shares[0.2], shares[0.3], shares[0.4], shares[0.5], shares[0.6], shares[0.7], shares[0.8], shares[0.9], shares[1.0] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for date in train_starts:
    
        file_name = nn_type.split(' ')[0] + '_Max_validation_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        thold = float(df_result.loc[df_result['combination'] == top_combination, which_gain[:-4] + 'treshold'])
        share = float(df_result.loc[df_result['combination'] == top_combination, which_gain[:-4] + 'share'])

        match thold:
            case 0.0:   tholds[0.0] += 1
            case 0.005: tholds[0.005] += 1
            case 0.01:  tholds[0.01] += 1
            case 0.025: tholds[0.025] += 1
            case 0.05:  tholds[0.05] += 1

        match share:
            case 0.1: shares[0.1] += 1
            case 0.2: shares[0.2] += 1
            case 0.3: shares[0.3] += 1
            case 0.4: shares[0.4] += 1
            case 0.5: shares[0.5] += 1
            case 0.6: shares[0.6] += 1
            case 0.7: shares[0.7] += 1
            case 0.8: shares[0.8] += 1
            case 0.9: shares[0.9] += 1
            case 1.0: shares[1.0] += 1

    thold_test = max(tholds, key=tholds.get)
    share_test = max(shares, key=shares.get)

    which_test_gain = 'gain' if which_gain == 'MG_gain' else 'avg_gain'

    return (top_combination, thold_test, share_test, which_test_gain)



def determine_best_combination_for_all_intervals(comb_list: list, nn_type: str, top_comb_share: float, combs_res_dir_name: str, best_one_by_one: tuple) -> tuple:
    """"
    Auxiliary function for determine_best_combination_and_evaluate (main description in determine_best_combination_and_evaluate)
    Determine the best combinations for main and side model

    :param comb_list: list
    :param nn_type: str, name of the approach type i.e. 'MAIN APPROACH' (main model) or 'SIDE APPROACH' (side model)
    :param top_comb_share: float, share of largest gain 
    :param combs_res_dir_name: str, directory where files are located
    :return: tuple
    """

    val_max_score, val_avg_max_score = 0, 0
    
    file_name = nn_type.split(' ')[0] + '_Max_validation_intervals.csv'
    df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
    
    # Determine whether the mean of 25% largest values is higher for an approach based on all columns or 
    # An approach treated as an average of predictions (func calc_metrics_for_combinations)
    val_max = df_result['MG_gain'].nlargest(int(np.floor(comb_list*0.25))).mean()
    val_avg_max = df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list*0.25))).mean()
    
    if val_max > val_avg_max: val_max_score += 1 
    else: val_avg_max_score += 1

    # Determine combinations for n largest values (top_comb_share) for approach with higher mean (val_max vs val_avg_max)
    val_result_max = df_result.loc[df_result['MG_gain'].nlargest(int(np.floor(comb_list*top_comb_share))).index]['combination'] if val_max > val_avg_max \
                else df_result.loc[df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list*top_comb_share))).index]['combination']

    # Determine the combination with the highest mean
    # If more than one combination exists, choose the most numerous combination
    # With the highest mean of column which 25% largest values were chosen more often (val_max_score vs val_avg_max)
    df_top_combinations = pd.Series(val_result_max).value_counts()
    top_combinations = df_top_combinations.loc[df_top_combinations.values == max(df_top_combinations)].index.to_list()
    top_combination_len = len(max(top_combinations, key=len))
    top_combinations_names = [comb for comb in top_combinations if len(comb) == top_combination_len]

    which_gain = 'MG_gain' if val_max_score > val_avg_max_score else 'AVG_MG_gain'
    top_combination = df_result.loc[df_result['combination'].isin(top_combinations_names)].sort_values(by=which_gain, ascending=False)['combination'].iloc[0]
    thold_test = float(df_result.loc[df_result['combination'] == top_combination, which_gain[:-4] + 'treshold'])
    share_test = float(df_result.loc[df_result['combination'] == top_combination, which_gain[:-4] + 'share'])

    which_test_gain = 'gain' if which_gain == 'MG_gain' else 'avg_gain'

    return (top_combination, thold_test, share_test, which_test_gain)



def evaluate_strategies(best_one_by_one: tuple, best_all_at_once: tuple, top_comb_share: float, train_starts: list, nn_type: str, combs_res_dir_name: str, max_combinations_results: list) -> None:
    """"
    Auxiliary function for determine_best_combination_and_evaluate (main description in determine_best_combination_and_evaluate)
    Evaluate the best model on test datasets according to one by one interval and all intervals at once
    Compare whether one strategy could be used to determine the best model for the other one

    :param best_one_by_one: tuple, the best combination, treshold, share of positive predictions and gain type (main or as average) for one by one strategy
    :param best_all_at_once: tuple, the best combination, treshold, share of positive predictions and gain type (main or as average) for all at once intervals strategy
    :param top_comb_share: float, share of largest gain  
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param nn_type: str, name of the approach type i.e. 'MAIN APPROACH' (main model) or 'SIDE APPROACH' (side model)
    :param combs_res_dir_name: str, directory where files are located
    :param max_combinations_results: list
    :return: None
    """
    
    ####################   ONE BY ONE   ####################

    # Having the best combination, approach to gain (standard gain or average gain), treshold and share
    # Evaluate test model summing gain for one by one strategy
    top_combination, thold_test, share_test, which_test_gain = best_one_by_one

    gain = 0
    for date in train_starts:
        
        file_name = nn_type.split(' ')[0] + '_All_test_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        gain += df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]
    
    #print('OBOM', top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', gain)

    df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ONE_BY_ONE'
    df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type.split(' ')[0]
    df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
    df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
    result = df_result.loc[df_result['combination'] == top_combination]

    max_combinations_results.append(result)


    ####################   ONE BY ONE TESTED ON ALL INTERVALS AT ONCE  ####################

    # Evaluate test model on best (according to one by one strategy) combination on all intervals dataset 
    file_name = nn_type.split(' ')[0] + '_All_test_intervals.csv'
    df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
    gain = df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]

    print('OBOT', top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', gain)

    df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ONE_BY_ONE_TESTED'
    df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type.split(' ')[0]
    df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
    df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
    result = df_result.loc[df_result['combination'] == top_combination]

    max_combinations_results.append(result)


    ####################   ALL AT ONCE   ####################

    # Having the best combination, approach to gain (standard gain or average gain), treshold and share
    # Evaluate test model for all intervals at once strategy
    top_combination, thold_test, share_test, which_test_gain = best_all_at_once

    file_name = nn_type.split(' ')[0] + '_All_test_intervals.csv'
    df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
    gain = df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]

    print('AAOM', top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', gain)

    df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ALL_AT_ONCE'
    df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type.split(' ')[0]
    df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
    df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
    result = df_result.loc[df_result['combination'] == top_combination]

    max_combinations_results.append(result)


    ####################   ALL AT ONCE TESTED ON ONE BY ONE   ####################

    # Evaluate test model on best (according to all at once strategy) combination on one by one interval datasets
    gain = 0
    for date in train_starts:
        
        file_name = nn_type.split(' ')[0] + '_All_test_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        gain += df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]
    
    #print('AAOT',top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', gain)

    df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ONE_BY_ONE'
    df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type.split(' ')[0]
    df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
    df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
    result = df_result.loc[df_result['combination'] == top_combination]

    max_combinations_results.append(result)
    
    print()



def main() -> None:

    initializers_labels = ['GN', 'GU', 'HN', 'HU', 'LU', 'OR', 'RN', 'RU', 'TN', 'VS']
    train_starts = ['2001-04-02', '2001-07-02', '2001-10-01', '2002-01-02', '2002-04-01', '2002-07-01', '2002-10-01', '2003-01-02', '2003-04-01']
    combinations_list = sum([list(map(list, combinations(initializers_labels, i))) for i in range(len(initializers_labels) + 1)], [])

    combinations_results_dir_name = r'MAX_PREDICTIONS/'

    calc_combinations(combinations_list, train_starts, initializers_labels, combinations_results_dir_name, 'ALL_AT_ONCE')
    calc_combinations(combinations_list, train_starts, initializers_labels, combinations_results_dir_name, 'ONE_BY_ONE')
    determine_best_combination_and_evaluate(combinations_list, train_starts, combinations_results_dir_name) 


if __name__ == '__main__':

    main()

        


