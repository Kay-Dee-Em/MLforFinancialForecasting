import pandas as pd
import os
import numpy as np



def determine_best_combination_for_one_by_one_interval(comb_list_len: int, train_starts: list, nn_type: str, top_comb_share: float, combs_res_dir_name: str) -> tuple:
    """"
    Auxiliary function for determine_best_combination_and_evaluate (main description in determine_best_combination_and_evaluate)
    Determine the best combinations for main and side model

    :param comb_list_len: int, length of combination list 
    :param train_starts: list, list od start dates in format 'YYYY-MM-DD'
    :param nn_type: str, name of the approach type i.e. 'MAIN' (main model) or 'SIDE' (side model)
    :param top_comb_share: float, share of largest gain 
    :param combs_res_dir_name: str, directory where files are located
    :return: tuple
    """

    val_result_maxs = []

    # Including all neural networks intervals (train_starts)
    val_max_score, val_avg_max_score = 0, 0
    for date in train_starts:
    
        file_name = nn_type + '_Max_validation_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        
        # For each neural network determine
        # Whether the mean of 25% largest values is higher for an approach based on all columns or 
        # An approach treated as an average of predictions (func calc_metrics_for_combinations)
        val_max = df_result['MG_gain'].nlargest(int(np.floor(comb_list_len*0.25))).mean() 
        val_avg_max = df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list_len*0.25))).mean()

        if val_max > val_avg_max: val_max_score += 1 
        else: val_avg_max_score += 1

        # Determine combinations for n largest values (top_comb_share) for approach with higher mean (val_max vs val_avg_max)
        val_result_max = df_result.loc[df_result['MG_gain'].nlargest(int(np.floor(comb_list_len*top_comb_share))).index]['combination'] if val_max > val_avg_max \
                    else df_result.loc[df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list_len*top_comb_share))).index]['combination']

        # And append them (i.e. combinations to list val_result_maxs)
        val_result_maxs.extend(val_result_max)

    # Determine the combination with the highest number of occurrences
    # If more than one combination exists, choose the most profitable one
    # And if still more than one combination exists, choose the longest one (the highest number of initial weights) 
    # Profit is calculated from the highest mean of column which 25% largest values were chosen more often (val_max_score vs val_avg_max)
    df_top_combinations = pd.Series(val_result_maxs).value_counts()
    top_combinations = df_top_combinations.loc[df_top_combinations.values == max(df_top_combinations)].index.to_list()

    which_gain = 'MG_gain' if val_max_score > val_avg_max_score else 'AVG_MG_gain'
    
    top_combinations_temp = []
    for date in train_starts:
    
        file_name = nn_type + '_Max_validation_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        top_comb_temp = df_result.loc[df_result['combination'].isin(top_combinations)].sort_values(by=which_gain, ascending=False)[['combination', which_gain]]
        top_combinations_temp.append(top_comb_temp)

    df_top_combination = pd.concat(top_combinations_temp)
    top_combinations_mean = df_top_combination.groupby('combination').mean()
    top_gain = top_combinations_mean[which_gain].nlargest(comb_list_len).max()
    top_of_the_top_combinations = top_combinations_mean.loc[top_combinations_mean[which_gain] == top_gain].index.to_list()
    top_combination = max(top_of_the_top_combinations, key=len)

    # Determine which treshold was the most common for chosen top combination
    tholds, shares = [], []
    for date in train_starts:
    
        file_name = nn_type + '_Max_validation_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))

        thold_counts = df_result.loc[:, which_gain[:-4] + 'treshold'].value_counts()
        share_counts = df_result.loc[:, which_gain[:-4] + 'share'].value_counts()

        tholds.append(thold_counts)
        shares.append(share_counts)

    df_tholds = pd.concat(tholds)
    tholds_counts_all = df_tholds.groupby(df_tholds.index).sum()
    thold_test = list(df_tholds[tholds_counts_all == tholds_counts_all.max()].index)[0]

    df_shares = pd.concat(shares)
    shares_counts_all = df_shares.groupby(df_shares.index).sum() * (1-abs(0.5-df_shares.groupby(df_shares.index).sum().index))
    share_test = list(shares_counts_all[shares_counts_all == shares_counts_all.max()].index)[0]

    which_test_gain = 'gain' if which_gain == 'MG_gain' else 'avg_gain'

    return (top_combination, thold_test, share_test, which_test_gain)



def determine_best_combination_for_all_intervals(comb_list_len: int, nn_type: str, top_comb_share: float, combs_res_dir_name: str, best_one_by_one: tuple) -> tuple:
    """"
    Auxiliary function for determine_best_combination_and_evaluate (main description in determine_best_combination_and_evaluate)
    Determine the best combinations for main and side model

    :param comb_list_len: int, length of combination list 
    :param nn_type: str, name of the approach type i.e. 'MAIN' (main model) or 'SIDE' (side model)
    :param top_comb_share: float, share of largest gain 
    :param combs_res_dir_name: str, directory where files are located
    :return: tuple
    """
    
    file_name = nn_type + '_Max_validation_intervals.csv'
    df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
    
    # Determine whether the mean of 25% largest values is higher for an approach based on all columns or 
    # An approach treated as an average of predictions (func calc_metrics_for_combinations)
    val_max = df_result['MG_gain'].nlargest(int(np.floor(comb_list_len*0.25))).mean() 
    val_avg_max = df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list_len*0.25))).mean()
    
    # Determine combinations for n largest values (top_comb_share) for approach with higher mean (val_max vs val_avg_max)
    top_combinations = df_result.loc[df_result['MG_gain'].nlargest(int(np.floor(comb_list_len*top_comb_share))).index]['combination'] if val_max > val_avg_max \
                else df_result.loc[df_result['AVG_MG_gain'].nlargest(int(np.floor(comb_list_len*top_comb_share))).index]['combination']

    # Determine the combination with the highest mean
    # If more than one combination exists, choose the most numerous combination
    # With the highest mean of column which 25% largest values were chosen more often (val_max_score vs val_avg_max)
    which_gain = 'MG_gain' if val_max > val_avg_max else 'AVG_MG_gain' #add change if mean is the same
    top_gain = df_result.loc[df_result['combination'].isin(top_combinations)].sort_values(by=which_gain, ascending=False)[which_gain].iloc[0]
    top_of_the_top_combinations = df_result.loc[df_result[which_gain] == top_gain, 'combination'].to_list()
    top_combination = max(top_of_the_top_combinations, key=len)

    thold_test = df_result.loc[:, which_gain[:-4] + 'treshold'].value_counts().index[0]
    share_counts = df_result.loc[:, which_gain[:-4] + 'share'].value_counts()
    share_counts_share = share_counts * (1-abs(0.5-share_counts.index))
    share_test = list(share_counts_share[share_counts_share == share_counts_share.max()].index)[0]

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
    :param nn_type: str, name of the approach type i.e. 'MAIN' (main model) or 'SIDE' (side model)
    :param combs_res_dir_name: str, directory where files are located
    :param max_combinations_results: list
    :return: None
    """
    
    ####################   ONE BY ONE   ####################

    # Having the best combination, approach to gain (standard gain or average gain), treshold and share
    # Evaluate test model summing gain for one by one strategy
    top_combination, thold_test, share_test, which_test_gain = best_one_by_one
    which_test_accuracy = 'accuracy' if which_test_gain == 'gain' else 'avg_accuracy'

    gain = 0
    for date in train_starts:
        
        file_name = nn_type + '_All_test_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        gain = df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]
        accuracy = df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_accuracy].iloc[0]
    
        print('OBOM', top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'WHICH ACCURACY:', which_test_accuracy, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', round(gain,2), 'ACCURACY:', round(accuracy,4), 'DATE:', date)

        df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ONE_BY_ONE'
        df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type
        df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
        df_result.loc[df_result['combination'] == top_combination, 'which_accuracy'] = which_test_accuracy
        df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
        df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), 'which_best'] = 'BEST'
        df_result.loc[df_result['combination'] == top_combination, 'which_date'] = date
        result = df_result.loc[df_result['combination'] == top_combination]

        max_combinations_results.append(result)
    print()

    ####################   ONE BY ONE TESTED ON ALL INTERVALS AT ONCE  ####################
    
    # Evaluate test model on best (according to one by one strategy) combination on all intervals dataset 
    file_name = nn_type + '_All_test_intervals.csv'
    df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
    gain = df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]
    accuracy = df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_accuracy].iloc[0]

    print('OBOT', top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'WHICH ACCURACY:', which_test_accuracy, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', round(gain,2), 'ACCURACY:', round(accuracy,4))

    df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ONE_BY_ONE_TESTED_ON_ALL_AT_ONCE'
    df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type
    df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
    df_result.loc[df_result['combination'] == top_combination, 'which_accuracy'] = which_test_accuracy
    df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
    df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), 'which_best'] = 'BEST'
    result = df_result.loc[df_result['combination'] == top_combination]

    max_combinations_results.append(result)


    ####################   ALL AT ONCE   ####################

    # Having the best combination, approach to gain (standard gain or average gain), treshold and share
    # Evaluate test model for all intervals at once strategy
    top_combination, thold_test, share_test, which_test_gain = best_all_at_once
    which_test_accuracy = 'accuracy' if which_test_gain == 'gain' else 'avg_accuracy'
    
    file_name = nn_type + '_All_test_intervals.csv'
    df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
    gain = df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]
    accuracy = df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_accuracy].iloc[0]

    print('AAOM', top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'WHICH ACCURACY:', which_test_accuracy, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', round(gain,2), 'ACCURACY:', round(accuracy,4))

    df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ALL_AT_ONCE'
    df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type
    df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
    df_result.loc[df_result['combination'] == top_combination, 'which_accuracy'] = which_test_accuracy
    df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
    df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), 'which_best'] = 'BEST'
    result = df_result.loc[df_result['combination'] == top_combination]

    max_combinations_results.append(result)
    print()

    ####################   ALL AT ONCE TESTED ON ONE BY ONE   ####################

    # Evaluate test model on best (according to all at once strategy) combination on one by one interval datasets
    gain = 0
    for date in train_starts:
        
        file_name = nn_type + '_All_test_' + date + '.csv'
        df_result = pd.read_csv(os.path.join(combs_res_dir_name, file_name))
        gain = df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_gain].iloc[0]
        accuracy = df_result.loc[((df_result['combination'] == top_combination) & (df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), which_test_accuracy].iloc[0]

        print('AAOT',top_combination, nn_type, 'WHICH GAIN:', which_test_gain, 'WHICH ACCURACY:', which_test_accuracy, 'TOP SHARE:', top_comb_share, 'THOLD:', thold_test, 'SHARE:', share_test, 'GAIN:', round(gain,2), 'ACCURACY:', round(accuracy,4), 'DATE:', date)

        df_result.loc[df_result['combination'] == top_combination, 'which_strategy'] = 'ALL_AT_ONCE_TESTED_ON_ONE_BY_ONE'
        df_result.loc[df_result['combination'] == top_combination, 'which_approach'] = nn_type
        df_result.loc[df_result['combination'] == top_combination, 'which_gain'] = which_test_gain
        df_result.loc[df_result['combination'] == top_combination, 'which_accuracy'] = which_test_accuracy
        df_result.loc[df_result['combination'] == top_combination, 'which_top_comb_share'] = top_comb_share
        df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test) & (df_result['share'] == share_test)), 'which_best'] = 'BEST'
        df_result.loc[df_result['combination'] == top_combination, 'which_date'] = date
        result = df_result.loc[df_result['combination'] == top_combination]

        max_combinations_results.append(result)
    
    print()