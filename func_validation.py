import pandas as pd
import os
import numpy as np
import datetime



def calc_metrics_for_combinations(df: pd.DataFrame, combinations_list: list, initializers_labels: list, train_starts: list, dataset_name: str, nn_type: str, dir_name: str,
                                  calc_one_by_one_or_all: str, save_specific_combination: bool = False, specific_combination_params: tuple = None) -> None:
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
    :param nn_type: str, name of the approach type i.e. 'MAIN' (main model) or 'SIDE' (side model)
    :param dir_name: str, directory where files will be saved
    :param calc_one_by_one_or_all: str, calculate one by one neural network (one interval) or group and then calculate (all intervals at the same time) (options: 'ONE_BY_ONE', 'ALL_AT_ONCE')
    :param save_specific_combination: bool, default: False, save predictions for given combination
    :param specific_combination_params: tuple(combination, thold, share, dir_name, dataset_name, nn_type, method, top_comb_share, gain)
    :return: None
    """

    base_col_name = 'Prediction_' + dataset_name + '_'
    df_base_columns = ['DateTime', 'Close', 'Change', 'Decision']

    comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options, comb_max_options = [], [], [], [], [], [], [], []

    if save_specific_combination:
        
        combination = specific_combination_params[0]
        for combination_el in combination:
            for train_start in train_starts:
                column_in_comb = base_col_name + combination_el + '_' + train_start
                df_base_columns.append(column_in_comb)

        df_combination = df.loc[:, df_base_columns].copy()
        df_base_columns = ['DateTime', 'Close', 'Change', 'Decision']
        df_combination.fillna(-9999, inplace=True)

        thold = specific_combination_params[1]
        calc_metrics_acc_thold(df_combination, combination, df_base_columns, initializers_labels, base_col_name, train_starts, thold, comb_accs_opt, comb_gains_opt,
                               comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options, save_specific_combination, specific_combination_params)

    else:
        
        if calc_one_by_one_or_all == 'ONE_BY_ONE': print('START:', nn_type, dataset_name, train_starts[0], datetime.datetime.now())
        elif calc_one_by_one_or_all == 'ALL_AT_ONCE': print('START:', nn_type, dataset_name, datetime.datetime.now())

        # For each combination of models
        for combination in combinations_list[1:]:

            # Create temporary DataFrame with combination columns (predictions) and base columns i.e. 'Change' and 'Decision'
            for combination_el in combination:
                for train_start in train_starts:
                    column_in_comb = base_col_name + combination_el + '_' + train_start
                    df_base_columns.append(column_in_comb)

            df_combination = df.loc[:, df_base_columns].copy()
            df_base_columns = ['DateTime', 'Close', 'Change', 'Decision']
            df_combination.fillna(-9999, inplace=True)

            # For each interval of uncertainty 
            for thold in [0, 0.005, 0.01, 0.025, 0.05]:
                calc_metrics_acc_thold(df_combination, combination, df_base_columns, initializers_labels, base_col_name, train_starts, thold, comb_accs_opt, comb_gains_opt,
                                       comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options, save_specific_combination, specific_combination_params)


            # Determine max values for combination, append to list comb_max_options
            df_temp_max = pd.DataFrame([tholds, shares, comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt]).T
            df_temp_max.columns = ['treshold', 'share', 'accuracy', 'gain', 'avg_accuracy', 'avg_gain']

            df_gain = df_temp_max[df_temp_max['gain'] == df_temp_max['gain'].max()]
            df_gain = df_gain[df_gain['share'] == min(df_gain['share'], key=lambda x:abs(x-df_gain['share'].median()))]
            df_gain = df_gain[df_gain['treshold'] == min(df_gain['treshold'], key=lambda x:abs(x-df_gain['treshold'].median()))] 
            df_gain_list = np.array(df_gain[['treshold', 'share', 'accuracy', 'gain']])[0] 

            df_acc = df_temp_max[df_temp_max['accuracy'] == df_temp_max['accuracy'].max()]
            df_acc = df_acc[df_acc['share'] == min(df_acc['share'], key=lambda x:abs(x-df_acc['share'].median()))]
            df_acc = df_acc[df_acc['treshold'] == min(df_acc['treshold'], key=lambda x:abs(x-df_acc['treshold'].median()))]
            df_acc_list = np.array(df_acc[['treshold', 'share', 'accuracy', 'gain']])[0] 

            df_avg_gain = df_temp_max[df_temp_max['avg_gain'] == df_temp_max['avg_gain'].max()]
            df_avg_gain = df_avg_gain[df_avg_gain['share'] == min(df_avg_gain['share'], key=lambda x:abs(x-df_avg_gain['share'].median()))]
            df_avg_gain = df_avg_gain[df_avg_gain['treshold'] == min(df_avg_gain['treshold'], key=lambda x:abs(x-df_avg_gain['treshold'].median()))]
            df_avg_gain_list = np.array(df_avg_gain[['treshold', 'share', 'avg_accuracy', 'avg_gain']])[0] 

            df_avg_acc = df_temp_max[df_temp_max['avg_accuracy'] == df_temp_max['avg_accuracy'].max()]
            df_avg_acc = df_avg_acc[df_avg_acc['share'] == min(df_avg_acc['share'], key=lambda x:abs(x-df_avg_acc['share'].median()))]
            df_avg_acc = df_avg_acc[df_avg_acc['treshold'] == min(df_avg_acc['treshold'], key=lambda x:abs(x-df_avg_acc['treshold'].median()))]
            df_avg_acc_list = np.array(df_avg_acc[['treshold', 'share', 'avg_accuracy', 'avg_gain']])[0] 

            max_option_row = [combination, *df_gain_list, *df_acc_list, *df_avg_gain_list, *df_avg_acc_list]

            comb_max_options.append(max_option_row)
            comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares = [], [], [], [], [], []

        if calc_one_by_one_or_all == 'ONE_BY_ONE':
            
            file_name_all = nn_type + '_All_' + dataset_name + '_' + train_starts[0] + '.csv'
            file_name_max = nn_type + '_Max_' + dataset_name + '_' + train_starts[0] + '.csv'

        elif calc_one_by_one_or_all == 'ALL_AT_ONCE':

            file_name_all = nn_type + '_All_' + dataset_name + '_intervals.csv'
            file_name_max = nn_type + '_Max_' + dataset_name + '_intervals.csv'

        # Crete and save DataFrames for all and max values for each combination
        df_comb_options = pd.DataFrame(comb_options, columns=['combination', 'treshold', 'share', 'accuracy', 'gain', 'avg_accuracy', 'avg_gain'])
        df_comb_max_options = pd.DataFrame(comb_max_options, columns=['combination', 'MG_treshold', 'MG_share', 'MG_accuracy', 'MG_gain', 'MA_treshold', 'MA_share', 'MA_accuracy', 'MA_gain',
                                                                                    'AVG_MG_treshold', 'AVG_MG_share', 'AVG_MG_accuracy', 'AVG_MG_gain', 'AVG_MA_treshold', 'AVG_MA_share', 'AVG_MA_accuracy', 'AVG_MA_gain'])
        
        df_comb_options.to_csv(os.path.join(dir_name, file_name_all), index=None)
        df_comb_max_options.to_csv(os.path.join(dir_name, file_name_max), index=None)

        print('END:', datetime.datetime.now())
    


def calc_metrics_acc_thold(df_combination: pd.DataFrame, combination: list, df_base_columns: list, initializers_labels: list, base_col_name: str, train_starts: list, thold: float, 
                           comb_accs_opt: list, comb_gains_opt: list, comb_avg_accs_opt: list, comb_avg_gains_opt: list, tholds: list, shares: list, comb_options: list,
                           save_specific_combination: bool = False, specific_combination_params: tuple = None) -> None:
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
    :param save_specific_combination: bool, default: False, save predictions for given combination
    :param specific_combination_params: tuple(combination, thold, share, dir_name, dataset_name, nn_type, method, top_comb_share, gain)
    :return: None
    """

    # Create temporary DataFrame
    # Calculate Average Prediction ('Prediction_avg) as an average of all predictive columns
    # For predictions between uncertainty interval (0.5 +- treshold ('thold')) assign np.nan
    # Round predictions according to math's rule (columns 'Predicition_avg')  
    df_comb_opt = df_combination.loc[:, df_combination.columns != 'DateTime'].copy()
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

    df_comb_smry = pd.DataFrame([df_combination['DateTime'], df_comb_opt['Close'], df_comb_opt['Change'], df_comb_opt['Decision'], decision_sum_cols, decision_count_cols, decision_count_nan_cols],
                                    index=['DateTime', 'Close', 'Change', 'Decision', 'Long Predictions', 'How many NNs', 'NaNs Predictions'])

    # Calculate share of LONG predictions in all decisions made ('Share Long Predictions')
    # Assign to df_comb_smry column with average predictions ('Prediction_avg')
    df_comb_smry = df_comb_smry.T

    if (df_comb_smry['Long Predictions'] > df_comb_smry['How many NNs']).sum() >0:
        print('ERROR:', specific_combination_params)

    df_comb_smry.loc[df_comb_smry['How many NNs'] == 0, 'Share Long Predictions'] = np.nan
    df_comb_smry.loc[df_comb_smry['How many NNs'] != 0, 'Share Long Predictions'] = \
                     df_comb_smry.loc[df_comb_smry['How many NNs'] != 0, 'Long Predictions'] / df_comb_smry.loc[df_comb_smry['How many NNs'] != 0, 'How many NNs']

    df_comb_smry['Share Long Predictions'] = df_comb_smry['Long Predictions'] / df_comb_smry['How many NNs']
    df_comb_smry['Average Prediction'] = df_comb_opt['Prediction_avg']

    if save_specific_combination:

        share = specific_combination_params[2]
        calc_metrics_acc_thold_and_share(df_comb_smry, combination, thold, round(share,3), comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt,
                                         tholds, shares, comb_options, train_starts, save_specific_combination, specific_combination_params)

    else:

        # For each share of positive (1) predictions (LONG)
        for share in np.linspace(0, 1, len(initializers_labels)+1)[1:]:
            calc_metrics_acc_thold_and_share(df_comb_smry, combination, thold, round(share,3), comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt,
                                             tholds, shares, comb_options, train_starts, save_specific_combination, specific_combination_params)



def calc_metrics_acc_thold_and_share(df_summary: pd.DataFrame, combination: list, thold: int, share: float,
                                     comb_accs_opt: list, comb_gains_opt: list, comb_avg_accs_opt: list, comb_avg_gains_opt: list, tholds: list, shares: list, comb_options: list,
                                     train_starts: list, save_specific_combination: bool = False, specific_combination_params: tuple = None) -> None:
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
    :param train_starts: list, list of start date in format 'YYYY-MM-DD' (for saving)
    :param save_specific_combination: bool, default: False, save predictions for given combination
    :param specific_combination_params: tuple(combination, thold, share, dir_name, dataset_name, nn_type, method, top_comb_share, gain)
    :return: None
    """

    # Calculate final prediction ('Final Prediction')
    # If number of Nans columns (columns where decision was not determined in func calc_metrics_acc_thold) is greater or equals 0.5, assign np.nan
    # Else determine final prediction based on share value
    df_smry = df_summary.copy()
    df_smry['Final Prediction'] = 0
    df_smry.loc[df_smry['Share Long Predictions'] >= share, 'Final Prediction'] = 1
    df_smry.loc[(df_smry['NaNs Predictions']/(df_smry['NaNs Predictions']+df_smry['How many NNs'])) >= 0.5, 'Final Prediction'] = np.nan
    
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

    if save_specific_combination:

        df_smry['Final Prediction Change'] = (-abs(df_smry_wo_nans['Decision']-df_smry_wo_nans['Final Prediction']) + (df_smry_wo_nans['Decision'] == df_smry_wo_nans['Final Prediction'])) * abs(df_smry_wo_nans['Change'])
        df_smry.loc[df_smry['Final Prediction Change'].isnull(), 'Final Prediction Change'] = df_smry_w_nans['Change']
        df_smry['Average Prediction Change'] = (-abs(df_smry_wo_nans_avg['Decision']-df_smry_wo_nans_avg['Average Prediction']) + (df_smry_wo_nans_avg['Decision'] == df_smry_wo_nans_avg['Average Prediction'])) * abs(df_smry_wo_nans_avg['Change'])
        df_smry.loc[df_smry['Average Prediction Change'].isnull(), 'Average Prediction Change'] = df_smry_w_nans_avg['Change']
        
        if specific_combination_params[6] in ['OBOM', 'AAOT']:
            file_name = specific_combination_params[5] + '_Predictions_for_' + specific_combination_params[6] + '_' + str(specific_combination_params[7]) + '_' + specific_combination_params[8] + '_' + train_starts[0] + '.xlsx'
        else:
            file_name = specific_combination_params[5] + '_Predictions_for_' + specific_combination_params[6] + '_' + str(specific_combination_params[7]) + '_' + specific_combination_params[8] + '.xlsx'

        df_smry.to_excel(os.path.join(specific_combination_params[3], file_name), index=None)


    else:

        # Append results to lists
        comb_accs_opt.append(dec_accuracy); comb_gains_opt.append(dec_gain); comb_avg_accs_opt.append(dec_accuracy_by_avg); comb_avg_gains_opt.append(dec_gain_by_avg); tholds.append(thold); shares.append(share)

        option_row = [combination, thold, share, dec_accuracy, dec_gain, dec_accuracy_by_avg, dec_gain_by_avg]
        comb_options.append(option_row)


