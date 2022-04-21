import os
import pandas as pd
from itertools import combinations
import numpy as np
import datetime


def join_external_dfs_from_one_interval(path, initializers_labels, train_start, dataset_name):

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


def calc_metrics_for_combinations(df, train_start, nn_no, dataset_name, nn_type, dir_name):

    
    base_col_name = 'Prediction_' + dataset_name + '_' + str(nn_no) if nn_type == 'MAIN APPROACH' else  'Prediction_' + dataset_name + '_'
    df_base_columns = ['Change', 'Decision']

    comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares, comb_options = [], [], [], [], [], [], []
    comb_max_accs, comb_max_gains, comb_max_options = [], [], []

    print('START:', train_start, datetime.datetime.now())
    for comb_no, combination in enumerate(combinations_list[1:]):

        for combination_el in combination:
            column_in_comb = base_col_name + combination_el + '_' + train_start
            df_base_columns.append(column_in_comb)

        df_combination = df.loc[:, df_base_columns].copy()
        df_base_columns = ['Change', 'Decision']

        for thold in [0, 0.25, 0.5, 1, 2.5, 5]:

            df_comb_opt = df_combination.copy()
            pred_col_names = list(set(df_comb_opt.columns.to_list()).difference(set(df_base_columns)))
            df_comb_opt['Prediction_avg'] = np.nanmean(df_comb_opt[pred_col_names], axis=1)
            df_comb_opt.loc[(df_comb_opt['Prediction_avg'] >= (50-thold)/100) & (df_comb_opt['Prediction_avg'] <= (50+thold)/100), 'Prediction_avg'] = np.nan
            df_comb_opt['Prediction_avg']  = round(df_comb_opt['Prediction_avg'])

            for combination_el in combination:

                col_el_opt = base_col_name + combination_el + '_' + train_start

                df_comb_opt.loc[(df_comb_opt[col_el_opt] > (50+thold)/100), col_el_opt] = 1
                df_comb_opt.loc[(df_comb_opt[col_el_opt] >= (50-thold)/100) & (df_comb_opt[col_el_opt] <= (50+thold)/100), col_el_opt] = np.nan
                df_comb_opt.loc[(df_comb_opt[col_el_opt] < (50-thold)/100), col_el_opt] = 0

            decision_sum_cols = pd.Series(np.nansum(df_comb_opt.loc[:, df_comb_opt.columns.str.startswith(base_col_name)], axis=1))
            decision_count_cols = pd.Series(np.nansum(df_comb_opt.loc[:, ~df_comb_opt.columns.isin(df_base_columns)], axis=1))
            decision_count_nan_cols = df_comb_opt.loc[:, df_comb_opt.columns.str.startswith(base_col_name)].isnull().sum(axis=1)
            df_comb_smry = pd.DataFrame([df_comb_opt['Change'], df_comb_opt['Decision'], decision_sum_cols, decision_count_cols, decision_count_nan_cols],
                                            index=['Change', 'Decision', 'Long Predictions', 'How many NN', 'NaNs Predictions'])

            df_comb_smry = df_comb_smry.T
            df_comb_smry['Share Long Predictions'] = df_comb_smry['Long Predictions'] / df_comb_smry['How many NN']
            df_comb_smry['Average Prediction'] = df_comb_opt['Prediction_avg']


            for share in np.linspace(0, 1, len(initializers_labels)+1)[5:]:

                df_comb_smry['Final Prediction'] = 0
                df_comb_smry.loc[(df_comb_smry['NaNs Predictions']/(df_comb_smry['NaNs Predictions']+df_comb_smry['How many NN'])) >= 0.5, 'Final Prediction'] = np.nan
                df_comb_smry.loc[df_comb_smry['Share Long Predictions'] >= share/100, 'Final Prediction'] = 1
                
                df_comb_smry_wo_nans = df_comb_smry.loc[df_comb_smry['Final Prediction'].notnull()].copy()
                df_comb_smry_w_nans = df_comb_smry.loc[df_comb_smry['Final Prediction'].isnull()].copy()

                decision_accuracy = np.nan if len(df_comb_smry_wo_nans) == 0 else (df_comb_smry_wo_nans['Decision'] == df_comb_smry_wo_nans['Final Prediction']).sum()/len(df_comb_smry_wo_nans['Decision'])
                decision_gain = ((-abs(df_comb_smry_wo_nans['Decision']-df_comb_smry_wo_nans['Final Prediction']) + (df_comb_smry_wo_nans['Decision'] == df_comb_smry_wo_nans['Final Prediction'])) * abs(df_comb_smry_wo_nans['Change'])).sum()
                decision_gain += df_comb_smry_w_nans['Change'].sum()

                df_comb_smry_wo_nans_avg = df_comb_smry.loc[(df_comb_smry['Final Prediction'].notnull()) & (df_comb_smry['Average Prediction'].notnull())].copy()
                df_comb_smry_w_nans_avg = df_comb_smry.loc[(df_comb_smry['Final Prediction'].isnull()) | (df_comb_smry['Average Prediction'].isnull())].copy()

                decision_accuracy_by_avg =  np.nan if len(df_comb_smry_wo_nans_avg) == 0 else (df_comb_smry_wo_nans_avg['Decision'] == df_comb_smry_wo_nans_avg['Average Prediction']).sum()/len(df_comb_smry_wo_nans_avg['Decision'])
                decision_gain_by_avg = ((-abs(df_comb_smry_wo_nans_avg['Decision']-df_comb_smry_wo_nans_avg['Average Prediction']) + (df_comb_smry_wo_nans_avg['Decision'] == df_comb_smry_wo_nans_avg['Average Prediction'])) * abs(df_comb_smry_wo_nans_avg['Change'])).sum()
                decision_gain_by_avg += df_comb_smry_w_nans_avg['Change'].sum()


                comb_accs_opt.append(decision_accuracy)
                comb_gains_opt.append(decision_gain)
                comb_avg_accs_opt.append(decision_accuracy_by_avg)
                comb_avg_gains_opt.append(decision_gain_by_avg)
                tholds.append(thold)
                shares.append(share)

                option_row = [combination, thold, share, decision_accuracy, decision_gain, decision_accuracy_by_avg, decision_gain_by_avg]
                comb_options.append(option_row)


        max_acc = max(comb_accs_opt)
        max_gain = max(comb_gains_opt)

        max_avg_acc = max(comb_avg_accs_opt)
        max_avg_gain = max(comb_avg_gains_opt)

        comb_max_accs.append(max_acc)
        comb_max_gains.append(max_gain)


        # print(comb_no, combination, '(GAIN) Thold:', tholds[comb_gains_opt.index(max_gain)], '(GAIN) Share:', round(shares[comb_gains_opt.index(max_gain)],2),
        #                             'ACC:', round(comb_accs_opt[comb_gains_opt.index(max_gain)],2), 'Max GAIN:', round(max_gain,2), '\n', 
        #                             '(ACC) Thold:', tholds[comb_accs_opt.index(max_acc)], '(ACC) Share:', round(shares[comb_accs_opt.index(max_acc)],2), 
        #                             'Max ACC:', round(max_acc,2), 'GAIN:', round(comb_gains_opt[comb_accs_opt.index(max_acc)],2), '\n',
        #                             '(AVG GAIN) Thold:', tholds[comb_avg_gains_opt.index(max_avg_gain)], '(AVG GAIN) Share:', round(shares[comb_avg_gains_opt.index(max_avg_gain)],2), 
        #                             'AVG ACC:', round(comb_avg_accs_opt[comb_avg_gains_opt.index(max_avg_gain)],2), 'AVG Max GAIN:', round(max_avg_gain,2), '\n', 
        #                             '(AVG ACC) Thold:', tholds[comb_avg_accs_opt.index(max_avg_acc)], '(AVG ACC) Share:', round(shares[comb_avg_accs_opt.index(max_avg_acc)],2), 
        #                             'AVG Max ACC:', round(max_avg_acc,2), 'AVG GAIN:', round(comb_avg_gains_opt[comb_avg_accs_opt.index(max_avg_acc)],2), '\n')

        
        max_option_row = [combination, tholds[comb_gains_opt.index(max_gain)], shares[comb_gains_opt.index(max_gain)], comb_accs_opt[comb_gains_opt.index(max_gain)], max_gain,
                                       tholds[comb_accs_opt.index(max_acc)], shares[comb_accs_opt.index(max_acc)], max_acc, comb_gains_opt[comb_accs_opt.index(max_acc)],
                                       tholds[comb_avg_gains_opt.index(max_avg_gain)], shares[comb_avg_gains_opt.index(max_avg_gain)], comb_avg_accs_opt[comb_avg_gains_opt.index(max_avg_gain)], max_avg_gain,
                                       tholds[comb_avg_accs_opt.index(max_avg_acc)], shares[comb_avg_accs_opt.index(max_avg_acc)], max_avg_acc, comb_avg_gains_opt[comb_avg_accs_opt.index(max_avg_acc)]]
        
        
        comb_max_options.append(max_option_row)
        
        comb_accs_opt, comb_gains_opt, comb_avg_accs_opt, comb_avg_gains_opt, tholds, shares = [], [], [], [], [], []

    file_name_all = nn_type.split(' ')[0] + '_All_' + dataset_name + '_' + train_start + '.csv'
    file_name_max = nn_type.split(' ')[0] + '_Max_' + dataset_name + '_' + train_start + '.csv'

    df_comb_options = pd.DataFrame(comb_options, columns=['combination', 'treshold', 'share', 'accuracy', 'gain', 'avg_accuracy', 'avg_gain'])
    df_comb_max_options = pd.DataFrame(comb_max_options, columns=['combination', 'MG_treshold', 'MG_share', 'MG_accuracy', 'MG_gain',
                                                                                 'MA_treshold', 'MA_share', 'MA_accuracy', 'MA_gain',
                                                                                 'AVG_MG_treshold', 'AVG_MG_share', 'AVG_MG_accuracy', 'AVG_MG_gain',
                                                                                 'AVG_MA_treshold', 'AVG_MA_share', 'AVG_MA_accuracy', 'AVG_MA_gain'])
    
    df_comb_options.to_csv(os.path.join(dir_name, file_name_all), index=None)
    df_comb_max_options.to_csv(os.path.join(dir_name, file_name_max), index=None)

    print('END:', datetime.datetime.now())
    


if __name__ == '__main__':

    initializers_labels = ['GN', 'GU', 'HN', 'HU', 'LU', 'OR', 'RN', 'RU', 'TN', 'VS']
    train_starts = ['2001-04-02', '2001-07-02', '2001-10-01', '2002-01-02', '2002-04-01', '2002-07-01', '2002-10-01', '2003-01-02', '2003-04-01']
    combinations_list = sum([list(map(list, combinations(initializers_labels, i))) for i in range(len(initializers_labels) + 1)], [])
    combinations_results_dir_name = r'17042022_2_MAX_PREDICTIONS2/'
    val_result_maxs = []

    if not os.path.isdir(combinations_results_dir_name):
        os.mkdir(combinations_results_dir_name)

    #for nn_path_type in list(zip(['PREDICTIONS', 'PREDICTIONS_SIDE'], ['MAIN APPROACH', 'SIDE APPROACH'])):
    for nn_path_type in list(zip(['17042022_2'], ['MAIN APPROACH'])):
        for dataset_name in ['validation', 'test']:
            for no, date in enumerate(train_starts):


                if nn_path_type[1] == 'MAIN APPROACH':

                    file_name = 'Prediction_' + dataset_name + '_VS_' + date + '.csv'
                    df_10nn = pd.read_csv(os.path.join(nn_path_type[0], file_name))
                
                else:
                    df_10nn = join_external_dfs_from_one_interval(nn_path_type[0], initializers_labels, date, dataset_name)
                
                calc_metrics_for_combinations(df_10nn, date, no, dataset_name, nn_path_type[1], combinations_results_dir_name)


# START

    
    # for nn_type in ['MAIN APPROACH', 'SIDE APPROACH']:
        
    #     val_result_maxs = []
    #     val_max_score, val_avg_max_score = 0, 0

    #     for no, date in enumerate(train_starts):
        
    #         file_name = nn_type.split(' ')[0] + '_Max_validation_' + date + '.csv'
    #         df_result = pd.read_csv(os.path.join(combinations_results_dir_name, file_name))
            
    #         val_max = df_result['MG_gain'].mean()#.nlargest(250).mean()
    #         val_avg_max = df_result['AVG_MG_gain'].mean()#.nlargest(250).mean()
            
    #         if val_max > val_avg_max: val_max_score += 1 
    #         else: val_avg_max_score += 1

    #         val_result_max = df_result.loc[df_result['MG_gain'].nlargest(100).index]['combination'] if val_max > val_avg_max \
    #                     else df_result.loc[df_result['AVG_MG_gain'].nlargest(100).index]['combination']

    #         val_result_maxs.extend(val_result_max)


    #     df_top_combinations = pd.Series(val_result_maxs).value_counts()
    #     top_combinations = df_top_combinations.loc[df_top_combinations.values == max(df_top_combinations)].index.to_list()
    #     top_combination_len = len(max(top_combinations, key=len))
    #     top_combinations_names = [comb for comb in top_combinations if len(comb) == top_combination_len]

    #     which_gain = 'MG_gain' if val_max_score > val_avg_max_score else 'AVG_MG_gain'
    #     top_combination = df_result.loc[df_result['combination'].isin(top_combinations_names)].sort_values(by=which_gain, ascending=False)['combination'].iloc[0]

    #     tholds = {}
    #     tholds[0.0], tholds[0.25], tholds[0.5], tholds[1.0], tholds[2.5], tholds[5.0] = 0, 0, 0, 0, 0, 0
    #     for no, date in enumerate(train_starts):
        
    #         file_name = nn_type.split(' ')[0] + '_Max_validation_' + date + '.csv'
    #         df_result = pd.read_csv(os.path.join(combinations_results_dir_name, file_name))
    #         thold = int(df_result.loc[df_result['combination'] == top_combination, which_gain[:-4] + 'treshold'])

    #         if thold == 0: tholds[0.0] += 1
    #         elif thold == 0.25: tholds[0.25] += 1
    #         elif thold == 0.5: tholds[0.5] += 1
    #         elif thold == 1.0: tholds[1.0] += 1
    #         elif thold == 2.5: tholds[2.5] += 1
    #         elif thold == 5.0: tholds[5.0] += 1

    #     thold_test = max(tholds, key=tholds.get)

    #     which_test_gain = 'gain' if which_gain == 'MG_gain' else 'avg_gain'
    #     gain = 0
    #     for no, date in enumerate(train_starts):
            
    #         file_name = nn_type.split(' ')[0] + '_All_test_' + date + '.csv'
    #         df_result = pd.read_csv(os.path.join(combinations_results_dir_name, file_name))

    #         #print(df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test)), which_test_gain])
    #         gain += df_result.loc[(df_result['combination'] == top_combination) & ((df_result['treshold'] == thold_test)), which_test_gain].iloc[0]
        
    #     print(nn_type, gain)
        


        


