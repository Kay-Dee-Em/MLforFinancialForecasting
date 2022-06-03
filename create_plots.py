import ImageGenerator as IG
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import re

""""
Auxiliary script to create plots. Default plots are plotted for OIH dataset.

- create visualization of preprocessing the data (linear plot, polar plot) for both strategies for chosen days i.e.:
    LONG - 2021-12-31 (default for OIH)
    SHORT - 2021-11-09 (default for OIH)
- create visualization of cross-validation (CV) intervals for n neural networks
- create visualization of cross-validation (CV) intervals for n neural networks and show only test intervals
- create histogram of daily closing prices changes
- create heatmap for each overall (all interavals - AAO) the best (according to type) combination
- create heatmap for each (one by one - OBO) the best (accoring to type) combination
- create visualization of daily changes according to choosen combination
- create visualization of daily strategies according to choosen combination

"""

def strategy_visualization(df: pd.DataFrame, date_long_start: str = '2021-12-03', date_long_end_next_day: str = '2022-01-01',
                                             date_long_pred_start: str = '2021-12-31', date_long_pred_end_next_day: str = '2022-01-04',
                                             date_short_start: str = '2021-10-13', date_short_end_next_day: str = '2021-11-10',
                                             date_short_pred_start: str = '2021-11-09', date_short_pred_end_next_day: str = '2021-11-11', plots_dir_name: str = 'PLOTS') -> None:

    df_long = df[(df['DateTime'].dt.date.astype(str) >= date_long_start) & (df['DateTime'].dt.date.astype(str) < date_long_end_next_day)]
    df_long_next_day = df[(df['DateTime'].dt.date.astype(str) >= date_long_pred_start) & (df['DateTime'].dt.date.astype(str) < date_long_pred_end_next_day)][-9:] #9 - number of last observations (1 + 8)
    df_short = df[(df['DateTime'].dt.date.astype(str) >= date_short_start) & (df['DateTime'].dt.date.astype(str) < date_short_end_next_day)]
    df_short_next_day = df[(df['DateTime'].dt.date.astype(str) >= date_short_pred_start) & (df['DateTime'].dt.date.astype(str) < date_short_pred_end_next_day)][-9:] #9 - number of last observations (1 + 8)

    dfs_strategy = [df_long, df_short]
    dfs_strategy_next_day = [df_long_next_day, df_short_next_day]

    # assuming that default frequencies were used
    offset = '9h'
    freqs = ['1h', '2h', '4h', '1d']
    freqs_break = [1, 2, 4, 8]

    for strategy in dfs_strategy:
        strategy.reset_index(drop=True, inplace=True)
        for i, freq in enumerate(freqs):
            df_agg = strategy.groupby(pd.Grouper(key='DateTime', freq=freq, offset=offset)).mean().dropna().reset_index(drop=True)['Close'].tolist()
            nans = np.full(len(strategy), np.nan)
            nans[::freqs_break[i]] = df_agg
            strategy.loc[:, freq] = nans

    
    fig_names = ['LONG_' + date_long_pred_start.replace('-', '_') + '_line', 'SHORT_' + date_short_pred_start.replace('-', '_') + '_line',
                 'LONG_' + date_long_pred_start.replace('-', '_') + '_polar', 'SHORT_' + date_short_pred_start.replace('-', '_') + '_polar']


    ####################   LINEAR PLOTS   ####################
    for number, strategy in enumerate(dfs_strategy):
        plt.figure(figsize=(10,10))

        for i in [(1, '1h','blue'), (2, '2h','orange'), (4, '4h','green'), (8, '1d','red')]:

            x = list(range(0, len(strategy), i[0]))
            y = strategy.loc[strategy[i[1]].notnull(), i[1]]
            x_20 = x[-20:]
            y_20 = y[-20:]

            plt.plot(x, y, color=i[2], label=i[1], linestyle='--')
            plt.plot(x_20, y_20, color=i[2], label=i[1] + ' - 20 ostatnich obs.', marker='*', markersize=16, linewidth=3, alpha=0.75)

        x = list(range(len(strategy)-1, len(strategy)+8, 1))
        y = dfs_strategy_next_day[number]['Close']
        plt.plot(x, y, color='black', label= fig_names[number].split('_')[0] + ' prediction', linestyle='-.')

        x_ticks = strategy['DateTime'].tolist() + dfs_strategy_next_day[number]['DateTime'].tolist()[1:]
        x_ticks = pd.Series(x_ticks).astype(str).apply(lambda x: x[:10]).tolist()
        x_ticks = x_ticks[::32]

        plt.legend(fontsize=11)
        plt.xticks(ticks = list(range(0, max(x)+1, 32)), labels = x_ticks)
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir_name, fig_names[number]), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()


    ####################   POLAR PLOTS   ####################
    for number, strategy in enumerate(dfs_strategy):
        fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})

        for i in [(1, '1h','blue'), (2, '2h','orange'), (4, '4h','green'), (8, '1d','red')]:

            r = list(range(0, len(strategy), i[0]))[-20:]
            y_20 = strategy.loc[strategy[i[1]].notnull(), i[1]][-20:]

            y_norm = ((y_20 - max(y_20)) + (y_20 - min(y_20)))/(max(y_20) - min(y_20))
            theta = np.arccos(y_norm)
            ax.plot(theta, r, color=i[2], label=i[1] + ' - 20 ostatnich obs.', marker='*', markersize=16, linewidth=3, alpha=0.75)

        ax.set_rlabel_position(-22.5)
        fig.legend(fontsize=16, loc='lower left')
        fig.savefig(os.path.join(plots_dir_name, fig_names[number+2]), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()



def cross_validation_visualization(df: pd.DataFrame, train_start: str = '2001-04-02', len_train: int = 13*12, len_valdidation: int = 3.75*12, len_test: int = 2*12,
                                   interval_in_month: int = 3, NN_number: int = 9, plots_dir_name: str = 'PLOTS') -> None:

    plt.rc('axes', axisbelow=True)
    fig, axs = plt.subplots(2, 1, figsize=(9,6), gridspec_kw={'height_ratios': [3, 1]})
    x = df['DateTime']
    y = df['Close']
    axs[0].plot(x, y, color='#BFDCF7')
    axs[0].fill_between(x, y, color='#BFDCF7')
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].margins(0)
    axs[0].grid(True)

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].yaxis.set_ticks_position('left')
    axs[0].xaxis.set_ticks_position('bottom')

    train_start = pd.to_datetime(dt.date.fromisoformat(train_start))
    start_value = 5
    for i in range(NN_number):

        train_end = pd.to_datetime(train_start + relativedelta(months=len_train)) 
        validation_end = pd.to_datetime(train_end + relativedelta(months=len_valdidation)) 
        test_end = pd.to_datetime(validation_end + relativedelta(months=len_test)) 

        df_train_chunk = df.loc[df['DateTime'].between(train_start, train_end, inclusive='left')].reset_index(drop=True)
        df_validation_chunk = df.loc[df['DateTime'].between(train_end, validation_end, inclusive='left')].reset_index(drop=True)
        df_test_chunk = df.loc[df['DateTime'].between(validation_end, test_end, inclusive='left')].reset_index(drop=True)

        x_train = df_train_chunk['DateTime']
        y_train = [start_value] * len(x_train)
        x_validation = df_validation_chunk['DateTime']
        y_validation = [start_value] * len(x_validation)
        x_test = df_test_chunk['DateTime']
        y_test = [start_value] * len(x_test)

        if i == NN_number-1:
            axs[1].plot(x_train, y_train, color='red', linewidth=4, label='zbiór treningowy')
            axs[1].plot(x_validation, y_validation, color='orange', linewidth=4, label='zbiór walidacyjny')
            axs[1].plot(x_test, y_test, color='green', linewidth=4, label='zbiór testowy')
        
        else:
            axs[1].plot(x_train, y_train, color='red', linewidth=4)
            axs[1].plot(x_validation, y_validation, color='orange', linewidth=4)
            axs[1].plot(x_test, y_test, color='green', linewidth=4)

        train_start += relativedelta(months=interval_in_month)
        start_value += 5

    axs[1].margins(0)
    axs[1].set_ylim([0, start_value])
    axs[1].axis('off')
    handles, labels = axs[1].get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc='upper right', fontsize=14, bbox_to_anchor=(0.96, 0.97)) 
    fig.savefig(os.path.join(plots_dir_name, 'cross_validation.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()



def cross_validation_test_visualization(df: pd.DataFrame, train_start: str = '2001-04-02', test_start: str = '2018-01-01', test_end_next_day: str = '2022-01-01',
                                        len_train: int = 13*12, len_valdidation: int = 3.75*12, len_test: int = 2*12,
                                        interval_in_month: int = 3, NN_number: int = 9, plots_dir_name: str = 'PLOTS') -> None:

    plt.rc('axes', axisbelow=True)
    fig, axs = plt.subplots(2, 1, figsize=(10,7), gridspec_kw={'height_ratios': [3, 1]})

    x = df.loc[df['DateTime'].between(test_start, test_end_next_day, inclusive='left')]['DateTime']
    y = df.loc[df['DateTime'].between(test_start, test_end_next_day, inclusive='left')]['Close']
    axs[0].plot(x, y, color='#BFDCF7')
    axs[0].fill_between(x, y, color='#BFDCF7')
    axs[0].tick_params(axis='x', labelsize=12)
    axs[0].tick_params(axis='y', labelsize=12)
    axs[0].margins(0)
    axs[0].grid(True)

    train_start = pd.to_datetime(dt.date.fromisoformat(train_start))

    start_value = 5
    for i in range(NN_number):

        train_end = pd.to_datetime(train_start + relativedelta(months=len_train)) 
        validation_end = pd.to_datetime(train_end + relativedelta(months=len_valdidation))
        test_end_next_day = pd.to_datetime(validation_end + relativedelta(months=len_test))

        df_test_chunk = df.loc[df['DateTime'].between(validation_end, test_end_next_day, inclusive='left')].reset_index(drop=True)

        x_test = df_test_chunk['DateTime']
        y_test = [start_value] * len(x_test)
        y_vals = df_test_chunk['Close']

        if i == NN_number-1:
            axs[1].plot(x_test, y_test, color='green', linewidth=4, label='zbiór testowy')
            axs[0].vlines(x=x_test[0], ymin=10, ymax=y_vals[0]-10, colors='#009699', ls=':', lw=2, label='start')
            axs[0].vlines(x=x_test[len(x_test)-1], ymin=10, ymax=y_vals[len(y_vals)-1]-10, colors='#06474D', ls='-.', lw=2, label='end')
        
        else:
            axs[1].plot(x_test, y_test, color='green', linewidth=4)
            axs[0].vlines(x=x_test[0], ymin=10, ymax=y_vals[0]-10, colors='#009699', ls=':', lw=2)
            axs[0].vlines(x=x_test[len(x_test)-1], ymin=10, ymax=y_vals[len(y_vals)-1]-10, colors='#06474D', ls='-.', lw=2)

        train_start += relativedelta(months=interval_in_month)
        start_value += 5

    axs[1].margins(0)
    axs[1].set_ylim([0, start_value])
    axs[1].axis('off')
    handles_labels = [axs.get_legend_handles_labels() for axs in fig.axes]
    handles, labels = [sum(label, []) for label in zip(*handles_labels)]
    fig.legend(handles, labels, loc='upper right', fontsize=14, bbox_to_anchor=(0.90, 0.8825)) 
    fig.savefig(os.path.join(plots_dir_name, 'cross_validation_test.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()



def test_data_changes_histogram(df: pd.DataFrame, test_start: str = '2018-01-01', test_end_next_day: str = '2022-01-01', plots_dir_name: str = 'PLOTS') -> None:

    plt.figure(figsize=(10,7))

    df_x = df.loc[df['DateTime'].between(test_start, test_end_next_day, inclusive='left')]
    plt.hist(df_x.loc[df_x['Change'] > 0]['Change'], color='#7FC866', edgecolor='black', linewidth=1.2, bins=10) 
    plt.hist(df_x.loc[df_x['Change'] < 0]['Change'], color='#EE613F', edgecolor='black', linewidth=1.2, bins=10) 
    
    plt.axvline(df_x['Change'].mean(), color='black', linestyle='solid', linewidth=5)
    min_ylim, max_ylim = plt.ylim()
    plt.text(df_x['Change'].mean()*-5, max_ylim*0.85, 'Średnia: {:.2f}'.format(df_x['Change'].mean()), fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    sns.despine()

    plt.savefig(os.path.join(plots_dir_name, 'hist_test_data.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()



def predictions_heatmap_overall(df: pd.DataFrame, hists_dir_name: str, approach_type: str = 'MAIN', additional_file_name_prefix: str = '') -> None:

    min = df.describe().loc['min',['gain', 'avg_gain']].min()
    max = df.describe().loc['max',['gain', 'avg_gain']].max()

    xticks = ['0%', '0.5%', '1%', '2.5%', '5%']
    yticks = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', 'AVG']

    rows_dict = {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 'AVG': 10}
    cols_dict = {0.000: 0, 0.005: 1, 0.010: 2, 0.025: 3, 0.050: 4}

    norm = plt.Normalize(min,max)

    no = 0
    for strategy in ['ONE_BY_ONE_TESTED_ON_ALL_AT_ONCE', 'ALL_AT_ONCE']:
        for top_share in [0.1, 0.05, 0.01]:

            df_cut = df.loc[(df['which_approach'] == approach_type) & (df['which_strategy'] == strategy) & (df['which_top_comb_share'] == top_share)].copy()
            combination = re.sub('[^A-Z ]+', '', df_cut['combination'].unique()[0])

            if strategy == 'ALL_AT_ONCE' and top_share in [0.05, 0.01]:
                continue
            
            df_avg = df_cut.loc[df_cut['share'] == 0.1].copy()
            df_avg['share'] = 'AVG'
            df_avg['gain'] = df_avg['avg_gain']
            df_heatmap = pd.concat([df_cut, df_avg])

            heatmap = df_heatmap.pivot(index='share', columns='treshold', values='gain')
            h_row = float(df_heatmap.loc[df_heatmap['which_best'] == 'BEST']['share'])
            h_col = float(df_heatmap.loc[df_heatmap['which_best'] == 'BEST']['treshold'])

            h_row = rows_dict[h_row]
            h_col = cols_dict[h_col]
            
            ax = sns.heatmap(heatmap, cmap='RdYlGn', linewidths=1, square=True, cbar_kws={'shrink': .9}, annot=True, fmt='.0f', annot_kws={'fontsize':8}, norm=norm)
            ax.add_patch(Rectangle((h_col, h_row), 1, 1, fill=False, edgecolor='#8936BB', lw=2))

            plt.xticks(plt.xticks()[0], labels=xticks, rotation=0, fontsize=9)
            plt.xlabel('Przedział niepewności', fontsize=9)
            plt.ylabel('Udział LONG', fontsize=9)
            plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=9)
            plt.title(combination, fontsize=9)

            top_share = str(top_share).replace('.', '_')
            file_name = top_share + '_' + combination + '.png'
            plt.savefig(os.path.join(hists_dir_name, additional_file_name_prefix + file_name), bbox_inches='tight', pad_inches=0, transparent=True, dpi=200)
            plt.close()
            no += 1



def predictions_heatmap_by_interval(df: pd.DataFrame, hists_dir_name: str, train_starts: list, approach_type: str = 'MAIN', additional_file_name_prefix: str = '') -> None:

    min = df.describe().loc['min',['gain', 'avg_gain']].min()
    max = df.describe().loc['max',['gain', 'avg_gain']].max()

    xticks = ['0%', '0.5%', '1%', '2.5%', '5%']
    yticks = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', 'AVG']

    rows_dict = {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 'AVG': 10}
    cols_dict = {0.000: 0, 0.005: 1, 0.010: 2, 0.025: 3, 0.050: 4}

    norm = plt.Normalize(min,max)

    no = 0
    for date in train_starts:
        for strategy in ['ONE_BY_ONE', 'ALL_AT_ONCE_TESTED_ON_ONE_BY_ONE']:
            for top_share in [0.1, 0.05, 0.01]:

                df_cut = df.loc[(df['which_approach'] == approach_type) & (df['which_strategy'] == strategy) & (df['which_top_comb_share'] == top_share) & (df['which_date'] == date)].copy()
                combination = re.sub('[^A-Z ]+', '', df_cut['combination'].unique()[0])

                if strategy == 'ALL_AT_ONCE_TESTED_ON_ONE_BY_ONE' and top_share in [0.05, 0.01]:
                    continue
                
                df_avg = df_cut.loc[df_cut['share'] == 0.1].copy()
                df_avg['share'] = 'AVG'
                df_avg['gain'] = df_avg['avg_gain']
                df_heatmap = pd.concat([df_cut, df_avg])

                heatmap = df_heatmap.pivot(index='share', columns='treshold', values='gain')
                h_row = float(df_heatmap.loc[df_heatmap['which_best'] == 'BEST']['share'])
                h_col = float(df_heatmap.loc[df_heatmap['which_best'] == 'BEST']['treshold'])

                h_row = rows_dict[h_row]
                h_col = cols_dict[h_col]

                ax = sns.heatmap(heatmap, cmap='RdYlGn', linewidths=1, square=True, cbar_kws={'shrink': .9}, annot=True, fmt='.0f', annot_kws={'fontsize':8}, norm=norm)
                ax.add_patch(Rectangle((h_col, h_row), 1, 1, fill=False, edgecolor='#8936BB', lw=2))

                plt.xticks(plt.xticks()[0], labels=xticks, rotation=0, fontsize=9)
                plt.xlabel('Przedział niepewności', fontsize=9)
                plt.ylabel('Udział LONG', fontsize=9)
                plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=9)

                title = date + '\n' + combination
                plt.title(title, fontsize=9)

                top_share = str(top_share).replace('.', '_')
                file_name = date + '_' + top_share + '_' + combination + '.png'
                plt.savefig(os.path.join(hists_dir_name, additional_file_name_prefix + file_name), bbox_inches='tight', pad_inches=0, transparent=True, dpi=200)
                plt.close()
                no += 1

    

def prediction_change_visualization(data_dir_name: str, plots_dir_name: str = 'PLOTS') -> None:

    files_in_dir = os.listdir(data_dir_name)
    predictions_files = [file for file in files_in_dir if 'Predictions_for' in file]

    plt.figure(figsize=(16,11))
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    colors = ['royalblue', 'orange', 'green', 'red', 'purple', 'deeppink', 'darkblue', 'limegreen', 'turquoise', 'teal', 'goldenrod', 'lightcoral', 'indigo', 'magenta', 'palevioletred', 'orangered']

    for file_no, pred_file in enumerate(predictions_files):

        df = pd.read_excel(os.path.join(data_dir_name, pred_file))

        gain = 'Average Prediction Change' if 'avg_gain' in pred_file else 'Final Prediction Change'
        df['Prediction'] = df[gain]
        df.loc[0, 'Prediction'] = df.loc[0, 'Close'] + df.loc[0, gain]
        df['Prediction'] = df['Prediction'].cumsum(axis=0)

        x = df['DateTime'].apply(lambda x: x[:10])[::10]
        y = df['Prediction'][::10]
        label = pred_file.split('_')
        label = label[0] + ': ' + label [3] + ': ' + str(int(float(label[4])*100)) + '%'

        x_neg = x[y <= df['Close'][::10]] 
        y_neg = y[y <= df['Close'][::10]]

        plt.plot(x, y, color=colors[file_no], label=label, linewidth=3, linestyle='-', alpha=0.75)
        plt.scatter(x_neg, y_neg, color=colors[file_no], marker='H')

    y = df['Close'][::10]
    label = 'OIH: BUY & HOLD'

    plt.plot(x, y, color='black', label=label, linewidth=3, linestyle='-.', alpha=0.75)

    plt.legend(fontsize=16, loc ='upper left', bbox_to_anchor=(-0.005, -0.05), ncol=4)

    x_ticks = x[::20]
    plt.xticks(ticks = list(range(0, len(x)+1, 20)), labels = x_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir_name, 'MAIN_prediction.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()



def prediction_visualization(data_dir_name: str, plots_dir_name: str = 'PLOTS') -> None:

    files_in_dir = os.listdir(data_dir_name)
    predictions_files = [file for file in files_in_dir if 'Predictions_for' in file]

    plt.figure(figsize=(16,12)) 
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.left'] = False

    colors = ['royalblue', 'orange', 'green', 'red', 'purple', 'deeppink', 'darkblue', 'limegreen', 'turquoise', 'teal', 'goldenrod', 'lightcoral', 'indigo', 'magenta', 'palevioletred', 'orangered']

    for file_no, pred_file in enumerate(predictions_files):

        df = pd.read_excel(os.path.join(data_dir_name, pred_file))

        decision = 'Average Prediction' if 'avg_gain' in pred_file else 'Final Prediction'
        
        df.loc[df[decision].isnull(), decision] = df.loc[df[decision].isnull(), 'Decision']
        df['Prediction'] = df[decision]

        step = round(len(set(df['DateTime'].apply(lambda x: x[:7]))) / 12)
        step = 1
        step_label = int(200/int(step))
        bh_label = 'OIH: BUY & HOLD'
        x = df['DateTime'].apply(lambda x: x[:10])[::step]
        y = df['Prediction'][::step]
        label = pred_file.split('_')
        label = label[0] + ': ' + 'GAIN: ' + label [3] + ': ' + str(int(float(label[4])*100)) + '%'

        x_true = np.ma.masked_where(y != df['Decision'][::step], x)
        y_true = np.ma.masked_where(y != df['Decision'][::step], y)

        y_scatter_dec = df['Decision'][::step] + (file_no+2)*6+2
        y_scatter_pred = y + (file_no+2)*6+2
        y_scatter = np.ma.masked_where(y_scatter_pred != y_scatter_dec, y_scatter_pred)

        plt.plot(x, y + (file_no+2)*6, color=colors[file_no], label=label, linewidth=3, linestyle='-', alpha=1)
        plt.plot(x_true, y_true + (file_no+2)*6+2, color=colors[file_no], linewidth=3, linestyle='-', alpha=1)
        plt.plot(x_true, y_scatter, color=colors[file_no], linewidth=0.1, alpha=0.4, marker='.')

    for file_no, pred_file in enumerate(predictions_files):

        file_no += 4
        df = pd.read_excel(os.path.join(data_dir_name, pred_file))

        decision = 'Average Prediction' 
        
        df.loc[df[decision].isnull(), decision] = df.loc[df[decision].isnull(), 'Decision']
        df['Prediction'] = df[decision]

        step = round(len(set(df['DateTime'].apply(lambda x: x[:7]))) / 12)
        step = 1
        step_label = int(200/int(step))
        bh_label = 'OIH: BUY & HOLD'
        x = df['DateTime'].apply(lambda x: x[:10])[::step]
        y = df['Prediction'][::step]
        label = pred_file.split('_')
        label = label[0] + ': ' + 'AVG GAIN: ' + label [3] + ': ' + str(int(float(label[4])*100)) + '%'

        x_true = np.ma.masked_where(y != df['Decision'][::step], x)
        y_true = np.ma.masked_where(y != df['Decision'][::step], y)

        y_scatter_dec = df['Decision'][::step] + (file_no+2)*6+2
        y_scatter_pred = y + (file_no+2)*6+2
        y_scatter = np.ma.masked_where(y_scatter_pred != y_scatter_dec, y_scatter_pred)

        plt.plot(x, y + (file_no+2)*6, color=colors[file_no], label=label, linewidth=3, linestyle='-', alpha=1)
        plt.plot(x_true, y_true + (file_no+2)*6+2, color=colors[file_no], linewidth=3, linestyle='-', alpha=1)
        plt.plot(x_true, y_scatter, color=colors[file_no], linewidth=0.1, alpha=0.4, marker='.')

    y = df['Decision'][::step]
    

    plt.plot(x, y, color='black', label=bh_label, linewidth=3, linestyle='-', alpha=0.75)

    plt.legend(fontsize=16, loc ='upper left', bbox_to_anchor=(-0.005, -0.05), ncol=3)

    x_ticks = x[::step_label]
    plt.xticks(ticks = list(range(0, len(x)+1, step_label)), labels = x_ticks, fontsize=16)
    plt.yticks([])
    
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir_name, 'MAIN_prediction_raw.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()



def main() -> None:

    plots_dir_name = 'PLOTS'
    
    if not os.path.isdir(plots_dir_name):
        os.mkdir(plots_dir_name)

    heatmaps_dir_name = 'HEATMAPS'
    
    if not os.path.isdir(heatmaps_dir_name):
        os.mkdir(heatmaps_dir_name)

    ig = IG.ImageGenerator(data_name = 'OIH_adjusted.txt', generate_only_df_data=True)
    ig.generate_images()

    train_starts = ['2001-04-02', '2001-07-02', '2001-10-01', '2002-01-02', '2002-04-01', '2002-07-01', '2002-10-01', '2003-01-02', '2003-04-01']

    strategy_visualization(ig.df, plots_dir_name = plots_dir_name)
    cross_validation_visualization(ig.df, plots_dir_name = plots_dir_name)
    cross_validation_test_visualization(ig.df, plots_dir_name = plots_dir_name)
    test_data_changes_histogram(ig.df_closing_prices, plots_dir_name = plots_dir_name)
    
    predictions_daily_dir_name = 'PREDICTIONS_DAILY_BEST_COMBINATIONS'
    prediction_change_visualization(predictions_daily_dir_name, plots_dir_name)
    prediction_visualization(predictions_daily_dir_name, plots_dir_name)

    predictions_dir_name = 'PREDICTIONS_MAX_PREDICTIONS'
    df_predictions = pd.read_csv(os.path.join(predictions_dir_name, 'MAX_combinations_results.csv'))
    predictions_heatmap_overall(df_predictions, heatmaps_dir_name)  
    predictions_heatmap_by_interval(df_predictions, heatmaps_dir_name, train_starts)




if __name__ == '__main__':

    main()


    


        








