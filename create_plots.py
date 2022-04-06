import ImageGenerator as IG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
from dateutil.relativedelta import relativedelta

""""
Auxiliary script to:
- create visualization of preprocessing the data (linear plot, polar plot) for both strategies for chosen days i.e.:
    LONG - 2021-12-31
    SHORT - 2021-11-09

- create visualization of cross-validation (CV) intervals for 9 neural networks 

"""

def strategy_visualization(df: pd.DataFrame()):

    df_long = df[(df['DateTime'].dt.date.astype(str) >= '2021-12-03') & (df['DateTime'].dt.date.astype(str) < '2022-01-01')]
    df_long_next_day = df[(df['DateTime'].dt.date.astype(str) >= '2021-12-31') & (df['DateTime'].dt.date.astype(str) < '2022-01-04')][-9:]
    df_short = df[(df['DateTime'].dt.date.astype(str) >= '2021-10-13') & (df['DateTime'].dt.date.astype(str) < '2021-11-10')]
    df_short_next_day = df[(df['DateTime'].dt.date.astype(str) >= '2021-11-09') & (df['DateTime'].dt.date.astype(str) < '2021-11-11')][-9:]

    dfs_strategy = [df_long, df_short]
    dfs_strategy_next_day = [df_long_next_day, df_short_next_day]

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


    fig_names = ['LONG_31_12_2021_line', 'SHORT_09_11_2021_line', 'LONG_31_12_2021_polar', 'SHORT_09_11_2021_polar']

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


def cross_validation_visualization(df: pd.DataFrame()):

    plt.figure(figsize=(10,7))
    plt.rc('axes', axisbelow=True)
    x = df['DateTime']
    y = df['Close']
    plt.plot(x, y, color='#BFDCF7')
    plt.fill_between(x, y, color='#BFDCF7')
    plt.margins(0)

    train_start = pd.to_datetime(dt.date.fromisoformat('2001-04-01'))
    NN_number = 9
    start_value = 15
    for i in range(NN_number):

        train_end = pd.to_datetime(train_start + relativedelta(months=13*12))
        validation_end = pd.to_datetime(train_end + relativedelta(months=3.75*12))
        test_end = pd.to_datetime(validation_end + relativedelta(months=2*12))

        df_train_chunk = df.loc[df['DateTime'].between(train_start, train_end, inclusive='left')].reset_index(drop=True)
        df_validation_chunk = df.loc[df['DateTime'].between(train_end, validation_end, inclusive='left')].reset_index(drop=True)
        df_test_chunk = df.loc[df['DateTime'].between(validation_end, test_end, inclusive='left')].reset_index(drop=True)


        x_train = df_train_chunk['DateTime']
        y_train = [start_value] * len(x_train)
        x_validation = df_validation_chunk['DateTime']
        y_validation = [start_value] * len(x_validation)
        x_test = df_test_chunk['DateTime']
        y_test = [start_value] * len(x_test)

        plt.plot(x_train, y_train, color='red', linewidth=2.5, linestyle='-')
        plt.plot(x_validation, y_validation, color='orange', linewidth=2.5, linestyle='-')
        plt.plot(x_test, y_test, color='green', linewidth=2.5)

        train_start += relativedelta(months=3)
        start_value += 30


    plt.grid(True)
    plt.savefig(os.path.join(plots_dir_name, 'cross_validation.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


if __name__ == '__main__':

    ig = IG.ImageGenerator(data_name = 'OIH_adjusted.txt', generate_only_df_data=True)
    ig.generate_images()

    plots_dir_name = 'PLOTS'
    os.mkdir(plots_dir_name)

    strategy_visualization(ig.df)
    cross_validation_visualization(ig.df)



    


        








