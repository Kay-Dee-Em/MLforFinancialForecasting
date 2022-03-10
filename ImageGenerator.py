import os
import pandas as pd
from pandas import Timestamp as ts
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from datetime import time
from multiprocessing import Pool
from funcy import join_with
from pyts.image import GramianAngularField
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import Logger as log
import logging

matplotlib.use('Agg')

logger = logging.getLogger(os.path.basename(__file__))
custom_loggs = logging.StreamHandler()
custom_loggs.setFormatter(log.CustomFormatter())
logger.addHandler(custom_loggs)
setattr(logger, 'success', lambda message, *args: logger._log(logging.SUCCESS, message, args))
setattr(logger, 'info', lambda message, *args: logger._log(logging.INFO, message, args))


class ImageGenerator:
  """
  Image Generator for processing, cleaning and creating GAFs (Gramian Angular Field) images for time series.
  
  Parameters
  ------------

  data_name: str,
      Data name

  project_path: str, default: os.getcwd(),
      Path where the files will be created

  imgs_dir_name: str default: 'GAF',
      Dir name for images 

  base_freq: str, default: '1h,
      Base frequency for data grouping 

  h_start: str, default: '9:00',
      Start trading hour

  h_end: str, default: '16:00',
      End trading hour

  interval: int, default: 20,
      Data interval (preprocessed subsequent unique days)
  
  generate_only_df_data: bool, default: False,
      Generate only DataFrame data (do not create images)

  freqs: list, default: ['1h', '2h', '4h', '1d'],
      Frequencies for data grouping where array length equals number of images in one image grid

  GAF_method: str, default: 'difference',
      GAF method, possible options: ['difference', 'summation']

  image_matrix: tuple, default: (2,2),
      Image matrix dimensions, nrows * ncols >= len(freqs)

  cmap: str, default: 'rainbow',
      Color map for images
  """


  def __init__(self,
               data_name: str,
               project_path: str=os.getcwd(),
               imgs_dir_name: str='GAF', 
               base_freq: str='1h', 
               h_start: str='9:00', 
               h_end: str='16:00', 
               interval: int=20,
               generate_only_df_data: bool=False,
               freqs: list=['1h', '2h', '4h', '1d'],
               GAF_method: str='difference',
               image_matrix: tuple=(2,2),
               cmap: str='rainbow'): 


    self.data_name = data_name
    self.project_path = project_path
    self.imgs_dir_name = imgs_dir_name
    self.base_freq = base_freq
    self.h_start = h_start
    self.h_end = h_end
    self.interval = interval
    self.generate_only_df_data = generate_only_df_data
    self.freqs = freqs
    self.image_matrix = *image_matrix,
    self.GAF_method = GAF_method
    self.cmap = cmap


  def __str__(self) -> str:

    return(f'This Image Generator is initialized for {self.data_name} data in {self.project_path} path')


  def __repr__(self) -> str:

    return(f'\nImageGenerator(data_name = {self.data_name},\n \
              project_path = {self.project_path},\n \
              imgs_dir_name = {self.imgs_dir_name},\n \
              base_freq = {self.base_freq},\n \
              h_start = {self.h_start},\n \
              h_end = {self.h_end},\n \
              interval = {self.interval},\n \
              generate_only_df_data = {self.generate_only_df_data},\n \
              freqs = {self.freqs},\n \
              GAF_method = {self.GAF_method},\n \
              freqs = {self.freqs},\n \
              image_matrix = {self.image_matrix},\n \
              cmap = {self.cmap})')


  def generate_images(self) -> None:

    if self.generate_only_df_data:  
        self.preprocess_data()

    else:
        self.create_directories()
        self.preprocess_data()
        self.create_images()  


  def create_directories(self) -> None:
    """ 
    Create directories for images
    :return: None
    """

    GAF = os.path.join(self.project_path, self.imgs_dir_name)
    LONG = os.path.join(self.project_path, self.imgs_dir_name, 'LONG')
    SHORT = os.path.join(self.project_path, self.imgs_dir_name, 'SHORT')
    os.makedirs(LONG)
    os.makedirs(SHORT)
    self.imgs_path = GAF


  def preprocess_data(self) -> None:
    """
    Preprocess data i.e. delete unnecessary columns, group data by defined interval, delete non-trading times (days and hours),
    determine intervals and changes in the closing prices
    :return: None
    """

    logger.info('PROCESSING DATA')

    col_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(os.path.join(self.project_path, self.data_name), names=col_names, header=None)
    df = df[['Date', 'Time', 'Close']]
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], infer_datetime_format=True)
    df = df.groupby(pd.Grouper(key='DateTime', freq=self.base_freq)).mean().reset_index()     
    df['Close'].replace(to_replace=0, method='ffill', inplace=True)

    df = df[df['DateTime'].dt.weekday < 5].set_index('DateTime').between_time(self.h_start,self.h_end).reset_index()
    
    data_start = ts(df['DateTime'].min().year,1,31)
    data_end = ts(df['DateTime'].max().year+1,1,1)
    df = df[~df['DateTime'].isin(Calendar().holidays(start=data_start, end=data_end))].fillna(method='ffill')
    self.df = df

    self.dates = self.df['DateTime'].dt.date.drop_duplicates()
    list_dates = self.dates.apply(str).tolist()
    self.days_start = list_dates[0:-self.interval]
    self.days_end = list_dates[self.interval-1:-1]
    self.days_next_after_end = list_dates[self.interval:]


    df_closing_prices = self.df.loc[(self.df['DateTime'].dt.date.astype(str).isin(self.days_end)) & 
                                    (self.df['DateTime'].dt.hour == time(*map(int, self.h_end.split(':'))).hour)].reset_index(drop=True)
    df_closing_prices['Change'] = -df_closing_prices.Close.diff(periods=-1)
    df_closing_prices['Decision'] = df_closing_prices['Change'].apply(lambda x: 1 if x > 0 else 0)
    self.df_closing_prices = df_closing_prices


  def create_images(self) -> None:
    """
    Create images for preprocessed data, set decision long/short (buy/sell) and create images for defined frequencies 
    :return: None
    """
    
    global results
    results = []
    logger.info('SETTING LONG/SHORT DECISION')
    pool = Pool(os.cpu_count())
    for i in range(len(self.days_end)):
        pool.apply_async(self.set_decision, args=(i, self.days_start[i], self.days_end[i], self.days_next_after_end[i]), callback=self.__get_result)
    pool.close()
    pool.join()
    number, dicts = zip(*results)
    decision_map = join_with(list, list(dicts))
    logger.success("SETTING DECISION FINISHED SUCCESSFULLY")

    logger.info('GENERATING IMAGES')
    pool = Pool(os.cpu_count())
    for decision, data in decision_map.items():
        for image_data in data:
            pool.apply_async(self.generate_gaf, args=(image_data, decision)).get() 
    pool.close()
    total_days = self.dates.shape[0]
    total_short = len(decision_map['SHORT'])
    total_long = len(decision_map['LONG'])
    imgs_created = total_short + total_long
    logger.success(f"GENERATING IMAGES FINISHED SUCCESSFULLY\nTotal Days: {total_days}\nTotal Images Created: {imgs_created}\
        \nTotal LONG: {total_long}\nTotal SHORT: {total_short}")


  @staticmethod
  def __get_result(result: list) -> None:
      """
      Auxiliary function for gathering setting decision results 
      :param result: list
      :return: None
      """
      global results
      results.append(result)


  def set_decision(self, i: int, day_start: str, day_end: str, day_next_after_end: str) -> tuple:
      """
      Set decision for given interval (self.interval: default = 20) and 
      for given frequencies (self.freqs: default ['1h', '2h', '4h', '1d']),
      i.e. determine whether long or short strategy should be taken based on
      the Close value the next day (self.h_end: default = 16:00)
      For each freqs group and determine the last 20 timestamps and append list GAFs
      Return tuple(i, decision_map) for each interval
      
      :param i: int, auxiliary variable for tracking data order
      :param day_start: str, first day of the defined interval 
      :param day_end: str, last day of the defined interval 
      :param day_next_after_end: str, next day after the last day of the defined interval 
      :return: tuple
      """

      df_interval = self.df.loc[(self.df['DateTime'] >= day_start) & (self.df['DateTime'] < day_next_after_end)]
      
      gafs = []
      for freq in self.freqs:
              group_dt = df_interval.groupby(pd.Grouper(key='DateTime', freq=freq)).mean().reset_index().dropna()
              gafs.append(group_dt['Close'].tail(20))
      
      future_value = self.df[self.df['DateTime'].dt.date.astype(str) == day_next_after_end]['Close'].iloc[-1]
      current_value = df_interval['Close'].iloc[-1]
      decision = 'LONG' if future_value > current_value else 'SHORT'
      decision_map = {decision: (day_end, gafs)} 
      return(i, decision_map)


  def generate_gaf(self, one_day_data: list, decision: str) -> None:
      """
      Call functions to generate GAF images i.e. call create_gaf to create list of DataFrames
      and then calling join_gafs will result saving images
      :param one_day_data: list
      :param decision: str
      :return: None
      """

      gafs = [self.create_gaf(x)['gaf'] for x in one_day_data[1]]
      self.join_gafs(images=gafs, image_name='{0}'.format(one_day_data[0].replace('-', '_')), destination=decision)
  

  def create_gaf(self, time_series: list) -> list:
      """
      Create GAF (more explicit GADF or GASF (Gramian Angular Difference/Summation Field)) data for n prior defined frequencies
      Return a list of n DataFrames with the time series shape (square matrix)
      :param time_series: list
      :return: list
      """

      data = dict()
      gaf = GramianAngularField(method=self.GAF_method, image_size=time_series.shape[0])
      data['gaf'] = gaf.fit_transform(pd.DataFrame(time_series).T)[0]  
      return data


  def join_gafs(self, images: list, image_name: str, destination: str) -> None:  
      """
      Join GAFs' data and create Image Grid 
      :param images: list
      :param image_name: str
      :param destination: str
      :return: None
      """

      fig = plt.figure(figsize=[img * self.image_matrix[0] * self.image_matrix[1] for img in self.image_matrix])
      grid = ImageGrid(fig, 111, axes_pad=0, nrows_ncols=self.image_matrix, share_all=True)
      for image, ax in zip(images, grid):
          ax.set_xticks([])
          ax.set_yticks([])
          ax.imshow(image, cmap=self.cmap, origin='lower')

      repo = os.path.join(self.imgs_path, destination)
      fig.savefig(os.path.join(repo, image_name), bbox_inches='tight')
      plt.close(fig)

