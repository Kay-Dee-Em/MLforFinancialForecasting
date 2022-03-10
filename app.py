import ImageGenerator as IG
#import Model as M

if __name__ == '__main__':

  ig = IG.ImageGenerator(data_name = 'OIH_adjusted.txt', imgs_dir_name='GAF_binary', cmap='binary')#, generate_only_df_data=True)
  ig.generate_images()
  #model = M.Model(df_closing_prices = ig.df_closing_prices)
  #model.create_and_evalute_model()


  

 

