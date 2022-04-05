import ImageGenerator as IG
import NNModel as NN


if __name__ == '__main__':

  ig = IG.ImageGenerator('OIH_adjusted.txt', generate_only_df_data=True)
  ig.generate_images()
  nn = NN.NNModel(df_closing_prices=ig.df_closing_prices, NN_number=9, patience=20, verbose=0)
  nn.create_train_and_evalute_model()
