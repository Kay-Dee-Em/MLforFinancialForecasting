import ImageGenerator as IG

if __name__ == '__main__':

  ig = IG.ImageGenerator(data_name = 'OIH_adjusted.txt', imgs_dir_name='GAF_binary', cmap='binary')
  ig.generate_images()

  

 

