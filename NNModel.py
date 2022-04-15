import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import os 
import glob

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, LeakyReLU, BatchNormalization, Dropout, Lambda
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal, HeUniform, LecunNormal, LecunUniform
from tensorflow.keras.initializers import Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from keras_self_attention import SeqSelfAttention
from NNModelAuxiliary import ChannelAttention, SpatialAttention, StopOnPoint, ReshapeLayer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



class NNModel:
    """
    ...
    """


    def __init__(self,
                 project_path: str=os.getcwd(),
                 imgs_path: str=os.path.join(os.getcwd(), 'GAF'), 
                 models_dir_name: str='MODELS',
                 predictions_dir_name: str='PREDICTIONS',
                 df_closing_prices: pd.DataFrame=pd.DataFrame(),
                 train_start: str='2001-04-01',
                 len_train: int=13*12,
                 len_validation: int=3.75*12,
                 len_test: int=2*12,
                 interval_in_month: int=3,
                 pass_model: bool=False,
                 models_list: list=None,
                 NN_number: int=9,
                 NN_initializers_labels_custom: list=['GN'],
                 epochs: int=100,
                 loss: str='binary_crossentropy',
                 metric: str='acc',
                 patience: int=20,
                 verbose: int=1):


        self.project_path = project_path
        self.imgs_path = imgs_path
        self.models_dir_name = models_dir_name
        self.predictions_dir_name = predictions_dir_name
        self.df_closing_prices = df_closing_prices
        self.train_start = train_start
        self.len_train = len_train
        self.len_validation = len_validation
        self.len_test = len_test
        self.interval_in_month = interval_in_month
        self.pass_model = pass_model
        self.models_list = models_list
        self.NN_number = NN_number
        self.NN_initializers_labels_custom = NN_initializers_labels_custom
        self.epochs = epochs
        self.loss = loss
        self.metric = metric
        self.patience = patience
        self.verbose = verbose


    def __str__(self) -> str:

        return(f'This Model is initialized for {self.NN_number} neural networks for data in {self.imgs_path} path')


    def __repr__(self) -> str:

        return(f'\Model(project_path = {self.project_path},\n \
                imgs_path = {self.imgs_path},\n \
                models_dir_name = {self.models_dir_name},\n \
                predictions_dir_name = {self.predictions_dir_name},\n \
                df_closing_prices = {self.df_closing_prices},\n \
                train_start = {self.train_start},\n \
                len_train = {self.len_train},\n \
                len_validation = {self.len_validation},\n \
                len_test = {self.len_test},\n \
                interval_in_month = {self.interval_in_month},\n \
                pass_model = {self.pass_model},\n \
                models_list = {self.models_list},\n \
                NN_number = {self.NN_number},\n \
                NN_initializers_labels_custom = {self.NN_initializers_labels_custom},\n \
                epochs = {self.epochs},\n \
                loss = {self.loss},\n \
                metric = {self.metric},\n \
                patience = {self.patience},\n \
                verbose = {self.verbose})')


    def create_train_and_evalute_model(self) -> None:
        """
        ...
        """


        if not os.path.isdir(self.models_dir_name):
            os.mkdir(self.models_dir_name)

        if not os.path.isdir(self.predictions_dir_name):
            os.mkdir(self.predictions_dir_name)


        self.gather_data_path_into_df()

        self.train_img_data = ImageDataGenerator(rescale=1/255)  
        self.validation_img_data = ImageDataGenerator(rescale=1/255) 
        self.test_img_data = ImageDataGenerator(rescale=1/255)
        self.NNs_data = self.split_data_into_N_NNs()

        if not self.pass_model:

            GN, GU, HN, HU = GlorotNormal(seed=42), GlorotUniform(seed=42), HeNormal(seed=42), HeUniform(seed=42)
            LN, LU, ON, OR = LecunNormal(seed=42), LecunUniform(seed=42), Ones, Orthogonal(seed=42)
            RN, RU, TN, VS = RandomNormal(seed=42), RandomUniform(seed=42), TruncatedNormal(seed=42), VarianceScaling(seed=42)

            self.initializers = [GN, GU, HN, HU, LN, LU, ON, OR, RN, RU, TN, VS]
            self.initializers_labels = ['GN', 'GU', 'HN', 'HU', 'LN', 'LU', 'ON', 'OR', 'RN', 'RU', 'TN', 'VS']

            models = []
            for NN_no in range(self.NN_number):
                for initializer in self.initializers:

                    model = self.create_model(initializer)
                    models.append(model)
                   
            self.models_list = models

            train_model_no = 0
            for NN_no in range(len(self.NNs_data)):
                for initializer in self.initializers_labels:

                    self.train_model(train_model_no, NN_no, initializer)
                    train_model_no += 1
                    
        else:

            train_model_no = 0
            for NN_no in range(len(self.NNs_data)):
                for initializer in self.NN_initializers_labels_custom:

                    self.train_model(train_model_no, NN_no, initializer)
                    train_model_no += 1


    def gather_data_path_into_df(self) -> None:
        """
        ...
        """

        dfs = []
        for number, subfolder in enumerate(['SHORT', 'LONG']):
            images = glob.glob(self.imgs_path + '/{}/*.png'.format(subfolder))  
            dates = [dt.split('/')[-1].split('\\')[-1].split('.')[0].replace('_', '-') for dt in images]
            data_class = pd.DataFrame({'Image': images, 'Label': [str(number) + '_' + subfolder] * len(images), 'Date': dates})
            data_class['Date'] = pd.to_datetime(data_class['Date'])
            dfs.append(data_class)
        data = pd.concat(dfs).reset_index(drop=True)
        self.df_closing_prices['Date'] = pd.to_datetime(self.df_closing_prices['DateTime'].dt.date)
        data = data.merge(self.df_closing_prices, how='inner', on='Date').sort_values(by='Date').reset_index(drop=True)
        del data['Date']
        self.df_data_path = data


    def split_data_into_N_NNs(self) -> list:
        """
        ...
        """

        train_start = pd.to_datetime(dt.date.fromisoformat(self.train_start))

        NNs = []
        for i in range(self.NN_number):

            train_end = pd.to_datetime(train_start + relativedelta(months=self.len_train))
            validation_end = pd.to_datetime(train_end + relativedelta(months=self.len_validation))
            test_end = pd.to_datetime(validation_end + relativedelta(months=self.len_test))

            df_train_chunk = self.df_data_path.loc[self.df_data_path['DateTime'].between(train_start, train_end, inclusive='left')].reset_index(drop=True)
            df_validation_chunk = self.df_data_path.loc[self.df_data_path['DateTime'].between(train_end, validation_end, inclusive='left')].reset_index(drop=True)
            df_test_chunk = self.df_data_path.loc[self.df_data_path['DateTime'].between(validation_end, test_end, inclusive='left')].reset_index(drop=True)

            df_chunk = [df_train_chunk, df_validation_chunk, df_test_chunk]
            print(i, 'TRAIN:', str(df_train_chunk['DateTime'].dt.date[0]), str(df_train_chunk['DateTime'].dt.date[len(df_train_chunk)-1]),
                     'VALIDATION:', str(df_validation_chunk['DateTime'].dt.date[0]), str(df_validation_chunk['DateTime'].dt.date[len(df_validation_chunk)-1]),
                     'TEST:', str(df_test_chunk['DateTime'].dt.date[0]), str(df_test_chunk['DateTime'].dt.date[len(df_test_chunk)-1]))

            NNs.append(df_chunk)
            train_start += relativedelta(months=self.interval_in_month)

        return NNs
    

    def create_model(self, initializer) -> tf.keras.models.Sequential:
        """
        ...
        """
        
        model=Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), kernel_initializer=initializer,  input_shape=(40, 40, 3), activation=LeakyReLU(alpha=0.1), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3,3), activation=LeakyReLU(alpha=0.1), padding='same'))
        model.add(BatchNormalization())
        model.add(ChannelAttention(32, 8))
        model.add(SpatialAttention(40))
        model.add(Dropout(0.4))

        model.add(Conv2D(64, kernel_size=(3,3), activation=LeakyReLU(alpha=0.1), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3,3), activation=LeakyReLU(alpha=0.1), padding='same'))
        model.add(BatchNormalization())
        model.add(ChannelAttention(64, 8))
        model.add(SpatialAttention(40))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, kernel_size=(3,3), activation=LeakyReLU(alpha=0.1), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3,3), activation=LeakyReLU(alpha=0.1), padding='same'))
        model.add(BatchNormalization())
        model.add(ChannelAttention(128, 8))
        model.add(SpatialAttention(40))
        
        model.add(Lambda(ReshapeLayer))
        model.add(LSTM(32, dropout=0.4, return_sequences=True))
        model.add(SeqSelfAttention(attention_width=16))
        model.add(Flatten())
        model.add(Dense(1024, activation=LeakyReLU(alpha=0.1)))
        model.add(Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True) , loss=self.loss, metrics=self.metric)
        
        return model


    def train_model(self, train_model_no, NN_no, initializer) -> None:
        """
        ...
        """

        print(f'Net: {NN_no} initializer: {initializer}')
        print(dt.datetime.now())
        df_train = self.NNs_data[NN_no][0]
        df_validation = self.NNs_data[NN_no][1]
        df_test = self.NNs_data[NN_no][2]

        train_data = self.train_img_data.flow_from_dataframe(
            dataframe=df_train,
            directory=self.imgs_path,
            target_size=(40, 40),
            x_col='Image',
            y_col='Label',
            class_mode='binary',
            batch_size=32,
            shuffle=False,
            validate_filenames=True)
        
        validation_data = self.validation_img_data.flow_from_dataframe(
            dataframe=df_validation,
            directory=self.imgs_path,
            target_size=(40, 40),
            x_col='Image',
            y_col='Label',
            class_mode='binary',
            batch_size=32,
            shuffle=False,
            validate_filenames=True)
        
        test_data = self.test_img_data.flow_from_dataframe(
            dataframe=df_test,
            directory=self.imgs_path,
            target_size=(40, 40),
            x_col='Image',
            y_col='Label',
            class_mode='binary',
            batch_size=32,
            shuffle=False,
            validate_filenames=True)

        start_NN_date = '_' + str(df_train['DateTime'][0])[:10]
        print('Start_date', start_NN_date)
        print(self.models_list[train_model_no].summary())
        model_name = os.path.join(self.models_dir_name, 'Model_' + str(NN_no) + '_' + initializer + start_NN_date + '.h5')
        model_ckpoint = ModelCheckpoint(model_name, monitor='val_acc', mode='max', verbose = self.verbose, save_best_only=True, save_weights_only=False)
        early_stoping = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=self.patience, verbose=self.verbose)

        model_fit = self.models_list[train_model_no].fit(train_data,
                                                         epochs=self.epochs,
                                                         validation_data=validation_data,
                                                         callbacks=[early_stoping, model_ckpoint, StopOnPoint(0.65)],
                                                         verbose=self.verbose)

        max_acc_train = max(model_fit.history['acc'])
        max_acc_validation = max(model_fit.history['val_acc'])
        print(f'Net: {NN_no} initializer: {initializer}\nTrain Accuracy: {max_acc_train*100}%\nValidation Accuracy: {max_acc_validation*100}%')


        self.evaluate_model(NN_no, initializer, validation_data, df_validation, test_data, df_test, model_name, start_NN_date)
                                                        
                                                        
    def evaluate_model(self, NN_no, initializer, validation_data, df_validation, test_data, df_test, model_name, start_NN_date) -> None:
        """
        ...
        """

        if not self.pass_model:
            best_model = load_model(model_name, custom_objects = {"ChannelAttention": ChannelAttention,
                                                                  "SpatialAttention": SpatialAttention, 
                                                                  "SeqSelfAttention": SeqSelfAttention})
        else:
            best_model = load_model(model_name)


        ####################   VALIDATION   ####################

        scores_validation = best_model.evaluate(validation_data)
        print(f'Validation (Best Model) Accuracy: {scores_validation[1]*100}%')

        model_prediction_validation = best_model.predict(validation_data)
        col_val_name = 'Prediction_validation_' + str(NN_no) + str(initializer) + start_NN_date
        df_validation[col_val_name] = model_prediction_validation

        df_validation.to_csv(os.path.join(self.predictions_dir_name, col_val_name + '.csv'), index=False)


        ####################   TEST   ####################

        scores_test = best_model.evaluate(test_data)
        print(f'Test Accuracy: {scores_test[1]*100}%')

        model_prediction_test = best_model.predict(test_data)
        col_test_name = 'Prediction_test_' + str(NN_no) + str(initializer) + start_NN_date
        df_test[col_test_name] = model_prediction_test  
        
        df_test.to_csv(os.path.join(self.predictions_dir_name, col_test_name + '.csv'), index=False)


    