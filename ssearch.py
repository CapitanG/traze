import sys
#Please, change the following path to where convnet2 can be located
#sys.path.append("/home/jsaavedr/Research/git/tensorflow-2/convnet2")
from sklearn.svm import SVC
from sklearn.svm._libsvm import predict_proba

sys.path.append("/Users/G/Downloads/VAE/Git/convnet2")

import tensorflow as tf
import datasets.data as data
import utils.configuration as conf #sets utils.configuration.py as conf
import utils.imgproc as imgproc #sets utils.imgproc.py as imgproc
import skimage.io as io
import skimage.transform as trans
import os
import argparse
import numpy as np
from tensorflow.keras.utils import plot_model
from numpy import savetxt
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold
from statistics import mean
from sklearn import svm
from sklearn import metrics
#import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer



class SSearch :
    def __init__(self, config_file, model_name): #constructor, receives config_file, model_name
        
        self.configuration = conf.ConfigurationFile(config_file, model_name) #
        #defining input_shape image height,
        self.input_shape =  (self.configuration.get_image_height(), #create a tuple with height, width and channels (1 or 3)
                             self.configuration.get_image_width(),
                             self.configuration.get_number_of_channels())

        #loading the model
        model = tf.keras.applications.ResNet50(include_top=True, 
                                               weights='imagenet', 
                                               input_tensor=None, 
                                               input_shape =self.input_shape, # using a tuple (height, width, channels)
                                               pooling=None, 
                                               classes=1000)

        #redefining the model to get the hidden output
        #self.output_layer_name = 'conv5_block3_out' #layer 5
        #self.output_layer_name = 'conv4_block6_out' #layer 4
        #self.output_layer_name = 'conv3_block4_out' #layer 3
        #self.output_layer_name = 'conv2_block3_out' #layer 2
        self.output_layer_name = 'pool1_pool'   #Layer 1
        output = model.get_layer(self.output_layer_name).output # model.
        #print('these are the dimensions of output b4 GAP ', output)
        output = tf.keras.layers.GlobalAveragePooling2D()(output)
        #print('these are the dimensions of output fter GAP  ', output)
        self.sim_model = tf.keras.Model(model.input, output)


        self.sim_model.summary()

        #model variable for layer printing
        self.modelForPrint = tf.keras.Model(model.input, output) # i added this to print the graph

        #defining image processing function
        self.process_fun =  imgproc.process_image_visual_attribute  # creates local function of process_fun to change image size and aspect ratio

        #loading catalog

        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), '') #loads path to data directory

        #/datasets/Labeled_Images
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt') # loads path specifically for ssearch directory
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file) #verifies existence of catalog file in directory
        print('loading catalog ...')
        self.load_catalog(catalog_file) # calls function that creates array w name of files & variable with length
        print(  'loading catalog ok ...')
        self.enable_search = False
        ## end of constructor

    #read_image
    def read_image(self, filename):

        im = self.process_fun(data.read_image(filename, self.input_shape[2]), (self.input_shape[0], self.input_shape[1]))

        #for resnet
        im = tf.keras.applications.resnet50.preprocess_input(im) # processes image according to resnet specs

        return im

    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")                        
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype = np.int32)
        self.features = np.fromfile(fvs_file, dtype = np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')
        
    def load_catalog(self, catalog):
        with open(catalog) as f_in :
            self.filenames = [filename.strip() for filename in f_in ]
        self.data_size = len(self.filenames)


    def StratKFold(self, path, k):
        fold_save_path = '/Users/G/Downloads/VAE/Git/Datasets/Visual_Attributes/ssearch/folds_csvs' #path to save csvs of the folds
        df = self.csv_to_dataframe(path) # loads the csv from the path parameter

        num_features = df.shape[1] # here i determine the number of columns? what is df.shape[1]


        y = df.iloc[:, 0].ravel()
        X = df.iloc[:, 1:num_features]
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        return_dict = {}

        for fold, (entrenamiento, validacion) in enumerate(kf.split(X=X, y=y)):
            saved_file_name = 'validacion_'+str(fold)+'.csv'
            complete_path_save = os.path.join(fold_save_path, saved_file_name)
            savetxt(complete_path_save , validacion, fmt='%s', delimiter=',')
            return_dict['validacion_'+str(fold)] = validacion

            saved_file_name = 'entrenamiento_'+str(fold)+'.csv'
            complete_path_save = os.path.join(fold_save_path, saved_file_name)
            savetxt(complete_path_save , entrenamiento, fmt='%s', delimiter=',')
            return_dict['entrenamiento_'+str(fold)] = entrenamiento

            #Test with reduced dataset, work in progress
            mylist1=[]
            for keys in return_dict:
              mylist1.append(keys)
            return_dict_short = dict.fromkeys(mylist1)

            mylist2 =[]
            for values in return_dict.values():
              mylist2.append(list(values))

            for i in range(len(return_dict)):
              return_dict_short[mylist1[i]]=mylist2[i][:50]
        return return_dict

    def csv_to_dataframe(self, path):
        le = preprocessing.LabelEncoder() #object of encoder class
        self.accuracy_layers_df = pd.read_csv(path, header=None) #creating dataframe with csv data
        feature_columns_headers = ["f"+str(i) for i in range(1, self.accuracy_layers_df.shape[1])] #creating headers for data columns
        label_column_header = ['label'] #creating header for label column
        self.all_headers = label_column_header + feature_columns_headers # creating file with all headers
        self.accuracy_layers_df.columns = self.all_headers #assigning layer column headers
        self.accuracy_layers_df['label'] = le.fit_transform(self.accuracy_layers_df['label']) #converting colors in label column to integers
        return self.accuracy_layers_df

    def find_layer_accuracy(self, k, folds):

        layers = ['Color__1.csv', 'Color__2.csv', 'Color__3.csv', 'Color__4.csv', 'Color__5.csv']
        ## SELECT LAYERS TO PROCESS
        #layers = ['Color_1.csv', 'Color_2.csv', 'Color_3.csv', 'Color_4.csv', 'Color_5.csv']
        #layers = ['Texture_1.csv', 'Texture_2.csv', 'Texture_3.csv', 'Texture_4.csv', 'Texture_5.csv']
        #layers = ['Color_1.csv', 'Color_2.csv', 'Color_3.csv', 'Color_4.csv', 'Color_5.csv',
                  #'Texture_1.csv', 'Texture_2.csv', 'Texture_3.csv', 'Texture_4.csv', 'Texture_5.csv',
                  #'Acce_1.csv', 'Acce_2.csv', 'Acce_3.csv', 'Acce_4.csv']


        ## LOAD DATA AND INDICES
        csvs_dir = "/Users/G/Downloads/VAE/Git/Datasets/Visual_Attributes/ssearch/features_csv_format_Accuracy"
        path_to_csvs = os.path.join(csvs_dir, layers[0])
        training_size_df = pd.read_csv(path_to_csvs, header=None)
        all_80_20_indices = list(range(len(training_size_df)))
        train_80_20_indices, test_80_20_indices = train_test_split(all_80_20_indices, test_size=0.2)
        all_straKfold_indx_dict = self.StratKFold(path_to_csvs, folds)


        ##STRATIFIED

        ## KNN
        accuracy_results = 'Stratified kNN'+str(folds)+'-Folds \n '

        for i in range(len(layers)):   #looping thru layers for Stratified accuracy kNN
            csv_file_name = layers[i]
            path_to_csvs = os.path.join(csvs_dir, csv_file_name )
            self.df_concatenated = self.csv_to_dataframe(path_to_csvs)

            kNN_fold_accuracies = []

        
            for j in range(folds): #looping thru folds obtained thru StratKFold() kNN
                print()
                X_train_stratKfold_values1 = self.df_concatenated.iloc[all_straKfold_indx_dict['entrenamiento_'+str(j)], 1:]
                #X_train_stratKfold_values1 = StandardScaler().fit_transform(X_train_stratKfold_values1)
                #X_train_stratKfold_values1 = self.square_root_norm(X_train_stratKfold_values1)
                y_train_stratKfold_labels1 = self.df_concatenated.iloc[all_straKfold_indx_dict['entrenamiento_'+str(j)], 0]

                x_test_stratKfold_values1 = self.df_concatenated.iloc[all_straKfold_indx_dict['validacion_'+str(j)], 1:]
                #x_test_stratKfold_values1 = StandardScaler().fit_transform(x_test_stratKfold_values1)
                #x_test_stratKfold_values1 = self.square_root_norm(x_test_stratKfold_values1)
                x_test_stratKfold_valuesNP1 = x_test_stratKfold_values1.to_numpy()
                y_test_stratKfold_labels1 = self.df_concatenated.iloc[all_straKfold_indx_dict['validacion_'+str(j)], 0]
                X_train_stratKfold_valuesNP1 = X_train_stratKfold_values1.to_numpy()


                '''
                #CODE KNN FOR PARAMETER TUNING 
               
                #List Hyperparameters that we want to tune.
                #leaf_size = list(range(1,51)) #no usar
                n_neighbors = [1,3,5,7,9,11,13,15,17,19] 
                p=[1,2] 
                weight_options = ['uniform', 'distance']
                #Convert to dictionary
                hyperparameters = dict(n_neighbors=n_neighbors, p=p, weights = weight_options )
                #Create new KNN object
                knn_2 = KNeighborsClassifier()
                #Use GridSearch
                clf = GridSearchCV(knn_2, hyperparameters, cv=10, scoring='accuracy', n_jobs = 4)
                #Fit the model
                best_model = clf.fit(X_train_stratKfold_valuesNP1, y_train_stratKfold_labels1.ravel())
                #Print The value of best Hyperparameters
                #print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
                print('Best p:', best_model.best_estimator_.get_params()['p'])
                print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
                print('Best Weight:', best_model.best_estimator_.get_params()['weights'])
                                
                #END PARAMETER TUNING

                '''

                knnST = KNeighborsClassifier(n_neighbors = k, p=2, weights='uniform')
                #knnST = KNeighborsClassifier(n_neighbors = k, p=2, weights='distance')
                knnST.fit(X_train_stratKfold_valuesNP1, y_train_stratKfold_labels1.ravel())
                y_pred_stratKfold_knn = knnST.predict(x_test_stratKfold_valuesNP1)
                fold_accuracy_knn = accuracy_score(y_test_stratKfold_labels1, y_pred_stratKfold_knn)
                kNN_fold_accuracies.append(fold_accuracy_knn)
                
            accuracy_results += layers[i] + ' ' + str(mean(kNN_fold_accuracies)) + '\n '


        accuracy_results += '\nStratified SVM'+str(folds)+'-Folds \n '


        for p in range(len(layers)): #looping thru layers for Stratified accuracy SVM
            csv_file_name = layers[p]
            path_to_csvs = os.path.join(csvs_dir, csv_file_name )
            self.df_concatenated = self.csv_to_dataframe(path_to_csvs)
            svm_fold_accuracies = []

            for q in range(folds):  #looping thru folds obtained thru StratKFold() SVM
                X_train_stratKfold_values2 = self.df_concatenated.iloc[all_straKfold_indx_dict['entrenamiento_'+str(q)], 1:]
                #X_train_stratKfold_values2 = StandardScaler().fit_transform(X_train_stratKfold_values2)
                #X_train_stratKfold_values2 = self.square_root_norm(X_train_stratKfold_values2)
                y_train_stratKfold_labels2 = self.df_concatenated.iloc[all_straKfold_indx_dict['entrenamiento_'+str(q)], 0]

                x_test_stratKfold_values2 = self.df_concatenated.iloc[all_straKfold_indx_dict['validacion_'+str(q)], 1:]
                #x_test_stratKfold_values2 = StandardScaler().fit_transform(x_test_stratKfold_values2)
                #x_test_stratKfold_values2 = self.square_root_norm(x_test_stratKfold_values2)
                x_test_stratKfold_valuesNP2 = x_test_stratKfold_values2.to_numpy()
                y_test_stratKfold_labels2 = self.df_concatenated.iloc[all_straKfold_indx_dict['validacion_'+str(q)], 0]
                X_train_stratKfold_valuesNP2 = X_train_stratKfold_values2.to_numpy()

                '''
                #Code for PARAMETER TUNING
                #List Hyperparameters that we want to tune.
                param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001, 0.0001],'kernel': ['rbf', 'poly', 'sigmoid']}
                
                #Create new svm object
                #knn_2 = KNeighborsClassifier()
                
                #Use GridSearch
                grid = GridSearchCV(SVC(), param_grid, cv=5, refit=True, verbose=2, scoring='accuracy', n_jobs = 4)
                grid.fit(X_train_stratKfold_valuesNP2, y_train_stratKfold_labels2.ravel())
    
                #Print The value of best Hyperparameters                
                print(grid.best_estimator_)
                
                #END PARAMETER TUNING
                '''

                #Create a svm Classifier
                #clf = svm.SVC(C=100, gamma=1, kernel='poly') # Linear Kernel
                clf = svm.SVC(C=10, gamma=0.01, kernel='rbf')

                #Train the model using the training sets
                clf.fit(X_train_stratKfold_valuesNP2, y_train_stratKfold_labels2.ravel())
                #Predict the response for test dataset
                y_pred_stratKfold_svm= clf.predict(x_test_stratKfold_valuesNP2)

                # Model Accuracy: how often is the classifier correct?
                fold_accuracy_svm = metrics.accuracy_score(y_test_stratKfold_labels2, y_pred_stratKfold_svm)
                svm_fold_accuracies.append(fold_accuracy_svm)

            accuracy_results += layers[p] + ' ' + str(mean(svm_fold_accuracies)) + '\n '


        ### NAIVE BAYES
        accuracy_results += '\nStratified Naive Bayes'+str(folds)+'-Folds \n '
        trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform') ###!!!!

        for v in range(len(layers)): #looping thru layers for Stratified naive bayes
            csv_file_name = layers[v]

            path_to_csvs = os.path.join(csvs_dir, csv_file_name )
            self.df_concatenated = self.csv_to_dataframe(path_to_csvs)
            NB_fold_accuracies = []

            for w in range(folds):  #looping thru folds obtained thru StratKFold() SVM
                X_train_stratKfold_values3 = self.df_concatenated.iloc[all_straKfold_indx_dict['entrenamiento_'+str(w)], 1:]

                #trans.fit(X_train_stratKfold_values3)
                #X_train_stratKfold_values3 = trans.transform(X_train_stratKfold_values3)
                y_train_stratKfold_labels3 = self.df_concatenated.iloc[all_straKfold_indx_dict['entrenamiento_'+str(w)], 0]


                x_test_stratKfold_values3 = self.df_concatenated.iloc[all_straKfold_indx_dict['validacion_'+str(w)], 1:]
                trans.fit(x_test_stratKfold_values3)

                #x_test_stratKfold_values3 = trans.transform(x_test_stratKfold_values3)
                #x_test_stratKfold_valuesNP3 = x_test_stratKfold_values3.to_numpy()
                y_test_stratKfold_labels3 = self.df_concatenated.iloc[all_straKfold_indx_dict['validacion_'+str(w)], 0]
                #X_train_stratKfold_valuesNP3 = X_train_stratKfold_values3.to_numpy()

                '''
                ### NAIVE BAYES PARAMETER TUNING

                param_grid_nb = {    'var_smoothing': np.logspace(0,-9, num=100) }
                #from sklearn.naive_bayes import GaussianNB
                #from sklearn.model_selection import GridSearchCV
                nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
                nbModel_grid.fit(X_train_stratKfold_values3, y_train_stratKfold_labels3.ravel())
                print(nbModel_grid.best_estimator_)

                ###END PARAMETER TUNING
                '''

                #Create a naives gaussian Classifier
                gnb = GaussianNB(priors=None, var_smoothing=1e-09)
                #gnb = GaussianNB(priors=None, var_smoothing=0.001) ## PARAMETERS FROM TUNING
                #gnb = MultinomialNB()#alpha=1.0, fit_prior=True, class_prior=None)
                #gnb = ComplementNB()
                #Train the model using the training sets
                gnb.fit(X_train_stratKfold_values3, y_train_stratKfold_labels3.ravel())
                #Predict the response for test dataset
                y_predNB = gnb.predict(x_test_stratKfold_values3)
                probabilities = gnb.predict_proba(x_test_stratKfold_values3)

                timestr = time.strftime("%Y%m%d-%H%M%S")
                csv_file_name_prob = 'Prob_layer_'+str(v)+'_fold_'+str(w)+'_'+ timestr + '.csv'
                csv_file_name_test_features = 'features_layer_'+str(v)+'_fold_'+str(w)+'_'+ timestr + '.csv'
                csv_file_name_test_labels = 'labels_layer_'+str(v)+'_fold_'+str(w)+'_'+ timestr + '.csv'
                save_dir_prob ='/Users/G/Downloads/VAE/Git/Datasets/Visual_Attributes/ssearch/probabilities'
                self.saved_file_name_test_features = os.path.join(save_dir_prob, csv_file_name_test_features )
                self.saved_file_name_prob = os.path.join(save_dir_prob, csv_file_name_prob )
                self.saved_file_name_test_labels = os.path.join(save_dir_prob, csv_file_name_test_labels )
                savetxt(self.saved_file_name_prob, probabilities, fmt='%s', delimiter=',')
                savetxt(self.saved_file_name_test_features, x_test_stratKfold_values3, fmt='%s', delimiter=',')
                savetxt(self.saved_file_name_test_labels, y_test_stratKfold_labels3, fmt='%s', delimiter=',')

                # Model Accuracy: how often is the classifier correct?
                fold_accuracy_NB = metrics.accuracy_score(y_test_stratKfold_labels3, y_predNB)
                NB_fold_accuracies.append(fold_accuracy_NB)


            accuracy_results += layers[v] + ' ' + str(mean(NB_fold_accuracies)) + '\n '


        accuracy_results += '\n kNN with simple 80/20 Split \n '

        for i in range(len(layers)):  #looping thru layers for 80% Train 20% Test accuracy kNN
            csv_file_name = layers[i]
            path_to_csvs = os.path.join(csvs_dir, csv_file_name )
            self.df_concatenated = self.csv_to_dataframe(path_to_csvs)
            train_80_20_data = self.df_concatenated.iloc[train_80_20_indices, :] #slicing the dataframe to select the train_80_20_data rows
            X_train_80_20_values = train_80_20_data.iloc[:, 1:].to_numpy() # separating data for training
            #X_train_80_20_values = StandardScaler().fit_transform(X_train_80_20_values)
            X_train_80_20_values = self.square_root_norm(X_train_80_20_values)
            y_train_80_20_labels = train_80_20_data.iloc[:, 0].to_numpy() # separating labels for training
            test_80_20_data = self.df_concatenated.iloc[test_80_20_indices, :] #slicing the dataframe to select the test_80_20_data rows
            x_test_80_20_values = test_80_20_data.iloc[:, 1:].to_numpy() # separating data for testing
            #x_test_80_20_values = StandardScaler().fit_transform(x_test_80_20_values)
            x_test_80_20_values = self.square_root_norm(x_test_80_20_values)
            y_test_80_20_labels = test_80_20_data.iloc[:, 0].to_numpy() # separating labels for testing

            knn82 = KNeighborsClassifier(n_neighbors = k, p=2, weights='distance')
            knn82.fit(X_train_80_20_values, y_train_80_20_labels.ravel())
            y_pred = knn82.predict(x_test_80_20_values)
            accuracy_results += layers[i] + ' ' + str(accuracy_score(y_test_80_20_labels, y_pred)) + '\n '

        accuracy_results += '\n Naive Bayes with simple 80/20 Split \n '

        for p in range(len(layers)):  #looping thru layers for 80% Train 20% Test accuracy kNN
            csv_file_name = layers[p]
            path_to_csvs = os.path.join(csvs_dir, csv_file_name )
            self.df_concatenated = self.csv_to_dataframe(path_to_csvs)
            train_80_20_data = self.df_concatenated.iloc[train_80_20_indices, :] #slicing the dataframe to select the train_80_20_data rows
            X_train_80_20_values = train_80_20_data.iloc[:, 1:].to_numpy() # separating data for training
            #X_train_80_20_values = StandardScaler().fit_transform(X_train_80_20_values)
            X_train_80_20_values = self.square_root_norm(X_train_80_20_values)
            y_train_80_20_labels = train_80_20_data.iloc[:, 0].to_numpy() # separating labels for training
            test_80_20_data = self.df_concatenated.iloc[test_80_20_indices, :] #slicing the dataframe to select the test_80_20_data rows
            x_test_80_20_values = test_80_20_data.iloc[:, 1:].to_numpy() # separating data for testing
            #x_test_80_20_values = StandardScaler().fit_transform(x_test_80_20_values)
            x_test_80_20_values = self.square_root_norm(x_test_80_20_values)
            y_test_80_20_labels = test_80_20_data.iloc[:, 0].to_numpy() # separating labels for testing

            #knn82 = KNeighborsClassifier(n_neighbors = k, p=2, weights='distance')
            gnb = GaussianNB()
            #knn82.fit(X_train_80_20_values, y_train_80_20_labels.ravel())
            gnb.fit(X_train_80_20_values, y_train_80_20_labels.ravel())
            #y_pred = knn82.predict(x_test_80_20_values)
            y_pred = gnb.predict(x_test_80_20_values)
            #accuracy_results += layers[p] + ' ' + str(accuracy_score(y_test_80_20_labels, y_pred)) + '\n '
            accuracy_results += layers[p] + ' ' + str(accuracy_score(y_test_80_20_labels, y_pred)) + '\n '


        return accuracy_results

    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]
        
    def compute_features(self, image, expand_dims = False):
        #image = image - self.mean_imag
        if expand_dims :
            image = tf.expand_dims(image, 0)  # add a missing dimension so it fits the predict?
        fv = self.sim_model.predict(image) # run the batch of images thru the model get predictions return
        return fv
    
    def normalize(self, data) :
        """
        unit normalization
        """
        norm = np.sqrt(np.sum(np.square(data), axis = 1)) # square element, sum rows, square root of total result in norm
        norm = np.expand_dims(norm, 0) #the axis determines where the new dimension will be inserted
        data = data / np.transpose(norm)
        return data

    def square_root_norm(self, data) :
        return self.normalize(np.sign(data)*np.sqrt(np.abs(data))) # returns an array square elements with original sign

    def search(self, im_query, metric = 'l2', norm = 'None'):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims = True)
        #it seems that Euclidean performs better than cosine ## makes me think of knn
        if metric == 'l2' :
            data = self.features
            query =q_fv            
            if norm == 'square_root' :  # it was set to norm on the config right after pargs when it runs
                data = self.square_root_norm(data)
                print('this is the type for data after norm \n', type(data), data.shape)
                query = self.square_root_norm(query)
            d = np.sqrt(np.sum(np.square(data - query[0]), axis = 1))
            idx_sorted = np.argsort(d)
            print('this is the fist 20 of d sorted')
            neighbors = d[idx_sorted][:5]
            print(d[idx_sorted][:20])
        elif metric == 'cos' : 
            sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
            sim = np.reshape(sim, (-1))            
            idx_sorted = np.argsort(-sim)
            print(sim[idx_sorted][:20])                
        return idx_sorted[:90]
        
                                
    def compute_features_from_catalog(self):

        n_batch = self.configuration.get_batch_size()
        images = np.empty((self.data_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = np.float32)

        for i, filename in enumerate(self.filenames) :
            if i % 1000 == 0:
                print('reading {}'.format(i))
                sys.stdout.flush()
            images[i, ] = self.read_image(filename)        
        n_iter = np.int(np.ceil(self.data_size / n_batch))
        result = []
        for i in range(n_iter) :
            print('iter {} / {}'.format(i, n_iter))  
            sys.stdout.flush()             
            batch = images[i*n_batch : min((i + 1) * n_batch, self.data_size), ]
            result.append(self.compute_features(batch)) # here it calls the function that runs it thru the model
        fvs = np.concatenate(result)
        timestr = time.strftime("%Y%m%d-%H%M%S")

        # start compute csv
        fvs_for_csv = np.empty((0, fvs.shape[1]+1), object)
        for i, filename in enumerate(self.filenames):
            if i % 1000 == 0:
                print('processing csv files {}'.format(i))
                sys.stdout.flush()
            fvs_temp = fvs[i].astype(object)
            fvs_temp = np.expand_dims(fvs_temp, axis=0)
            arr_filename = [filename]
            fvs_temp = np.insert(fvs_temp, 0, arr_filename, axis=1)

            fvs_for_csv = np.append(fvs_for_csv, fvs_temp, axis=0)
            #timestr = time.strftime("%Y%m%d-%H%M%S")
            csv_file_name = '_'+self.output_layer_name+'_' + timestr + '.csv'

            self.saved_file_name = os.path.join(self.ssearch_dir, csv_file_name )

        savetxt(self.saved_file_name, fvs_for_csv, fmt='%s', delimiter=',')
        print('csv saved at {}'.format(self.saved_file_name))
        # end compute csv

        print('fvs {}'.format(fvs.shape))

        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)       
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def draw_result(self, filenames):
        w = 1000
        h = 1000
        w_i = np.int(w / 10)
        h_i = np.int(h / 10)
        image_r = np.zeros((w,h,3), dtype = np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames) :
            pos = (i * w_i)
            x = pos % w
            y = np.int(np.floor(pos / w)) * h_i
            image = data.read_image(filename, 3) #
            image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
            image_r[y:y+h_i, x : x +  w_i, :] = image              
        return image_r


    def plotModelfun(self):
        plot_model(self.modelForPrint, to_file='VAEmodel_plot.png', show_shapes=True, show_layer_names=True)


#unit test
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)                
    parser.add_argument("-mode", type=str, choices = ['search', 'compute'], help=" mode of operation", required = True)
    parser.add_argument("-list", type=str,  help=" list of image to process", required = False)
    parser.add_argument("-odir", type=str,  help=" output dir", required = False, default = '.')
    pargs = parser.parse_args()     
    configuration_file = pargs.config        
    ssearch = SSearch(pargs.config, pargs.name) #passing parameters from terminal + model name
    metric = 'l2'
    norm = 'square_root'

    if pargs.mode == 'compute' :        
        ssearch.compute_features_from_catalog()
        ssearch.load_features()
        print('layers accuracy: \n', ssearch.find_layer_accuracy(11, 5))


    if pargs.mode == 'search' :
        print('went into search')
        ssearch.load_features()        
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames = [ item.strip() for item in f_list]


            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query, metric, norm)
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)#
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)

                def most_common(lst):
                    return max(set(lst), key=lst.count)

                print('result saved at {} '.format(output_name))

        else :
            print('went into else')
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                print_im_query = np.expand_dims(im_query, axis=0) # added the dimension it expects
                idx = ssearch.search(im_query, metric, norm)
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_{}_{}_result.png'.format(metric, norm, ssearch.output_layer_name)
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)

                print('result saved at {}'.format(output_name))

                fquery = input('Query:')
        
