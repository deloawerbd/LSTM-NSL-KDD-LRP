import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, Normalizer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
import h5py

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Dropout, Activation, Embedding


def data_normalization():
    ##### preprocess the NSL-KDD dataset ####
    dataset_train=pd.read_csv('kdd_train.csv')
    col_names = list(dataset_train.columns)
    dataset_test=pd.read_csv('kdd_test.csv')

    ####Identify categorical features############

    # colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
    # explore categorical features

    for col_name in dataset_train.columns:
        if dataset_train[col_name].dtypes == 'object' :
            unique_cat = len(dataset_train[col_name].unique())
            #print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    #isee how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
    dataset_train['service'].value_counts().sort_values(ascending=False).head()

    #Test set
    #print('Test set:')
    for col_name in dataset_test.columns:
        if dataset_test[col_name].dtypes == 'object' :
            unique_cat = len(dataset_test[col_name].unique())
            #print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    ########  LabelEncoder.  Insert categorical features into a 2D numpy array ###########
    categorical_columns=['protocol_type', 'service', 'flag']
    # insert code to get a list of categorical columns into a variable, categorical_columns
    categorical_columns=['protocol_type', 'service', 'flag']



    dataset_train_categorical_values = dataset_train[categorical_columns]
    dataset_train_categorical_values.head()


    dataset_test_categorical_values = dataset_test[categorical_columns]
    dataset_test_categorical_values.head()

    # print("============== End Categorial 2d array   ========================")

    ######### Column protocol type ##########

    unique_protocol=sorted(dataset_train.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2=[string1 + str(x) for x in unique_protocol]

    # service training set
    unique_service=sorted(dataset_train.service.unique())
    string2 = 'service_'
    unique_service2=[string2 + str(x) for x in unique_service]


    # service for test set
    unique_service_t=sorted(dataset_test.service.unique())
    string_t = 'service_'
    unique_service2_t=[string_t + str(x) for x in unique_service_t]


    # flag
    unique_flag=sorted(dataset_train.flag.unique())
    string3 = 'flag_'
    unique_flag2=[string3 + str(x) for x in unique_flag]


    #put together
    datacols=unique_protocol2 + unique_service2 + unique_flag2

    #do same for test set
    unique_service_test=sorted(dataset_test.service.unique())
    unique_service2_test=[string2 + x for x in unique_service_test]


    testdatacols=unique_protocol2 + unique_service2_test + unique_flag2

    ########### Transform categorical features into numbers using LabelEncoder()   ###############
    #Transform categorical features into numbers using LabelEncoder()
    dataset_train_categorical_values_enc=dataset_train_categorical_values.apply(LabelEncoder().fit_transform)

    ### test set ####
    dataset_test_categorical_values_enc=dataset_test_categorical_values.apply(LabelEncoder().fit_transform)

    ################## One-Hot-Encoding   ##########################

    enc = OneHotEncoder()
    dataset_train_categorical_values_encenc = enc.fit_transform(dataset_train_categorical_values_enc)

    dataset_train_categorical_values_encenc.toarray()

    dataset_train_cat_data = pd.DataFrame(dataset_train_categorical_values_encenc.toarray(),columns=datacols)

    # test set
    dataset_test_categorical_values_encenc = enc.fit_transform(dataset_test_categorical_values_enc)

    dataset_test_cat_data = pd.DataFrame(dataset_test_categorical_values_encenc.toarray(),columns=testdatacols)


    ############## Add 6 missing categories from train set to test set ( 06-06-2021)  ###################
    #print("############### Add 6 missing categories from train set to test set  ###############")
    trainservice=dataset_train['service'].tolist()
    testservice= dataset_test['service'].tolist()


    #Join encoded categorical dataframe with the non-categorical dataframe
    dataset_new=dataset_train.join(dataset_train_cat_data)
    dataset_new.drop('flag', axis=1, inplace=True)
    dataset_new.drop('protocol_type', axis=1, inplace=True)
    dataset_new.drop('service', axis=1, inplace=True)
    dataset_new.shape

    #test data

    dataset_new_test=dataset_test.join(dataset_test_cat_data)

    dataset_new_test.drop('flag', axis=1, inplace=True)
    dataset_new_test.drop('protocol_type', axis=1, inplace=True)
    dataset_new_test.drop('service', axis=1, inplace=True)


    dataset_new_test.insert(loc = 64, column = 'service_harvest',value = 0)
    dataset_new_test.insert(loc = 67, column = 'service_http_2784',value = 0)
    dataset_new_test.insert(loc = 69, column = 'service_http_8001',value = 0)
    dataset_new_test.insert(loc = 103, column = 'service_tftp_u',value = 0)

    ### Rename every attack label: 0=normal, 1=DoS, 2=Probe, 3=R2L and 4=U2R  ###

    # take label column
    label_rn_data=dataset_new['labels']
    label_rn_data_test=dataset_new_test['labels']
    # change the label column
    new_label_data=label_rn_data.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                               'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                               ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                               'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
    new_label_data_test=label_rn_data_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                               'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                               ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                               'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
    # put the new label column back
    dataset_new['labels'] = new_label_data
    dataset_new_test['labels'] = new_label_data_test

    X1 = dataset_new.iloc[:,0:38]
    X2 = dataset_new.iloc[:,39:]
    X = X1.join(X2)
    Y = dataset_new.iloc[:,38]

    T1 = dataset_new_test.iloc[:,0:38]
    T2 = dataset_new_test.iloc[:,39:]
    T = T1.join(T2)
    C = dataset_new_test.iloc[:,38]

    scaler = Normalizer().fit(X)
    trainX = scaler.transform(X)

    # summarize transformed data
    np.set_printoptions(precision=3)

    scaler = Normalizer().fit(T)
    testT = scaler.transform(T)

    ##  summarize transformed data
    np.set_printoptions(precision=3)

    ## normal, Dos, Probe, R2L, U2R
    ## Ormalizing the train label 
    main_label_name = ['normal', 'Dos', 'Probe', 'R2L','U2R']
    normal_arr = []
    Dos_arr = []
    Probe_arr = []
    R2L_arr = []
    U2R_arr = []

    all_labels = pd.DataFrame(Y)

    for ite in range(len(all_labels)):

        if all_labels.loc[ite, "labels"] == 0:
            normal_arr.append(1)
            Dos_arr.append(0)
            Probe_arr.append(0)
            R2L_arr.append(0)
            U2R_arr.append(0)
        elif all_labels.loc[ite, "labels"] == 1:    
            normal_arr.append(0)
            Dos_arr.append(1)
            Probe_arr.append(0)
            R2L_arr.append(0)
            U2R_arr.append(0)
        elif all_labels.loc[ite, "labels"] == 2:
            normal_arr.append(0)
            Dos_arr.append(0)
            Probe_arr.append(1)
            R2L_arr.append(0)
            U2R_arr.append(0)
        elif all_labels.loc[ite, "labels"] == 3:
            normal_arr.append(0)
            Dos_arr.append(0)
            Probe_arr.append(0)
            R2L_arr.append(1)
            U2R_arr.append(0)
        elif all_labels.loc[ite, "labels"] == 4:
            normal_arr.append(0)
            Dos_arr.append(0)
            Probe_arr.append(0)
            R2L_arr.append(0)
            U2R_arr.append(1)
        else:
            normal_arr.append(0)
            Dos_arr.append(0)
            Probe_arr.append(0)
            R2L_arr.append(0)
            U2R_arr.append(0)
    label_y_train_data = pd.DataFrame(list(zip(normal_arr,Dos_arr,Probe_arr,R2L_arr,U2R_arr))) 
    label_y_train_data.columns = main_label_name

    test_normal_arr = []
    test_Dos_arr = []
    test_Probe_arr = []
    test_R2L_arr = []
    test_U2R_arr = []

    test_all_labels = pd.DataFrame(C)

    for it in range(len(test_all_labels)):
        if test_all_labels.loc[it, "labels"] == 0:
            test_normal_arr.append(1)
            test_Dos_arr.append(0)
            test_Probe_arr.append(0)
            test_R2L_arr.append(0)
            test_U2R_arr.append(0)
        elif test_all_labels.loc[it, "labels"] == 1:
            test_normal_arr.append(0)
            test_Dos_arr.append(1)
            test_Probe_arr.append(0)
            test_R2L_arr.append(0)
            test_U2R_arr.append(0)
        elif test_all_labels.loc[it, "labels"] == 2:
            test_normal_arr.append(0)
            test_Dos_arr.append(0)
            test_Probe_arr.append(1)
            test_R2L_arr.append(0)
            test_U2R_arr.append(0)
        elif test_all_labels.loc[it, "labels"] == 3:
            test_normal_arr.append(0)
            test_Dos_arr.append(0)
            test_Probe_arr.append(0)
            test_R2L_arr.append(1)
            test_U2R_arr.append(0)
        elif test_all_labels.loc[it, "labels"] == 4:
            test_normal_arr.append(0)
            test_Dos_arr.append(0)
            test_Probe_arr.append(0)
            test_R2L_arr.append(0)
            test_U2R_arr.append(1)
        else:
            test_normal_arr.append(0)
            test_Dos_arr.append(0)
            test_Probe_arr.append(0)
            test_R2L_arr.append(0)
            test_U2R_arr.append(0)

    label_y_test_data = pd.DataFrame(list(zip(test_normal_arr,test_Dos_arr,test_Probe_arr,test_R2L_arr,test_U2R_arr)))
    label_y_test_data.columns = main_label_name  

    y_train= label_y_train_data
    y_test= label_y_test_data

    X_train = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
    X_test = np.reshape(testT, (testT.shape[0],1, testT.shape[1]))
    
    return X_train, X_test, y_train, y_test

def my_lstm_network(batch_size, epochs, input_neurons, hidden_neurons, output_neurons,dropout, X_train, X_test, y_train, y_test):
    model = Sequential()
    # Input layers (LSTM)
    model.add(LSTM(input_neurons, input_shape=(1, 122), return_sequences=True, return_state = False))
    model.add(LSTM(input_neurons, return_sequences=False, return_state=False))
    model.add(Dropout(dropout))
    # Hidden layers
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dropout(dropout))
    # Output layers
    model.add(Dense(output_neurons, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])
    history =  model.fit(X_train, y_train, batch_size=batch_size, epochs = epochs, validation_data=(X_test, y_test) )
    loss, accuracy = model.evaluate(X_test, y_test)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

    ## Plot the loss  ##
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history.history['val_loss'], 'r', label='Validation', linewidth=2)

    ax.set_title('Model Loss', fontsize=20)
    ax.set_ylabel('Loss (MAE)')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper right')
    plt.show()

    
 
#get the normalize data 
X_train, X_test, y_train, y_test = data_normalization()


batch_size = 122
epochs = 5
neurons_in_input_layers = 64
neurons_in_hidden_layers = 64
neurons_in_output_layers = 5
dropout_rate = .2
# Run calling funtion to run LSTM netwrok
my_lstm_network(batch_size, epochs, neurons_in_input_layers, neurons_in_hidden_layers, neurons_in_output_layers,dropout_rate, X_train, X_test, y_train, y_test)

