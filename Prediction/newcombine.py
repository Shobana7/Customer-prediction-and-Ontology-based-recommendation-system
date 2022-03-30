import warnings
warnings.filterwarnings("ignore")


#setting up libraries and base configuration

import tensorflow as tf 
tf.compat.v1.set_random_seed(42)
from sklearn import metrics
from sklearn import svm
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np

# session_conf = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=1,
#     inter_op_parallelism_threads=1
# )

# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)


def load_datasets():

    # print("inside load datasets")

    LSTMTrain = pd.read_csv("LSTMTrainData.csv")
    LSTMTest = pd.read_csv("LSTMTestData.csv")

    # print(LSTMTrain.head(2))
    
    LSTMTrain.drop(LSTMTrain.columns[[0]], axis = 1, inplace = True)
    LSTMTest.drop(LSTMTest.columns[[0]], axis = 1, inplace = True)
     
    #testDataset.drop(testDataset.columns[[0]], axis = 1, inplace = True)
    
    return LSTMTrain, LSTMTest

def prepare_datasets(LSTMTrain, LSTMTest):
    print("inside prepare datasets")
    LSTMTrainX = LSTMTrain[LSTMTrain.columns[0:7]]
    LSTMTrainY = LSTMTrain[LSTMTrain.columns[7:8]]
    
    LSTMTestX = LSTMTest[LSTMTest.columns[0:7]]
    LSTMTestY = LSTMTest[LSTMTest.columns[7:8]]
    

    return LSTMTrainX, LSTMTrainY, LSTMTestX, LSTMTestY

def define_lstm_model():
    
    timesteps = 7
    input_dim = 1
    n_classes = 2
    epoch = 30
    batch_size = 16
    n_hidden = 30

    INPUT_ACTION_TYPES = [
        "Administrative",
        "Informational",
        "ProductRelated",
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
        "PageValues"
        ]

    LABELS = [
        "False",
        "True"
    ]

    lstm_model = Sequential()
    lstm_model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(1, activation='sigmoid'))
    
    lstm_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return lstm_model

def running_lstm_model(lstm_model, LSTMTrainX, LSTMTrainY, LSTMTestX, LSTMTestY):
    
    from sklearn.metrics import classification_report
    
    lstm_model.fit(LSTMTrainX, LSTMTrainY, batch_size = 16, epochs = 30)
    print("model fitted")
    predicted = lstm_model.predict(LSTMTestX)
    
    predicted[predicted<0.5]=0
    predicted[predicted>0.5]=1
    
    print(classification_report(LSTMTestY, predicted))
    print("finished with running_lstm module")
    return lstm_model

def call_for_lstm_train():
    #training the LSTM
    LSTMTrain, LSTMTest = load_datasets()
    LSTMTrainX, LSTMTrainY, LSTMTestX, LSTMTestY = prepare_datasets(LSTMTrain, LSTMTest)
    LSTMTrainX = LSTMTrainX.to_numpy()
    LSTMTestX = LSTMTestX.to_numpy()
# print("before reshape")
    
    LSTMTrainX = LSTMTrainX.reshape(8631, 7,1)
    LSTMTestX = LSTMTestX.reshape(3699,7,1)
# print("reshape success")
    lstm_model = define_lstm_model()
# print("model defined")
    lstm_model = running_lstm_model(lstm_model, LSTMTrainX, LSTMTrainY, LSTMTestX, LSTMTestY)
# print("comeback from running lstm")
    return lstm_model
 
def run_lstm_for_input(lstm_model):

    #runnign LSTM on interested datapoint
    testInput = pd.read_csv("TestInput.csv")
    # print("have read test input")

    #doing only the relevant steps of preprocessing
    testInput.ProductRelated[testInput.ProductRelated != 0] = 1
    testInput.ProductRelated[testInput.ProductRelated == 0] = 0

    testInput.Informational[testInput.Informational != 0] = 1
    testInput.Informational[testInput.Informational == 0] = 0

    testInput.Administrative[testInput.Administrative != 0] = 1
    testInput.Administrative[testInput.Administrative == 0] = 0  

    # print("some alterations")

    testInput = testInput.loc[:, ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "PageValues"]]
    
    # print("some more changes")

    testInput = np.array(testInput)
    testInput = testInput.reshape(1,7,1)

    # print("after reshaping inputdata")

    testOutput = lstm_model.predict(testInput)
    testOutput[testOutput<0.5]=0
    testOutput[testOutput>0.5]=1

    # print("after predicting before returning")
    print(testOutput)

    return testOutput


# outputLSTM, lstm_model = run_lstm_for_input()


def rf_load_datasets(trainFile, testFile):
    
    import pandas as pd
    trainDataset = pd.read_csv(trainFile)
    testDataset = pd.read_csv(testFile)
    
    trainDataset.drop(trainDataset.columns[[0]], axis = 1, inplace = True)
    testDataset.drop(testDataset.columns[[0]], axis = 1, inplace = True)
    
    return trainDataset, testDataset

#seperating test and train into features and target
def rf_prepare_datasets(trainDataset, testDataset):
    
    trainX = trainDataset[trainDataset.columns[0:15]]
    trainY = trainDataset[trainDataset.columns[15:16]]

    testX = testDataset[testDataset.columns[0:15]]
    testY = testDataset[testDataset.columns[15:16]]

    return trainX, trainY, testX, testY

def randomForest(DataTrainX, DataTrainY, DataTestX, DataTestY):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    rf = RandomForestClassifier(bootstrap= True, max_depth= 80, max_features= 3, min_samples_leaf= 3, min_samples_split= 8, n_estimators= 100)
    rf.fit(DataTrainX, DataTrainY)
    predictedData = rf.predict(DataTestX)
    
    print(classification_report(DataTestY, predictedData))
    
    return rf

def call_for_rf_train():
     #training the dataset
    trainData, testData = rf_load_datasets("RFTrainingData.csv", "RFTestingData.csv")    
    trainDataX, trainDataY, testDataX, testDataY = rf_prepare_datasets(trainData, testData)
 
    rf = randomForest(trainDataX, trainDataY, testDataX, testDataY)

    return rf


def run_rf_for_input(rf_model):

    #running the model on input instance
    testInput = pd.read_csv("TestInput.csv")
    testInput = testInput.loc[:, ["Month_Nov", "TrafficType_2", "TrafficType_3", "Month_May", "SpecialDay", "TrafficType_13", "OS_3", "TrafficType_1", "Month_Mar", "TrafficType_8", "BounceRates", "ExitRates", "Month_Feb", "ProductRelated_Duration", "PageValues"]]
    testOutput = rf_model.predict(testInput)
    
    index = np.where(testOutput==1)
    
    print(index)
    
    return testOutput

# testOutput, rf = run_rf_for_input()

##Case 1. LSTM should predict 0 for this customer
##Data Point : 8916,3,142.5,0,0.0,48,1052.255952,0.004347826,0.013043478,0.0,0.0,False,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0


# testInput = pd.read_csv("TestInput.csv")

# testInput.ProductRelated[testInput.ProductRelated != 0] = 1
# testInput.ProductRelated[testInput.ProductRelated == 0] = 0

# testInput.Informational[testInput.Informational != 0] = 1
# testInput.Informational[testInput.Informational == 0] = 0

# testInput.Administrative[testInput.Administrative != 0] = 1
# testInput.Administrative[testInput.Administrative == 0] = 0  

# testInput = testInput.loc[:, ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "PageValues"]]
    
# testInput = np.array(testInput)
# testInput = testInput.reshape(1,7,1)

# testOutput = lstm_model.predict(testInput)
# testOutput[testOutput<0.5]=0
# testOutput[testOutput>0.5]=1

# #will be easier for front end if all the outputs are 1d numpy arrays
# testOutput = testOutput.flatten()

# print(testOutput)


# ##Case 2. RF should return 0 for this customer
# ##Data Point : 8916,3,142.5,0,0.0,48,1052.255952,0.004347826,0.013043478,0.0,0.0,False,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0

# testInput = pd.read_csv("TestInput.csv")

# testInput = testInput.loc[:, ["Month_Nov", "TrafficType_2", "TrafficType_3", "Month_May", "SpecialDay", "TrafficType_13", "OS_3", "TrafficType_1", "Month_Mar", "TrafficType_8", "BounceRates", "ExitRates", "Month_Feb", "ProductRelated_Duration", "PageValues"]]

# testOutput = rf.predict(testInput)

# print(testOutput)

# ##Case 3. RF should return 1 for this customer
# #Data Point : 6290,1,29.2,0,0.0,0,0.0,0.0,0.066666667,0.0,0.0,True,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1


# testInput = pd.read_csv("TestInput.csv")

# testInput = testInput.loc[:, ["Month_Nov", "TrafficType_2", "TrafficType_3", "Month_May", "SpecialDay", "TrafficType_13", "OS_3", "TrafficType_1", "Month_Mar", "TrafficType_8", "BounceRates", "ExitRates", "Month_Feb", "ProductRelated_Duration", "PageValues"]]

# testOutput = rf.predict(testInput)

# print(testOutput)
