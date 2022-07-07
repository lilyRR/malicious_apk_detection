import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model, load_model
from dbn.tensorflow import SupervisedDBNClassification
from dbn.models import UnsupervisedDBN
from sklearn.svm import SVC, LinearSVC, NuSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

#from preTrain import *

#Execute pre-train
#pre_Train = PreTrain(DATASET_PATH)

#Get dataframes of ben/mal
benign_df = pd.read_csv('./Data/gram/begine/3_gram_b.csv')
malign_df = pd.read_csv('./Data/gram/malware/3_gram_malware.csv')

#--------------------------------------------------------------------------------------
# concatenating dataframe of benign and malign
frames = [benign_df,malign_df]
df = pd.concat(frames)
df.reset_index()

# dropping the first 2 columns as the first 2 columns are useless
df.drop(df.columns[[0, 0]], axis=1, inplace=True)
X = df.values
length = X.shape[0]
y = [1]*(length//2) + [0]*(length//2)

# Actual dimensions of feature
print("Actual Dimenions of feature space is ", X.shape)

# splitting the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
#----------------------------------------------------------------------------------------

#Process for training AutoEncoder
if LOAD_MODEL_TRAIN == True:
    start_time = dt.now()
    encoding_dim = 1000# dimension of the output
    # this is our input placeholder
    input_img = Input(shape=(X.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    #print(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(X.shape[1], activation='relu')(encoded)
    #print(decoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    #print(autoencoder)
    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
    #print(autoencoder)
    autoencoder.fit(
                    X_train, X_train,
                    validation_data=(X_test, X_test),
                    epochs=5,
                    batch_size=5,
                    shuffle=True,
                    validation_split=0.4,
                   )

    # saving the encoder with encoded as the output - we need this to readuce the dimension of the feature vector space
    our_encoder = Model(input_img, encoded)
    our_encoder.save('pickled/autoEncoder.mod')
    loaded_encoder = our_encoder
    print("Time for training AutoEncoder: ", dt.now() - start_time)

# reducing the dimension of the feature vector space from (12*186) to (12*8)
X_with_reduced_dimension = loaded_encoder.predict(X)
#print('Now the dimensions of the encoded input feature space is -- ',X_with_reduced_dimension.shape)

#Process for training DBN classifier
if LOAD_MODEL_TRAIN == True:
    start_time = dt.now()
    # Creating the classifier for the reduced dimension input to classify 
    # the Clasifier is a hybrid model which comprises of Unsupervised DBN followed by a SVM Classfier to predict the label
    svm = SVC()
    dbn = UnsupervisedDBN(hidden_layers_structure=[512, 256, 256, 512],
                          batch_size=10,
                          learning_rate_rbm=0.06,
                          n_epochs_rbm=20,
                          activation_function='relu')
    classifier = Pipeline(steps=[('dbn', dbn),
                                 ('svm', svm)])
    classifier.fit(X_with_reduced_dimension, y)
    f = open("pickled/DBNClassifier.pkl", "wb")
    pickle.dump(classifier, f)
    f.close()
    print("Time for training DBNClassifier: ", dt.now() - start_time)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8)
print("[*] Classification report is ")
#print(loaded_encoder.predict(X_test))
#print(Y_test)
print("[+] Accuracy score: ", accuracy_score(Y_test, classifier.predict(loaded_encoder.predict(X_test))))
print("[+] Classification Report")
print(classification_report(Y_test, classifier.predict(loaded_encoder.predict(X_test))))
print("[*] classification_report end")