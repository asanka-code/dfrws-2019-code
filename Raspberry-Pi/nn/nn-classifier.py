from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


trainingTraces = "../data/training-samples"


###############################################################################
# need a function to load data from .npy trace files into the X 2d array and fill
# the relevant element in Y array.
def loadDataToXY():
    pathToNpyFiles = trainingTraces

    # A dictionary to encode the crypto algorithm label with a number 
    cryptoDict = dict()
    # A variable to keep a unique number for each crypto algorithm
    cryptoAlgoCounter = 0
    
    X = []
    Y = [] 

    listOfFiles = os.listdir(pathToNpyFiles)  

    print("number of traces: %d" % len(listOfFiles))

    for fileName in listOfFiles:
        #print (fileName)
        cryptoName, sequenceNumber, extension = fileName.split(".")    
        
        if cryptoName not in cryptoDict:
            cryptoDict[cryptoName] = cryptoAlgoCounter
            cryptoAlgoCounter = cryptoAlgoCounter + 1
            
        fftloaded = np.load(pathToNpyFiles+"/"+fileName)
        fftTrace = fftloaded.tolist()
        X.extend([fftTrace])
        Y.append(cryptoDict[cryptoName])
            
        #if len(X)==800:
        #    break

    print("cryptoDict=", cryptoDict)
    #print("type(X)=", type(X))
    #print("len(X)=", len(X))
    #print("type(Y)=", type(Y))
    #print("len(Y)=", len(Y))
    #print("X=",X)  
    #print("Y=",Y)       
    return X, Y


###############################################################################
# The function to perform training and testing of the model
def trainAndTest(X, Y):
    print("Selecting training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    print("Training the classifier...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict (X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    

###############################################################################
# The function to perform 10 fold cross-validation
def tenFoldCrossValidation(classifier, X_data, Y_labels):
    scores = cross_val_score(clf, X_data, Y_labels, cv=10)
    print scores




# training samples
print("Loading data...")
X, Y = loadDataToXY()

print("Creating classifier...")
# classifier
# Following setting worked perfectly for AES256 vs AES128 vs 3DES vs None
clf = MLPClassifier(solver='lbfgs', alpha=1e-20, hidden_layer_sizes=(10, 5), random_state=1)

# 10-fold cross-validation
tenFoldCrossValidation(clf, X, Y)

# Training and testing
#trainAndTest(X, Y)






