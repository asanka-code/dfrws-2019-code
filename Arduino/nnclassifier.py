from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
import os


def loadDataToXY(trainingFilePath):
    '''
    This function loads data from .npy trace files from the directory specified
    into the 2d array called X and fill the relevant class label in Y array.
    '''
    pathToNpyFiles = trainingFilePath
    X = []
    Y = [] 
    listOfFiles = os.listdir(pathToNpyFiles)  
    #print("number of traces: %d" % len(listOfFiles))
    for fileName in listOfFiles:
        cryptoName, sequenceNumber, extension = fileName.split(".")    
        fftloaded = np.load(pathToNpyFiles+"/"+fileName)
        fftTrace = fftloaded.tolist()
        X.extend([fftTrace])
        Y.append(cryptoName)
        #if len(X)==800:
        #    break      
    return X, Y

def trainAndTest(classifier, X, Y):
    '''
    The function to perform training and testing of the model
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict (X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
def tenFoldCrossValidation(classifier, X_data, Y_labels):
    '''
    The function to perform 10 fold cross-validation
    '''
    scores = cross_val_score(classifier, X_data, Y_labels, cv=10)
    print(scores)
    
def predictClass(classifier, x):
    y = classifier.predict(x)
    return y

def createClassifier():
    '''
    This function creates the the neural network classifier
    '''
    clf = MLPClassifier(solver='lbfgs', alpha=1e-20, hidden_layer_sizes=(10, 5), random_state=1)
    return clf





