import seciqlib
import numpy as np
import os

bucketSize = 500

#cryptoAlgoName="none"

#if cryptoAlgoName == "aes":

'''
###############################################################################
# CRYPTO  
path_to_em_traces = "../data/em-traces/crypto"
path_to_training_samples = "../data/training-samples"
    
listOfFiles = os.listdir(path_to_em_traces)
print("number of traces: %d" % len(listOfFiles))
traceCounter = 0
    
for fileName in listOfFiles:
    data = seciqlib.getData(path_to_em_traces+"/"+fileName)
    fftVector = seciqlib.getBucketedNormalizedFFTVector(data)
    print(len(fftVector[0:bucketSize]))
    print(type(fftVector[0:bucketSize]))
    print(min(fftVector[0:bucketSize]))
    print(max(fftVector[0:bucketSize]))
    np.save(path_to_training_samples+"/crypto."+str(traceCounter), fftVector[0:bucketSize])  
    #np.save(path_to_training_samples+"/crypto."+str(traceCounter), fftVector) 
    traceCounter = traceCounter + 1
'''




###############################################################################
# AES-256
path_to_em_traces = "../data/em-traces/aes256"
path_to_training_samples = "../data/training-samples"
    
listOfFiles = os.listdir(path_to_em_traces)
print("number of traces: %d" % len(listOfFiles))
traceCounter = 0
    
for fileName in listOfFiles:
    data = seciqlib.getData(path_to_em_traces+"/"+fileName)
    fftVector = seciqlib.getBucketedNormalizedFFTVector(data)
    print(len(fftVector[0:bucketSize]))
    print(type(fftVector[0:bucketSize]))
    print(min(fftVector[0:bucketSize]))
    print(max(fftVector[0:bucketSize]))
    np.save(path_to_training_samples+"/aes256."+str(traceCounter), fftVector[0:bucketSize])  
    #np.save(path_to_training_samples+"/aes."+str(traceCounter), fftVector)  
    traceCounter = traceCounter + 1



#elif cryptoAlgoName == "3des":

###############################################################################
# 3DES  
path_to_em_traces = "../data/em-traces/3des"
path_to_training_samples = "../data/training-samples"
    
listOfFiles = os.listdir(path_to_em_traces)
print("number of traces: %d" % len(listOfFiles))
traceCounter = 0
    
for fileName in listOfFiles:
    data = seciqlib.getData(path_to_em_traces+"/"+fileName)
    fftVector = seciqlib.getBucketedNormalizedFFTVector(data)
    print(len(fftVector[0:bucketSize]))
    print(type(fftVector[0:bucketSize]))
    print(min(fftVector[0:bucketSize]))
    print(max(fftVector[0:bucketSize]))
    np.save(path_to_training_samples+"/3des."+str(traceCounter), fftVector[0:bucketSize])  
    #np.save(path_to_training_samples+"/3des."+str(traceCounter), fftVector) 
    traceCounter = traceCounter + 1

#elif cryptoAlgoName == "none":



###############################################################################
# AES-128
path_to_em_traces = "../data/em-traces/aes128"
path_to_training_samples = "../data/training-samples"
    
listOfFiles = os.listdir(path_to_em_traces)
print("number of traces: %d" % len(listOfFiles))
traceCounter = 0
    
for fileName in listOfFiles:
    data = seciqlib.getData(path_to_em_traces+"/"+fileName)
    fftVector = seciqlib.getBucketedNormalizedFFTVector(data)
    print(len(fftVector[0:bucketSize]))
    print(type(fftVector[0:bucketSize]))
    print(min(fftVector[0:bucketSize]))
    print(max(fftVector[0:bucketSize]))
    np.save(path_to_training_samples+"/aes128."+str(traceCounter), fftVector[0:bucketSize])  
    #np.save(path_to_training_samples+"/aes."+str(traceCounter), fftVector)  
    traceCounter = traceCounter + 1




###############################################################################
# NONE  
path_to_em_traces = "../data/em-traces/none"
path_to_training_samples = "../data/training-samples"
    
listOfFiles = os.listdir(path_to_em_traces)
print("number of traces: %d" % len(listOfFiles))
traceCounter = 0
    
for fileName in listOfFiles:
    data = seciqlib.getData(path_to_em_traces+"/"+fileName)
    fftVector = seciqlib.getBucketedNormalizedFFTVector(data)
    print(len(fftVector[0:bucketSize]))
    print(type(fftVector[0:bucketSize]))
    print(min(fftVector[0:bucketSize]))
    print(max(fftVector[0:bucketSize]))
    np.save(path_to_training_samples+"/none."+str(traceCounter), fftVector[0:bucketSize])  
    #np.save(path_to_training_samples+"/none."+str(traceCounter), fftVector) 
    traceCounter = traceCounter + 1

#else:
#    print("Nothing to do!")
