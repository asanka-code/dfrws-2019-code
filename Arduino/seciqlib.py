# Reference: http://witestlab.poly.edu/~ffund/el9043/labs/lab1.html


# includes core parts of numpy, matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn import preprocessing
import os

sampleRate=20000000.0

'''
# practice reading in complex values stored in a file
# Read in data that has been stored as raw I/Q interleaved 32-bit float samples
dat = np.fromfile("./data/with-aes.cfile", dtype="float32")
sampleRate=20000000

# Turn the interleaved I and Q samples into complex values
# the syntax "dat[0::2]" means "every 2nd value in 
# array dat starting from the 0th until the end"
dat = dat[0::2] + 1j*dat[1::2]

# Note: a quicker way to turn the interleaved I and Q samples  into complex values
# (courtesy of http://stackoverflow.com/a/5658446/) would be:
# dat = dat.astype(np.float32).view(np.complex64)
'''

#########################################################################

def getData(cfileName):
    """
    Given a name of a *.cfile, this function extracts the interleaved
    Inphase-Quadrature data samples and convert it into a numpy array of complex
    data elements. *.cfile format has interleaved I and Q samples where each sample
    is a float32 type. GNURadio Companion (GRC) scripts output data into a file
    though a file sink block in this format.
    Read more in SDR data types: https://github.com/miek/inspectrum
    """
    # Read the *.cfile which has each element in float32 format.
    data = np.fromfile(cfileName, dtype="float32")
    # Take each consecutive interleaved I sample and Q sample to create a single complex element.
    data = data[0::2] + 1j*data[1::2]
    
    print("data type=", type(data))
    
    # Return the complex numpy array.
    return data

def getSegmentData(cFileName, offsetTime, windowTime):
    '''
    Given a I-Q data file name, an offset value as a time, and an window time,
    this function extracts the required segment from the file and return it as
    a complx numpy array.    
    '''
    # converting offset and window from time to index values (sample_points = time x sample_rate)
    offset_index = offsetTime * sampleRate
    window_index = windowTime * sampleRate
    
    # converting the offset and window index values into byte values of the I-Q data file
    #Original bytes in file (offset_bytes) => array of float32 (4bytes in each element)
    #data = np.fromfile(cfileName, dtype="float32")
    #each pair of elements of the array is combined to create a single complex numpy array element.
    #data = data[0::2] + 1j*data[1::2]
    #Therefore, origina_file_size = numpy_array_length x 2 x 4
    offset_bytes = int(offset_index * 2 * 4)
    window_bytes = int(window_index * 2 * 4)
    
    #print("offset_bytes = ", offset_bytes)
    #print("window_index = ", window_index)
    #print("window_bytes = ", window_bytes)
    
    # reading the required segment of bytes from the file
    f = open(cFileName, 'rb')
    f.seek(offset_bytes,1)
    segment = f.read(window_bytes)
    #print("segment length = ", len(segment))

    data = np.frombuffer(segment, dtype="float32")
    data = data[0::2] + 1j*data[1::2]
    return data

def getTimeDuration(cFileName):
    '''
    Calculate the total time duration represented in an I-Q file.
    '''
    size_bytes = os.path.getsize(cFileName)

    #Original bytes in file (offset_bytes) => array of float32 (4bytes in each element)
    #data = np.fromfile(cfileName, dtype="float32")
    #each pair of elements of the array is combined to create a single complex numpy array element.
    #data = data[0::2] + 1j*data[1::2]
    #Therefore, origina_file_size = numpy_array_length x 2 x 4
    size_index = size_bytes / ( 2 * 4 )
    
    # sample_points = time x sample_rate
    size_time = size_index / sampleRate
    return size_time

def plotWaveform(data, show=1):
    """
    Given a data set as a complex numpy array, this function returns the waveform plot.
    """
    plt.figure()
    # Plot the waveform of the selected sample range of the numpy data array
    #plt.plot(data)
    plt.plot(np.abs(data))
    #plt.plot(np.real(data))
    #plt.plot(np.imag(data))
    
    if(show==1):
        plt.show()
    else:
        # Save the waveform into a PDF file
        plt.savefig('./wavform.pdf', fotmat='pdf', bbox_inches='tight')
        
    return 1

def plotScatter(data, show=1):
    """
    Given a data set as a complex numpy array, this function returns the scatter plot.
    """
    plt.figure()
    # plot the scatter plot of the selected sample range of the numpy data array
    plt.scatter(np.real(data), np.imag(data))
    plt.title("Constellation of the 'signal' loaded from file")
    
    if(show==1):
        plt.show()
    else:
        # save the scatter plot into a PDF file
        plt.savefig('./scatter.pdf', fotmat='pdf', bbox_inches='tight')
        
    return 1


def plotPSD(data, show=1):
    """
    Given a data set as a complex numpy array, this function returns the power spectral density (PSD) plot.
    """
    plt.figure()
    # plot the PSD of the selected sample range of the numpy data array
    plt.psd(data, NFFT=1024, Fs=sampleRate)
    
    if(show==1):
        plt.show()
    else:
        # save the PSD plot into a PDF file
        plt.savefig('./psd.pdf', fotmat='pdf', bbox_inches='tight')
    return 1

    
def plotFFT(data, show=1):
    """
    Given a data set as a complex numpy array, this function returns the FFT plot.
    """
    # get the length of the selected data sample range        
    N = len(data)
    # get the time interval beteween each sample
    T = 1.0 / sampleRate
    # calculate the FFT of the selected sample range. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data)
    # get the vector with frequencies of the sample range. But the output contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    freqs = fftfreq(N, T)
    # shift the frequencies to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    shifted_freqs = fftshift(freqs)
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[int(N/2):int(N)], yf[0:int(N/2)]))
    # plot the FFT vector against the frequencies
    plt.plot(shifted_freqs, np.abs(new_yf))    
    #print('len(shifted_freqs)=%d' % len(shifted_freqs))    
    #print('len(new_yf)=%d' % len(new_yf))    

    if(show==1):
        plt.show()
    else:
        # save theFFT plot as a PDF file.
        plt.savefig('./fft.pdf', fotmat='pdf', bbox_inches='tight')
        
    return 1


def plotSpectrogram(data, show=1):
    """
    Given a data set as a complex numpy array, this function returns the spectrogram plot.
    """
    # plot the spectrogram of the selected sample range
    #plt.specgram(data, NFFT=4096, Fs=sampleRate, cmap=plt.cm.get_cmap("Greys"))
    plt.specgram(data, NFFT=4096, Fs=sampleRate)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    #plt.axis('off')
    #ax = plt.axes()
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    
    # zoom in to the middle of the y-axis because 4 MHz band-pass in the GRC script has caused the
    # other y-axis frequency ranges to be highly attenuated and useless.
    plt.ylim(-2000000, 2000000)    
    
    if(show==1):
        plt.show()
    else:
        # save the spectrogram into a PDF file.
        plt.savefig('./spectrogram.pdf', fotmat='pdf', bbox_inches='tight', pad_inches=0)
        #plt.savefig('spectrogram-from-iq.pdf', fotmat='pdf', bbox_inches='tight')
        
    return 1


def getFeatureVector(data):
    """
    Given a data set as a complex numpy array, this function returns a 500 elements long feature vector.
    """
    N = len(data)
    # calculate the FFT of the selected sample range. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data)
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[int(N/2):int(N)], yf[0:int(N/2)]))
    fftdata = np.abs(new_yf)
    
    # DC spike at the center due to the nature of SDR should be removed
    N = len(fftdata)
    fftdata[int(N/2)] = 0
    
    # Use only the middle portion of the FFT vector as a feature vector
    #featureVector = fftdata[int(N/4):int(3*N/4)]
    #featureVector = fftdata[3*N/8:5*N/8]
    featureVector = fftdata       
       
    # Make the feature vector small by breaking and averaging into 500 buckets.   
    # lenth of the FFT vector we are considering
    L = len(featureVector)
    # number of buckets
    #l = 500
    l = 1000
    index = 0
    bucketSize = L/l
    vector = []
    while index<len(featureVector):
        #avg = sum(featureVector[index:index+int(bucketSize)])/len(featureVector[index:index+int(bucketSize)])
        #vector.append(avg)
        maxi = max(featureVector[index:index+int(bucketSize)])
        vector.append(maxi)    
    
        index = index + int(bucketSize)
    
    fft_normalized = preprocessing.normalize([vector], norm='l2')

    # get the normalized numpy array (we take the first dimention which is the correct array)
    feature_vector = fft_normalized[0]
    return feature_vector[0:l]

'''
def getFeatureVector(data):
    """
    Given a data set as a complex numpy array, this function returns a 500 elements long feature vector.
    """
    N = len(data)
    # calculate the FFT of the selected sample range. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data)
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[int(N/2):int(N)], yf[0:int(N/2)]))
    fftdata = np.abs(new_yf)
    
    # DC spike at the center due to the nature of SDR should be removed
    N = len(fftdata)
    fftdata[int(N/2)] = 0
    
    # Use only the middle portion of the FFT vector as a feature vector
    featureVector = fftdata[int(N/4):int(3*N/4)]
    #featureVector = fftdata[3*N/8:5*N/8]
       
    # Make the feature vector small by breaking and averaging into 500 buckets.   
    # lenth of the FFT vector we are considering
    L = len(featureVector)
    # number of buckets
    l = 500
    index = 0
    bucketSize = L/l
    vector = []
    while index<len(featureVector):
        avg = sum(featureVector[index:index+int(bucketSize)])/len(featureVector[index:index+int(bucketSize)])
        vector.append(avg)
        index = index + int(bucketSize)
    
    fft_normalized = preprocessing.normalize([vector], norm='l2')

    # get the normalized numpy array (we take the first dimention which is the correct array)
    feature_vector = fft_normalized[0]
    return feature_vector[0:l]
'''

###############################################################################
#                         Depreciated function                                #
###############################################################################

def getSegment(timeOffset, window):
    """
    Given a starting time offset (seconds) and a time window (seconds), this function
    returns the starting and ending sample indexes of a complex numpy array.
    """
    # Segment starting offset (sample points)
    start = timeOffset * sampleRate
    # Segment ending offset (sample points)
    end = start + (window * sampleRate)
    #print("start=%d", int(start))
    #print("end=%d", int(end))
    #Return the starting index and ending index
    return int(start), int(end)

def getFFTVector(data, timeOffset, window):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function returns the FFT vector as a numpy array.
    """
    # Get the desired starting and ending index of the numpy data array
    start, end = getSegment(timeOffset, window)    
    # get the length of the selected data sample range        
    N = len(data[start:end])
    
    print("segment length = ", N)
    
    # calculate the FFT of the selected sample range. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data[start:end])
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[int(N/2):int(N)], yf[0:int(N/2)]))
    # return the absolute values of the FFT vector.
    return np.abs(new_yf)


def getNormalizedFFTVector(data, timeOffset, window):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function generates the FFT vector as a numpy array and
    normalize it before returning it.
    """
    # get the FFT vector as a numpy array
    fftdata = getFFTVector(data,timeOffset, window)

    # DC spike at the center due to the nature of SDR should be removed
    N = len(fftdata)
    fftdata[N/2] = 0    
    
    # normalize the numpy array (note that we input the fftdata inside []. So, the
    # input data is basically a 2-D vector)
    fft_normalized = preprocessing.normalize([fftdata], norm='l2')
    # return normalized numpy array (we take the first dimention which is the correct array)
    return fft_normalized[0]

def getFullFFTVector(data):
    """
    Given a data set as a complex numpy array, this function returns the FFT vector as a numpy array.
    """
    # get the length of the data sample        
    N = len(data)
    # calculate the FFT of the sample. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data)
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[N/2:N], yf[0:N/2]))
    # return the absolute values of the FFT vector.
    return np.abs(new_yf)

def getBucketedNormalizedFFTVector(data):
    # time window in seconds
    window_time = 0.01
    window_length = window_time * sampleRate
    timeOffset = ((len(data)-1) - window_length) / sampleRate
    start, end = getSegment(timeOffset, window_time)
    fftdata = getFFTVector(data,timeOffset, window_time)
    
    
    # get the FFT vector as a numpy array
    #fftdata = getFullFFTVector(data)

    # DC spike at the center due to the nature of SDR should be removed
    N = len(fftdata)
    fftdata[int(N/2)] = 0
    
    # Use only the middle portion of the FFT vector as a feature vector
    featureVector = fftdata[int(N/4):int(3*N/4)]
    #featureVector = fftdata[3*N/8:5*N/8]
    
    
    # Make the feature vector small by breaking and averaging into 500 buckets.   

    # lenth of the FFT vector we are considering
    L = len(featureVector)
    # number of buckets
    l = 500
    
    index = 0
    bucketSize = L/l
    vector = []
    while index<len(featureVector):
        avg = sum(featureVector[index:index+int(bucketSize)])/len(featureVector[index:index+int(bucketSize)])
        vector.append(avg)
        index = index + int(bucketSize)
    
    #print("len(vector)=%d" % len(vector))
    #print("vector=", vector)
    fft_normalized = preprocessing.normalize([vector], norm='l2')
        
    
    # normalize the numpy array (note that we input the fftdata inside []. So, the
    # input data is basically a 2-D vector)
    #fft_normalized = preprocessing.normalize([fftdata], norm='l2')
    
    # return normalized numpy array (we take the first dimention which is the correct array)
    return fft_normalized[0]

def getBucketedNormalizedFFTVectorFromFile():
    '''
    # time window in seconds
    window_time = 0.01
    window_length = window_time * sampleRate
    timeOffset = ((len(data)-1) - window_length) / sampleRate
    #fftdata = getFFTVector(data,timeOffset, window_time)
    '''    
    # user defined variables
    offset_time = 0
    window_time = 0.01
    cFileName = "3des.dat"
    
    # converting offset and window from time to index values (sample points)
    offset_index = offset_time * sampleRate
    window_index = window_time * sampleRate
    
    # converting the offset and window index values into byte values of the I-Q data file
    offset_bytes = int(offset_index * 2 * 4)
    window_bytes = int(window_index * 2 * 4)
    
    print("offset_bytes = ", offset_bytes)
    print("window_index = ", window_index)
    print("window_bytes = ", window_bytes)
    
    # reading the required segment of bytes from the file
    f = open(cFileName, 'rb')
    f.seek(offset_bytes,1)
    segment = f.read(window_bytes)
    print("segment length = ", len(segment))

    data = np.frombuffer(segment, dtype="float32")
    data = data[0::2] + 1j*data[1::2]
    print("data type=", type(data))
    print("data length=", len(data))
    
    N = len(data)
     
    # calculate the FFT of the selected sample range. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data)
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[int(N/2):int(N)], yf[0:int(N/2)]))
    fftdata = np.abs(new_yf)
    
    # get the FFT vector as a numpy array
    #fftdata = getFullFFTVector(data)

    # DC spike at the center due to the nature of SDR should be removed
    N = len(fftdata)
    fftdata[int(N/2)] = 0
    
    # Use only the middle portion of the FFT vector as a feature vector
    featureVector = fftdata[int(N/4):int(3*N/4)]
    #featureVector = fftdata[3*N/8:5*N/8]
    
    
    # Make the feature vector small by breaking and averaging into 500 buckets.   

    # lenth of the FFT vector we are considering
    L = len(featureVector)
    # number of buckets
    l = 500
    
    index = 0
    bucketSize = L/l
    vector = []
    while index<len(featureVector):
        avg = sum(featureVector[index:index+int(bucketSize)])/len(featureVector[index:index+int(bucketSize)])
        vector.append(avg)
        index = index + int(bucketSize)
    
    #print("len(vector)=%d" % len(vector))
    #print("vector=", vector)
    fft_normalized = preprocessing.normalize([vector], norm='l2')
        
    
    # normalize the numpy array (note that we input the fftdata inside []. So, the
    # input data is basically a 2-D vector)
    #fft_normalized = preprocessing.normalize([fftdata], norm='l2')
    
    # return normalized numpy array (we take the first dimention which is the correct array)
    return fft_normalized[0]

###############################################################################
#                         Deleted function                                #
###############################################################################
    
'''
def getTimeDuration(data):
    """
    Given a data array, this function returns the time duration (seconds) of the data array.
    This return value of duration is a float.
    """
    # number of samples divided by sample rate gives the time duration of the total samples
    duration = len(data)/sampleRate
    #print("duration=", duration)
    # return the time duration of the dataset
    return duration
    
def plotWaveform(data, timeOffset, window, wavformFileName):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function returns the waveform plot as a PDF file.
    """
    # Get the desired starting and ending index of the numpy data array
    start, end = getSegment(timeOffset, window)
    plt.figure()
    # Plot the waveform of the selected sample range of the numpy data array
    plt.plot(data[start:end])
    # Save the waveform into a PDF file
    plt.savefig('./spectrograms/' + wavformFileName +'.pdf', fotmat='pdf', bbox_inches='tight')
    return 1

def plotScatter(data, timeOffset, window, scatterFileName):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function returns the scatter plot as a PDF file.
    """
    # Get the desired starting and ending index of the numpy data array
    start, end = getSegment(timeOffset, window)
    # And let's look at it on the complex plane
    # Note that showing *every* data point would be time- and processing-intensive
    # so we'll just show a few
    plt.figure()
    # plot the scatter plot of the selected sample range of the numpy data array
    plt.scatter(np.real(data[start:end]), np.imag(data[start:end]))
    plt.title("Constellation of the 'signal' loaded from file")
    # save the scatter plot into a PDF file
    plt.savefig('./spectrograms/' + scatterFileName +'.pdf', fotmat='pdf', bbox_inches='tight')
    return 1

def plotPSD(data, timeOffset, window, psdFileName):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function returns the power spectral density (PSD) plot as a PDF file.
    """
    # Get the desired starting and ending index of the numpy data array
    start, end = getSegment(timeOffset, window)
    plt.figure()
    # plot the PSD of the selected sample range of the numpy data array
    plt.psd(data[start:end], NFFT=1024, Fs=sampleRate)
    # save the PSD plot into a PDF file
    plt.savefig('./spectrograms/' + psdFileName +'.png', fotmat='pdf', bbox_inches='tight')
    return 1
    
def plotFFT(data, timeOffset, window, fftFileName):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function returns the FFT plot as a PDF file.
    """
    # Get the desired starting and ending index of the numpy data array
    start, end = getSegment(timeOffset, window)
    # get the length of the selected data sample range        
    N = len(data[start:end])
    # get the time interval beteween each sample
    T = 1.0 / sampleRate
    # calculate the FFT of the selected sample range. But the FFT x axis contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    yf = fft(data[start:end])
    # get the vector with frequencies of the sample range. But the output contains data
    # in the range from 0 to positive values first and at the end the negative values
    # like 0, 1, 2, 3, 4, -4, -3, -2, -1
    freqs = fftfreq(N, T)
    # shift the frequencies to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    shifted_freqs = fftshift(freqs)
    # rearrange the FFT vector to have it zero-centered, e.g., -4, -3, -2, -1, 0, 1, 2, 3, 4
    new_yf = np.concatenate((yf[N/2:N], yf[0:N/2]))
    # plot the FFT vector against the frequencies
    plt.plot(shifted_freqs, np.abs(new_yf))    
    #print('len(shifted_freqs)=%d' % len(shifted_freqs))    
    #print('len(new_yf)=%d' % len(new_yf))        
    # save theFFT plot as a PDF file.
    plt.savefig('./' + fftFileName +'.png', fotmat='pdf', bbox_inches='tight')
    return 1

def plotSpectrogram(data, timeOffset, window, specFileName):
    """
    Given a data set as a complex numpy array, a time offset (seconds), a time window (seconds)
    and a file name for the graph, this function returns the spectrogram plot as a PDF file.
    """
    # Get the desired starting and ending index of the numpy data array
    start, end = getSegment(timeOffset, window)   
    # plot the spectrogram of the selected sample range
    plt.specgram(data[start:end], NFFT=4096, Fs=sampleRate, cmap=plt.cm.get_cmap("Greys"))
    #plt.xlabel("Time (s)")
    #plt.ylabel("Frequency (MHz)")
    plt.axis('off')
    ax = plt.axes()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)    
    # zoom in to the middle of the y-axis because 4 MHz band-pass in the GRC script has caused the
    # other y-axis frequency ranges to be highly attenuated and useless.
    plt.ylim(-2000000, 2000000)    
    # save the spectrogram into a PDF file.
    plt.savefig('./spectrograms/' + specFileName +'.pdf', fotmat='PDF', bbox_inches='tight', pad_inches=0)
    #plt.savefig('spectrogram-from-iq.pdf', fotmat='pdf', bbox_inches='tight')
    return 1
'''