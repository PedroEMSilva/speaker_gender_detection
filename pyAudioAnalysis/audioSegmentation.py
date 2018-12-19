import numpy
import sklearn.cluster
import time
import scipy
import os
import pyAudioAnalysis.audioFeatureExtraction as aF
from audioTrainTest import trainSVM
from audioTrainTest import normalizeFeatures
import audioBasicIO
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.discriminant_analysis
import csv
import os.path
import sklearn
import sklearn.cluster
import hmmlearn.hmm
import cPickle
import glob

""" General utility functions """




def silenceRemoval(x, Fs, stWin, stStep, smoothWindow=0.5, Weight=0.5, plot=False):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - Fs:               sampling freq
         - stWin, stStep:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - Weight:           (optinal) weight factor (0 < Weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - segmentLimits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                    the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''

    if Weight >= 1:
        Weight = 0.99
    if Weight <= 0:
        Weight = 0.01

    # Step 1: feature extraction
    x = audioBasicIO.stereo2mono(x)                        # convert to mono
    ShortTermFeatures = aF.stFeatureExtraction(x, Fs, stWin * Fs, stStep * Fs)        # extract short-term features

    # Step 2: train binary SVM classifier of low vs high energy frames
    EnergySt = ShortTermFeatures[1, :]                  # keep only the energy short-term sequence (2nd feature)
    E = numpy.sort(EnergySt)                            # sort the energy feature values:
    L1 = int(len(E) / 10)                               # number of 10% of the total short-term windows
    T1 = numpy.mean(E[0:L1]) + 0.000000000000001                 # compute "lower" 10% energy threshold
    T2 = numpy.mean(E[-L1:-1]) + 0.000000000000001                # compute "higher" 10% energy threshold
    Class1 = ShortTermFeatures[:, numpy.where(EnergySt <= T1)[0]]         # get all features that correspond to low energy
    Class2 = ShortTermFeatures[:, numpy.where(EnergySt >= T2)[0]]         # get all features that correspond to high energy
    featuresSS = [Class1.T, Class2.T]                                    # form the binary classification task and ...

    [featuresNormSS, MEANSS, STDSS] = normalizeFeatures(featuresSS)   # normalize and ...
    SVM = trainSVM(featuresNormSS, 1.0)                               # train the respective SVM probabilistic model (ONSET vs SILENCE)

    # Step 3: compute onset probability based on the trained SVM
    ProbOnset = []
    for i in range(ShortTermFeatures.shape[1]):                    # for each frame
        curFV = (ShortTermFeatures[:, i] - MEANSS) / STDSS         # normalize feature vector
        ProbOnset.append(SVM.predict_proba(curFV.reshape(1,-1))[0][1])           # get SVM probability (that it belongs to the ONSET class)
    ProbOnset = numpy.array(ProbOnset)
    ProbOnset = smoothMovingAvg(ProbOnset, smoothWindow / stStep)  # smooth probability

    # Step 4A: detect onset frame indices:
    ProbOnsetSorted = numpy.sort(ProbOnset)                        # find probability Threshold as a weighted average of top 10% and lower 10% of the values
    Nt = ProbOnsetSorted.shape[0] / 10
    T = (numpy.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * numpy.mean(ProbOnsetSorted[-Nt::]))

    MaxIdx = numpy.where(ProbOnset > T)[0]                         # get the indices of the frames that satisfy the thresholding
    i = 0
    timeClusters = []
    segmentLimits = []

    # Step 4B: group frame indices to onset segments
    while i < len(MaxIdx):                                         # for each of the detected onset indices
        curCluster = [MaxIdx[i]]
        if i == len(MaxIdx)-1:
            break
        while MaxIdx[i+1] - curCluster[-1] <= 2:
            curCluster.append(MaxIdx[i+1])
            i += 1
            if i == len(MaxIdx)-1:
                break
        i += 1
        timeClusters.append(curCluster)
        segmentLimits.append([curCluster[0] * stStep, curCluster[-1] * stStep])

    # Step 5: Post process: remove very small segments:
    minDuration = 0.2
    segmentLimits2 = []
    for s in segmentLimits:
        if s[1] - s[0] > minDuration:
            segmentLimits2.append(s)
    segmentLimits = segmentLimits2

    if plot:
        timeX = numpy.arange(0, x.shape[0] / float(Fs), 1.0 / Fs)

        plt.subplot(2, 1, 1)
        plt.plot(timeX, x)
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.subplot(2, 1, 2)
        plt.plot(numpy.arange(0, ProbOnset.shape[0] * stStep, stStep), ProbOnset)
        plt.title('Signal')
        for s in segmentLimits:
            plt.axvline(x=s[0])
            plt.axvline(x=s[1])
        plt.title('SVM Probability')
        plt.show()

    return segmentLimits

def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = numpy.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1], inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
    w = numpy.ones(windowLen, 'd')
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[windowLen:-windowLen+1]
