"""
October 25, 2018
Author: Steven Block

TOF_Analyzer_DRS4.py
This program takes data from a DRS4 data file and computes:
Per Channel:
    Pulse height distributions
    Rise/Fall Times based on polarity

A Combination of possible Time of Flights to establish best two detectors.

Future work:
    - Multiple peaks per events
    - more configuration settings?
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import struct
import array
from uncertainties import ufloat
import pandas as pd
from math import *
from tkinter.filedialog import askopenfilename
import tkinter as tk
import os
import scipy.signal as scisig
from drs4 import DRS4BinaryFile





def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.nanmean(values)
    # Fast and numerically precise:
    std = np.nanstd(values)
    return [np.round(average,3), np.round(std,3)]


def Interpolator(X, Y, YValue, dTime, center):
    """
    Interpolate exact time Y == YValue using 4 points around closest data point to YValue

    Returns a tuple with time and error associated.
    """
    if YValue > np.mean(Y[:25]):
        TimeRightIndex = np.where(Y > YValue)[0][0] + 1
        TimeleftIndex = TimeRightIndex - 3
    else:
        TimeRightIndex = np.where(Y < YValue)[0][0] + 1
        TimeleftIndex= TimeRightIndex - 3
    Y1 = Y[TimeleftIndex]
    Y2 = Y[TimeRightIndex]
    X2 = X[TimeRightIndex]
    X1 = X[TimeleftIndex]
    slope = (Y2 - Y1) / (X2 - X1)
    if slope != 0:
        X0 = (YValue - Y1) / slope + X1
    else:
        X0 = np.nan
    return X0
def PulseHeightFinder(Y):
    return max(abs(np.mean(Y[:20]) - Y))

def PeakCalculation(X, Y):
    """
    Find peak location for proper interpolotion

    Returns a returns time that waveform hit 40% of peak value
    """
    # Channel1Data is from first TOF
    # Channel2Data is from second TOF
    dTimes = X
    # defines the positive or negative pulse and the correction to positive
    Ymag = np.abs(Y - np.mean(Y[:25]))
    index = np.where(Ymag == np.max(Ymag))[0][0]

    if Y[index] < np.mean(Y[:25]):
        YValue = np.min(Y)
    else:
        YValue = np.max(Y)
    #YValueIndex = np.where(Yprime == YValue)[0][0]
    if Y[index] < 0:
        index = np.where(Y == np.min(Y))[0]
        riseTimeend = Interpolator(
            X, Y, .4*(Y[index]-np.mean(Y[:25])) + np.mean(Y[:25]), dTimes, False)
    else:
        riseTimeend = Interpolator(
            X, Y, .4*(Y[index]-np.mean(Y[:25])) + np.mean(Y[:25]) , dTimes, True)
    if riseTimeend > 65 and riseTimeend < 80:
        return riseTimeend
    else:
        return np.nan
def hitfinder(Y):
    NoiseSigma = 2
    baseline = np.mean(Y[:20])
    noiserms = np.sqrt(np.mean(Y[:20]**2))
    p = scisig.savgol_filter(x=Y, window_length=13, polyorder=7)
    durationTheshold=5
    adjDurationThreshold=5
    #plt.plot(Y)
    #plt.show()
    print(baseline,NoiseSigma,noiserms)
    hitLogic = np.array([(True if abs(pi) > abs(baseline + NoiseSigma * noiserms) else False) for pi in p])
    for i in range(1, np.size(hitLogic)):
        if ((not hitLogic[i - 1]) and hitLogic[i]) and hitLogic[i]:
            countDuration = 0
            for j in range(i, np.size(hitLogic) - 1):
                if hitLogic[j]:
                    countDuration = countDuration + 1
                if not hitLogic[j + 1]:
                    break

            if countDuration < durationTheshold:
                for j in range(i, i + countDuration):
                    hitLogic[j] = False
    for i in range(1, np.size(hitLogic)):
        if (hitLogic[i - 1] and (not hitLogic[i])) and (not hitLogic[i]):
            countDuration = 0
            for j in range(i, np.size(hitLogic) - 1):
                if (not hitLogic[j]):
                    countDuration = countDuration + 1
                if hitLogic[j + 1]:
                    break

            if countDuration < adjDurationThreshold:
                for j in range(i, i + countDuration):
                    hitLogic[j] = True


    hitStartIndexList = []
    hitPeakAmplitude = []
    hitPeakIndexArray = []
    for i in range(1, np.size(hitLogic)):
        if ((not hitLogic[i - 1]) and hitLogic[i]) and hitLogic[i]:
            hitAmplitude = 1E100
            hitPeakIndex = i
            for j in range(i, np.size(hitLogic) - 1):
                if p[j] < hitAmplitude:
                    hitAmplitude = p[j]
                    hitPeakIndex = j
                if not hitLogic[j + 1]:
                    break
            ThresholdADC = baseline - (.3 * (baseline - hitAmplitude))

            hitStartIndex = i
            for j in range(hitPeakIndex, 0, -1):
                if (p[j] <= ThresholdADC and p[j - 1] > ThresholdADC):
                    hitStartIndex = j - 0.5
                    break

            hitStartIndexList = np.append(hitStartIndexList, hitStartIndex)
            hitPeakAmplitude = np.append(hitPeakAmplitude, hitAmplitude)
            hitPeakIndexArray = np.append(hitPeakIndexArray, hitPeakIndex)
    print([hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray, baseline, NoiseSigma])
    return [hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray, hitLogic, baseline, NoiseSigma]


def RisetimeFinder(X, Y):
    """
    Find peak location for proper interpolotion

    Returns a returns time that waveform hit 40% of peak value
    """
    # Channel1Data is from first TOF
    # Channel2Data is from second TOF
    dTimes = X
    # defines the positive or negative pulse and the correction to positive
    Ymag = np.abs(Y - np.mean(Y[:25]))
    index = np.where(Ymag == np.max(Ymag))[0][0]

    if Y[index] < np.mean(Y[:25]):
        YValue = np.min(Y)
    else:
        YValue = np.max(Y)
    #YValueIndex = np.where(Yprime == YValue)[0][0]
    if Y[index] < 0:
        index = np.where(Y == np.min(Y))[0]
        riseTimeend = Interpolator(
            X, Y, .1*(Y[index]-np.mean(Y[:25])) + np.mean(Y[:25]), dTimes, False)
        riseTimestart = Interpolator(
            X, Y, .9*(Y[index]-np.mean(Y[:25])) + np.mean(Y[:25]), dTimes, False)

    else:
        riseTimeend = Interpolator(
            X, Y, .1*(Y[index]-np.mean(Y[:25])) + np.mean(Y[:25]) , dTimes, True)
        riseTimestart = Interpolator(
            X, Y, .9*(Y[index]-np.mean(Y[:25])) + np.mean(Y[:25]), dTimes, True)
    if abs(riseTimeend-riseTimestart) < 4:
        return abs(riseTimeend-riseTimestart)
    else:
        return np.nan


root = tk.Tk()
root.withdraw()

def Peakfinder(X,Y,*argv):
    """
    Peak Finding function that tells code whether a peak exists in the waveform.

    Returns a boolean stating existance of peak that satisfies hardcoded requirements.
    """
    mean = np.mean(Y[:25])
    MaxPeak = False
    MinPeak = False
    Max = np.max(Y)
    MaxIndex = np.where(Y==Max)[0]
    Min = np.min(Y)
    MinIndex = np.where(Y==Min)[0]
    Yprime = np.diff(Y)
    YprimeMax = np.max(Yprime)
    YprimeMin = np.min(Yprime)
    YprimeMaxIndex = np.where(Yprime == YprimeMax)[0][0]
    YprimeMinIndex = np.where(Yprime == YprimeMin)[0][0]
    if len(MaxIndex) > 1 or len(MinIndex) > 1: #Pulses left range of the ADC -> not all data is present
        return False
    else:
        MaxIndex = MaxIndex[0]
        MinIndex = MinIndex[0]
    #Pulses must be larger than 20 channels which is equivalent to 5mV, the location of the signal peak must also be between the max and min locations of the derivative of the waveform, with these values being above a threshold of 40 channels in a tick.
    peakheight = .05
    primepeakheight = .002
    if (bool(abs(Y[MaxIndex] - mean) > peakheight) and bool(MaxIndex in range(YprimeMaxIndex,YprimeMinIndex)) and bool(abs(Yprime[YprimeMaxIndex]) > primepeakheight) and bool(abs(Yprime[YprimeMinIndex]) > primepeakheight)):
        MaxPeak = True
    else:
        MaxPeak = False

    if (bool(abs(Y[MinIndex] - mean) > peakheight) and bool(MinIndex in range(YprimeMinIndex,YprimeMaxIndex)) and bool(abs(Yprime[YprimeMaxIndex]) > primepeakheight) and bool(abs(Yprime[YprimeMinIndex]) > primepeakheight)):
        MinPeak = True
    else:
        MinPeak = False

    if MaxPeak and MinPeak: #If there are both positive an negative peaks which match all criteria, the signal is not considered stable and is ignored
        return False
    elif MaxPeak or MinPeak:
        #print('Signal Found in Event {}!'.format(argv[0]))
        return True
    else:
        return False



def reject_outliers(TimeDeltas,TimeRes, m):
    """
    Conducts m-sigma rejection for a given dataset to ensure statistical outliers do not affect results

    Returns inputs with all statistical outliers removed
    """

    mean,stdev =  weighted_avg_and_std(TimeDeltas, TimeRes)
    maskMin = mean - stdev * m
    maskMax = mean + stdev * m
    Indexes = np.where(abs(TimeDeltas-mean)>m*stdev)[0]
    TimeDeltas = np.delete(TimeDeltas,Indexes)
    TimeRes = np.delete(TimeRes,Indexes)
    return TimeDeltas,TimeRes

def ChargeCalculator(Y):
    return np.trapz(Y,dx = .2E-9)


FileName = askopenfilename(
    filetypes=[("Binary Files", "*.dat")])
Data1 = pd.DataFrame()
Data2 = pd.DataFrame()
Data3 = pd.DataFrame()
Data4 = pd.DataFrame()
Divider = 1
with DRS4BinaryFile(FileName) as f:
    BoardID = f.board_ids[0]
    NumberofChannels = f.channels[BoardID]

    if len(NumberofChannels) > 1:
            ReferenceChannel = NumberofChannels[0]
            TimeWidths = f.time_widths[f.board_ids[0]][ReferenceChannel]
    Time = np.arange(0,1023)*.2

    eventNumber = 0

    for event in list(f):
        RC = event.range_center
        ADCData = event.adc_data
        triggerCell = event.trigger_cells[BoardID]
        for i in NumberofChannels:
            if (eventNumber % Divider == 0):
                Data =  ADCData[BoardID][i]/65535 + (RC*1000 - .5)
                [hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray, hitLogic, baseline, NoiseSigma] = hitfinder(Data)
                if Peakfinder(Time,Data):
                    RiseTime = RisetimeFinder(Time,Data)
                    PulseHeight = PulseHeightFinder(Data)
                    Charge = ChargeCalculator(Data)
                    PeakTime =PeakCalculation(Time,Data)
                    TempData = pd.DataFrame(data = {'0':[RiseTime],'1':[PulseHeight],'2':[Charge],'3':[PeakTime]})
                    if i == 1:
                        Data1 = Data1.append(TempData,ignore_index=True)
                    if i == 2:
                        Data2 = Data2.append(TempData,ignore_index=True)
                    if i == 3:
                        Data3 = Data3.append(TempData,ignore_index=True)
                    if i == 4:
                        Data4 = Data4.append(TempData,ignore_index=True)
        eventNumber = eventNumber + 1
columnNames = []

for i in NumberofChannels:
    if i == NumberofChannels[0]:
        columnNames = ["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i)]
    else:
        columnNames.append(["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i)])

if 1 == NumberofChannels[0]:
    Data = Data1
if 2 == NumberofChannels[0]:
    Data = Data2
if 3 == NumberofChannels[0]:
    Data = Data3
if 4 == NumberofChannels[0]:
    Data = Data4


if 1 in NumberofChannels and 1 != NumberofChannels[0]:
    Data = pd.concat([Data,Data1],axis=1,ignore_index=True)
if 2 in NumberofChannels and 2 != NumberofChannels[0]:
    Data = pd.concat([Data,Data2],axis=1,ignore_index=True)
if 3 in NumberofChannels and 3 != NumberofChannels[0]:
    Data = pd.concat([Data,Data3],axis=1,ignore_index=True)
if 4 in NumberofChannels and 4 != NumberofChannels[0]:
    Data = pd.concat([Data,Data4],axis=1,ignore_index=True)

Data.columns = columnNames
print(columnNames)
PulseHeightColumns = []
PulseHeightColumns = [column for column in columnNames if "Pulse Height" in column]
print(PulseHeightColumns)
histPulseHieghts = Data.plot.hist(y = PulseHeightColumns,bins = 100,alpha = .3,subplots=False,title = 'Pulse Height Distributions')
plt.xlabel('Pulse Height (V)')
plt.legend(['Channel 1','Channel 2','Channel 3','Channel 4'])

Text = []
if 1 in NumberofChannels:
    [ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 1 Rise Time'].values,np.ones(len(Data.index)))
    Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(1,ToFMean,TofStd))
if 2 in NumberofChannels:
    [ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 2 Rise Time'].values,np.ones(len(Data.index)))
    Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(2,ToFMean,TofStd))
if 3 in NumberofChannels:
    [ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 3 Rise Time'].values,np.ones(len(Data.index)))
    Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(3,ToFMean,TofStd))
if 4 in NumberofChannels:
    [ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 4 Rise Time'].values,np.ones(len(Data.index)))
    Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(4,ToFMean,TofStd))
ristimeColumns = [column for column in columnNames if "Rise Time" in column]
histRiseTimes = Data.plot.hist(y =ristimeColumns,bins = 100,alpha = .3,subplots=False,title = 'Rise Time Distributions')
plt.legend(Text)
plt.xlabel('Rise Times (ns)')

plt.show()
