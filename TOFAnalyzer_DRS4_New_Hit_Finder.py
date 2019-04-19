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
import scipy
from time import sleep


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.nanmean(values)
    # Fast and numerically precise:
    std = np.nanstd(values)
    return [np.round(average,3), np.round(std,3)]


def Interpolator(X, Y, TimeleftIndex, TimeRightIndex,YValue):
    """
    Interpolate exact time Y == YValue using 4 points around closest data point to YValue

    Returns a tuple with time and error associated.
    """
    Y1 = Y[TimeleftIndex]
    Y2 = Y[TimeRightIndex]
    X2 = X[TimeRightIndex]
    X1 = X[TimeleftIndex]
    slope = (Y2 - Y1) / (X2 - X1)
    if slope != 0:
        X0 = (YValue - Y1) / slope + X1
        return X0
    else:
        return 0
def PulseHeightFinder(Y):
    return max(abs(np.mean(Y[:20]) - Y))

def hitfinder(Y):

    #noiserms = np.std(Y[:50])**2
    p = scisig.savgol_filter(x=Y, window_length=25, polyorder=5)
    NoiseSigma = 3
    baseline = np.mean(p[:50])
    noiserms = np.abs(p - np.median(p))
    mdev = np.median(noiserms)
    s = noiserms / mdev if mdev else 0.
    noiserms = np.std(p[s < 2])**2
    durationTheshold=5
    adjDurationThreshold=5
    #plt.plot(Y)
    #plt.show()
    #print(baseline,NoiseSigma,abs(baseline) + NoiseSigma * noiserms)
    p_diff = np.diff(p)
    #1mV per tick = .001
    #2mV per tick = .002
    #etc...

    if abs(min(Y)) > abs(max(Y)):
        hitLogic = np.array([(True if pi < baseline - NoiseSigma * noiserms else False) for pi in p])
    else:
        hitLogic = np.array([(True if pi > baseline + NoiseSigma * noiserms else False) for pi in p])
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
    hitEndIndexList = []
    hitPeakAmplitude = []
    hitPeakIndexArray = []
    hitEndIndex = 0
    hitStartIndex = 0
    hitPeakIndex = 0
    eventNumber = 0
    global SubDivider
    global PersistanceData
    global PersistanceTime

    for i in range(1, np.size(hitLogic)):
        if ((not hitLogic[i - 1]) and hitLogic[i]) and hitLogic[i]:
            hitAmplitude = 0
            hitPeakIndex = i
            for j in range(i, np.size(hitLogic) - 1):
                    if abs(p[j]) > abs(hitAmplitude):
                        hitAmplitude = p[j]
                        hitPeakIndex = j
                    if not hitLogic[j + 1]:
                        break
            ThresholdADC = baseline - (.3 * (baseline - hitAmplitude))
            hitStartIndex = i
            for j in range(hitPeakIndex, 0, -1):
                if (abs(p[j]) >= abs(ThresholdADC) and abs(p[j - 1]) < abs(ThresholdADC)):
                    hitStartIndex = int(j)
                    break
            for j in range(hitPeakIndex,  np.size(hitLogic) - 1, 1):
                if not hitLogic[j]:
                    hitEndIndex = int(j)
                    break
            #print(hitStartIndex,hitEndIndex,hitPeakIndex,hitAmplitude)
            #print(bool(hitStartIndex-hitEndIndex > 3))
            if abs(hitEndIndex-hitStartIndex) > 10 and abs(hitAmplitude) < .5 and hitStartIndex != 0 and hitEndIndex !=0 and hitPeakIndex !=0 and hitEndIndex < 1023 and hitPeakIndex < hitEndIndex and hitPeakIndex > hitStartIndex:
                if eventNumber % SubDivider == 0:
                    PersistanceData.append(Data)
                    PersistanceTime.append(np.arange(0,1024)*.2)
                hitStartIndexList = np.append(hitStartIndexList, hitStartIndex)
                hitEndIndexList = np.append(hitEndIndexList,hitEndIndex)
                hitPeakAmplitude = np.append(hitPeakAmplitude, hitAmplitude)
                hitPeakIndexArray = np.append(hitPeakIndexArray, hitPeakIndex)
            i = hitEndIndex
    #print([hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray, baseline, NoiseSigma])
    if len(hitPeakAmplitude) > 1:
        minpeak = min(hitPeakAmplitude)
        maxpeak = max(hitPeakAmplitude)
        if abs(maxpeak) > abs(minpeak):
            Indexes = np.nonzero(hitPeakAmplitude > 0)
        else:
            Indexes = np.nonzero(hitPeakAmplitude < 0)

        hitStartIndexList = hitStartIndexList[Indexes]
        hitEndIndexList = hitEndIndexList[Indexes]
        hitPeakAmplitude = hitPeakAmplitude[Indexes]
        hitPeakIndexArray = hitPeakIndexArray[Indexes]

    return [[int(x) for x in hitStartIndexList], hitPeakAmplitude, [int(x) for x in hitPeakIndexArray],[int(x) for x in hitEndIndexList], hitLogic, baseline, noiserms]


def RisetimeFinder(X, Y,startIndex,peakIndex,baseline):
    """
    Find peak location for proper interpolotion

    Returns a returns time that waveform hit 40% of peak value
    """
    # Channel1Data is from first TOF
    # Channel2Data is from second TOF
    UpperThreshold = baseline - (.7 * (baseline - hitAmplitude))
    LowerThreshold = baseline - (.3 * (baseline - hitAmplitude))
    riseTimestart = 0
    riseTimeend = 0
    for i in range(peakIndex,startIndex-10,-1):
        if abs(Y[i]) > UpperThreshold and abs(Y[i-1] <=UpperThreshold):
            riseTimestart = Interpolator(X, Y, i-2,i,UpperThreshold)
        if abs(Y[i]) > LowerThreshold and abs(Y[i-1] <=LowerThreshold):
            riseTimeend = Interpolator(X, Y, i-2,i,LowerThreshold)
        if riseTimestart and riseTimeend and riseTimestart != riseTimeend:
            break
    return riseTimestart-riseTimeend



root = tk.Tk()
root.withdraw()
def Lowpass(Y):
    CutoffFreq = 5000
    pedestle = list(Y[:50])
    pedestle.extend(np.zeros(len(Y)-50))
    fftpedestle= scipy.fft(pedestle)# (G) and (H)
    fft= scipy.fft(Y)
    newfft = fft-fftpedestle
    bp=newfft[:]
    # for i in range(len(bp)): # (H-red)
    #     if i>=CutoffFreq:bp[i]=0
    ibp=scipy.ifft(bp) # (I), (J), (K) and (L)
    return ibp


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

def ChargeCalculator(Y,startIndex,EndIndex):
    C = 40E-12
    return np.trapz(Y[startIndex-5:EndIndex+5],dx = .2E-9)


FileName = askopenfilename(
    filetypes=[("Binary Files", "*.dat")])
directory = os.path.dirname(FileName)
newDirectory = os.path.join(directory,FileName[:-4])
if not os.path.exists(newDirectory):
    os.mkdir(newDirectory)

Data1 = pd.DataFrame()
Data2 = pd.DataFrame()
Data3 = pd.DataFrame()
Data4 = pd.DataFrame()
Divider = 1
SubDivider = 1000
PersistanceData = []
PersistanceTime = []
with DRS4BinaryFile(FileName) as events:
    length = len(list(events))
with DRS4BinaryFile(FileName) as f:

    BoardID = f.board_ids[0]
    NumberofChannels = f.channels[BoardID]

    if len(NumberofChannels) > 1:
            ReferenceChannel = NumberofChannels[0]
            TimeWidths = f.time_widths[f.board_ids[0]][ReferenceChannel]
    Time = np.arange(0,1024)*.2

    eventNumber = 0
    plt.figure(1)
    printProgressBar(0, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for event in list(f):

        RC = event.range_center
        ADCData = event.adc_data
        triggerCell = event.trigger_cells[BoardID]
        for i in NumberofChannels:
            if (eventNumber % Divider == 0):
                Data =  ADCData[BoardID][i]/65535 + (RC/1000 - .5)
                [hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray,hitEndIndexList, hitLogic, baseline, rmsnoise] = hitfinder(Data)
                if hitStartIndexList:

                    # plt.plot(Data)
                    # plt.plot(scisig.savgol_filter(x=Data, window_length=33, polyorder=7))
                    # plt.show()
                    for (startIndex,EndIndex,hitAmplitude,hitAmplitudeIndex) in zip(hitStartIndexList,hitEndIndexList,hitPeakAmplitude,hitPeakIndexArray):

                        #print(startIndex,EndIndex,hitAmplitude,hitAmplitudeIndex)
                        RiseTime = RisetimeFinder(Time,Data,startIndex,EndIndex,baseline)
                        PulseHeight = hitAmplitude
                        Charge = ChargeCalculator(Data,startIndex,EndIndex)
                        PeakTime =  Time[hitAmplitudeIndex]
                        TempData = pd.DataFrame(data = {'0':[RiseTime],'1':[PulseHeight],'2':[Charge],'3':[PeakTime],'4':[rmsnoise]})
                        if eventNumber % SubDivider == 0:
                            plt.plot(Time,Data,'k')
                            plt.axvline(Time[startIndex],color = 'r',ymax = 1,linewidth=.2)
                            plt.axvline(Time[hitAmplitudeIndex],color = 'g',ymax = 1,linewidth=.2)
                            plt.axvline(Time[EndIndex],color = 'b',ymax = 1,linewidth=.2)
                        if i == 1:
                            Data1 = Data1.append(TempData,ignore_index=True)
                        if i == 2:
                            Data2 = Data2.append(TempData,ignore_index=True)
                        if i == 3:
                            Data3 = Data3.append(TempData,ignore_index=True)
                        if i == 4:
                            Data4 = Data4.append(TempData,ignore_index=True)
        #sleep(0.001)
        printProgressBar(eventNumber + 1, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        eventNumber = eventNumber + 1
columnNames = []
plt.savefig(os.path.join(newDirectory,'Persistance.png'))
for i in NumberofChannels:
    if i == NumberofChannels[0]:
        columnNames = ["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i),"Channel {} RMS Noise".format(i)]
    else:
        columnNames.extend(["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i),"Channel {} RMS Noise".format(i)])

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
print(Data.head(30),[e for e in columnNames if 'Channel' in e])
Data.columns = [e for e in columnNames if 'Channel' in e]
PulseHeightColumns = []
PulseHeightColumns = [column for column in columnNames if "Pulse Height" in column]
PulseandNoiseColumns = [column for column in columnNames if "Pulse Height" in column or "Noise" in column]
ChargeColumns = [column for column in columnNames if "Charge" in column]

histPulseHieghts = Data.plot.hist(y = PulseandNoiseColumns,bins =1000,alpha = .3,subplots=False,title = 'Pulse Height Distributions',log=True)
plt.xlabel('Pulse Height (V)')

plt.legend(PulseandNoiseColumns)
plt.savefig(os.path.join(newDirectory,'Pulse_Height_Distribution.png'))

histCharge = Data.plot.hist(y = ChargeColumns,bins =1000,alpha = .3,subplots=False,title = 'Pulse Area Distribution',log=True)
plt.xlabel('Area (V*s)')
plt.legend(ChargeColumns)
plt.savefig(os.path.join(newDirectory,'Pulse_Area_Distribution.png'))


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
histRiseTimes = Data.plot.hist(y =ristimeColumns,bins = 1000,alpha = .3,subplots=False,title = 'Rise Time Distributions')
plt.legend(Text)
plt.xlabel('Rise Times (ns)')
plt.savefig(os.path.join(newDirectory,'Rise_Time_Distribution.png'))
plt.show()
