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
    peakheight = .020
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




FileName = askopenfilename(
    filetypes=[("Binary Files", "*.dat")])
with DRS4BinaryFile(FileName) as f:
    NumberofChannels = f.channels[f.board_ids[0]]

    columnNames = []
    nanarray = []
    for i in NumberofChannels:
        columnNames.append(["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Pulse Time".format(i)])
        nanarray.append(np.nan)
        nanarray.append(np.nan)
        nanarray.append(np.nan)
    columnNames = [item for sublist in columnNames for item in sublist]
    Data = pd.DataFrame(columns = columnNames)
    #Iterate over data
    TimeWidths1 = f.time_widths[f.board_ids[0]][1]
    TimeWidths2 = f.time_widths[f.board_ids[0]][2]
    TimeWidths3 = f.time_widths[f.board_ids[0]][3]
    TimeWidths4 = f.time_widths[f.board_ids[0]][4]
    Time1 = np.arange(0,1023)*.2
    Time2 = np.arange(0,1023)*.2
    Time3 = np.arange(0,1023)*.2
    Time4 = np.arange(0,1023)*.2
    eventNumber = 0
    BoardID = f.board_ids[0]
    for event in list(f):
        triggerCell = event.trigger_cells[BoardID]
        RC = event.range_center
        ADCData = event.adc_data
        if (eventNumber % 1 == 0):
            # for j in range(0,len(TimeWidths1)):
            #         for i in range(0,j):
            #             Time1[j] = Time1[j] + TimeWidths1[(i + triggerCell)%1024]
            #             Time2[j] = Time2[j] + TimeWidths2[(i + triggerCell)%1024]
            #             Time3[j] = Time3[j] + TimeWidths3[(i + triggerCell)%1024]
            #             Time4[j] = Time4[j] + TimeWidths4[(i + triggerCell)%1024]
            #

            # TimeOffset1 = Time1[int(1024-triggerCell)%1024]
            #
            # Time2 = Time2 - (TimeOffset1-Time2[0])
            # Time3 = Time3 - (TimeOffset1-Time3[0])
            # Time4 = Time4 - (TimeOffset1-Time4[0])
            RiseTime1,RiseTime2,RiseTime3,RiseTime4 = np.nan,np.nan,np.nan,np.nan
            PulseHeight1,PulseHeight2,PulseHeight3,PulseHeight4 = np.nan,np.nan,np.nan,np.nan
            PeakTime1,PeakTime2,PeakTime3,PeakTime4 = np.nan,np.nan,np.nan,np.nan
            try:
                Data1 =  ADCData[BoardID][1]/65535 + RC*1000

                if Peakfinder(Time1,Data1):
                    RiseTime1 = RisetimeFinder(Time1,Data1)
                    PulseHeight1 = PulseHeightFinder(Data1)
                    PeakTime1 =PeakCalculation(Time1,Data1)
            except:
                pass
            try:
                Data2 = ADCData[BoardID][2]/65535 + RC*1000
                if Peakfinder(Time2,Data2):
                    RiseTime2 = RisetimeFinder(Time2,Data2)
                    PulseHeight2 = PulseHeightFinder(Data2)
                    PeakTime2 =PeakCalculation(Time2,Data2)
            except:
                pass
            try:

                Data3 = ADCData[BoardID][3]/65535 + RC*1000

                if Peakfinder(Time3,Data3):
                    RiseTime3 = RisetimeFinder(Time3,Data3)
                    PulseHeight3 = PulseHeightFinder(Data3)
                    PeakTime3 =PeakCalculation(Time3,Data3)
            except:
                pass
            try:

                Data4 = ADCData[BoardID][4]/65535 + RC*1000
                if Peakfinder(Time4,Data4):
                    RiseTime4 = RisetimeFinder(Time4,Data4)
                    PulseHeight4 = PulseHeightFinder(Data4)
                    PeakTime4 =PeakCalculation(Time4,Data4)
            except:
                pass

            Data.loc[eventNumber] = [RiseTime1,PulseHeight1,PeakTime1,RiseTime2,PulseHeight2,PeakTime2,RiseTime3,PulseHeight3,PeakTime3,RiseTime4,PulseHeight4,PeakTime4]

        eventNumber = eventNumber + 1
#PulseHeightColumns = columnNames[]
histPulseHieghts = Data.plot.hist(y = ['Channel 1 Pulse Height','Channel 2 Pulse Height','Channel 3 Pulse Height','Channel 4 Pulse Height'],bins = 100,alpha = .3,subplots=False,title = 'Pulse Height Distributions')
plt.xlabel('Pulse Height (V)')
plt.legend(['Channel 1','Channel 2','Channel 3','Channel 4'])

Text = []
[ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 1 Rise Time'].values,np.ones(len(Data.index)))
Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(1,ToFMean,TofStd))

[ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 2 Rise Time'].values,np.ones(len(Data.index)))
Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(2,ToFMean,TofStd))

[ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 3 Rise Time'].values,np.ones(len(Data.index)))
Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(3,ToFMean,TofStd))

[ToFMean, TofStd] = weighted_avg_and_std(Data['Channel 4 Rise Time'].values,np.ones(len(Data.index)))
Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(4,ToFMean,TofStd))

histRiseTimes = Data.plot.hist(y = ['Channel 1 Rise Time','Channel 2 Rise Time','Channel 3 Rise Time','Channel 4 Rise Time'],bins = 100,alpha = .3,subplots=False,title = 'Rise Time Distributions')
plt.legend(Text)
plt.xlabel('Rise Times (ns)')


Text = []
DataToF = Data[['Channel 1 Pulse Time','Channel 2 Pulse Time','Channel 3 Pulse Time','Channel 4 Pulse Time']]
try:
    DataToF = DataToF.assign(Delta12 = DataToF['Channel 1 Pulse Time'].values - DataToF['Channel 2 Pulse Time'].values)
    [ToFMean, TofStd] = weighted_avg_and_std(DataToF['Delta12'].values,np.ones(len(DataToF.index)))
    #print('Delta 12 Stats: \n Mean = {}\n Standard Deviation = {}'.format(ToFMean,TofStd))
    Text.append(r'$\Delta_{}: \mu = {}ns; \sigma = {}ns$'.format({12},ToFMean,TofStd))
except:
    pass
try:

    DataToF = DataToF.assign(Delta13 = DataToF['Channel 1 Pulse Time'].values - DataToF['Channel 3 Pulse Time'].values)
    [ToFMean, TofStd] = weighted_avg_and_std(DataToF['Delta13'].values,np.ones(len(DataToF.index)))
    #print(r'$\Delta 13: \n \mu = {}ns \n \sigma = {}ns$'.format(ToFMean,TofStd))
    Text.append(r'$\Delta_{}: \mu = {}ns; \sigma = {}ns$'.format({13},ToFMean,TofStd))
except:
    pass
try:
    DataToF = DataToF.assign(Delta14 =DataToF['Channel 1 Pulse Time'].values - DataToF['Channel 4 Pulse Time'].values)
    [ToFMean, TofStd] = weighted_avg_and_std(DataToF['Delta14'].values,np.ones(len(DataToF.index)))
    #print('Delta 14 Stats: \n Mean = {}\n Standard Deviation = {}'.format(ToFMean,TofStd))
    Text.append(r'$\Delta_{}: \mu = {}ns; \sigma = {}ns$'.format({14},ToFMean,TofStd))
except:
    pass
try:

    DataToF = DataToF.assign(Delta23 =DataToF['Channel 2 Pulse Time'].values- DataToF['Channel 3 Pulse Time'].values)
    [ToFMean, TofStd] = weighted_avg_and_std(DataToF['Delta23'].values,np.ones(len(DataToF.index)))
    #print('Delta 23 Stats: \n Mean = {}\n Standard Deviation = {}'.format(ToFMean,TofStd))
    Text.append(r'$\Delta_{}: \mu = {}ns; \sigma = {}ns$'.format({23},ToFMean,TofStd))
except:
    pass
try:
    DataToF = DataToF.assign(Delta24 =DataToF['Channel 2 Pulse Time'].values- DataToF['Channel 4 Pulse Time'].values)
    [ToFMean, TofStd] = weighted_avg_and_std(DataToF['Delta24'].values,np.ones(len(DataToF.index)))
    #print('Delta 24 Stats: \n Mean = {}\n Standard Deviation = {}'.format(ToFMean,TofStd))
    Text.append(r'$\Delta_{}: \mu = {}ns; \sigma = {}ns$'.format({24},ToFMean,TofStd))

except:
    pass
try:
    DataToF = DataToF[['Delta12','Delta13','Delta14','Delta23','Delta24']]
    HistToF = DataToF.plot.hist(bins = 100,alpha = .3,title = 'Time fo Flight Distributions')
    plt.gca().legend(Text)

    # build a rectangle in axes coords
    left,right = plt.gca().get_xlim()
    bottom,top = plt.gca().get_ylim()
    #plt.text(left+.3,.9*top,Text)
    plt.xlabel('Time of Flight (ns)')
    plt.tight_layout()
    #HistToF =
except:
    pass
plt.show()
