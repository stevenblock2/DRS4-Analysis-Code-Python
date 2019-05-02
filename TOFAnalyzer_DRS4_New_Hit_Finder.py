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
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3!!!!")
def install_and_import(package):
    import importlib

    try:
        importlib.import_module(package)
    except ImportError:
        import subprocess
        subprocess.call(['pip', 'install', package])
    finally:
        globals()[package] = importlib.import_module(package)

install_and_import('matplotlib')
import matplotlib.pyplot as plt
install_and_import('numpy')
import numpy as np
import struct
import array
install_and_import('uncertainties')
from uncertainties import ufloat
install_and_import('pandas')
import pandas as pd
from math import *
install_and_import('scipy')
from scipy.stats import poisson
from tkinter.filedialog import askopenfilename,askopenfilenames
import tkinter as tk
import os
import scipy.signal as scisig
from drs4 import DRS4BinaryFile
from scipy import stats
from time import sleep
from matplotlib.ticker import EngFormatter
from scipy.optimize import curve_fit,least_squares
from scipy.misc import factorial
from scipy.optimize import minimize
from lmfit.models import GaussianModel
# Print iterations progress
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def poisson(p,k):
    lamb,amp=p[0],p[1]
    return amp*(lamb**(k))/factorial(k) * np.exp(-lamb)

def poissonMinimizer(p,k,Y):
    lamb,amp= p[0],p[1]
    lnl = amp*((lamb**(k))/factorial(k) * np.exp(-lamb)-Y)
    return np.log(lnl**2)

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

def filterData(Y):
    return scisig.savgol_filter(x=Y, window_length=51, polyorder=11)
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
def gaussMinimizer(p,x,Y):
    A, mu, sigma = p[0],p[1],p[2]
    return np.log((A*np.exp(-(x-mu)**2/(2.*sigma**2))-Y)**2+1)

def hifinderScipy(p):
    #p = filterData(Y)
    NoiseSigma = 7
    baseline = np.mean(p[:50])
    noiserms = np.std(p[:50])
    hitStartIndexList = []
    hitEndIndexList = []
    hitPeakAmplitude = []
    hitPeakIndexArray = []
    Max = abs(max(p))
    Min = abs(min(p))
    if Max > Min and max(p) > baseline + NoiseSigma*noiserms:
        peaks, properties = scipy.signal.find_peaks(p, prominence=.003, width=6,height = baseline - (baseline-NoiseSigma*noiserms))
        # plt.plot(Y)
        # plt.fill_between(np.arange(0,1024),y1 = - NoiseSigma*noiserms,y2 =  NoiseSigma*noiserms,alpha = .1)
        # plt.show()
    elif Min > Max and min(p) < baseline - NoiseSigma*noiserms:
        peaks, properties = scipy.signal.find_peaks(-p, prominence=.003, width=6,height = baseline - (baseline-NoiseSigma*noiserms))
    else:
        peaks, properties = [],{'widths':[]}

    for (peak,width) in zip(peaks,properties['widths']):
        hitAmplitude = p[peak]
        #ThresholdADC = baseline - (.3 * (baseline - hitAmplitude))
        hitEndIndex = peak + int(width)
        hitStartIndex = peak - int(width)
        if abs(hitAmplitude) < 500 and hitStartIndex != 0 and hitEndIndex !=0 and peak !=0 and hitEndIndex < 1023 and peak < hitEndIndex and peak > hitStartIndex and peak - int(width) > 100  and peak + int(width) < 900:
            if eventNumber % SubDivider == 0:
                PersistanceData.append(Data)
                PersistanceTime.append(np.arange(0,1024)*.2)
            hitStartIndexList = np.append(hitStartIndexList, hitStartIndex)
            hitEndIndexList = np.append(hitEndIndexList,hitEndIndex)
            hitPeakAmplitude = np.append(hitPeakAmplitude, hitAmplitude)
            hitPeakIndexArray = np.append(hitPeakIndexArray, peak)

    return [[int(x) for x in hitStartIndexList], hitPeakAmplitude, [int(x) for x in hitPeakIndexArray],[int(x) for x in hitEndIndexList], baseline, noiserms]
def hitfinder(Y):
    #noiserms = np.std(Y[:50])**2
    p = Y# scisig.savgol_filter(x=Y, window_length=25, polyorder=5)
    NoiseSigma = 3
    baseline = np.mean(Y[:50])
    noiserms = np.std(Y[:50])

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
        hitLogic = np.array([(True if pi < baseline - NoiseSigma * noiserms else False) for pi in Y])
    else:
        hitLogic = np.array([(True if pi > baseline + NoiseSigma * noiserms else False) for pi in Y])
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

    return [[int(x) for x in hitStartIndexList], hitPeakAmplitude, [int(x) for x in hitPeakIndexArray],[int(x) for x in hitEndIndexList],  baseline, noiserms]


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
    riseIndex = 0
    fallIndex = 0
    diffs = Y[startIndex:peakIndex]-UpperThreshold
    value = np.min(abs(diffs))
    #print(value,diffs)
    #print(np.where(value == abs(diffs))[0][0])
    riseIndex = int(np.where(value == abs(diffs))[0][0]) + startIndex
    diffs = Y[startIndex:peakIndex]-LowerThreshold
    value = np.min(abs(diffs))
    fallIndex =  int(np.where(value == abs(diffs))[0][0]) + startIndex
    # plt.plot(X,Y)
    # plt.axvline(x = X[riseIndex],color = 'r')
    # plt.axvline(x = X[peakIndex],color = 'g')
    # plt.axvline(x = X[startIndex],color = 'b')`
    # plt.show()`
    riseTimestart = Interpolator(X, Y, riseIndex-1,riseIndex+1,UpperThreshold)
    riseTimeend = Interpolator(X, Y, fallIndex-1,fallIndex+1,LowerThreshold)
    # stop1 = 0
    # stop2 = 0
    # for i in range(peakIndex,startIndex,-1):
    #     if Y[i] > UpperThreshold and Y[i-1] <= UpperThreshold and Y[peakIndex] > 0 and stop1 == 0:
    #         riseIndex = i
    #         stop1 = 1
    #         riseTimestart = Interpolator(X, Y, riseIndex-1,riseIndex+1,UpperThreshold)
    #     if Y[i] > LowerThreshold and Y[i-1] <= LowerThreshold and Y[peakIndex] > 0 and stop2 == 0:
    #         fallIndex = i
    #         stop2 = 1
    #         riseTimeend = Interpolator(X, Y, fallIndex-1,fallIndex+1,LowerThreshold)
    #
    # for i in range(peakIndex,startIndex,-1):
    #     if Y[i] < UpperThreshold and Y[i-1] >= UpperThreshold and Y[peakIndex] < 0 and stop1 == 0:
    #         riseIndex = i
    #         stop1 = 1
    #         riseTimestart = Interpolator(X, Y, riseIndex-1,riseIndex+1,UpperThreshold)
    #     if Y[i] < LowerThreshold and Y[i-1] >= LowerThreshold and Y[peakIndex] < 0 and stop2 == 0:
    #         fallIndex = i
    #         stop2 = 1
    #         riseTimeend = Interpolator(X, Y, fallIndex-1,fallIndex+1,LowerThreshold)
    #print(riseTimestart,riseTimeend)
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
def DCS(Y):
    step = 80
    newY = np.ones(len(Y)-step-1)
    for i in range(0,len(Y)-step-1):
        newY[i] = Y[i+step] - Y[i]
    plt.plot(Y,label= 'RAW')
    plt.plot(newY,label= 'DCS')
    plt.legend(loc='best')
    plt.show()

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
    C = 1
    Gain = 31
    e = 1.602E-19
    return (np.trapz(Y[startIndex:EndIndex],dx = .2E-9)*C/e)

def get_hist(ax):
    n,bins = [],[]
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0) # left edge of each bin
    bins.append(x1) # also get right edge of last bin

    return np.asarray(n,dtype=np.float32),np.asarray(bins,dtype=np.float32)

def FindHistPeaks(Y):
    peaks, properties  = scipy.signal.find_peaks(Y, width=10,height =5,prominence= 5,distance = 15)
    return peaks,properties

FileNames = askopenfilenames(
    filetypes=[("Binary Files", "*.dat")])
with DRS4BinaryFile(FileNames[0]) as events:
    length = len(list(events))
itertor = 1
for FileName in FileNames:
    directory = os.path.dirname(FileName)
    newDirectory = os.path.join(directory,FileName[:-4])
    path,name = os.path.split(FileName)
    if not os.path.exists(newDirectory):
        os.mkdir(newDirectory)
    print('Processing: {} - ({} of {})'.format(os.path.basename(FileName),itertor,len(FileNames)))
    with DRS4BinaryFile(FileName) as events:
        length = len(list(events))
    Data1 = pd.DataFrame()
    Data2 = pd.DataFrame()
    Data3 = pd.DataFrame()
    Data4 = pd.DataFrame()
    Divider = 1
    SubDivider = 1000
    PersistanceData = []
    PersistanceTime = []

    with DRS4BinaryFile(FileName) as f:

        BoardID = f.board_ids[0]
        NumberofChannels = f.channels[BoardID]

        if len(NumberofChannels) > 1:
                ReferenceChannel = NumberofChannels[0]
                TimeWidths = f.time_widths[f.board_ids[0]][ReferenceChannel]
        Time = np.arange(0,1024)*.2

        eventNumber = 0
        printProgressBar(0, length, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for event in list(f):
            RC = event.range_center
            ADCData = event.adc_data
            triggerCell = event.trigger_cells[BoardID]
            for i in NumberofChannels:
                if (eventNumber % Divider == 0):
                    Data =  (ADCData[BoardID][i]/65535 + (RC/1000 - .5))
                    #DCS(Data)
                    Data = filterData(Data)
                    [hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray,hitEndIndexList, baseline, rmsnoise] = hifinderScipy(Data) #hitfinder(Data)
                    #print(hitStartIndexList)
                    if hitStartIndexList:
                        for (startIndex,EndIndex,hitAmplitude,hitAmplitudeIndex) in zip(hitStartIndexList,hitEndIndexList,hitPeakAmplitude,hitPeakIndexArray):
                            #print(startIndex,EndIndex,hitAmplitude,hitAmplitudeIndex)
                            RiseTime = RisetimeFinder(Time,Data,startIndex,hitAmplitudeIndex,baseline)
                            PulseHeight = hitAmplitude
                            Charge = ChargeCalculator(Data,startIndex,EndIndex)
                            PeakTime =  Time[hitAmplitudeIndex]
                            TempData = pd.DataFrame(data = {'0':[RiseTime],'1':[PulseHeight],'2':[Charge],'3':[PeakTime],'4':[rmsnoise],'5':[baseline],'6':[baseline+rmsnoise]})
                            #print(TempData)
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
            columnNames = ["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i),"Channel {} RMS Noise".format(i),"Channel {} Baseline".format(i),"Channel {} Pedestle".format(i)]
        else:
            columnNames.extend(["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i),"Channel {} RMS Noise".format(i),"Channel {} Baseline".format(i),"Channel {} Pedestle".format(i)])

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
    # print(Data.head(30),[e for e in columnNames if 'Channel' in e])
    #Data= Data[(np.abs(stats.zscore(Data)) < 3).all(axis=1)]
    #Data= Data[(np.abs(stats.zscore(Data)) < 3).all(axis=1)]
    Data.columns = [e for e in columnNames if 'Channel' in e]

    PulseHeightColumns = []
    PulseHeightColumns = [column for column in columnNames if "Pulse Height" in column]
    PulseandNoiseColumns = [column for column in columnNames if "Pulse Height" in column]
    ChargeColumns = [column for column in columnNames if "Charge" in column]
    histPulseHieghts = Data.plot.hist(y = PulseandNoiseColumns,bins =1000,alpha = .3,subplots=False,title = 'Pulse Height Distributions',log=False)
    plt.xlabel('Pulse Height (mV)')
    plt.savefig(os.path.join(newDirectory,'Pulse_Height_Distribution.png'))

    histCharge = Data.plot.hist(y = ChargeColumns,bins =1000,alpha = .3,subplots=False,title = 'Pulse Area Distribution',log=False)
    plt.xlabel(r'Area ($\frac{pC}{e}$)')
    plt.legend(ChargeColumns)


    #print(n)
    # for column in ChargeColumns:
    #     print("{} Statistics: \nMean Charge: {}\nVariance of Charge: {}".format(column,Data[column].mean(),Data[column].std()**2))

    text = []
    mu = []
    variance = []
    lammaList = []

    amp = []
    n, bins = get_hist(histCharge)
    bincenters = np.asarray([(bins[i]+bins[i-1])/2 for i in range(1,len(bins))],np.float32)
    peaks,properties = FindHistPeaks(n)
    widths = scipy.signal.peak_widths(n, peaks, rel_height=0.5)
    j = 0
    scale = .5
    for (peak,width) in zip(peaks,widths[0]):
        #true_width = abs(bincenters[int(peak - width/2)]-bincenters[int(peak + width/2)])
        X = bincenters[int(peak - width):int(peak + width)]
        Y = n[int(peak - width):int(peak + width)]
        mean,std = weighted_avg_and_std(X,Y)
        p0 = [n[peak], mean, std]
        bounds = [(0,.5*mean,.5*std),(n[peak],1.5*mean,1.5*std)]
        res = least_squares(gaussMinimizer, p0, loss='linear', f_scale=scale,args=(X,Y),bounds = bounds,xtol = 1E-20,ftol = 1E-15,x_scale = 'jac',tr_solver = 'lsmr',max_nfev=1E4)
        #coeff, var_matrix = curve_fit(gauss, bincenters[int(peak - width/2):int(peak + width/2)], n[int(peak - width/2):int(peak + width/2)], p0=p0,bounds=bounds)
        fixed_range = bincenters[int(peak - 2*width):int(peak + 2*width)]
        hist_fit = gauss(fixed_range, *res.x)
        #plt.plot(fixed_range, hist_fit,linewidth=2.0,label = r"$\mu_{}$ = {:.3E}, $\sigma$ = {:.3E}".format(j,np.round(res.x[1],5),np.round(res.x[2],5)))
        mu.append(res.x[1])
        variance.append(res.x[2]**2)
        amp.append(res.x[0])

        #         true_width = abs(bincenters[int(peak - width/2)]-bincenters[int(peak + width/2)])
        #         p0 = [n[peak], bincenters[peak], .7*true_width]
        #         bounds = [(0,bincenters[peak]-.8*abs(bincenters[peak]),0),(1.5*n[peak],bincenters[peak]+.8*abs(bincenters[peak]),.8*true_width)]
        #         coeff, var_matrix = curve_fit(gauss, bincenters[int(peak - width/2):int(peak + width/2)], n[int(peak - width/2):int(peak + width/2)], p0=p0,bounds=bounds)
        #         fixed_range = bincenters[int(peak - 2*width):int(peak + 2*width)]
        #         hist_fit = gauss(fixed_range, *coeff)
        #         print(coeff)
        #         plt.plot(fixed_range, hist_fit,linewidth=2.0,label = r"$\mu_{}$ = {}, $\sigma$ = {}".format(j,np.round(coeff[1],3),np.round(coeff[2],3)))
        #         mu.append(coeff[1])
        #         variance.append(coeff[2]**2)
        #         amp.append(coeff[0])
        # >>>>>>> a657a6e154855458946bc6be4ebb0f321b58aaee
        # if j == 0:
        #     lamma = -np.log(coeff[1])
        # if j != 0:
        #     lamma = -np.log(coeff[1])+ coeff[1]
        print(mu,len(mu))
        j = j+1

    for i in range(0,len(mu)):
        if i == 0:
            mod = GaussianModel(prefix = 'f{}_'.format(i))
            pars = mod.guess(n,x=bincenters, sigma=np.sqrt(variance[i]),height = amp[i],center = mu[i])
            pars.add('G',value = mu[i+1]-mu[i],brute_step=.01*mu[i],min = .1*(mu[i+1]-mu[i]),max = 5*(mu[i+1]-mu[i]))
        else:
            tempmod =  GaussianModel(prefix = 'f{}_'.format(i))
            temppars = tempmod.guess(n,x=bincenters,center = mu[i], sigma=np.sqrt(variance[i]),height = amp[i])
            pars += temppars
            pars['f{}_center'.format(i)].set(expr='G+f{}_center'.format(i-1))
            #pars['f{}_center'.format(i-1)].set(expr='G-f{}_center'.format(i))
            print(pars)
            mod += tempmod

    if len(mu):
        result = mod.fit(n, pars, x=bincenters)
        print(result.fit_report())
        #plt.plot(bincenters,n,'y')
        #plt.plot(mu,amp,'k+')
        #print(pars.valuesdict())
        print(result.params['G'].stderr)
        vals = pars.valuesdict()
        #plt.plot(bincenters,result.init_fit,'r--')
        plt.plot(bincenters,result.best_fit,'k--',label = 'Gain = {:.2E} +/- {:.2E}'.format(result.params['G'].value,result.params['G'].stderr))
        plt.legend(loc = 'best')
    # if len(mu) > 3:
    #     X = mu
    #     NewX = mu/abs(mu[1]) #this alters the behavior of the distribution!!!!
    #     #plt.plot(newmu[1:],amp[1:],'k+')
    #     area = sum(amp*NewX)
    #     Y = amp/area
    #     p0= [NewX[1],Y[1]]
    #     bounds = [(NewX[0],0),(NewX[-1],10)]
    #     #parameters, cov_matrix = curve_fit(poisson, NewX, Y,p0=p0,sigma = 1/np.sqrt(Y),bounds = bounds)
    #     res = least_squares(poissonMinimizer, p0, loss='linear', f_scale=scale,args=(NewX, Y),bounds=bounds,xtol = 1E-20,gtol = 1E-50,ftol = 1E-20,x_scale = 'jac',tr_solver = 'lsmr',max_nfev=1E4)
    #     print(res)
    #     x_plot = np.linspace(0,2*NewX[-1], 1000)
    # else:
    #     X = bincenters #this alters the behavior of the distribution!!!!
    #     NewX = bincenters/bincenters[np.where(n==np.max(n))[0][0]]#plt.plot(newmu[1:],amp[1:],'k+')
    #     area = np.trapz(n,dx = abs(bincenters[1]-bincenters[0]))
    #     Y = n/area
    #     p0= [NewX[1],Y[1]]
    #     bounds = [(NewX[0],0),(NewX[-1],np.inf)]
    #     #parameters, cov_matrix = curve_fit(poisson, NewX, Y,p0=p0,sigma = 1/np.sqrt(Y+.0001),bounds = bounds)
    #     res = least_squares(poissonMinimizer, p0, loss='linear', f_scale=scale,args=(NewX, Y),bounds=bounds,gtol = 1E-50,xtol = 1E-50,ftol = 1E-50,x_scale = 'jac',tr_solver = 'lsmr',max_nfev=1E4)
    #     print(res)
    #     x_plot = np.linspace(0,2*NewX[-1], 1000)
    # true_x = np.linspace(X[0],2*X[-1], 1000)
    # p = poisson(res.x,x_plot)
    # p = p*(max(n)/max(p))
    # plt.plot(true_x,p, 'r--', lw=2,label =r"<$\mu$> = {}".format(np.round(res.x[0],3)))

    # if len(mu) > 3:
    #     newmu = mu/mu[0] #this alters the behavior of the distribution!!!!
    #     #plt.plot(newmu[1:],amp[1:],'k+')
    #     amp = amp
    #     p0= [1,amp[1]*10]
    #     bounds = [(0,0),(5,10*amp[1])]
    #     parameters, cov_matrix = curve_fit(poisson, newmu, amp,p0=p0,sigma = 1/np.sqrt(amp))
    #     x_plot = np.linspace(0,2*newmu[-1], 1000)
    #     true_x = np.linspace(0,2* mu[-1], 1000)
    #     plt.plot(true_x, poisson(x_plot, *parameters), 'r--', lw=2,label =r"<$\mu$> = {}".format(np.round(parameters[0],3)))
    #
    # plt.legend(loc='best')
    # plt.savefig(os.path.join(newDirectory,'Pulse_Area_Distribution.png'))
    # if len(mu) > 3:
    #     plt.figure()
    #     plt.ylabel(r'$\sigma^2$ $(\frac{pC}{e})^2$')
    #     plt.xlabel(r'$\mu$ ($\frac{pC}{e}$)')
    #     #plt.gca().yaxis.set_major_formatter(formatter1)
    #     p = np.polyfit(mu, variance, 1)
    #
    #     plt.plot(mu,variance,label = 'Raw Data')
    # print(p)

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
    itertor = itertor +1
    if len(FileNames) != 1:
        plt.close('all')
if len(FileNames) == 1:
    plt.show()
else:
    print("Analysis of Files Complete!")
