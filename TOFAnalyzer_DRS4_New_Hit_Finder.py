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
#from scipy.stats import poisson
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
from lmfit.models import GaussianModel,Model
# Print iterations progress
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def poisson(p,k):
    lamb,amp=p[0],p[1]
    lnl = amp*(lamb**(k))/factorial(k) * np.exp(-lamb)
    return lnl
def PEModel(x,A,sigma,gain,mu,n):
    prob = np.exp(-mu)*mu**n/factorial(n)
    print(A,sigma,gain,mu,n)
    return A*prob*[1/np.sqrt(2*np.pi*sigma)*np.exp((x-n*gain)**2/(2*sigma**2))]
def poissonMinimizer(p,k,Y):
    lamb,amp=p[0],p[1]
    lnl = amp*(lamb**(k))/factorial(k) * np.exp(-lamb)-Y
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
    Max = abs(max(p[100:900]))
    Min = abs(min(p[100:900]))
    #print(baseline - (baseline-NoiseSigma*noiserms))
    if Max > Min and max(p[100:900]) > baseline + NoiseSigma*noiserms:
        peaks, properties = scipy.signal.find_peaks(p, prominence=.005, width=6,height = baseline - (baseline-NoiseSigma*noiserms))
    elif Min > Max and min(p[100:900]) < baseline - NoiseSigma*noiserms:
        peaks, properties = scipy.signal.find_peaks(-p, prominence=.005, width=6,height = baseline - (baseline-NoiseSigma*noiserms))
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
    hitAmplitude = Y[peakIndex]
    UpperThreshold = baseline - (.7 * (baseline - hitAmplitude))
    LowerThreshold = baseline - (.3 * (baseline - hitAmplitude))
    riseTimestart = 0
    riseTimeend = 0
    riseIndex = 0
    fallIndex = 0
    diffs = Y[startIndex:peakIndex]-UpperThreshold
    value = np.min(abs(diffs))
    noiserms = np.std(Y[:50])*5
    YStart = Y[startIndex]
    YSign  =np.sign(Y[startIndex])
    #print(value,diffs)
    #print(np.where(value == abs(diffs))[0][0])
    riseIndex = int(np.where(value == abs(diffs))[0][0]) + startIndex
    diffs = Y[startIndex:peakIndex]-LowerThreshold
    value = np.min(abs(diffs))
    fallIndex =  int(np.where(value == abs(diffs))[0][0]) + startIndex
    riseTimestart = Interpolator(X, Y, riseIndex-1,riseIndex+1,UpperThreshold)
    riseTimeend = Interpolator(X, Y, fallIndex-1,fallIndex+1,LowerThreshold)
    #print(UpperThreshold,LowerThreshold)
    if riseTimestart < X[startIndex] or riseTimestart > X[EndIndex] or riseTimeend < X[startIndex] or riseTimeend > X[EndIndex]:
        return False
    if riseTimestart - riseTimeend > (X[EndIndex] - X[startIndex]):
        return False
    if riseTimestart - riseTimeend <= 0:
        return False
    if riseIndex == 0 or fallIndex ==0:
        return False
    if YSign > 0:
        if(YStart > baseline + noiserms):
            return False
    if YSign < 0:
        if(YStart < baseline - noiserms):
            return False
    if len(np.unique(np.sign(np.diff(Y[fallIndex:startIndex])))) > 1:
        return False

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

def get_hist(ax,nbins):
    n,bins = [],[]
    finaln,finalbins = [],[]
    bin = 0
    iteration = 0
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0) # left edge of each bin
    finaln = [n[i:i + nbins] for i in range(0, len(n), nbins)]
    finalbins = [bins[i:i + nbins] for i in range(0, len(bins), nbins)]
    print(finaln)
    i = 0
    for (arrayn,arrabins) in zip(finaln,finalbins):
        if i ==0:
            n = arrayn
            bins = arrabins
        else:
            n = [n[i]+arrayn[i] for i in range(0,len(arrayn))]
        i+=1


    bins.append(x1) # also get right edge of last bin

    return np.asarray(n,dtype=np.float32),np.asarray(bins,dtype=np.float32)

def FindHistPeaks(Y):
    peaks, properties  = scipy.signal.find_peaks(Y, width=10,height =5,prominence= 2,distance = 15)
    return peaks,properties

FileNames = askopenfilenames(
    filetypes=[("Binary Files", "*.dat")])
with DRS4BinaryFile(FileNames[0]) as events:
    length = len(list(events))
itertor = 1
GainArray = []
GainErrorArray = []
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
                            if RiseTime == False:
                                continue
                            PulseHeight = hitAmplitude
                            Charge = ChargeCalculator(Data,startIndex,EndIndex)
                            PeakTime =  Time[hitAmplitudeIndex]
                            ChargePedestle = ChargeCalculator(Data,0,50)
                            TempData = pd.DataFrame(data = {'0':[RiseTime],'1':[PulseHeight],'2':[Charge],'3':[PeakTime],'4':[rmsnoise],'5':[baseline],'6':[baseline+rmsnoise],'7':[ChargePedestle]})
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
            columnNames = ["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i),"Channel {} RMS Noise".format(i),"Channel {} Baseline".format(i),"Channel {} Pedestle".format(i),"Channel {} Charge Pedestle".format(i)]
        else:
            columnNames.extend(["Channel {} Rise Time".format(i),"Channel {} Pulse Height".format(i),"Channel {} Cummulative Charge".format(i),"Channel {} Pulse Time".format(i),"Channel {} RMS Noise".format(i),"Channel {} Baseline".format(i),"Channel {} Pedestle".format(i),"Channel {} Charge Pedestle".format(i)])

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
    ChargeColumns = [column for column in columnNames if "Cummulative Charge" in column]
    histPulseHieghts = Data.plot.hist(y = PulseandNoiseColumns,bins =1000,alpha = .3,subplots=False,title = 'Pulse Height Distributions',log=False,sharex = True)
    plt.xlabel('Pulse Height (mV)')
    plt.savefig(os.path.join(newDirectory,'Pulse_Height_Distribution.png'))

    histCharge = Data.plot.hist(y = ChargeColumns,bins =1000,alpha = .3,subplots=False,title = 'Pulse Area Distribution',log=False,sharex = True)
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

    n, bins = get_hist(histCharge,1000)

    bincenters = np.asarray([(bins[i]+bins[i-1])/2 for i in range(1,len(bins))],np.float32)

    peaks,properties = FindHistPeaks(n)
    #plt.plot(bincenters[peaks],n[peaks],'g+')
    widths = scipy.signal.peak_widths(n, peaks, rel_height=0.5)
    j = 0
    scale = .5
    for (peak,width) in zip(peaks,widths[0]):
        try:
            true_width = abs(bincenters[int(peak - width/2)]-bincenters[int(peak + width/2)])
            X = bincenters[int(peak - width):int(peak + width)]
            Y = n[int(peak - width):int(peak + width)]
            mean,std = weighted_avg_and_std(X,Y)
            p0 = [n[peak], mean, std]
            bounds = [(0,.5*mean,.5*std),(n[peak],1.5*mean,1.5*std)]
            res = least_squares(gaussMinimizer, p0, loss='linear', f_scale=scale,args=(X,Y),bounds = bounds,xtol = 1E-20,ftol = 1E-15,x_scale = 'jac',tr_solver = 'lsmr',max_nfev=1E4)
            mu.append(res.x[1])
            variance.append(res.x[2]**2)
            amp.append(res.x[0])

        except:
            pass
        print(mu,len(mu))
        j = j+1

    # for i in range(0,len(mu)+1):
    #     if i==0 and i < len(mu):
    #         mod = Model(PEModel,prefix = 'f{}_'.format(i))
    #         pars = mod.make_params(verbose = True)
    #         pars['f{}_n'.format(i)].set(value = i+1,vary = False)
    #         pars['f{}_A'.format(i)].set(value = amp[i],min = 0)
    #         pars['f{}_mu'.format(i)].set(value = 1,min = 0,max = 100)
    #         pars['f{}_sigma'.format(i)].set(min = 0,value = np.sqrt(variance[i]))
    #         pars['f{}_gain'.format(i)].set(brute_step= 1E5,min = 1E5,max = 1E10,value = 1E8)
    #     elif i == 0 and not len(mu):
    #         mod = Model(PEModel,prefix = 'f{}_'.format(i))
    #         pars = mod.make_params(verbose = True)
    #         pars['f{}_n'.format(i)].set(value = i+1,vary = False)
    #         pars['f{}_A'.format(i)].set(value = amp[i],min = 0)
    #         pars['f{}_mu'.format(i)].set(value = 1,min = 0,max = 100)
    #         pars['f{}_sigma'.format(i)].set(min = 0)
    #         pars['f{}_gain'.format(i)].set(brute_step= 1E5,min = 1E5,max = 1E10,value = 1E8)
    #     else:
    #         tempmod = Model(PEModel,prefix = 'f{}_'.format(i))
    #         temppars = tempmod.make_params(verbose = True)
    #         temppars['f{}_n'.format(i)].set(value = i+1,vary = False)
    #         mod+=tempmod
    #         pars+=temppars
    #         pars['f{}_gain'.format(i)].set(expr ='f{}_gain'.format(i-1) )
    #         pars['f{}_mu'.format(i)].set(expr = 'f{}_mu'.format(i-1))
    #         pars['f{}_sigma'.format(i)].set(min = 0)
    #         pars['f{}_A'.format(i)].set(value = 50,min = 0)
    #WORKING
    for i in range(0,len(mu)+1):
        if i == 0 and i < len(mu):
            mod = GaussianModel(prefix = 'f{}_'.format(i))
            pars = mod.guess(n,x=bincenters, sigma=np.sqrt(variance[i]),height = amp[i],center = mu[i])
            if len(mu) >=2:
                pars.add('G',value = mu[i+1]-mu[i],brute_step=.01*mu[i],min = .1*(mu[i+1]-mu[i]),max = 5*(mu[i+1]-mu[i]))
            else:
                pars.add('G',value = 1E8,min=1E5,max=1E10,brute_step = .1E5)
        elif i >= len(mu) and i !=0:
            tempmod =  GaussianModel(prefix = 'f{}_'.format(i))
            temppars = tempmod.guess(n,x=bincenters)
            pars += temppars
            pars['f{}_center'.format(i)].set(expr='G+f{}_center'.format(i-1))
            mod += tempmod
        elif i < len(mu):
            tempmod =  GaussianModel(prefix = 'f{}_'.format(i))
            temppars = tempmod.guess(n,x=bincenters,center = mu[i], sigma=np.sqrt(variance[i]),height = amp[i])
            pars += temppars
            pars['f{}_center'.format(i)].set(expr='G+f{}_center'.format(i-1))
            #pars['f{}_center'.format(i-1)].set(expr='G-f{}_center'.format(i))
            mod += tempmod
        elif not len(mu) and i==0:
            mod = GaussianModel(prefix = 'f{}_'.format(i))
            pars = mod.guess(n,x=bincenters)
            pars.add('G',value = 1E8,min=1E5,max=1E10,brute_step = .1E5)
        else:
            pass

    result = mod.fit(n, pars, x=bincenters)
    print(result.fit_report())
    #plt.plot(bincenters,n,'y')
    #plt.plot(mu,amp,'k+')
    vals = result.params.valuesdict()
    mu = []
    amp = []
    for (key,value) in vals.items():
        if 'center' in key:
            mu.append(value)
        if 'height' in key:
            amp.append(value)
    mu = [0] +mu
    amp = [0]+amp
    #print(mu,amp)
    plt.plot(mu,amp,'g+')
    vals = pars.valuesdict()
    #print(result.params['G'].value)
    GainError = result.params['G'].stderr
    Gain = result.params['G'].value
    #print(result.params["*_center"])
    if GainError == None:
        GainError = 1
        print('Error approximation failed!')
    GainArray.append(Gain)
    GainErrorArray.append(GainError)
    lamd = 0
    if len(mu) >= 2:
        newmu =  np.arange(0,len(mu)) #print(result.params['G'].stderr)
        newamp = np.array(amp)#/np.linalg.norm(amp) #[x/np.linalg.norm(amp) for x in amp]
        p0= [1,newamp[1]*10]
        bounds = [(0,newamp[1]),(newmu[-1],20*newamp[1])]
        res = least_squares(poissonMinimizer, p0, loss='linear',args=(newmu, newamp),bounds=bounds,gtol = 1E-50,xtol = 1E-50,ftol = 1E-50,x_scale = 'jac',tr_solver = 'lsmr',max_nfev=1E4)
        fit = poisson(res.x,newmu)
        lamd = res.x[0]
        #plt.stem(newmu*Gain, fit, 'r--',label =r"<$\mu$> = {}".format(np.round(res.x[0],3)))
    plt.plot(bincenters,result.best_fit,'k',label = 'Gain = {:.2e} +/- {:.2e}, <$\mu$> = {}'.format(Gain,GainError,np.round(lamd,3)))

    plt.legend(loc = 'best')
    plt.savefig(os.path.join(newDirectory,'Pulse_Area_Distribution.png'))
    Text = []
    for i in NumberofChannels:
        values = Data['Channel {} Rise Time'.format(i)].values
        values = [x for x in values if abs(x - np.mean(values)) < 3*np.std(values)]
        [ToFMean, TofStd] = weighted_avg_and_std(values,np.ones(len(Data.index)))
        Text.append(r'$\tau_{}: \mu = {}ns; \sigma = {}ns$'.format(i,ToFMean,TofStd))
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
    GainPlotting = str(input("Voltage Based Gain Plotting? (y/n): "))
    plt.figure()
    if GainPlotting == 'y':
        LowVoltage = float(input("Lower Voltage?: "))
        HighVoltage = float(input("Higher Voltage?: "))
        Breakdown = float(input("Breakdown Voltage?: "))
        X = np.linspace(LowVoltage,HighVoltage,len(GainArray)) - Breakdown
        plt.xlabel('Breakdown Voltage (V)')
    else:
        X = [1,2,3,4,5.8,8]
        print(X,GainArray,GainErrorArray)
        #plt.errorbar(voltages, GainArray, yerr=GainErrorArray, fmt='o')
        plt.errorbar(X, GainArray, yerr=GainErrorArray, fmt='o')
        p = np.polyfit(X,GainArray,deg = 1)
        p = np.poly1d(p)
        plt.plot(X,p(X),label =r'Fit: m = {:.2E}, b = {:.2E}, $\mu$ = {:.2E}'.format(p[1],p[0],np.mean(GainArray)))

        plt.legend(loc = 'best')
        plt.ylabel('Gain')
        plt.xlabel('Light Intensity (A.U)')
        plt.ylim(1E6,2E8)
        plt.savefig('Gain_Plot.png')
    print("Analysis of Files Complete!")
    plt.show()
