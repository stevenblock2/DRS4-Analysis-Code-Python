import numpy as np
import scipy.signal as scisig

def FindPedestal(p, m):
    noOutlier = RejectOutliers(p, m=m)
    return np.mean(noOutlier)



def RejectOutliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def WaveformDiscriminator(p,
                          noiseSigma,
                          nNoiseSigmaThreshold=1,
                          sgFilter=True,
                          sgWindow=15,
                          sgPolyOrder=3):
    [baselineVal, noiseInADC] = FindDigitizedPedestal(p=p, m=3, nBits=12, dynamicRange=1, noiseSigma=noiseSigma)
    if sgFilter:
        filter_p = scisig.savgol_filter(x=p, window_length=sgWindow, polyorder=sgPolyOrder)
        hitLogic = np.array(
            [(True if pi < baselineVal - nNoiseSigmaThreshold * noiseInADC else False) for pi in filter_p])
    else:
        hitLogic = np.array([(True if pi < baselineVal - nNoiseSigmaThreshold * noiseInADC else False) for pi in p])
    return [hitLogic, baselineVal, noiseInADC]
def DiscriminatorConditioning(p,
                              noiseSigmaInVolt,
                              durationTheshold=5,
                              adjDurationThreshold=5,
                              nNoiseSigmaThreshold=1,
                              sgFilter=True,
                              sgWindow=15,
                              sgPolyOrder=3):

    [baseline, noiseInADC] = [np.mean(p[:50]),np.std(p[:50])]
    hitLogic = hitLogic = np.array(
        [(True if pi < baseline - nNoiseSigmaThreshold * noiseSigmaInVolt else False) for pi in p])
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

    return [hitLogic, baseline, noiseSigmaInVolt]

def startTimeFinder(p,
              noiseSigmaInVolt,
              cfdThreshold=0.2,
              durationTheshold=10,
              adjDurationThreshold=5,
              nNoiseSigmaThreshold=3,
              sgFilter=True,
              sgWindow=15,
              sgPolyOrder=3):
    [hitLogic, baseline, noiseSigma] = DiscriminatorConditioning(p=p,
                                                                 noiseSigmaInVolt=noiseSigmaInVolt,
                                                                 durationTheshold=durationTheshold,
                                                                 adjDurationThreshold=adjDurationThreshold,
                                                                 nNoiseSigmaThreshold=nNoiseSigmaThreshold,
                                                                 sgFilter=sgFilter,
                                                                 sgWindow=sgWindow,
                                                                 sgPolyOrder=sgPolyOrder)

    hitStartIndexList = []
    hitPeakAmplitude = []
    hitPeakIndexArray = []
    hitStartIndex = 0
    hitAmplitude=0
    hitPeakIndex=0
    for i in range(1, np.size(hitLogic)):
        if ((not hitLogic[i - 1]) and hitLogic[i]) and hitLogic[i] and hitStartIndex == 0:
            hitAmplitude = 1E100
            hitPeakIndex = i
            for j in range(i, np.size(hitLogic) - 1):
                if p[j] < hitAmplitude:
                    hitAmplitude = p[j]
                    hitPeakIndex = j
                if not hitLogic[j + 1]:
                    break
            ThresholdADC = baseline + (cfdThreshold * (hitAmplitude-baseline ))

            hitStartIndex = i
            for j in range(hitPeakIndex, 0, -1):
                if (p[j-1] <= ThresholdADC and p[j ] > ThresholdADC):
                    hitStartIndex = j-1
                    break

            #hitStartIndexList = np.append(hitStartIndexList, hitStartIndex)
            #hitPeakAmplitude = np.append(hitPeakAmplitude, hitAmplitude)
            #hitPeakIndexArray = np.append(hitPeakIndexArray, hitPeakIndex)
    return hitStartIndex
