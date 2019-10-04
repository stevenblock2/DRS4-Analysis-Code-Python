import numpy as np
import scipy.signal as scisig

def DiscriminatorConditioning(p,
                              noiseSigmaInVolt,
                              durationTheshold=5,
                              adjDurationThreshold=5,
                              nNoiseSigmaThreshold=1,
                              sgFilter=True,
                              sgWindow=15,
                              sgPolyOrder=3):
    [hitLogic, baseline, noiseSigma] = WaveformDiscriminator(p=p,
                                                             noiseSigma=noiseSigmaInVolt,
                                                             nNoiseSigmaThreshold=nNoiseSigmaThreshold,
                                                             sgFilter=sgFilter,
                                                             sgWindow=sgWindow,
                                                             sgPolyOrder=sgPolyOrder)

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

    return [hitLogic, baseline, noiseSigma]
    
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
            ThresholdADC = baseline - (cfdThreshold * (baseline - hitAmplitude))

            hitStartIndex = i
            for j in range(hitPeakIndex, 0, -1):
                if (p[j] <= ThresholdADC and p[j - 1] > ThresholdADC):
                    hitStartIndex = j - 0.5
                    break

            hitStartIndexList = np.append(hitStartIndexList, hitStartIndex)
            hitPeakAmplitude = np.append(hitPeakAmplitude, hitAmplitude)
            hitPeakIndexArray = np.append(hitPeakIndexArray, hitPeakIndex)

    return [hitStartIndexList, hitPeakAmplitude, hitPeakIndexArray, hitLogic, baseline, noiseSigma]
