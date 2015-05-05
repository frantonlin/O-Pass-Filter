import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import argrelmax, argrelmin
import scikits.audiolab

def findFormants(y, Fs, plot=False):
    # number of samplepoints
    n = len(y)

    # sample spacing
    T = n/float(Fs)
    k = np.arange(n)
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fftpack.rfft(y) # fft computing
    Y = Y[range(n/2)] # one side
    Y = Y[:0.3*n]

    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,n*Ts,Ts) # time vector
    
    absY = abs(Y)

    peak_locs = argrelmax(absY, order=2)[0]
    trough_locs = argrelmin(absY, order=4)[0]
    extrema_locs = np.sort(np.concatenate((peak_locs, trough_locs)))

    #peak_frq = [frq[i] for i in peak_locs]
    #peak_Y = [absY[i] for i in peak_locs]

    extrema_frq = [frq[i] for i in extrema_locs]
    extrema_Y = [absY[i] for i in extrema_locs]

    k_locs = argrelmax(np.array(extrema_Y))[0]
    k_frq = [extrema_frq[i] for i in k_locs]
    k_Y = [extrema_Y[i] for i in k_locs]

    if plot:
        plt.subplot(2,2,1)
        plt.plot(t,y,'b-',markersize=0.3)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        plt.subplot(2,2,2)
        plt.hold(True)
        for i in range(len(absY)):
            plt.plot([frq[i],frq[i]],(0,absY[i]),'r',markersize=0.1)
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')

        plt.subplot(2,2,4)
        plt.plot(extrema_frq, extrema_Y, 'gx-')
        plt.plot(k_frq, k_Y, '*')

        plt.subplot(2,2,3)
        plt.plot(k_frq[1], k_frq[0], 'x')
        plt.xlabel("F2 (Hz)")
        plt.ylabel("F1 (Hz)")

        plt.show()

    return k_frq[:3]

def parseAudio(data, Fs, plot=False):
    formants = []
    """for i in range(len(data)/320):
        frame = data[i*320:i*320+320]
        window = np.hanning(len(frame))
        hanned = frame*window
        formants.append(findFormants(hanned, Fs))

    if plot:
        for frame in formants:
            plt.plot(frame[1],frame[0],'rx')
        plt.show()
    return formants"""
    i = 40
    print len(data)
    framelen = 1000
    frame = data[i*framelen:i*framelen+framelen]
    window = np.hanning(len(frame))
    hanned = frame*window
    #scikits.audiolab.play(data, fs=Fs)
    formants.append(findFormants(hanned, Fs, True))
    return formants


"""Fs, sig = wavfile.read("men/m01ae.wav")
print Fs

n = len(sig)
Ts = 1.0/Fs; # sampling interval

# cut to only vowel
sig = sig[int(n*0.45):int(n*0.474)]
print sig

norm = sig/float(max(sig))
print type(norm)

# hanning
window = np.hanning(len(norm))
hanned = norm*window
print hanned

scikits.audiolab.play(hanned, fs=Fs)

print findFormants(hanned, Fs, plot=True)"""


"""Fs, sig = wavfile.read("men/m01ae.wav")
norm = sig/float(max(sig)*2)
scikits.audiolab.play(norm, fs=Fs)
formants = parseAudio(norm, Fs)"""
#for frame in formants:
#    plt.plot(frame[1],frame[0],'rx')
#plt.show()