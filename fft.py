import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import find_peaks_cwt
from utility import pcm2float
import scikits.audiolab
from numpy.random import normal

def plotSpectrum(y, Fs):
    # number of samplepoints
    n = len(y)

    # sample spacing
    T = n/float(Fs)
    k = np.arange(n)
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fftpack.fft(y) # fft computing
    Y = Y[range(n/2)] # one side

    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,n*Ts,Ts) # time vector

    plt.subplot(2,2,1)
    plt.plot(t,y,'b-',markersize=0.3)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(2,2,3)
    plt.hold(True)
    absY = abs(Y)
    for i in range(len(absY)):
        plt.plot([frq[i],frq[i]],(0,absY[i]),'r',markersize=0.1)
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.xlim([0, 5000])

    peak_locs = find_peaks_cwt(absY,np.arange(2,3))
    print peak_locs

    plt.subplot(2,2,4)
    peak_frq = [frq[i] for i in peak_locs]
    peak_Y = [absY[i] for i in peak_locs]
    plt.plot(peak_frq, peak_Y, 'g')
    plt.xlim([0, 5000])

    plt.show()

Fs, sig = wavfile.read("ee.wav")

n = len(sig)
Ts = 1.0/Fs; # sampling interval

# cut to only vowel
sig = sig[int(n*0.11/(n*Ts)):int(n*0.3/(n*Ts))]


#norm = pcm2float(sig, 'float32')
norm = sig/float(max(sig))

# hanning
window = np.hanning(len(norm))
hanned = norm*window

scikits.audiolab.play(hanned, fs=Fs)

plotSpectrum(hanned, Fs)