import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import find_peaks_cwt
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
    print len(Y)
    Y = Y[:900]

    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,n*Ts,Ts) # time vector

    plt.subplot(2,2,1)
    plt.plot(t,y,'b-',markersize=0.3)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(2,2,2)
    plt.hold(True)
    absY = abs(Y)
    for i in range(len(absY)):
        plt.plot([frq[i],frq[i]],(0,absY[i]),'r',markersize=0.1)
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    #plt.xlim([0, 5000])

    peak_locs = find_peaks_cwt(absY,np.arange(1,3),noise_perc=10)
    print peak_locs

    plt.subplot(2,2,4)
    peak_frq = [frq[i] for i in peak_locs]
    peak_Y = [absY[i] for i in peak_locs]
    plt.plot(peak_frq, peak_Y, 'gx-')
    #plt.xlim([0, 5000])

    #k_locs = find_peaks_cwt(peak_Y,np.arange(2,3))#,noise_perc=8)
    k_locs = get_local_maxes(peak_Y)
    plt.subplot(2,2,3)
    k_frq = [peak_frq[i] for i in k_locs]
    k_Y = [peak_Y[i] for i in k_locs]
    print k_Y
    plt.plot(k_frq, k_Y, '*')
    #plt.xlim([0, 5000])

    plt.show()

def get_local_maxes(a):
    locs = []
    for i in range(len(a)):
        if i > 0 and i < len(a)-1:
            if a[i] > a[i-1] and a[i] > a[i+1] and a[i] > 20:
                locs.append(i)
    return locs

Fs, sig = wavfile.read("men/m01ae.wav")
print Fs

n = len(sig)
Ts = 1.0/Fs; # sampling interval

# cut to only vowel
sig = sig[int(n*0.4):int(n*0.45)]


#norm = pcm2float(sig, 'float32')
norm = sig/float(max(sig))

# hanning
window = np.hanning(len(norm))
hanned = norm*window

scikits.audiolab.play(hanned, fs=Fs)

plotSpectrum(hanned, Fs)