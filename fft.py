import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
from scipy.signal import argrelmax
import scikits.audiolab

def findFormants(y, Fs, plot=True):
    # number of samplepoints
    n = len(y)
    print n

    # sample spacing
    T = n/float(Fs)
    k = np.arange(n)
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scipy.fftpack.fft(y) # fft computing
    Y = Y[range(n/2)] # one side
    Y = Y[:900]

    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,n*Ts,Ts) # time vector
    
    absY = abs(Y)

    peak_locs = argrelmax(absY, order=2)[0]

    peak_frq = [frq[i] for i in peak_locs]
    peak_Y = [absY[i] for i in peak_locs]

    k_locs = argrelmax(np.array(peak_Y))[0]
    k_frq = [peak_frq[i] for i in k_locs]
    k_Y = [peak_Y[i] for i in k_locs]

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
        plt.plot(peak_frq, peak_Y, 'gx-')
        plt.plot(k_frq, k_Y, '*')

        plt.subplot(2,2,3)
        plt.plot(k_frq[1], k_frq[0], 'x')
        plt.xlabel("F2 (Hz)")
        plt.ylabel("F1 (Hz)")

        plt.show()

    return k_frq[:3]

    def parseAudio(data):
        pass


Fs, sig = wavfile.read("men/m01ae.wav")
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

print findFormants(hanned, Fs, plot=True)