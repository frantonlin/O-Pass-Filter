import alsaaudio
import pyglet
from pyglet.window import key
import scikits.audiolab
import numpy as np
import struct
from fft import parseAudio
from math import sqrt

from scipy.io import wavfile
import matplotlib.pyplot as plt

SF = 44100
PERIOD = 1024
#UH = [400,1360]
#UH = [370,880] # Nur's OH

class Filter(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, width=512, height=512,visible=True)
        self.recording = False
        self.data = []
        self.filter = []

    def on_draw(self):
        self.clear()
        #print "Recording: " + str(self.recording)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.recording = True
            self.data = []
            self.filter = []
            self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,0)
            self.inp.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)
            self.inp.setchannels(1)
            self.inp.setrate(SF) # Sets sampling rate to SF Hz
            self.inp.setperiodsize(PERIOD)

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.recording = False
            self.inp.pause(1)
            # perform filtering
            self.data = np.array(self.data)
            norm = self.data/max(self.data)
            if len(norm)%2048 != 0:
                norm = norm[:-1024]
            formants = parseAudio(norm, SF, True)
            #print "len of formants: " + str(len(formants))
            #print "expected len: " + str(len(norm)/2048.0)
            for frame in formants:
                #print frame
                dist = sqrt((frame[0] - UH[0])**2 + (frame[1] - UH[1])**2)
                if dist > 300:
                    self.filter.extend(np.ones(2048))
                else:
                    #self.filter.extend(np.ones(2048)*dist/600.0)
                    self.filter.extend(np.zeros(2048))

            #print "len of filter: " + str(len(self.filter)/2048.0)
            filtered = np.array(self.filter)*norm
            #print self.filter

            plt.subplot(2,1,1)
            plt.plot(norm)
            plt.subplot(2,1,2)
            plt.plot(filtered)
            plt.show()

            scikits.audiolab.play(filtered, fs=SF)

            """Fs, self.data = wavfile.read("nur.wav")
            print self.data
            norm = self.data/float(max(self.data))
            norm = norm[:-512]
            #print len(norm)%2048
            formants = parseAudio(norm, SF, False)
            for frame in formants:
                print frame
                dist = sqrt((frame[0] - UH[0])**2 + (frame[1] - UH[1])**2)
                if dist > 200:
                    self.filter.extend(np.ones(2048))
                else:
                    #self.filter.extend(np.ones(2048)*dist/600.0)
                    self.filter.extend(np.zeros(2048))

            filtered = np.array(self.filter)*norm

            scikits.audiolab.play(filtered, fs=SF)"""

            #pyglet.app.exit()


    def update(self, dt):
        if self.recording:
            l, data = self.inp.read()
            if data:
                floats = struct.unpack('f'*PERIOD,data)
                self.data.extend(floats)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1.0/SF)
        pyglet.app.run()

def process_wav(filename):
    Fs, sig = wavfile.read(filename)

    n = len(sig)
    Ts = 1.0/Fs; # sampling interval
    norm = sig/float(max(sig))
    if len(norm)%2048 != 0:
        norm = norm[:-1*(len(norm)%2048)]

    formants = parseAudio(np.array(norm), Fs, True)

    flt = []

    for frame in formants:
        dist = sqrt((frame[0] - UH[0])**2 + (frame[1] - UH[1])**2)
        if dist > 300:
            flt.extend(np.ones(2048))
        else:
            #self.filter.extend(np.ones(2048)*dist/200.0)
            flt.extend(np.zeros(2048))

    #print "len of filter: " + str(len(self.filter)/2048.0)
    filtered = np.array(flt)*norm
    #print self.filter

    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,len(norm)*Ts,Ts) # time vector

    plt.subplot(2,1,1)
    plt.plot(t, norm)
    plt.ylabel('Normalized Amplitude')
    plt.subplot(2,1,2)
    plt.plot(t, filtered)
    plt.ylabel('Amplitude after Filtering')
    plt.xlabel('Time (s)')
    plt.show()

    scikits.audiolab.play(filtered, fs=SF)

if __name__ == '__main__':
    #lt = Filter()
    #flt.run()
    process_wav('peanut_butter.wav')
