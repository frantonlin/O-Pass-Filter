import alsaaudio
import pyaudio
import pyglet
from pyglet.window import key
import scikits.audiolab
import numpy as np
import struct
from fft import parseAudio

SF = 44100
PERIOD = 470

class Filter(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, width=512, height=512,visible=True)
        self.recording = False
        self.data = []

    def on_draw(self):
        self.clear()
        #print "Recording: " + str(self.recording)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.recording = True
            self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,0)
            self.inp.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)
            self.inp.setchannels(1)
            self.inp.setrate(SF) # Sets sampling rate to SF Hz
            self.inp.setperiodsize(PERIOD)

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.recording = False
            self.inp.pause(1)
            #perform filtering
            self.data = np.array(self.data)
            norm = self.data/max(self.data)
            scikits.audiolab.play(norm, fs=SF)
            parseAudio(norm, SF, True)
            pyglet.app.exit()


    def update(self, dt):
        if self.recording:
            l, data = self.inp.read()
            if data:
                floats = struct.unpack('f'*PERIOD,data)
                self.data.extend(floats)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1.0/SF)
        pyglet.app.run()

if __name__ == '__main__':
    flt = Filter()
    flt.run()
