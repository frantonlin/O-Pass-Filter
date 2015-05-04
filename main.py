import alsaaudio
import pyglet
from pyglet.window import key
import scikits.audiolab
import numpy as np
import struct

class Filter(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, width=512, height=512,visible=True)
        self.recording = False
        self.inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,0)
        self.inp.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)
        self.inp.setchannels(1)
        self.inp.setrate(16000) # Sets sampling rate to 16000 Hz
        self.inp.setperiodsize(320)
        self.data = []

    def on_draw(self):
        self.clear()
        #print "Recording: " + str(self.recording)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.recording = True

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.recording = False
            #perform filtering
            self.data = np.array(self.data)
            norm = self.data/max(self.data)
            scikits.audiolab.play(norm, fs=16000)
            #print norm
            pyglet.app.exit()


    def update(self, dt):
        if self.recording:
            l, data = self.inp.read()
            if data:
                floats = struct.unpack('f'*320,data)
                self.data.extend(floats)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/16000.0)
        pyglet.app.run()

if __name__ == '__main__':
    flt = Filter()
    flt.run()
