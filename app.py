import pyaudio
import threading
import atexit
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget
from MainWindow import Ui_MainWindow
from msclap import CLAP
from dataset import ESC50
import torch.nn.functional as F

from scipy.io.wavfile import write
from  uuid import uuid4
import os


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

plt.style.use('dark_background')

class MicrophoneRecorder(object):
    def __init__(self, rate=44100, chunksize=1024, record_seconds=6):
        self.rate = rate
        self.chunksize = chunksize
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue
        
    def get_frames(self):
        with self.lock:
            frames = self.frames
            return frames
        
    def clear_frames(self):
        self.frames = []
    
    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()
        
        

class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)
        
        
class LiveFFTWidget(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        
        # customize the UI
        self.initUI()
        
        root_path = "dataset" # Folder with ESC-50-master/
        dataset = ESC50(root=root_path, download=True) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
        prompt = 'this is the sound of '
        
        self.clap = CLAP(version="2023", use_cuda=False)
        self.class_labels = [prompt + x for x in dataset.classes]
        self.text_embeddings = self.clap.get_text_embeddings(self.class_labels)
        # init class data
        self.initData()       
        
        # connect slots
        self.connectSlots()
        
        # init MPL widget
        self.initMplWidget()
        
    def initUI(self):

        hbox_gain = QtWidgets.QHBoxLayout()
        autoGain = QtWidgets.QLabel('Auto gain for frequency spectrum')
        autoGainCheckBox = QtWidgets.QCheckBox(checked=True)
        hbox_gain.addWidget(autoGain)
        hbox_gain.addWidget(autoGainCheckBox)
        
        # reference to checkbox
        self.autoGainCheckBox = autoGainCheckBox
        
        hbox_fixedGain = QtWidgets.QHBoxLayout()
        fixedGain = QtWidgets.QLabel('Manual gain level for frequency spectrum')
        fixedGainSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        hbox_fixedGain.addWidget(fixedGain)
        hbox_fixedGain.addWidget(fixedGainSlider)

        self.fixedGainSlider = fixedGainSlider

        vbox = self.verticalLayout
        vbox.addLayout(hbox_gain)
        vbox.addLayout(hbox_fixedGain)

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)
        
        # timer for callbacks, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        timer.start(100)
        # keep reference to timer        
        self.timer = timer
        
        timer1 = QtCore.QTimer()
        timer.timeout.connect(self.infer)
        timer1.start(7000)
        self.timer1 = timer1
        
        
     
    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()  

        # keeps reference to mic        
        self.mic = mic
        
        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize, 
                                         1./mic.rate)
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
                
    def connectSlots(self):
        pass
    
    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps 
        references for further use"""
        # top plot
        self.ax_top = self.main_figure.figure.add_subplot(211)
        self.ax_top.set_ylim(-3000, 3000)
        self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)
        #self.ax_top.figure.tight_layout()

        # bottom plot
        self.ax_bottom = self.main_figure.figure.add_subplot(212)
        self.ax_bottom.set_ylim(0, 1)
        self.ax_bottom.set_xlim(0, self.freq_vect.max())
        self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)
        #self.ax_bottom.figure.tight_layout()
        # line objects        
        self.line_top, = self.ax_top.plot(self.time_vect, 
                                         np.ones_like(self.time_vect))
        
        self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                               np.ones_like(self.freq_vect))
                                               
    def infer(self):
        frames = self.mic.get_frames()
        sounds = []
        if len(frames) > int(self.mic.rate / self.mic.chunksize * self.mic.record_seconds):
            for i in range(0, int(self.mic.rate / self.mic.chunksize * self.mic.record_seconds)):
                sounds.append(frames[-1*i])
            self.mic.clear_frames()
            sounds.reverse()
            data = np.concatenate(sounds, axis=0, out=None, dtype=None, casting="same_kind")
            file = f"wav\{uuid4()}.wav"
            write(file, self.mic.rate, data.astype(np.int16))
            # Extract audio embeddings
            a = r"C:\PythonCode\AudioClassifier\dataset\ESC-50-master\audio\1-4211-A-12.wav"
            audio_embeddings = self.clap.get_audio_embeddings([file])
            # Compute similarity between audio and text embeddings 
            similarity= self.clap.compute_similarity(audio_embeddings, self.text_embeddings)
            y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
            self.labelText.setText(self.class_labels[np.argmax(y_pred)])
            os.remove(file)
            

                                               
    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """        
        # gets the latest frames        
        frames = self.mic.get_frames()
        
        if len(frames) > 0:
            # keeps only the last frame
            current_frame = frames[-1]
            # plots the time signal
            self.line_top.set_data(self.time_vect, current_frame)
            # computes and plots the fft signal            
            fft_frame = np.fft.rfft(current_frame)
            if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                fft_frame /= np.abs(fft_frame).max()
            else:
                fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                #print(np.abs(fft_frame).max())
            self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))            
            
            # refreshes the plots
            self.main_figure.canvas.draw()
            
import sys
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LiveFFTWidget()
    window.show()
    sys.exit(app.exec_())