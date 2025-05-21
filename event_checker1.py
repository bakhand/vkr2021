from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import sys, os
import h5py
import matplotlib.pyplot as plt
import pickle
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QMainWindow, QFrame,
                             QTextEdit, QGridLayout, QApplication, QSplitter, QVBoxLayout,
                             QHBoxLayout, QPushButton)



class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        self.channel = None
        self.limit = 100000 # maximum number of samples to be plotted
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        
    def setHDF5(self, data, channel = None):
        self.hdf5 = data
        self.channel = channel
        self.updateHDF5Plot()
        
    def viewRangeChanged(self):
        self.updateHDF5Plot()
        
    def updateHDF5Plot(self):
        if self.hdf5 is None:
            self.setData([])
            return
        
        vb = self.getViewBox()
        if vb is None:
            return  # no ViewBox yet
        
        # Determine what data range must be read from HDF5
        xrange = vb.viewRange()[0]
        start = max(0,int(xrange[0])-1)
        stop = min(len(self.hdf5), int(xrange[1]+2))
        
        # Decide by how much we should downsample 
        ds = int((stop-start) / self.limit) + 1
        
        if ds == 1:
            # Small enough to display with no intervention.
            visible = self.hdf5[start:stop, self.channel]
            scale = 1
        else:
            # Here convert data into a down-sampled array suitable for visualizing.
            # Must do this piecewise to limit memory usage.        
            samples = 1 + ((stop-start) // ds)
            visible = np.zeros(samples*2, dtype=self.hdf5.dtype)
            sourcePtr = start
            targetPtr = 0
            
            # read data in chunks of ~1M samples
            chunkSize = (1000000//ds) * ds
            while sourcePtr < stop-1: 
                chunk = self.hdf5[sourcePtr:min(stop,sourcePtr+chunkSize), self.channel]
                sourcePtr += len(chunk)
                
                # reshape chunk to be integral multiple of ds
                chunk = chunk[:(len(chunk)//ds) * ds].reshape(len(chunk)//ds, ds)
                
                # compute max and min
                chunkMax = chunk.max(axis=1)
                chunkMin = chunk.min(axis=1)
                
                # interleave min and max into plot data to preserve envelope shape
                visible[targetPtr:targetPtr+chunk.shape[0]*2:2] = chunkMin
                visible[1+targetPtr:1+targetPtr+chunk.shape[0]*2:2] = chunkMax
                targetPtr += chunk.shape[0]*2
            
            visible = visible[:targetPtr]
            scale = ds * 0.5
            
        self.setData(visible) # update the plot
        self.setPos(start, 0) # shift to match starting index
        self.resetTransform()
        self.scale(scale, 1)  # scale to match downsampling

class EventChoose(QWidget):
   def __init__(self):
        super().__init__()
        
        self.ev_all = 0
        self.ev_done = 0
        self.events = None
        self.tr_level = None
        self.events_source = "NO EVENT LIST"
        
        
        self.initUI()
        

   def initUI(self):
        

        l_title = QLabel(f"Список: {self.events_source}")
        l_ev_all = QLabel(f'Событий всего: {self.ev_all}')
        l_ev_done = QLabel(f'Событий просмотрено: {self.ev_done}' )

        b_load_ev = QPushButton("Load", self)
        b_save_ev = QPushButton("Save", self)
        
        
        
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(l_title, 1, 0, 1, 2)
        grid.addWidget(l_ev_all, 2, 0, 1, 2)
        grid.addWidget(l_ev_done, 3, 0, 1, 2)
        grid.addWidget(b_load_ev, 4, 0)
        grid.addWidget(b_save_ev, 4, 1)
        
        self.setLayout(grid)
        self.show()


class DataChoose(QWidget):
   def __init__(self):
        super().__init__()
        
        self.ev_all = 0
        
        self.raw_data = None
        self.f3050 = None
        self.data_source = "NO EVENT LIST"
        self.f3050_source = "NO AVERAGED 30_50 ENARGY"
        self.hours_now = 0
        self.hours_all = 0
        self.initUI()
        

   def initUI(self):
        

        l_data = QLabel(f"Данные: {self.data_source}")
        l_f3050 = QLabel(f'Оценка f30_50: {self.f3050_source}')
        l_hours_now = QLabel(f'Часов с начала записи: {self.hours_now}' )
        l_hours_all = QLabel(f'Всего часов в записи: {self.hours_all}' )
        
        
        b_load_d = QPushButton("Load data", self)
        b_load_f = QPushButton("Load f3050", self)
        

        
        
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(l_data, 1, 0, 1, 2)
        grid.addWidget(l_f3050, 2, 0, 1, 2)
        grid.addWidget(l_hours_now, 3, 0, 1, 2)
        grid.addWidget(l_hours_all, 4, 0, 1, 2)
        
        grid.addWidget(b_load_d, 5, 0)
        grid.addWidget(b_load_f, 5, 1)
        
        
        
        self.setLayout(grid)
        self.show()


    
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()
        
    
    def initUI(self):
        self.eventchoose = EventChoose()
        self.datachoose = DataChoose()
        

        
        vbox = QVBoxLayout()
        vbox.addWidget(self.eventchoose)
        vbox.addWidget(self.datachoose)
        vbox.addStretch(1)
        
        
        labelsframe = QFrame()
        labelsframe.setLayout(vbox)
        
        
        self.plotwindow = PlotWindow()
        
        splitter1 = QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(self.plotwindow)
        splitter1.addWidget(labelsframe)

        
        self.setCentralWidget(splitter1)
        self.show()


class PlotWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        
        
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]

        # plot data: x, y values
        self.graphWidget.plot(hour, temperature)
        self.show()

events = list() #(start, stop, max_int, mean_int)

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


