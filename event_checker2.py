from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import sys, os
import h5py
import matplotlib.pyplot as plt
import pickle
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QMainWindow, QFrame,
                             QTextEdit, QGridLayout, QApplication, QSplitter, QVBoxLayout,
                             QHBoxLayout, QPushButton, QCheckBox, QComboBox, QShortcut)

import pyqtgraph.dockarea as dc

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

data_file = h5py.File('out_merge_f16.h5', 'r')
data_cumul = h5py.File('out_250.h5', 'r')


data = data_file['alldata']
f3050 = data_cumul['f30t50a250']

n_chan = 6
i_chan = 0

ult_max = len(data)

d_col= {0:(222, 184, 135),
          1:(0,0,255),
          2:(255,255,0),
          3:(125,0,0),
          4:(255,20,147),
          5:(255,0,0)}


vel = 10000


events = [(1000,2000, 'ttt'), (4000, 5000, "unknown"), (6000, 7000, "shite") ]




#%%%

class MovableRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def moveDist(self, dist):
        self.setRegion([self.getRegion()[0]+dist,self.getRegion()[1]+dist])
        
    def jumpDist(self, start, stop):
        self.setRegion([start,stop])


class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        self.channel = None
        self.limit = 10000 # maximum number of samples to be plotted
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        
    def setHDF5(self, data, channel = None, shift = 0.0) :
        self.hdf5 = data
        self.channel = channel
        self.updateHDF5Plot()
        self.shift = shift
        
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
            
        self.setData(self.hdf5[start:stop, self.channel]) # update the plot
        self.setPos(start, self.shift) # shift to match starting index
        self.resetTransform()
        self.scale(scale, 1)  # scale to match downsampling
        
        
class HDF5Plot1CH(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        self.channel = None
        self.limit = 100000 # maximum number of samples to be plotted
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        
    def setHDF5(self, data):
        self.hdf5 = data
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
            visible = self.hdf5[start:stop]
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
                chunk = self.hdf5[sourcePtr:min(stop,sourcePtr+chunkSize)]
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
            
        self.setData(self.hdf5[start:stop]) # update the plot
        self.setPos(start, 0) # shift to match starting index
        self.resetTransform()
        self.scale(scale, 1) 
              
       

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




class EventDetail(QWidget):
   def __init__(self):
        super().__init__()
        
        
        self.number = 0
        self.status = "unknown"
        self.f3050max = 0
        self.f3050sum = 0
        self.f3050aver = 0
        self.length = 0
        self.saved = False
        
        
        
        self.initUI()

        
   def initUI(self):
        

        l_number = QLabel(f"Номер: {self.number}")
        l_length = QLabel(f'Продолжительность, с: {self.length}')
        l_status = QLabel(f'Статус: {self.status}' )
        
        t_comment = QTextEdit()
        
        c_autosave = QCheckBox('Auto:', self)
        
        b_prev = QPushButton("Prev.", self)
        b_next = QPushButton("Next", self)
        b_update = QPushButton("Update", self)

        
        combo = QComboBox(self)
        combo.addItem('unknown')
        combo.addItem('questionable')
        combo.addItem('preSWD')
        combo.addItem('SWD')


        
        
        grid = QGridLayout()
        grid.setSpacing(10)
        
        grid.addWidget(l_number, 1, 0, 1, 2)
        grid.addWidget(l_length, 2, 0, 1, 2)
        grid.addWidget(l_status, 3, 0, 1, 2)
        
        
        grid.addWidget(combo, 4, 0, 1, 2)
        
        grid.addWidget(t_comment, 5, 0, 1, 2)
        
        
        grid.addWidget(c_autosave, 6, 0)
        grid.addWidget(b_update, 6, 1)
        
        grid.addWidget(b_prev, 7, 0)
        grid.addWidget(b_next, 7, 1)
        
        
        
        self.setLayout(grid)
        self.show()



    
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()
        
    
    def initUI(self):
        self.eventchoose = EventChoose()
        self.datachoose = DataChoose()
        self.eventdetail = EventDetail()

        
        vbox = QVBoxLayout()
        
        vbox.addWidget(self.eventchoose)
        vbox.addWidget(self.datachoose)
        vbox.addStretch(1)
        vbox.addWidget(self.eventdetail)
        
        
        
        labelsframe = QFrame()
        labelsframe.setLayout(vbox)
        
        
        self.plotwindow = PlotWindow()
        
        splitter1 = QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(self.plotwindow)
        splitter1.addWidget(labelsframe)

        
        self.setWindowTitle("Просмотр заранее обнаруженных событий")
        self.setCentralWidget(splitter1)
        
        self.rightSc = QShortcut(QtGui.QKeySequence("right"), self)
        self.rightSc.activated.connect(lambda: self.plotwindow.plotMove(vel))
        
        self.rightSc = QShortcut(QtGui.QKeySequence("left"), self)
        self.rightSc.activated.connect(lambda: self.plotwindow.plotMove(-vel))
        
        self.show()


class PlotWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.initUI()
        self.updatePlot()
        
    def updatePlot(self):
        self.p1.setXRange(*self.region.getRegion(), padding=0)
        
        
    def updateRegion(self):
        self.region.setRegion(self.p1.getViewBox().viewRange()[0])

    def plotMove(self, dist):
        Xmin = self.p3.getViewBox().viewRange()[0][0]+dist
        Xmax = self.p3.getViewBox().viewRange()[0][1]+dist
        self.p3.setXRange(Xmin, Xmax, padding = 0)
        self.region.moveDist(dist) #при улетании в 0 крашится
        
        
    def plotJump(self, start, stop):
        Xmin = max(0, start-10000)
        Xmax = min(stop+10000, ult_max)
        
        
        self.p3.setXRange(Xmin, Xmax, padding = 0)
        self.region.jumpDist(start, stop) 



    def initUI(self):
        
        
        
        self.dockarea = dc.DockArea()
        self.setCentralWidget(self.dockarea)
        
        d1 = dc.Dock("График выбранного канала")
        d2 = dc.Dock("График критического параметра")
        d3 = dc.Dock("График всех каналов")
        
        d1.hideTitleBar()
        d2.hideTitleBar()
        d3.hideTitleBar()
        
        self.p1 = pg.PlotWidget(title = "Выбранный канал")
        curve1 = HDF5Plot()
        curve1.setHDF5(data, channel = i_chan)
        curve1.setPen(pg.mkPen('k'))
        self.p1.addItem(curve1)
        self.p1.setXRange(0, 10000)
        self.p1.getViewBox().invertY(True)
        self.p1.showGrid(x=True, y=True)
        self.p1.setYRange(-2.0, 2.0)
        
        
        
        p2 = pg.PlotWidget(title = "Критический параметр")
        curve2 = HDF5Plot1CH()
        curve2.setHDF5(f3050)
        curve2.setPen(pg.mkPen('r'))
        p2.addItem(curve2)
        curve3 = HDF5Plot()
        curve3.setHDF5(data, channel = 5)
        curve3.setPen(pg.mkPen('k'))
        p2.addItem(curve3)
        
        
        
        p2.setXRange(0, 10000)
        p2.showGrid(x=False, y=True)
        p2.setYRange(-2, 5, padding = 0)
        p2.setXLink(self.p1)
        
        
        
        
        
        self.p3 = pg.PlotWidget(title = "Все каналы")
        
        for i in range(n_chan):
            curve = HDF5Plot()
            curve.setHDF5(data, i, i*1.0)
            curve.setPen(pg.mkPen(d_col[i]))
            self.p3.addItem(curve)
        self.p3.setXRange(0, 10000)
        self.p3.getViewBox().invertY(True)
        self.p3.setMouseEnabled(y=False)
        
        
        self.region = MovableRegion([1000,2000])
        self.region.setZValue(-10)
        self.p3.addItem(self.region)
        
        self.region.sigRegionChanged.connect(self.updatePlot)
        self.p1.sigXRangeChanged.connect(self.updateRegion)
        
        # self.sec_Roi = pg.LineSegmentROI(positions=([10, 1.5], [1010, 1.5]), pen = pg.mkPen('k') )
        # self.p1.addItem(self.sec_Roi)
        
        
        
        
        
        
        
        
        d1.addWidget(self.p1)
        d2.addWidget(p2)
        d3.addWidget(self.p3)
        
        self.dockarea.addDock(d1, 'top')
        self.dockarea.addDock(d2, 'bottom', d1)
        self.dockarea.addDock(d3, 'bottom', d2)
        
        
        
        
        
        self.show()
        

        
        
        
        # self.graphWidget = pg.PlotWidget()
        # self.setCentralWidget(self.graphWidget)

        # hour = [1,2,3,4,5,6,7,8,9,10]
        # temperature = [30,32,34,32,33,31,29,32,35,45]

        # # plot data: x, y values
        # self.graphWidget.plot(hour, temperature)
        

 #(start, stop, max_int, mean_int)






def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


data_file.close()
data_cumul.close()