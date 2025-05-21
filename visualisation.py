from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import sys, os
import h5py

N = 0

N_CHAN = 6
d_chan = {1:"ML",
          2:'SL',
          3:'MR',
          4:'SR',
          5:'FR',
          6:'act'}

d_col= {1:(222, 184, 135),
          2:(0,0,255),
          3:(255,255,0),
          4:(255,0,0),
          5:(255,20,147),
          6:(255,250,250)}





VEL = 6000


class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
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
            
        self.setData(visible) # update the plot
        self.setPos(start, 0) # shift to match starting index
        self.resetTransform()
        self.scale(scale, 1)  # scale to match downsampling










class KeyPressWindow(pg.GraphicsWindow):
    sigKeyPress = QtCore.pyqtSignal(object)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


class MovableRegion(pg.LinearRegionItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def moveDist(self, dist):
        self.setRegion([self.getRegion()[0]+dist,self.getRegion()[1]+dist])



swd_flag = False
mouse_pos = 0
prev_mouse_pos = 0

def keyPressed(evt):
    global swd_flag
    global prev_mouse_pos
    if evt.key() == 16777236:
        print("Right")
        region.moveDist(VEL)
        plotMove(main_plot, VEL)
    elif evt.key() == 16777234:
        print("Left")
        region.moveDist(-VEL)
        plotMove(main_plot, -VEL)
    elif evt.key() == 83:
        if not swd_flag:
            swd_flag = True
            prev_mouse_pos = mouse_pos
        else:
            swd_flag = False
            log_file.write("%0.1f,%0.1f\n" % (prev_mouse_pos, mouse_pos))
    else:
        evt.ignore()
   

def plotMove(plot, dist):
        Xmin = main_plot.getViewBox().viewRange()[0][0]+dist
        Xmax = main_plot.getViewBox().viewRange()[0][1]+dist
        plot.setXRange(Xmin, Xmax, padding = 0)


data_file = h5py.File('out_f16.h5', 'r')
cwt_file = h5py.File('Out_cwt.h5', 'r')








app = pg.mkQApp()
win = KeyPressWindow()
win.sigKeyPress.connect(keyPressed)
label = pg.LabelItem(justify='right')
win.addItem(label)
label.setText("Проверка связи")

key = list(cwt_file.keys())[N]
log_file = open(key + "_log.txt", "a")
log_file.write("START,STOP\n")

data_cwt  = cwt_file[key][:,:]


cwt_plot = win.addPlot(row=1, col=0, title = "CWT_CH_0")
zoom_plot = win.addPlot(row=2, col=0, title = "Zoom")
main_plot = win.addPlot(row=3, col=0, title = "Scroll")







for i in range(N_CHAN):
    curve = HDF5Plot()
    curve.setHDF5(data_file[key][:,i]-2*i)
    curve.setPen(d_col[i+1])
    main_plot.addItem(curve)
main_plot.setXRange(0, 60000)
    
region = MovableRegion([1000,2000])
region.setZValue(-10)
main_plot.addItem(region)


for i in range(N_CHAN):
    curve = HDF5Plot()
    curve.setHDF5(data_file[key][:,i]-2*i)  #,  pen=d_col[i+1], name=d_chan[i+1])
    curve.setPen(d_col[i+1])
    zoom_plot.addItem(curve)

zoom_plot.setXRange(1000, 2000)




img = pg.ImageItem(data_cwt[1000:2000,:])
cwt_plot.addItem(img)

cwt_plot.addLine(y = 26.0, )


def updatePlot():
    zoom_plot.setXRange(*region.getRegion(), padding=0)
    img.setImage(data_cwt[np.int(region.getRegion()[0]):np.int(region.getRegion()[1]),:])
def updateRegion():
    region.setRegion(zoom_plot.getViewBox().viewRange()[0])
    
region.sigRegionChanged.connect(updatePlot)
zoom_plot.sigXRangeChanged.connect(updateRegion)
updatePlot()


#cross hair
vLine = pg.InfiniteLine(angle=90, movable=False)
zoom_plot.addItem(vLine, ignoreBounds=True)

vb = zoom_plot.vb



def mouseMoved(evt):
    global mouse_pos
    global swd_flag
    global prev_mouse_pos
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    if zoom_plot.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        # index = int(mousePoint.x())
        # if index > 0 and index < len(data1):
        mouse_pos = (mousePoint.x()/1000)
        if swd_flag:
            label.setText("<span style='font-size: 12pt'>x=%0.1f <span style='color: red'>START=%0.1f</span></span>" % (mouse_pos,
                                                                                                                     prev_mouse_pos))
        else:
            label.setText("<span style='font-size: 12pt'>x=%0.1f " % mouse_pos)
        vLine.setPos(mousePoint.x())








proxy = pg.SignalProxy(zoom_plot.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)
#p1.scene().sigMouseMoved.connect(mouseMoved)




app.exec_()


cwt_file.close()
data_file.close()
log_file.close()