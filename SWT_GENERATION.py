from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import sys, os
import h5py
from scipy import signal

#%%
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


data_file = h5py.File('Out_f16.h5', 'r')

def do_cwt(k):
    long_data = data_file[k][:,3]
    w = 30
    fs = 1000
    freq = np.linspace(3, 30, 100)
    widths = w*fs / (2*freq*np.pi)
    cwtm_swd = signal.cwt(long_data, signal.morlet2, widths, w=w)
    abs_swd = np.abs(cwtm_swd).astype(dtype = 'float16')
    cwt_file.create_dataset(k, data=abs_swd.T)




cwt_file = h5py.File("Out_cwt.h5", 'w')

                     

                     
counter = len(data_file.keys())

for k in data_file.keys() :
    do_cwt(k)
    counter -= 1
    print(k + "is processed, " + str(counter) + " left.")
    
cwt_file.close()
data_file.close()

