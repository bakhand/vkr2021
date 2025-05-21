import numpy as np
import h5py


DATA_OFFSET = 611 #end_of_data_text_in_bin
N_CHAN = 6
FILE_NAME = "KM_2_2021-03-11_06-21-41"


data_file = open(FILE_NAME + ".bin", "rb")


#ADC step = 0.33mV
data_raw = np.fromfile(data_file, dtype='<i2', count=-1, sep='', offset = DATA_OFFSET+5)
data_reshaped = data_raw.reshape((-1,N_CHAN))
data_reshaped = data_reshaped
data_reshaped_float = data_reshaped*0.0003333333


with open(FILE_NAME + '.npy', 'wb') as f:
    np.save(f, data_reshaped_float)



h5f = h5py.File(FILE_NAME + '_f.h5', 'w')
h5f.create_dataset(FILE_NAME, data=data_reshaped_float)
h5f.close()