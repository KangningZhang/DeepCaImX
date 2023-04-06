from model import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend
from tensorflow.keras.initializers import Constant
import scipy.io as sio
import numpy as np
import tensorflow as tf
import time
#with tf.device("cpu:0"):
#    print("tf.keras code in this scope will run on CPU")
start_time = time.time()

N_epoch = 10
batch_size = 5
model = LSTM_ISTA()

for i in range(N_epoch): # epoch
    index = np.random.permutation(81*15*6) + 1
    for j in range(int(81*15*6/batch_size)): # iteration
        index_batch = index[j*batch_size:(j+1)*batch_size]
        LSTM_Video = sio.loadmat('./LSTM_Dataset/Train/LSTM_Video_'+str(index_batch[0])+'.mat')
        LSTM_Video = LSTM_Video['LSTM_video']
        LSTM_Video = np.transpose(LSTM_Video,[2,0,1])
        LSTM_Video = np.expand_dims(np.expand_dims(LSTM_Video,3),0)

        LSTM_Data = sio.loadmat('./LSTM_Dataset/Train/LSTM_Data'+str(index_batch[0])+'.mat')
        LSTM_Data = LSTM_Data['LSTM_data']
        LSTM_Data = np.transpose(LSTM_Data,[2,0,1])
        LSTM_Data = np.expand_dims(np.expand_dims(LSTM_Data,3),0)

        LSTM_Masks = sio.loadmat('./LSTM_Dataset/Train/LSTM_Masks_'+str(index_batch[0])+'.mat')
        LSTM_Masks = LSTM_Masks['LSTM_mask']
        LSTM_Masks = np.float32(np.expand_dims(LSTM_Masks,0))

        LSTM_Trace = sio.loadmat('./LSTM_Dataset/Train/LSTM_Trace_'+str(index_batch[0])+'.mat')
        LSTM_Trace = LSTM_Trace['LSTM_trace']
        LSTM_Trace = np.transpose(LSTM_Trace,[1,0])
        LSTM_Trace = np.expand_dims(LSTM_Trace,0)


        for k in range(batch_size-1): # load
            Video = sio.loadmat('./LSTM_Dataset/Train/LSTM_Video_'+str(index_batch[k])+'.mat')
            Video = Video['LSTM_video']
            Video = np.transpose(Video,[2,0,1])
            LSTM_Video = np.append(LSTM_Video, np.expand_dims(np.expand_dims(Video,3),0), axis = 0)

            Data = sio.loadmat('./LSTM_Dataset/Train/LSTM_Data'+str(index_batch[k])+'.mat')
            Data = Data['LSTM_data']
            Data = np.transpose(Data,[2,0,1])
            LSTM_Data = np.append(LSTM_Data, np.expand_dims(np.expand_dims(Data,3),0), axis = 0)

            Masks = sio.loadmat('./LSTM_Dataset/Train/LSTM_Masks_'+str(index_batch[k])+'.mat')
            Masks = np.float32(Masks['LSTM_mask'])
            LSTM_Masks = np.append(LSTM_Masks, np.expand_dims(Masks,0), axis = 0)

            Trace = sio.loadmat('./LSTM_Dataset/Train/LSTM_Trace_'+str(index_batch[k])+'.mat')
            Trace = Trace['LSTM_trace']
            Trace = np.transpose(Trace,[1,0])
            LSTM_Trace = np.append(LSTM_Trace, np.expand_dims(Trace,0), axis = 0)
            del(Video)
            del(Masks)
            del(Trace)
            del(Data)

        model.train(LSTM_Video, LSTM_Data, LSTM_Masks, LSTM_Trace, i, j)
        del(LSTM_Video)
        del(LSTM_Masks)
        del(LSTM_Trace)    
print("--- %s seconds ---" % (time.time() - start_time))