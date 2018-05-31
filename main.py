import multiprocessing
from keras_model import get_model
import os
import tensorflow as tf
import traceback
from keras.backend.tensorflow_backend import get_session,clear_session,set_session


def do_job(iter_num):
    model = get_model()
    x = [12345,54321,00000,1111]
    y = [1,1,0,0]
    model.fit(x=x, y=y,validation_split=0.5,epochs=10,shuffle=True,batch_size=2,verbose=2)
    print("Done process "+str(iter_num))
    clear_session()
    print("close")

def func(iter_num):
    try:
        print("Start process "+str(iter_num))
        if(iter_num%2==0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            #with tf.device('/gpu:0'):
            do_job(iter_num)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            #with tf.device('/gpu:1'):
            do_job(iter_num)
    except Exception:
        print(traceback.format_exc())

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    for i in range(8):
        pool.apply_async(func,args=(i,))
    pool.close()
    pool.join()