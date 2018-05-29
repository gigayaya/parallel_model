import multiprocessing
from keras_model import get_model
import os
import tensorflow as tf

def do_job(model,iter_num):
    x = [12345,54321,00000,1111]
    y = [1,1,0,0]
    model.fit(x=x, y=y,validation_split=0.5,epochs=10,shuffle=True,batch_size=2,verbose=2)
    print("Done process "+str(iter_num))

def func(iter_num):
    print("Start process "+str(iter_num))
    if(iter_num%2==0):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with tf.device('/gpu:0'):
            model = get_model()
            do_job(model,iter_num)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        with tf.device('/gpu:1'):
            model = get_model()
            do_job(model,iter_num)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=2)
    for i in range(40):
        pool.apply_async(func,args=(i,))
    pool.close()
    pool.join()
