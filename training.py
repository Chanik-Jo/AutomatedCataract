import os
from keras import backend as K
from keras.callbacks import EarlyStopping
import cv2 as cv
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Convolution2D, Flatten, Dense
from keras.utils import np_utils
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings

warnings.filterwarnings(action='ignore')

#출처: https://haonly.tistory.com/38 [Haonly's Blog]
if __name__ == '__main__':
    path ="input_path"
    listing = os.listdir(path)
    im_random = cv.imread("input_path//{0}".format(listing[23]))
    #cv.imread는 넘파이 이미지를 불러오는 함수이다. 그것도 보니까 23번째 이미지네.
    print(im_random.shape)

    imatrix = np.array([cv.imread("input_path//{0}".format(img)).flatten() for img in listing])
    #설명을 읽어보니
    #이미지 하나에 한줄이란다.....  R+g+b 합쳐서 이미지의 한칸이 255+255+255인건가 아니면 rgb가 나눠져있나?
    #불필요한 질문이긴한데 괜히 궁금해진다.


    # To validate, print the shape of the matrix
    print(imatrix.shape)
    print(type(imatrix))
    print(imatrix.ndim)


    #imatrix의 세로길이는 이미지 갯수 즉 2000일것이고 imatrix의 가로길이를 따내야 count에 집어넣을텐데.
    """
    >>> list = [1,2,3,4,5]
    >>> len(list)
    5   #파이썬 배열 길이 구하기.
    """
    #count = len(imatrix[0])#이건 가로길인데 가로길이를 넣는게 맞나 모르겠다.
    count = len(imatrix)#총 2천개니까 세로길이를 넣는게 맞다고 생각한다.


    label = np.ones((count,), dtype=int)
    #그러면 label은 2천개의 가로1차배열이 될거고....
    label[0:3001] = 0 #파이썬 리스트가 기억으로 마지막번호 -1 이라고 알고있는데...
    label[3001:6001] = 1
    np.size(label)

    data, label = sklearn.utils.shuffle(imatrix, label, random_state=2)
    train_data = [data, label] # 일자로 배열된 이미지 결과 0/1



    (X, y) = (train_data[0], train_data[1]) #traindata[0]은 1차열배열로 변환된 이미지 traindata[1]은 정답.

    # Splitting X and y in training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    #80퍼센트는 학습용, 20%는 정답합습용.
    #https://ebbnflow.tistory.com/126  눈에 잘 와닿지 않으면 여기를 참고할것.


    # Splitting X_train and y_train in training and validation data
    #여기는 궃이 필요없는 부분,   주석처리 되었다.
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

    # Validating the individual sizes
    print("X_train : {0}".format(X_train.shape))
    print("y_train :{0}".format(y_train.shape))

    #print("X_val : {0}".format(X_val.shape))
    #print("y_val : {0}".format(y_val.shape))

    print("X_test : {0}".format(X_test.shape))
    print("y_test : {0}".format(y_test.shape))

    # Reshaping the data to pass to CNN
    X_train = X_train.reshape(X_train.shape[0], 3, 128,128)

    # X_val = X_val.reshape(X_val.shape[0], 3, 128, 128)
    X_test = X_test.reshape(X_test.shape[0], 3, 128, 128)

    X_train=np.swapaxes(X_train,1,3); # 0 2

    X_train=np.swapaxes(X_train,1,2); # 0 1

    X_test=np.swapaxes(X_test,1,3);
    X_test=np.swapaxes(X_test,1,2);


    # Keras Training Parameters
    batch_size = 50  # 한번에 몇개씩 풀것인지
    nb_classes = 2
    nb_epoch = 5  # 총 몇회 반복할것인지.

    img_rows, img_col = 128, 128
    img_channels = 3
    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    y_train = np_utils.to_categorical(y_train, nb_classes)
    # y_val = np_utils.to_categorical(y_val, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    strategy = tf.distribute.MirroredStrategy()
    #야 진짜 이젠 학습이다.
    #with strategy.scope():# gpu풀가동!
    with tf.device('/device:GPU:0'):
        model = Sequential()


        model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                         padding="valid",
                         activation='relu',
                         input_shape=( img_rows, img_col,img_channels),

                         data_format='channels_last'))
                         #data_format='channels_last'))
                        #데이터 포맷은 애초에 74행에서 3 ,128,128이었으니 어쩔수 없이 채널퍼스트가 맞다.

        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.50))

        model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.50))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        #------------------------------------------------------------------

        '''
        1. epoch의 증가는 과적합 해결을 위한 수단이 아니라 모니터링 수단이다.        
        단순하게 epoch만 늘려주면 train data의 loss는 줄고 accuracy는 높아지기 
        때문에 train data의 loss와 accuracy는 valid 결과와 비교하
        기 위한 참고사항일 뿐이다. epoch 수를 늘리면 valid와 train loss가 교차하는
         지점이 발생한다. 일반적으로 valid loss가 감소하다가 증가하는 시점을 과적합으로 정의하
         기 때문에 이 지점에서 적당한 epoch을 결정한다. 이는 early stopping과도 관련이 있다.        
        
        2. epoch이 증가하면서 train loss 와 valid loss가 수렴해야 가장 좋다.        
        교차점에서 epoch을 결정하더라도 train과 최종 test loss와 accuracy를 비교
        해야한다. 이러한 점검은 cross-validation으로 진행해야한다.        
        
        3. batch_size는 과적합과 관련이 없다. 모수의 수렴 문제와 관련이 있다.        
        배치사이즈가 작을 수록 수렴속도는 느리지만 local minimum에 빠질 가능성은 
        줄어든다. 반면 배치사이즈가 클 수록 학습진행속도와 수렴속도가 빨라지지만 항상 빨
        리 수렴하는 것은 아니다. 작은 데이터셋이라면 32가 적당하다고 하는데 이 역
        시도 구글링하다보면 수많은 의견들이 존재한다. 적당히 참고해가면서 실험해보면 좋겠다.


        
        '''

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                            verbose=1, validation_data=(X_test, y_test))

        model.save_weights("eye1.h5")
        model.save('eye2.h5')
    '''
    save() saves the weights and the model structure to a single HDF5 file. 
    I believe it also includes things like the optimizer state. 
    Then you can use that HDF5 file with load() to reconstruct 
    the whole model, including weights.    
    save_weights() only saves the weights to HDF5 and nothing else. 
    You need extra code to reconstruct the model from a JSON file.
    
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    model.save("whole_model.h5")
    print("Saved model to disk")
    
    '''

    #model.fit에서 오류.

    print("--------------")
    print(history.history)

    K.clear_session()#multiprocess.py 오류 안뜨게 만드려는 목적.  왜뜨는진 모르겠지만 일단 막아야지.