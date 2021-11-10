import os


import cv2 as cv
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Convolution2D, Flatten, Dense
from keras.utils import np_utils
import sklearn
from sklearn.model_selection import train_test_split
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
    label[0:1001] = 0
    label[1001:2001] = 1
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
    X_train = X_train.reshape(X_train.shape[0], 3, 128, 128)
    # X_val = X_val.reshape(X_val.shape[0], 3, 128, 128)
    X_test = X_test.reshape(X_test.shape[0], 3, 128, 128)

    # Keras Training Parameters
    batch_size = 50  # 한번에 몇개씩 풀것인지
    nb_classes = 2
    nb_epoch = 30  # 총 몇회 반복할것인지.
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


    #야 진짜 이젠 학습이다.

    model = Sequential()

    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding="valid",
                     activation='relu',
                     input_shape=(img_channels, img_rows, img_col),
                     data_format='channels_first'))

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

    #------------------------------------------------------------------

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=(X_test, y_test))
    #model.fit에서 오류.

    print("--------------")
    print(history.history)
