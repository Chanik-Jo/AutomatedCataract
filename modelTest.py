from keras.models import load_model
import cv2
import numpy as np
import os

imagePath ="info"

model = load_model('eye2.h5')

listing = os.listdir(imagePath)

for imgg in listing:
    print("img=  "+imgg)
    data =np.ndarray(shape=(1,128,128,3),dtype=np.float32)
    img = cv2.imread(imagePath+"/{0}".format(imgg),cv2.IMREAD_COLOR)

    img = cv2.resize(img,(128,128))
    image_array = np.asarray(img)
    data[0] = image_array

    prediction = model.predict(data)


    print(prediction[0]) #왼쪽것은 가지고 있는 이미지와 일치하는지 여부 오른쪽은 정답여부. 오른쪽 숫자가 낮은게 백내장.

    #그래도 결과가 영 신뢰성이 없는데?
    print("\n\n")

    #오차율이 거의 30%가까이 되서 영 도움이 되지를 않는다.


# 100% = 정상.

#오른쪽것 숫자가 1에 가까울수록 정상눈


#오른쪽것 숫자가 0에 가까울수록 비정상눈.