from keras.models import load_model
import cv2
import numpy as np
import os



imagePath ="info"

model = load_model('eye2.h5')

listing = os.listdir(imagePath)

for imgg in listing:
    print("img= 일치율 발병확률\n"+imgg)
    data =np.ndarray(shape=(1,128,128,3),dtype=np.float32)
    img = cv2.imread(imagePath+"/{0}".format(imgg),cv2.IMREAD_COLOR)

    img = cv2.resize(img,(128,128))
    data[0] = np.asarray(img)

    #print(data)
    prediction = model.predict(data)

    result1,result2 = prediction[0]
    print(result2)

    print("\n\n")

    #오차율이 거의 30%가까이 되서 영 도움이 되지를 않는다.


# 100% = 정상.




#숫자가 1에 가까울수록 비정상눈.