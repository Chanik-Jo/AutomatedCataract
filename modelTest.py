from keras.models import load_model
import cv2,os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #info warning 출력안함.
imagePath = "info"


model = load_model('eye2.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Read the image

img = cv2.imread(imagePath)
img = cv2.resize(img,(128,128))
#img = np.reshape(img,[1,3,128,128])
img = np.reshape(img,[1,128,128,3])

classes = model.predict_classes(img)

print(classes)