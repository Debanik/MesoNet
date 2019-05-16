import numpy as np
from classifiers import *
from pipeline import *
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1. / 255)
generator = dataGenerator.flow_from_directory(
    'test_images',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    subset='training')
y_pred = list()
y_true = list()
# 3 - Predict
for i in range(1000):
    X, y = generator.next()
    y_pred.append(1 if classifier.predict(X) > 0.5 else 0)
    y_true.append(y)
    # print('Predicted :', classifier.predict(X), '\nReal class :', y)

confusionmatrix = confusion_matrix(y_true=y_true, y_pred = y_pred)
print(confusionmatrix)

# 4 - Prediction for a video dataset
'''
classifier.load('weights/Meso4_F2F')

predictions = compute_accuracy(classifier, 'test_videos')
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
'''
