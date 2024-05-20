import pickle
import matplotlib.pyplot as plot
from joblib import load, dump
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# with open('./mnist.pkl', 'rb') as f:
#     mnist = pickle.load(f)

# train_data = mnist['data'][0:60000]
# train_labels = mnist['target'][0:60000]
# val_data = mnist['data'][60000:70000]
# val_labels = mnist['target'][60000:70000]

# y_train_large = (train_labels.astype(int) >= 7)
# y_train_odd = (train_labels.astype(int) % 2 == 1)
# y_train_multilabel = np.c_[y_train_large,y_train_odd]

# y_val_large = (val_labels.astype(int) >= 7)
# y_val_odd = (val_labels.astype(int) % 2 == 1)
# y_val_multilabel = np.c_[y_val_large,y_val_odd]

# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(train_data, y_train_multilabel)

# predictions = model.predict(val_data)
# print("EVALUATION ON TESTING DATA")
# print(classification_report(y_val_multilabel, predictions,zero_division=1))

# dump(model,'./KNN_MULTILABEL_MODEL.joblib')
def ls(pred):
    if(pred):
        return "large"
    else:
        return "small"

def oe(pred):
    if(pred):
        return "odd"
    else:
        return "even"  

multilabel_model = load('./KNN_MULTILABEL_MODEL.joblib')
knn_model = load('./KNN_FINAL_MODEL.joblib')
import cv2
import PIL

file = r"C:\Users\hozay\OneDrive\Desktop\HW_recognition\tests_final\test2.jpg"
original_img = PIL.Image.open(file)
img_resized = original_img.resize((28, 28), PIL.Image.Resampling.LANCZOS)
img_resized = np.array(img_resized)
img_resized = img_resized[:,:,0]
img_resized = np.invert(np.array([img_resized]))
img_resized = img_resized[0,:,:]
print(img_resized)
white_threshold = 127
mask = img_resized > white_threshold
img_resized[~mask] = 0
img_resized[mask] = 255

print(img_resized)
plot.imshow(img_resized, cmap=plot.cm.binary)
plot.show()

img_pre = np.array(img_resized)
print(img_pre.shape)
img_pre = img_pre.reshape(784)

digit_prediction = knn_model.predict([img_pre])
multilabel_prediction = multilabel_model.predict([img_pre])
print("the Digit",digit_prediction[0],"is a", ls(multilabel_prediction[0][0]),"and an",oe(multilabel_prediction[0][1]),"number")
plot.imshow(original_img, cmap=plot.cm.binary)
plot.show()

