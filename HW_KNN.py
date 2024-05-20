import pickle
import matplotlib.pyplot as plot
from joblib import load, dump
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
with open('./mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)


#mnist['data'] = mnist['data']/255.0

train_data = mnist['data'][0:60000]
train_labels = mnist['target'][0:60000]
val_data = mnist['data'][60000:70000]
val_labels = mnist['target'][60000:70000]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_data, train_labels)

predictions = model.predict(val_data)
# Evaluate performance of model for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(val_labels, predictions))

matrix = confusion_matrix(val_labels,predictions)
fig, ax = plot.subplots(figsize=(7.5, 7.5))
ax.matshow(matrix, cmap=plot.cm.Blues, alpha=0.3)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='larger')
plot.xlabel('Predictions', fontsize=18)
plot.ylabel('Actuals', fontsize=18)
plot.title('Confusion Matrix', fontsize=18)
plot.show()

# dump(model,'./KNN_FINAL_MODEL.joblib')


# model = load('./KNN_FINAL_MODEL.joblib')

# import cv2
# import PIL

# file = r"C:\Users\hozay\OneDrive\Desktop\HW_recognition\tests_final\test5.jpg"
# original_img = PIL.Image.open(file)
# img_resized = original_img.resize((28, 28), PIL.Image.Resampling.LANCZOS)
# img_resized = np.array(img_resized)
# img_resized = img_resized[:,:,0]
# img_resized = np.invert(np.array([img_resized]))
# img_resized = img_resized[0,:,:]
# print(img_resized)
# white_threshold = 127
# mask = img_resized > white_threshold
# img_resized[mask] = 255
# img_resized[~mask] = 0
# print(img_resized)
# plot.imshow(img_resized, cmap=plot.cm.binary)
# plot.show()

# # original_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
# # img_resized = cv2.resize(original_img, (28,28), interpolation=cv2.INTER_LINEAR)
# #img_resized = cv2.bitwise_not(img_resized)
# # img_resized = np.array(img_resized)
# # img_resized = img_resized / np.max(img_resized)
# # plot.imshow(img_resized, cmap=plot.cm.binary)
# # plot.show()
# # img_pre = np.array(img_resized)
# # img_pre = img_pre.reshape(784)
# # prediction = model.predict([img_pre])
# # print("the Digit is: ",prediction[0])
# img_pre = np.array(img_resized)
# print(img_pre.shape)
# img_pre = img_pre.reshape(784)
# prediction = model.predict([img_pre])
# print("the Digit is: ",prediction[0])

# #original_img = np.invert(np.array([original_img]))
# #original_img = original_img[0,:,:]
# plot.imshow(original_img, cmap=plot.cm.binary)
# plot.show()

