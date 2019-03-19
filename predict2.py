import numpy as np
import cv2
img = cv2.imread('dataset/validate/1/apple_6.jpg',0)
cv2.imshow("My image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.resize(img, (64,64))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(type(img))
x=np.array(img)
dataset = np.load('dataset/final_dataset001.npz')
X_test = dataset['X_validate']
y_test = dataset['y_validate']
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
model = load_model('models/weights.13-0.00.hdf5')
print(model.summary())
y_pred_test = model.predict(x)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
print(y_pred_test_classes)
#print("y_pred_test",y_pred_test)
#print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))
