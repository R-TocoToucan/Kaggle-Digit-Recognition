import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Function for digit visualization in 28x28 size
def plot_digit(row):
    some_digit = np.array(row)
    some_digit_image = some_digit.reshape((28, 28))
    plt.imshow(some_digit_image, cmap = plt.cm.binary)
    plt.axis('off')
    plt.show()

# Kaggle MNIST dataset
train = pd.read_csv('C:/digit-recognizer/train.csv')
test = pd.read_csv('C:/digit-recognizer/test.csv')
sub = pd.read_csv('C:/digit-recognizer/sample_submission.csv')

print(f"Training data size is {train.shape}\nTesting data size is {test.shape}")


# Train test split
print("train-test-split")

x = train.drop(labels = ["label"], axis = 1) 
y = train["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=0)

# Knn training
print('Training Started')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train) 

y_pred=knn.predict(x_test)

score=accuracy_score(y_test, y_pred)

print(score)


# Displaying a digit from the training data

plot_digit(x_train.loc[0])


# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()