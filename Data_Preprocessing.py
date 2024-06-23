import numpy as np

from modules import *
from folder_setup import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

classes = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(number_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(classes[action])

# X and y variables 
X = np.array(sequences)
y= to_categorical(labels).astype(int)
# train_sizes = np.linspace(0.1,0.9,10)
#accuracies = []
#x_flattened = X.reshape(X.shape[0], -1)
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,random_state=42)
'''
for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(x_flattened, y, test_size=0.05,random_state=42)
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Training Set Size (fraction)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Training Set Size')
plt.grid(True)
plt.axhline(y=0.7, color='r', linestyle='--', label='Target Accuracy = 0.7')
plt.legend()
plt.show()
'''
# print(acc)
# print(classes)
