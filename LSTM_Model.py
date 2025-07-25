from utils.modules import *
from Data_Preprocessing import X_train,y_train,X_test,y_test
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(15, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=1500)

model.save('lstm_model_fin1500.h5')
#print(model.summary())

