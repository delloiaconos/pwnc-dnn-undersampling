""" 
Article: Smart Water Meter based on Deep Neural Network and Under-Sampling for PWNC Detection
Authors: Marco Carratu, Salvatore Dello Iacono, Giuseppe Di Leo, Vincenzo Gallo, Consolatina Liguori and Antonio Pietrosanto

In case of doubt or questions contact: sdelloiacono[at]unisa.it or vgallo[at]unisa.it

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow import keras 
import sklearn.model_selection as sk

from DatasetCreator import IMG_SIZE_1, IMG_SIZE_2

# Check for GPU
gpus= tf.config.experimental.list_physical_devices('GPU')
print(gpus)


# Mem Limit
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=32*1024)])


x=pickle.load(open('../train/32secx.pickle','rb'))
y=pickle.load(open('../train/32secy.pickle','rb'))

X_train, X_test, y_train, y_test = sk.train_test_split(x, y, test_size=0.40, stratify=y)

#Riscaliamo il dataset ed applichiamo la ONE-HOT encoding
X_train=X_train/np.max(X_train)
X_test=X_test/np.max(X_test)
y_train_dummies=pd.get_dummies(y_train) 
y_test_dummies=pd.get_dummies(y_test)
y_train_dummies = np.array(y_train_dummies)
y_test_dummies = np.array(y_test_dummies)

SEQUENCE_IMAGES=5
CLASSES=4
epochs=100

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPool2D, AveragePooling2D, TimeDistributed, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()

#1 layer conv
model.add(TimeDistributed(Conv2D(input_shape=(SEQUENCE_IMAGES,IMG_SIZE_2,IMG_SIZE_1,1),filters=32,kernel_size=(3,3),padding="same", activation="relu")))
model.add(TimeDistributed(Dropout(0.2)))

# #2 layer conv
model.add(TimeDistributed(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))) 
model.add(TimeDistributed(Dropout(0.2))) 
model.add(TimeDistributed(MaxPool2D(pool_size=(2,2),strides=(2,2))))

#3 layer conv
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3,3),padding="same", activation="relu"))) 
model.add( TimeDistributed(Dropout(0.2)))

#4 layer conv
model.add(TimeDistributed(Flatten()))

#1 layer LSTM
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

#2 layer LSTM
model.add(LSTM(256, activation='relu', return_sequences=False))
model.add(Dropout(0.2))

#1 layer Dense
model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.2))

#2 layer Dense
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(units=CLASSES, activation="softmax"))

# Model compilation and optimization
opt=tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-4)
model.compile(optimizer=opt , loss='categorical_crossentropy', metrics=['accuracy'])
model.build(X_train.shape)
model.summary()

early_stopping_monitor= EarlyStopping(patience=15, monitor='val_loss')

model_ceckpoint_callback = ModelCheckpoint(filepath='../models/model.h5', 
                                          monitor='val_loss', 
                                          mode='auto', save_best_only=True,
                                          verbose=1, period=10)

#Addestro il modello
history= model.fit(X_train,y_train_dummies,
          validation_data=(X_test,y_test_dummies), batch_size=32,
          epochs=epochs, 
          callbacks=[ model_ceckpoint_callback, early_stopping_monitor])


epochs_range = range(epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

acc_df = pd.DataFrame.from_dict(acc)

val_acc_df=pd.DataFrame.from_dict(val_acc)
val_acc_df=val_acc_df.rename(columns={0:1})

loss_df=pd.DataFrame.from_dict(loss)
loss_df=loss_df.rename(columns={0:2})

val_loss_df=pd.DataFrame.from_dict(val_loss)
val_loss_df=val_loss_df.rename(columns={0:3})


final_data=pd.concat([acc_df,val_acc_df,loss_df,val_loss_df],axis=1)


final_data.to_excel( '../results/' + '32sec.xlsx' )  