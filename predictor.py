import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import df_drop
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.keras.models
import h5py
import matplotlib.pyplot as plt  # for 畫圖用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def Movement_Average(data,Window):
  for features in data.columns:
    if features != 'Phase':
      data[features] = data[features].copy().rolling(window=Window).mean()
      data = data.dropna()
  return data

############ Data Sets ##########################
# data_1 = Movement_Average(pd.read_csv('./7_phase/apple7.csv'),10)
# data_2 = Movement_Average(pd.read_csv('./7_phase/hank7.csv'),10)
# data_3 = Movement_Average(pd.read_csv('./7_phase/star7.csv'),10)
# data_4 = Movement_Average(pd.read_csv('./7_phase/jerry7.csv'),10)
# data_5 = Movement_Average(pd.read_csv('./7_phase/jia7.csv'),10)
# data_6 = Movement_Average(pd.read_csv('./7_phase/yue7.csv'),10)
# data_7 = Movement_Average(pd.read_csv('./7_phase/lai7.csv'),10)
# data_8 = Movement_Average(pd.read_csv('./7_phase/aws7.csv'),10)
# data_9 = Movement_Average(pd.read_csv('./7_phase/book7.csv'),10)
# data_10 = Movement_Average(pd.read_csv('./7_phase/wei7.csv'),10)
# # # # # # ############# Data Sets ##########################
data_1 = pd.read_csv('./7_phase/77/apple7F.csv')
data_2 = pd.read_csv('./7_phase/77/hank7F.csv')
data_3 = pd.read_csv('./7_phase/77/star7F.csv')
data_4 = pd.read_csv('./7_phase/77/jerry7F.csv')
data_5 = pd.read_csv('./7_phase/77/jia7F.csv')
data_6 = pd.read_csv('./7_phase/77/yue7F.csv')
data_7 = pd.read_csv('./7_phase/77/lai7F.csv')
data_8 = pd.read_csv('./7_phase/77/aws7F.csv')
data_9 = pd.read_csv('./7_phase/77/book7F.csv')
data_10 = pd.read_csv('./7_phase/77/wei7F.csv')
###########Train/Test Data Group##############
def buildTrain_Val(train_list, rate,pastFrame,futureFrame):## 4 return
  X_step, y_step,X_val_step, Y_val_step = [], [], [], []

  for df_count in range(len(train_list)):
    train, val = train_test_split(train_list[df_count], test_size = rate, shuffle = False)
    for i in range(train.shape[0]-futureFrame-pastFrame):
        x = np.array(train.iloc[i:i+pastFrame,0:(train.shape[1]-1)])
        X_step.append(np.array(x))
        y = np.array(train.iloc[i+pastFrame:i+pastFrame+futureFrame, (train.shape[1]-1)])
        y_step.append(y)
    for j in range(val.shape[0]-futureFrame-pastFrame):
        x = np.array(val.iloc[j:j+pastFrame,0:(val.shape[1]-1)])
        # import pdb;pdb.set_trace()
        X_val_step.append(np.array(x))
        y = np.array(val.iloc[j+pastFrame:j+pastFrame+futureFrame, (val.shape[1]-1)])
        Y_val_step.append(y)
  return np.array(X_step), np.array(y_step), np.array(X_val_step), np.array(Y_val_step)

def buildTrain_test(test_list,pastFrame,futureFrame):## 2 return
  X_step, y_step = [], []
  for df_count in range(len(test_list)):
    for i in range(test_list[df_count].shape[0]-futureFrame-pastFrame):
        x = np.array(test_list[df_count].iloc[
            i:i+pastFrame,
            0:(test_list[df_count].shape[1]-1)])
        X_step.append(np.array(x))
        y = np.array(test_list[df_count].iloc[
            i+pastFrame:i+pastFrame+futureFrame,
             (test_list[df_count].shape[1]-1)])
        y_step.append(y)
  return np.array(X_step), np.array(y_step)

def buildTrain_all(train_list, val_rate, test_rate,pastFrame,futureFrame):## 4 return
  X_step, y_step,X_val_step, Y_val_step,X_test_step, Y_test_step = [], [], [], [], [], []
  for df_count in range(len(train_list)):
    train, test = train_test_split(train_list[df_count], test_size = test_rate, shuffle = False)
    train, val = train_test_split(train, test_size = val_rate/(1-test_rate), shuffle = False)
    for i in range(train.shape[0]-futureFrame-pastFrame):
        x = np.array(train.iloc[i:i+pastFrame,0:(train.shape[1]-1)])
        X_step.append(np.array(x))
        y = np.array(train.iloc[i+pastFrame:i+pastFrame+futureFrame, (train.shape[1]-1)])
        y_step.append(y)
    for j in range(val.shape[0]-futureFrame-pastFrame):
        x = np.array(val.iloc[j:j+pastFrame,0:(val.shape[1]-1)])
        # import pdb;pdb.set_trace()
        X_val_step.append(np.array(x))
        y = np.array(val.iloc[j+pastFrame:j+pastFrame+futureFrame, (val.shape[1]-1)])
        Y_val_step.append(y)
    for k in range(test.shape[0]-futureFrame-pastFrame):
        x = np.array(test.iloc[k:k+pastFrame,0:(test.shape[1]-1)])
        # import pdb;pdb.set_trace()
        X_test_step.append(np.array(x))
        y = np.array(test.iloc[k+pastFrame:k+pastFrame+futureFrame, (test.shape[1]-1)])
        Y_test_step.append(y)
  return np.array(X_step), np.array(y_step), np.array(X_val_step), np.array(Y_val_step), np.array(X_test_step), np.array(Y_test_step)

###########Train/Test Data Group##############
all_frame = [data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_9,data_8,data_10]
train_frame = [data_1,data_2,data_3,data_6,data_7,data_9,data_8,data_10]
test_frame = [data_4,data_5]

########### Concat Groups ########################
X_train, Y_train, X_val, Y_val = buildTrain_Val(train_frame, 0.1, 30, 1)
X_test, y_test = buildTrain_test(test_frame, 30, 1)
# X_train, Y_train, X_val, Y_val,X_test, y_test = buildTrain_all(all_frame, 0.1, 0.25, 30, 1)
#####################################

print('Shape: '+str(X_train.shape))
print('Shape: '+str(X_val.shape))


def buildManyToManyModel(shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=(shape[1],shape[2]), return_sequences=True),input_shape=(shape[1],shape[2]), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64, input_shape=(shape[1],shape[2]), return_sequences=False),input_shape=(shape[1],shape[2]), merge_mode='sum'))
    model.add(Dropout(0.5))
    model.add(Dense(32,name='FC1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss="sparse_categorical_crossentropy" ,optimizer=RMSprop(),metrics = ['accuracy'])
    model.summary()
    return model

# model = buildManyToManyModel(X_train.shape)
# callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_data=(X_val, Y_val), callbacks=[callback])
# model.save('./7_phase/Model/model_new_all32.h5')
model = tensorflow.keras.models.load_model("./7_phase/Model/model_new_dense32.h5")##載入model




predict_phase = model.predict_classes(X_test)
accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
predict_phase = predict_phase.reshape(-1)
y_test = y_test.reshape(-1)
print('Predict_Shape: '+str(predict_phase.shape))
print('Test_Shape: '+str(y_test.shape))
print(pd.crosstab(y_test, predict_phase, rownames=['Actual'], colnames=['Predicted']))

##### TIME COMPARE######
real_phase_persentage = pd.DataFrame(y_test).value_counts(normalize=True).sort_index().rename_axis('Phase').reset_index(name='Real_counts')
predict_phase_persentage = pd.DataFrame(predict_phase).value_counts(normalize=True).sort_index().rename_axis('Phase').reset_index(name='Predict_counts')
final_persentage = pd.merge(real_phase_persentage,predict_phase_persentage)
final_persentage['Gap'] = final_persentage.apply(lambda x : x['Real_counts']-x['Predict_counts'], axis=1)
print(final_persentage)
pd.DataFrame(predict_phase).to_csv('./7_phase/Processed/predict.csv',index =False)
##### TIME COMPARE######

# Visualising the results
plt.plot(y_test, color = 'red', label = 'Real Phase')  # 
plt.plot(predict_phase, color = 'blue', label = 'Predicted Phase')  # 
plt.title('Phase Prediction')
plt.xlabel('Time')
plt.ylabel('Phase')
plt.legend()
plt.show()




