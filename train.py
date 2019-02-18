#coding:utf-8
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def print_cmx(y_true, y_pred):
    y_label = 'True label'
    x_label = 'Predicted label'
    title = 'Counfusion Matrix - Fashion MNIST test data'
    save_path = 'confusin_matrix.png'
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j,i, str(cm[i][j]), horizontalalignment="center")
    plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    ax = plt.gca() # get current axis
    ax.grid(False) 
    plt.savefig(save_path)
    
def print_acc(fit):
    plt.plot(fit.history['acc'])
    plt.plot(fit.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
    plt.show()

def print_loss(fit):
  plt.plot(fit.history['loss'])
  plt.plot(fit.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['training loss', 'validation loss'], loc='upper right')
  plt.show()

def print_rand_sample(x_test, y_test, y_test_pred):
    w_num = 5
    h_num = 5
    figure = plt.figure(figsize=(20, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    for i, index in enumerate(np.random.choice(x_test.shape[0], size=w_num * h_num, replace=False)):
        ax = figure.add_subplot(h_num, w_num, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[index]))
        pred_index = np.argmax(y_test_pred[index])
        true_index = np.argmax(y_test[index])
        ax.set_title("predict:[{}], true:[{}]".format(CLASS_NAMES[pred_index], CLASS_NAMES[true_index]), color=("green" if pred_index == true_index else "red"))

        
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# min-max 正規化(データ画像の画素の値を[0,1]に収める)
max_pix_val = 255
min_pix_val = 0
x_train = (x_train.astype('float32') - min_pix_val ) / (max_pix_val - min_pix_val) 
x_test  = ( x_test.astype('float32') - min_pix_val ) / (max_pix_val - min_pix_val)

# 10000枚を検証データに、50000枚を学習データにランダム分割
valid_num = 10000 #検証データの枚数
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=(valid_num/len(x_train)))

# 28 x 28の画像がgrayscaleで1chなので、28, 28, 1にreshapeする
width, height = 28, 28
x_train = x_train.reshape(x_train.shape[0], width, height, 1)
x_valid = x_valid.reshape(x_valid.shape[0], width, height, 1)
x_test  = x_test.reshape(x_test.shape[0], width, height, 1)

# 10種類のラベルをOne-hot表現に変更(損失関数の計算のため)
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

#モデルの構築
model = keras.Sequential()
# 入力層
model.add(keras.layers.InputLayer(input_shape=(width,height,1)))
model.add(keras.layers.Dropout(0.2))
# 中間層(畳み込み層→ReLU→プーリング層)
model.add(keras.layers.Conv2D(filters=128, kernel_size=2, strides=(1, 1), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(0.5))
# 中間層(畳み込み層→ReLU→プーリング層)
model.add(keras.layers.Conv2D(filters=128, kernel_size=2, strides=(1, 1), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(0.5))
# 全結合層→ReLU
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
# 全結合層→Softmax
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 各エポック終了後にモデルを保存(一番val_lossが少ないものを保存する)
epochs = 2
batch_size=64
checkpointer = keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, period=1)
fit = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_valid, y_valid),callbacks=[checkpointer])


# 評価
model.load_weights('model.weights.best.hdf5')
score = model.evaluate(x_test, y_test, verbose=1)
y_test_pred = model.predict(x_test)

# 結果等表示
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print_acc(fit)
print_loss(fit)
print_cmx(np.argmax(y_test, axis = 1), np.argmax(y_test_pred, axis = 1))
print_rand_sample(x_test, y_test, y_test_pred)




