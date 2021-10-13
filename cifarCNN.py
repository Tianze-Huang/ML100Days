from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
print('train:',len(x_img_train))
print('test:',len(x_img_test))

label_dict={0:"airplane",1:"autombile",2:"bird",3:"cat",4:"deer",5:"dog",6:"forg",7:"horse",8:"shep",9:"truck"}

# 建立顯示圖片與預測結果的函數
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf() #取得當前的figure
    fig.set_size_inches(12,14)  #將其設定大小
    if num>25:num=25  
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i) 
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
          title+='=>'+label_dict[prediction[i]]
      
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()

x_img_train_normalize=x_img_train.astype('float32')/255.0
x_img_test_noarmalize=x_img_test.astype('float32')/255.0

from keras.utils import np_utils
y_label_train_onehot=np_utils.to_categorical(y_label_train)
y_label_test_onehot=np_utils.to_categorical(y_label_test)


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
model=Sequential()
# layer1 Conv,Drop,Maxpool
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32,32,3),activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer2 Conv,Drop,Maxpool
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer3 Conv,Drop,Maxpool
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer4 Flatten, Hidden, output
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1500, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=10,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x_img_train_normalize,y_label_train_onehot,validation_split=0.2,epochs=50,batch_size=300,verbose=1)
scores=model.evaluate(x_img_test_noarmalize, y_label_test_onehot,verbose=0)
print("accuracy:",scores[1])

model.save_weights("cifarCnnModel.h5")
print("Model Saved") 
















