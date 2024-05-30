# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:36:09 2024

@author: Mehmet
"""

import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import cv2

with open('/kaggle/input/ships-in-satellite-imagery/shipsnet.json') as data_file:
    dataset = json.load(data_file)
df = pd.DataFrame(dataset)
print(len(df.data[0]))
print(df.columns)
print(df.head())
print(df.info()) 
def get_X_chips_rgb(shipsnet_df, data_col='data', channels=3, rows=80, cols=80):
    num_samples = len(shipsnet_df[data_col])
    len_data    = len(shipsnet_df[data_col][0])     
    assert len_data == channels * rows * cols
    X = np.zeros((num_samples, len_data), dtype=np.uint8)
    for idx in range(num_samples):
        X[idx,:] = np.array(shipsnet_df[data_col][idx]).astype(dtype=np.uint8)
    X_rgb_chips = X.reshape(-1, channels, rows, cols)
    X_chips_rgb = X_rgb_chips.transpose([0, 2, 3, 1])
    assert (X_chips_rgb[0][0][0] == np.array([shipsnet_df.data[0][0], shipsnet_df.data[0][6400], shipsnet_df.data[0][12800]])).all(), 'Data reshaping ERROR'
    return X_chips_rgb
X_chips_rgb = get_X_chips_rgb(df)
r_df = X_chips_rgb
print(X_chips_rgb.shape)
import matplotlib.gridspec as gridspec
image = r_df[1]
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, kernel)
gaus_image = cv2.GaussianBlur(image, (3, 3), 0)
median_image = cv2.medianBlur(image, 5)
bilateral_image = cv2.bilateralFilter(image, 9, 75, 75)
fig = plt.figure(constrained_layout=True, figsize=(24, 16))
gs = gridspec.GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Original Image')
ax1.imshow(image, cmap='gray')

ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(sharpened_image, cmap='gray')
ax2.set_title('Sharpened')

ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(gaus_image, cmap='gray')
ax3.set_title('Gaus')

ax4 = fig.add_subplot(gs[2, 0])
ax4.imshow(median_image, cmap='gray')
ax4.set_title('Median')

ax5 = fig.add_subplot(gs[2, 1])
ax5.imshow(bilateral_image, cmap='gray')
ax5.set_title('Bilateral')
plt.show()

image = r_df[1]

original_image_canny = cv2.Canny(image,90,180)
sharpened_image_canny = cv2.Canny(sharpened_image,120,240,L2gradient = True)
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original')

axs[0, 1].imshow(original_image_canny, cmap='gray')
axs[0, 1].set_title('Canny')

axs[1, 0].imshow(sharpened_image, cmap='gray')
axs[1, 0].set_title('Sharpened')

axs[1, 1].imshow(sharpened_image_canny, cmap='gray')
axs[1, 1].set_title('Canny')
shar_list = []
shar_cany_list = []
cany_list = []
a = 0
for image in r_df:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cany_list.append(cv2.Canny(image,90,180))
    temp_sh = cv2.filter2D(image, -1, kernel)
    shar_list.append(temp_sh)
    temp_sh = cv2.cvtColor(temp_sh,cv2.COLOR_BGR2RGB)
    shar_cany_list.append(cv2.Canny(temp_sh,150,300,L2gradient = True))
    
    
np_shar = np.array(shar_list, dtype=np.uint8).reshape([-1, 1, 80, 80]).transpose([0,2,3,1])
np_shar_cany = np.array(shar_cany_list, dtype=np.uint8).reshape([-1, 1, 80, 80]).transpose([0,2,3,1])
np_cany = np.array(cany_list, dtype=np.uint8).reshape([-1, 1, 80, 80]).transpose([0,2,3,1])
x_train_test, x_val, y_train_test, y_val = train_test_split(combined_array, y, test_size=0.20, random_state=42)
print(np_cany.shape)
print(np_shar_cany.shape)
print(np_shar.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(240, 80, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit([np_shar,np_shar_cany,np_cany], y_train_test, epochs=5, batch_size=16, validation_data=(x_val, y_val))



wrong_indexes = np.where(y_pred.squeeze() != y_test)[0]



plt.figure(figsize=(15, 5))
for i, idx in enumerate(wrong_indexes[36:42]):
    plt.subplot(1, 6, i+1)
    plt.imshow(x_test[idx].reshape(240, 80), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred[idx][0]}")
    plt.axis('off')
plt.show()


