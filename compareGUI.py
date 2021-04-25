""" for user to control"""

import sys
import pickle
import gzip

from tkinter import *
from tkinter import ttk
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import models, layers
from keras.optimizers import RMSprop
from PIL import Image, ImageDraw,ImageQt,ImageTk
import numpy as np
import cv2 as cv

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data


#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=6)
#model.summary()
#model.save_weights("model.h5")
model.load_weights("anotherModel.h5")


network = Sequential()
network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.AveragePooling2D((2, 2)))
network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
network.add(layers.AveragePooling2D((2, 2)))
network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(84, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
network.load_weights("convolution.h5")


class Controller(Frame):
    
    
    def __init__(self,master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.createWidgets()
        
        self.startFlag = False
        self.handWriting = Image.new("RGB",(200,200),(0,0,0))
        self.imgDraw = ImageDraw.Draw(self.handWriting)
    def createWidgets(self):
        label1 = Label(window, text ="Welcome to Handwritten Digit Recognition System").pack(side = TOP)
        label2 = Label(window, text ="below is the instruction:").pack(side = TOP,anchor = W)
        label3 = Label(window, text ="1. Enter a single digit using mouse").pack(side = TOP,anchor = W)
        label4 = Label(window, text ="2.Click the \"clear\" key to enter again.").pack(side = TOP,anchor = W)
        label5 = Label(window, text ="***This System Is Designed for Comparing CNN and ANN Result***").pack()
        label6 = Label(window, text ="").pack(side = TOP,anchor = W)

        clearButton = Button(window, text = "Clear", command = self.clear)
        clearButton.pack()
        
        writeFrame = LabelFrame(window,text="Data Entry Area")
        convoResultFrame = LabelFrame(window,text="CNN Result")
        artiResultFrame = LabelFrame(window,text="ANN Result")
        writeFrame.place(x=35,y=170,width=200,height=200)
        convoResultFrame.place(x=280,y=210,width=120,height=150)
        artiResultFrame.place(x=450,y=210,width=120,height=150)
       
        self.canvas = Canvas(writeFrame, bg="black",width=200,height=200 )
        self.canvas.bind("<B1-Motion>",self.writing)
        self.canvas.bind("<ButtonRelease>",self.stop)
        self.canvas.pack(fill=BOTH,expand=YES)

        self.resultCanvas = Canvas(artiResultFrame)
        self.resultCanvas.pack(fill=BOTH,expand=YES)
        self.resultCanvas1 = Canvas(convoResultFrame)
        self.resultCanvas1.pack(fill=BOTH,expand=YES)
        
    def writing(self,event):
        #self.modeCombobox.current(1)
        self.resultCanvas.delete("all")
        self.resultCanvas1.delete("all")
        if not self.startFlag:
            self.startFlag = True
            self.x=event.x
            self.y=event.y
        self.canvas.create_line((self.x,self.y,event.x,event.y),width = 8,fill="white")        
        self.imgDraw.line((self.x,self.y,event.x,event.y),fill="white",width = 19)
        self.x=event.x
        self.y=event.y
        self.imgArrOrigin = np.array(self.handWriting)       
        self.imgArr = cv.resize(self.imgArrOrigin,(28,28))#interpolation?
        self.imgArr = cv.cvtColor(self.imgArr,cv.COLOR_BGR2GRAY)
        self.imgArr = self.imgArr.reshape((1,28,28,1)).astype('float')/255
        #self.imgArr = self.imgArr/255.0
        #self.imgArr = self.imgArr.reshape(1,28,28) 
        artiResult = model.predict(self.imgArr)
        label = np.argmax(artiResult,axis =1)
        self.resultCanvas.create_text(60,55,text=str(label),fill="red",font=("Purisa", 25))
        convoResult = network.predict(self.imgArr)
        label1 = np.argmax(convoResult,axis =1)
        self.resultCanvas1.create_text(60,55,text=str(label1),fill="blue",font=("Purisa",25))
    def stop(self,event):
        self.startFlag = False       
  
    def clear(self):
        self.canvas.delete("all")
        self.canvas.delete("all")
        self.handWriting = Image.new("RGB",(200,200),(0,0,0))
        self.imgDraw = ImageDraw.Draw(self.handWriting)
        self.resultCanvas.delete("all")
        self.resultCanvas1.delete("all") 

window = Tk()
window.title("Digit Recognition System")
window.geometry("600x400")
controller = Controller(master=window)
window.mainloop()
