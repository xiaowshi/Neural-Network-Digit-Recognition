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

count = 0
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
        label1 = Label(window, text ="Welcome to Handwritten Digit Recognition System, below is the instruction").pack(side = TOP)
        label2 = Label(window, text ="1. Enter a single digit by either mouse or click\"extract picture from Minist\"").pack(side = TOP,anchor = W)
        label3 = Label(window, text = "2.Click the \"clear\" key to enter again.").pack(side = TOP,anchor = W)
        label4 = Label(window, text ="3. The save button can save a .jpg file of what you wrote on the canvas.").pack(side = TOP,anchor = W)
        label5 = Label(window, text ="").pack(side = TOP,anchor = W)

        self.modeCombobox = ttk.Combobox(window, width = 20)
        self.modeCombobox["values"] = ("1. Extract from Mnist","2. Handwrite")
        self.modeCombobox.current(0)
        self.modeCombobox.pack()

        writeFrame = LabelFrame(window,text="Data Entry Area")
        buttonFrame = Frame(window)
        resultFrame = LabelFrame(window,text="Recognize Result")
        writeFrame.place(x=35,y=170,width=200,height=200)
        buttonFrame.place(x=265,y=210,width=135,height=150)
        resultFrame.place(x=450,y=200,width=120,height=150)
       
        self.canvas = Canvas(writeFrame, bg="black",width=200,height=200 )
        self.canvas.bind("<B1-Motion>",self.writing)
        self.canvas.bind("<ButtonRelease>",self.stop)
        self.canvas.pack(fill=BOTH,expand=YES)

        clearButton = Button(buttonFrame, text = "Clear", command = self.clear)
        saveButton = Button(buttonFrame, text = "Save", command = self.save)
        mnisButton = Button(buttonFrame, text = "Extract from Mnist", command = self.extract)
        mnisButton.pack(side = TOP,anchor = W,expand=YES,fill=X )
        clearButton.pack(side = TOP,anchor = W,fill=X )
        saveButton.pack(side = TOP,anchor = W,expand=YES,fill=X)

        self.resultCanvas = Canvas(resultFrame)
        self.resultCanvas.pack(fill=BOTH,expand=YES)      
        
    def writing(self,event):
        self.modeCombobox.current(1)
        self.resultCanvas.delete("all")
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
        result = network.predict(self.imgArr)
        label = np.argmax(result,axis =1)
        self.resultCanvas.create_text(60,55,text=str(label),fill="red")
    def stop(self,event):
        self.startFlag = False       
    def clear(self):
        self.canvas.delete("all")
        self.handWriting = Image.new("RGB",(200,200),(0,0,0))
        self.imgDraw = ImageDraw.Draw(self.handWriting)
        self.resultCanvas.delete("all")
    def extract(self):
        self.canvas.delete("all")
        self.resultCanvas.delete("all")
        self.modeCombobox.current(0)
        randomInt = np.random.randint(0,9999)
        self.mnistArray = x_test[randomInt]
        mnistArrayBig = cv.resize(self.mnistArray,(200,200),interpolation = cv.INTER_LINEAR)       
        self.mnistImage = ImageTk.PhotoImage(Image.fromarray(mnistArrayBig))
        self.canvas.create_image(100,100,image=self.mnistImage)
        self.mnistArray = self.mnistArray/255.0
        self.mnistArray = self.mnistArray.reshape(1,28,28)
        result = model.predict(self.mnistArray)
        label = np.argmax(result,axis =1)
        self.resultCanvas.create_text(60,55,text = str(label),fill="blue")
    def save(self):
        self.imgArr = np.array(self.handWriting)       
        self.imgArr = cv.resize(self.imgArr,(28,28))
        global count
        cv.imwrite(str(count)+".jpg",self.imgArr)
        count = count+1
        cv.imshow(str(count)+".jpg",self.imgArr)
        print("file saved")
        

window = Tk()
window.title("Digit Recognition System")
window.geometry("600x400")
controller = Controller(master=window)
window.mainloop()
