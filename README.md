## We together built this, known as the "hello world" of Machine Learning, handwritten digit recognition system using Python and the network using Keras Deep Learning APIs. To achieve this, we used the MNIST public data set which containing 60000 training images and 10000 testing images.
# Results

This GUI  allows you to both extracts an image from the MNIST testing dataset and recognize it and input digits by mouse.

![图片](https://uploader.shimo.im/f/ByWQzQlcb94YsctT.gif?fileGuid=wddjjPjKCqywdPwW)

This GUI is to compare the practical accuracy of both networks.

![图片](https://uploader.shimo.im/f/CcGCe5qda9bSoJJl.gif?fileGuid=wddjjPjKCqywdPwW)


# How to run it:

1 (Optional) Run the "LeNet.ipynb" file in Jupyter Notebook to generate the trained network and save it as "TrainedNetwork.h5" file which is already uploaded with the code.

2 Run "gui.py", you will be greeted with the graphical user interface.

# LeNet

```plain
def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    network.add(layers.AveragePooling2D((2, 2)))
    network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
    network.add(layers.Flatten())
    network.add(layers.Dense(84, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    return network
```
After 15 Epoches, it can achieve an accuracy of 99.06%.

![图片](https://uploader.shimo.im/f/BhbZZ8cMWImIuqku.png!thumbnail?fileGuid=wddjjPjKCqywdPwW)

# Image Formatting

![图片](https://uploader.shimo.im/f/cqbgJJ7x8R2L54Ci.png!thumbnail?fileGuid=wddjjPjKCqywdPwW)



