import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as io
from PIL import Image

num_px = 50
nx = num_px*num_px*3
m = 1
data = np.zeros((nx,m))

for i in range(m):
    my_image = str(i+1)+".jpg"   # change this to the name of your image file
    # We preprocess the image to fit your algorithm.
    fname = "base/" + my_image
    image = plt.imread(fname)
    print(image.shape)
    my_image = np.array(Image.fromarray(image).resize((num_px, num_px)))
    print(my_image.shape)
    my_image = my_image.reshape((1, num_px*num_px*3)).T
    print(my_image.shape)
    data[:,i] = np.squeeze(my_image)
    print(np.squeeze(my_image).shape)
    #plt.imshow(image)
    #plt.show()

io.savemat('data',dict(data=data))