## crie uma pasta com o nome 'base', na raiz do projeto onde esse escript vai ser executado
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import glob

EXTESAO_IMG = '.jpg'
num_px = 50
nx = num_px*num_px*3
m = 120
data = np.zeros((nx,m))

rootPath = os.path.abspath('./base')# raiz da base de imagens
dir = os.listdir(rootPath)# lista todos arquivos/pastas do diretorio ./base

labels = {}

# cria os labels relacionando um dicetorio com um número
count = 1
for i in dir:

    if os.path.isdir(rootPath + '/'+i):
        labels[i] = count
        count += 1

print(labels)

for i in labels:
    imagens = glob.glob(i+EXTESAO_IMG)
    print(imagens)

#     my_image = str(i+1)+".jpeg"   # change this to the name of your image file
#     # We preprocess the image to fit your algorithm.
#     fname = "base/" + my_image
#     image = np.array(ndimage.imread(fname, flatten=False))
#     my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
#     data[:,i] = np.squeeze(my_image)
#     # plt.imshow(image)
#     # plt.show()
#
# sio.savemat('data',dict(data=data))