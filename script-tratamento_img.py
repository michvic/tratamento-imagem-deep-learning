## crie uma pasta com o nome 'base', na raiz do projeto onde esse escript vai ser executado
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import glob

EXTESAO_IMG = '.jpg'
num_px = 50
nx = num_px*num_px*3
m = 120
data = np.zeros((nx,m))

ROOT_PATH = os.path.abspath('./base')#retorna o caminho completo da pasta ./base
dir = os.listdir(ROOT_PATH)# lista todos arquivos/pastas do diretorio ./base

labels = {}


count = 1
for pasta in dir:
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels[pasta] = count # cria os labels relacionando um dicetorio com um número
        count += 1

        for pathImg in glob.glob(ROOT_PATH + '/' + pasta + '/*' + EXTESAO_IMG):
            imageNumpy = imageio.imread(pathImg)
            print("shape antes: ", imageNumpy.shape)
            # We preprocess the image to fit your algorithm.
            my_image = np.array(Image.fromarray(imageNumpy).resize((num_px, num_px)))
            print("shape depois: ", my_image.shape)
            plt.imshow(my_image)
            plt.show()
            break



print(labels)

# for label in labels:
#     imagens = glob.glob(rootPath+'/'+label+'/*'+EXTESAO_IMG)
#     print(imagens[0])
#
#     for img in imagens:
#         imageNumpy = imageio.imread(img)
#         print("shape antes: ", imageNumpy.shape)
#         # We preprocess the image to fit your algorithm.
#         my_image = np.array(Image.fromarray(imageNumpy).resize((num_px, num_px)))
#         print("shape depois: ", my_image.shape)
#         plt.imshow(my_image)
#         plt.show()
#         break
#     # sio.savemat('data',dict(data=data))
#
#
#
#     #  my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T