## crie uma pasta com o nome 'base', na raiz do projeto onde esse escript vai ser executado
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import glob

EXTESAO_IMG = '.jpg'

num_px = 90
num_py = 100
# nx = num_px*num_px*3
# m = 120
# data = np.zeros((nx,m))

ROOT_PATH = os.path.abspath('./base')#retorna o caminho completo da pasta ./base
NEW_BASE = os.path.abspath('./base-2')
dir = os.listdir(ROOT_PATH)# lista todosarquivos/pastas do diretorio ./base

labels = {}
data = {}

count = 1
for pasta in dir:
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels[pasta] = count # cria os labels relacionando um dicetorio com um número

        os.mkdir(NEW_BASE + '/class-' + str(count))

        countImg = 1
        for pathImg in glob.glob(ROOT_PATH + '/' + pasta + '/*' + EXTESAO_IMG):

            image = Image.open(pathImg)
            newImage = image.resize((num_px, num_py))
            newImage.save(NEW_BASE + '/class-' + str(count)+'/'+str(countImg)+EXTESAO_IMG )


            # imageNumpy = plt.imread(pathImg)
            # myImage = np.array(Image.fromarray(imageNumpy).resize((num_px, num_py))).reshape((1, num_px*num_py*3)).T


            # print("shape original: ", imageNumpy)
            # plt.imshow(imageNumpy)
            # plt.show()

        #     data[:, countImg] = np.squeeze(myImage)
            countImg += 1
        #
        # sio.savemat('data-'+str(count), dict(data=data))
        count += 1

