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


def validaImagem(imagem : Image) -> bool:
    print(imagem)
    if (imagem.size[0] >= num_px or imagem.size[1] >= num_py):
        return True
    return False

def paths(baseFolderName):
    ROOT_PATH = os.path.abspath('./'+baseFolderName)  # retorna o caminho completo da base de imagem
    name = 'new_base-1'

    while True:

        try:
            os.mkdir(name)
            break

        except FileExistsError:
            name = name.split('-')
            name = name[0]+'-'+str(int(name[1])+1)
            os.mkdir(name)
            break
        except:
            continue



    NEW_BASE = os.path.abspath(name)

    return ROOT_PATH, NEW_BASE


ROOT_PATH = os.path.abspath('./base')#retorna o caminho completo da pasta ./base
NEW_BASE = os.path.abspath('./base-2')
dir = os.listdir(ROOT_PATH)# lista todosarquivos/pastas do diretorio ./base

labels = {}

count = 1
for pasta in dir:
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels[pasta] = count # cria os labels relacionando um dicetorio com um número

        os.mkdir(NEW_BASE + '/class-' + str(count))

        countImg = 1
        for pathImg in glob.glob(ROOT_PATH + '/' + pasta + '/*' + EXTESAO_IMG):

            image = Image.open(pathImg)
            if validaImagem(image):
                newImage = image.resize((num_px, num_py))
                newImage.save(NEW_BASE + '/class-' + str(count)+'/'+str(countImg)+EXTESAO_IMG )
            else:
                continue

            # imageNumpy = plt.imread(pathImg)
            # myImage = np.array(Image.fromarray(imageNumpy).resize((num_px, num_py))).reshape((1, num_px*num_py*3)).T
            # print("shape original: ", imageNumpy)
            # plt.imshow(imageNumpy)
            # plt.show()

            countImg += 1
        count += 1

