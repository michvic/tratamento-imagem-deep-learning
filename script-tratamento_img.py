## crie uma pasta com o nome 'base', na raiz do projeto onde esse escript vai ser executado
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import glob

EXTESAO_IMG = '.jpg'
TAXA_DIFERENCA = 0.3

num_px = 90
num_py = 100


def validaImagem(imagem : Image, taxaDeDiferenca=None) -> bool:
    width, height = 0, 0

    if taxaDeDiferenca:
        width, height = (taxaDeDiferenca * num_px, taxaDeDiferenca * num_py)

    if imagem.size[0] >= (num_px - width) or imagem.size[0] >= (num_py - height):
        return True
    return False

def paths(baseFolderName):
    ROOT_PATH = ''
    name = 'new_' + baseFolderName + '-1'

    if os.path.exists('./'+baseFolderName):
        ROOT_PATH = os.path.abspath('./'+baseFolderName)  # retorna o caminho completo da base de imagem
    else:
        print("ErroFolder: '"+baseFolderName+"' Not Found")
        exit()

    while True:
        if os.path.exists('./'+name):
            name = name.split('-')
            name = name[0] + '-' + str(int(name[1]) + 1)
        else:
            os.mkdir(name)

            break

    NEW_BASE = os.path.abspath('./' + name)

    return ROOT_PATH, NEW_BASE



""""" Main """""

ROOT_PATH, NEW_BASE = paths(baseFolderName='base') # nome da pasta onde se encontra a base de imagens
dir = os.listdir(ROOT_PATH)# lista todosarquivos/pastas do diretorio ./base
dir.sort()

dataPre = []
dataPos = []
columns = []
labels = {}



count = 1
for pasta in dir:
    print(count,"/",len(dir))
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels[count] = ''.join(pasta.split('-')[0:2])  # cria os labels relacionando um dicetorio com um número
        os.mkdir(NEW_BASE + '/'+ labels[count] +'- '+ str(count))

        countImg = 0
        imgagens = glob.glob(ROOT_PATH + '/' + pasta + '/*' + EXTESAO_IMG)
        for pathImg in imgagens:

            image = Image.open(pathImg)
            if validaImagem(image, TAXA_DIFERENCA):
                newImage = image.resize((num_px, num_py))
                newImage.save(NEW_BASE + '/'+ labels[count] +'- '+ str(count)+'/'+str(countImg)+EXTESAO_IMG )
            else:
                continue

            # imageNumpy = plt.imread(pathImg)
            # myImage = np.array(Image.fromarray(imageNumpy).resize((num_px, num_py))).reshape((1, num_px*num_py*3)).T
            # print("shape original: ", imageNumpy.shape)
            # plt.imshow(imageNumpy)
            # plt.show()

            countImg += 1

        columns.append(labels[count])
        dataPos.append(countImg)
        dataPre.append(len(imgagens))
        count += 1


df = pd.DataFrame(data= dataPos,index=columns,columns=['Nº Imagens pós tratamento'])
df.to_csv(NEW_BASE+'/dados da base.csv')