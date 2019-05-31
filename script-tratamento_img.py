
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import glob

EXTESAO_IMG = '.jpg'
TAXA_DIFERENCA = 0.2

num_px = 64
num_py = 64

def validaImagem(imagem : Image, taxaDeDiferenca=None) -> bool:
    width, height = 0, 0

    if taxaDeDiferenca:
        width, height = (taxaDeDiferenca * num_px, taxaDeDiferenca * num_py)

    if imagem.size[1] >= (num_px - width) and imagem.size[0] >= (num_py - height):
        return True
    return False

def paths(baseFolderName):
    ROOT_PATH = ''
    name = str(num_px) + '-' + str(num_py)+ '-' +"new" + baseFolderName +'-'
    cont = 1
    if os.path.exists('./'+baseFolderName):
        ROOT_PATH = os.path.abspath('./'+baseFolderName)  # retorna o caminho completo da base de imagem
    else:
        print("ErroFolder: '"+baseFolderName+"' Not Found")
        exit()

    while True:
        if os.path.exists('./'+name + str(cont)):
            cont += 1
        else:
            os.mkdir(name + str(cont))

            break

    NEW_BASE = os.path.abspath('./' + name+str(cont))

    return ROOT_PATH, NEW_BASE

""""" Main """""

print("Initializing...")
ROOT_PATH, NEW_BASE = paths(baseFolderName='base') # nome da pasta onde se encontra a base de imagens
dir = os.listdir(ROOT_PATH)# lista todosarquivos/pastas do diretorio ./base
dir.sort()

data = [[],[],[]] #nameClass, preTratamento, posTratamento
columns = []
labels = {}

count = 1
for pasta in dir:
    print(count,"/",len(dir))
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels[count] = ''.join(pasta.split('-')[0:2])  # cria os labels relacionando um dicetorio com um número
        os.mkdir(NEW_BASE + '/'+ str(count) + "-" + labels[count].rstrip(' '))

        countImg = 0
        imagens = glob.glob(ROOT_PATH + '/' + pasta + '/*')
        for pathImg in imagens:

            image = Image.open(pathImg)
            if validaImagem(image, TAXA_DIFERENCA):
                newImage = image.resize((num_px, num_py))
                newImage.save(NEW_BASE + '/'+ str(count) + "-" + labels[count].rstrip(' ')+ "/"+str(countImg)+EXTESAO_IMG)
            else:
                continue

            # imageNumpy = plt.imread(pathImg)
            # myImage = np.array(Image.fromarray(imageNumpy).resize((num_px, num_py))).reshape((1, num_px*num_py*3)).T
            # print("shape original: ", imageNumpy.shape)
            # plt.imshow(imageNumpy)
            # plt.show()

            countImg += 1

        data[0].append(labels[count])
        data[1].append(len(imagens))
        data[2].append(countImg)

        count += 1


data[0].insert(0,'Total')
data[1].insert(0,sum(data[1]))
data[2].insert(0,sum(data[2]))

data[0].append('Detalhes')
data[1].append('taxa de deferença:' + str(TAXA_DIFERENCA))
data[2].append("Dimensão: ({},{})".format(num_px,num_py))

df = pd.DataFrame(data= np.array(data).T,columns=['Classes','Nº Imagens pré tratamento','Nº Imagens pós tratamento'])
df.to_csv(NEW_BASE+'/dados da base ({}x{}).csv'.format(num_px, num_py))

