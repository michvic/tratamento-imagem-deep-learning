
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import glob

EXTESAO_IMG = '.jpg'
TAXA_DIFERENCA = 0.15

num_px = 56
num_py = 56



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

    # while True:
    #     if os.path.exists('./'+name + str(cont)):
    #         cont += 1
    #     else:
    #         os.mkdir(name + str(cont))
    #
    #         break
    #
    # NEW_BASE = os.path.abspath('./' + name+str(cont))

    return ROOT_PATH, None


""""" Main """""

print("Initializing...")
ROOT_PATH, NEW_BASE = paths(baseFolderName='base') # nome da pasta onde se encontra a base de imagens
dir = os.listdir(ROOT_PATH)# lista todosarquivos/pastas do diretorio ./base
dir.sort()

imgStatistic = []
data = [[],[],[], [], []] #nameClass, maxSizeIMG, minSizeIMG, maxPixel, minPixel,
labels = {}

count = 1
for pasta in dir:
    print(count,"/",len(dir))
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels[count] = ''.join(pasta.split('-')[0:2])  # cria os labels relacionando um dicetorio com um número
        # os.mkdir(NEW_BASE + '/'+ str(count) + "-" + labels[count].rstrip(' '))

        countImg = 0
        maiorDimensao = 0
        menorDimensao = 9999999999
        sizeMaior = None
        sizeMenor = None
        imagens = glob.glob(ROOT_PATH + '/' + pasta + '/*')
        for pathImg in imagens:
            image = Image.open(pathImg)
            # if validaImagem(image, TAXA_DIFERENCA):
            #     newImage = image.resize((num_px, num_py))
            #     newImage.save(NEW_BASE + '/'+ str(count) + "-" + labels[count].rstrip(' ')+ "/"+str(countImg)+EXTESAO_IMG)
            # else:
            #     continue

            # imageNumpy = plt.imread(pathImg)
            # myImage = np.array(Image.fromarray(imageNumpy).resize((num_px, num_py))).reshape((1, num_px*num_py*3)).T
            # print("shape original: ", imageNumpy.shape)
            # plt.imshow(imageNumpy)
            # plt.show()
            imgStatistic.append(image.size[0] * image.size[1])

            if image.size[1] * image.size[0] > maiorDimensao:
                maiorDimensao = image.size[1] * image.size[0]
                sizeMaior = image.size
            if image.size[1] * image.size[0] < menorDimensao:
                menorDimensao = image.size[1] * image.size[0]
                sizeMenor = image.size

            countImg += 1

        data[0].append(labels[count])
        data[1].append(sizeMaior)
        data[2].append(sizeMenor)
        data[3].append(maiorDimensao)
        data[4].append(menorDimensao)

        count += 1

imgStatistic.sort()
print("menor imagem: " +  str(imgStatistic[0]))
print("maior imagem: " +str(imgStatistic[-1]))

df = pd.DataFrame(data= np.array(data).T, columns=['Classes','Maior IMG','Menor IMG','Maior pixel','Menor pixel'])
df.to_csv(ROOT_PATH+'/estatistica-base.csv')
