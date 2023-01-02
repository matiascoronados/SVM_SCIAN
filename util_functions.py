# Se crea un archivo en la direccion ingresada, esta se elimina si ya existe.

import os
import re
import shutil
import random
from constants import LABBELS_NAMES

def create_folder( folder_name, dest_path ):
    try:
        folder_path = dest_path+'/'+folder_name
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)   
        os.mkdir(folder_path)
        return True
    except OSError as err:
        print("OS error:", err)

# Se crea un archivo en la direccion ingresada
def create_folder_wremoving( folder_name, dest_path ):
    try:
        folder_path = dest_path+folder_name
        if not(os.path.exists(folder_path)):
            os.mkdir(folder_path)
        return True
    except OSError as err:
        print("OS error:", err)


def get_most_frecuent_value( arr ):
    count = 0
    aux = arr[0]
    for i in range(len(arr)-1):
        frec = arr.count(arr[i])
        if(frec > count):
            count = frec
            aux = arr[i]
    return aux


def remove_elements(arr, element):
    return [x for x in arr if x != element]


def remove_elements_from_list( arr, remove_list ):
    for element in remove_list:
        arr.remove( element )
    return arr

def read_file( file_path ):
    file = open(file_path, 'r')
    elements = []
    for line in file:
        name, labbel = line.split(' ')
        labbel = labbel.replace( '\n', '' )
        elements.append([name, labbel])
    return elements


def remove_repeated( arr ):
    aux = []
    for element in arr:
        if element not in aux:
            aux.append(element)
    return aux



# Se crea el data-set a partir de la informacion presente en expertAnotations.txt, y las partialImages de entrada.
def create_dataset(path_expertAnotations,path_partialImages, dest_path, dataset_name):
    try:
        file = open(path_expertAnotations, 'r')
        create_folder(dataset_name, dest_path)
        for labbel_name in LABBELS_NAMES:
            create_folder( labbel_name ,dest_path+'/'+dataset_name)
        for x in file:
            # Se obtiene el nombre y clase desde el archivo txt
            aux1 = x.split('	')
            aux2 = aux1[0].replace('\n','').split('-')
            aux3 = aux2[2].split('/')
            clase = int(aux1[4].replace('\n',''))
            p = aux2[0]
            pl = aux2[1]
            n_sample = int(re.split('(\d+)',aux3[0])[1])
            n_sperm = int(re.split('(\d+)',aux3[1])[1])
            # Se conforma el directorio de la imagen a partir de la informacion anterior.
            file = path_partialImages+'ch00_'+p+'-'+pl+'-sample'+str(n_sample)+'-sperm'+str(n_sperm)+'.tif'
            # Se conforma el directorio donde se va a copiar la imagen
            aux = dest_path+'/'+dataset_name+'/'
            if (clase == 0):
                aux=aux+'01-Normal'
            elif (clase == 1):
                aux=aux+'02-Tapered'
            elif (clase == 2):
                aux=aux+'03-Pyriform'
            elif (clase == 3):
                aux=aux+'04-Small'
            else:
                aux=aux+'05-Amorphous'
            # Se copia la imagen
            shutil.copy(file,aux)  
        return True
    except OSError as err:
        print("OS error:", err)


def copy_element(origin_path, dest_path):
    try:
        shutil.copy( origin_path, dest_path)
        return True
    except OSError as err:
        print("OS error:", err)

# Elige aleatoriamente un elemento de la lista, y luego lo elimina de esta.
def choose_random_element(elements_list):
    element = random.choice(elements_list)
    elements_list.remove(element)
    return element


def create_test_train_valid( experiment_path, experiment_number, origin_path, dest_path):
    create_folder('data', dest_path)
    dataset_path = dest_path+'/'+'data'

    aux = ['train', 'valid', 'test']
    image_validation = []
    for element in aux:
        create_folder(element, dataset_path)
        for labbel_name in LABBELS_NAMES:
            create_folder( labbel_name, dataset_path+'/'+element )

        text_path = experiment_path+'/experiment'+str(experiment_number)+'/'+element+'.txt'
        images = read_file( text_path )
        for image_name, labbel in images:
            copy_element( origin_path+'/'+image_name, dataset_path+'/'+element+'/'+labbel+'/'+image_name)
            image_validation.append(image_name)
    if len(image_validation) == 1132:
        return True
    else:
        return False
