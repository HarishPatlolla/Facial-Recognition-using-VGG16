import os, sys
import math, keras, datetime, pandas as pd, threading, json, collections
import glob
import shutil
import zipfile
import numpy as np
import tarfile
import re
import pickle
import requests
from tqdm import tqdm_notebook as tqdm
from numpy.random import random, permutation
from keras import backend as K
from keras.utils.data_utils import get_file
import tensorflow as tf
from itertools import islice
import matplotlib
from matplotlib import pyplot as plt, rcParams
import random
from PIL import Image, ImageOps, ImageFont, ImageDraw 
import imageio
from skimage import color

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def limit_gpu_mem():
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)


def make_path(path):
    if not os.path.exists(path): 
        os.makedirs(path)
        
def download(url, filename):
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
                
def download_data (url, data_path, overwrite=False):
    filename = url.split("/")[-1]
    dest = data_path + filename
    if not os.path.exists(dest):
        os.chdir(data_path)
        download(url, filename)
    elif os.path.exists(dest) and overwrite:
        download(url, filename)   
    return dest

def untar_data(path):
    for file in os.scandir(path):
        if file.name.endswith('.zip'):
            print(os.stat(path + file.name).st_size, file.name, "this is zip file")
            with zipfile.ZipFile(file.name, 'r') as z:
                z.extractall(path)
        elif file.name.endswith('.gz'):
            '''specail condition needs to be added for .bin files'''
            print(os.stat(path + file.name).st_size, file.name, "this is gz file")
            
            tf = tarfile.open(file.name, "r")
            tf.extractall()
            tf.close()
        elif file.name.endswith('.tar'):
            print(os.stat(path + file.name).st_size, file.name, "this is tar file")
            tf = tarfile.open(file.name, "r")
            tf.extractall()
            tf.close()
        
    for file in os.scandir(path):
        print(file.name)
        
def download_untar(url, path, overwrite=False):
    dest = download_data(url, path, overwrite=overwrite)
    untar_data(path)
    files = glob.glob(path + '/**/*.*', recursive=True)
    return files

def files_in_path(path):
    file_counts = {}
    for f in os.listdir(path):
        if os.path.isdir(path+f):      
            file_counts[f] = sum([len(files) for r, d, files in os.walk(path+f)])
    return file_counts


def files_in_dir(path):
    files = {}
    tfiles = 0
    tdirs = 0
    for i, f in enumerate(os.listdir(path)):       
        if os.path.isdir(path+f):
            tdirs = tdirs + 1
            for r, d, fil in os.walk(path+f):
                files[f] = len(fil)
                tfiles = tfiles + len(fil)
    return files, tfiles, tdirs

def n_images_from_dir(path, num):
    file_images = {}
    for f in os.listdir(path):
        if os.path.isdir(path+f):
            for r, d, fil in os.walk(path+f):
                idxs = random.sample(range(0, len(fil)), num)
                random_imgs = []
                for idx in idxs:
                    random_imgs.append(fil[idx])
                file_images[f] = random_imgs
    return file_images

def draw_bar_chart(labels, numbers, x_label='Class names', y_label='class count', 
                   chart_name='Class distribution',start=0, num=6, y_space=20, color='green'):
    
    end = start+num
    if end>len(numbers):end=len(numbers)
    ind = np.arange(end - start)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    plt.bar(ind, numbers[start:end], width, color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(chart_name)
    plt.xticks(ind, labels[start:end], rotation='vertical')
    plt.yticks(np.arange(0, max(numbers), y_space))

    plt.show

def display_class_imgs(class_img_list):
    
    fig = plt.figure(figsize=(len(class_img_list)*7, len(class_img_list)*3))
    columns = 1
    rows = len(class_img_list)
    # ax enables access to manipulate each of subplots
    ax = []

    for i, im in enumerate(class_img_list):
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("Class:"+class_img_list[i][0], fontsize=20)  # set title
        plt.grid(b=None)
        plt.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False, # ticks along the bottom edge are off
            left=False,
            top=False,         # ticks along the top edge are off
            labelleft=False,
            labelbottom=False)
        plt.imshow(class_img_list[i][1])
    plt.show()  # finally, render the plot

def create_valid_set(valid_pct, rand_seed):
    for dr in os.listdir(data_path+'train/'):

        if os.path.exists(data_path+'valid/'+dr):
            for fn in os.listdir(data_path+'valid/'+dr):
                shutil.move(data_path+'valid/'+dr+'/'+fn, data_path+'train/'+dr)
            shutil.rmtree(data_path+'valid/'+dr)
            
        if os.path.isdir(data_path+'train/'+dr):      
            f_count = sum([len(files) for r, d, files in os.walk(data_path+'train/'+dr)])
            val_file_numbers = int(f_count * valid_pct)
            np.random.seed(rand_seed)
            x = np.random.randint(f_count, size=val_file_numbers)
            if not os.path.exists(data_path+'valid/'+dr):
                os.makedirs(data_path+'valid/'+dr)
            for i, fn in enumerate(os.listdir(data_path+'train/'+dr)):
                if i in x:
                    shutil.move(data_path+'train/'+dr+'/'+fn, data_path+'valid/'+dr)


def merge_images(file_list_dict, path, classes, size = (224, 224)):
    
    class_img_list = []
    for cl in classes:
        img_list = ()
        images = file_list_dict[cl]
        
        tiled_img = Image.new('RGB', (len(images)*(size[0]), size[1]))
        width = 0
        
        for image in images:
            img_path = path+cl+'/'+image
            img = Image.open(img_path)
            thumb = ImageOps.fit(img, size)
            tiled_img.paste(im=thumb, box=(width, 0) )
            width +=size[0]
        img_list = (cl, tiled_img)
        class_img_list.append(img_list) 
    return class_img_list

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')



def imread(path, mode="RGB"):
    # Loads data in HxW format, then transposes to correct format
    img = np.array(imageio.imread(path, pilmode=mode))
    return img
    

def imresize(img, size, interp='bilinear'):
    """
    Resizes an image

    :param img:
    :param size: (Must be H, W format !)
    :param interp:
    :return:
    """
    if interp == 'bilinear':
        interpolation = Image.BILINEAR
    elif interp == 'bicubic':
        interpolation = Image.BICUBIC
    else:
        interpolation = Image.NEAREST

    # Requires size to be HxW
    size = (size[1], size[0])

    if type(img) != Image:
        img = Image.fromarray(img, mode='RGB')

    img = np.array(img.resize(size, interpolation))
    return img
    
    
def imsave(path, img):
    imageio.imwrite(path, img)
    return
    

def fromimage(img, mode='RGB'):
    if mode == 'RGB':
        img = color.lab2rgb(img)
    else:
        img = color.rgb2lab(img)
    return img


def toimage(arr, mode='RGB'):
    return Image.fromarray(arr, mode)





