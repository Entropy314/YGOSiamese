import requests
import pandas as pd 
import os 
from skimage.util import random_noise
from PIL import Image
import cv2 


url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'
ygo_data = requests.get(url)
ksize = (15,15)

# Regular Image Paths
REGULAR_PATH = 'Images/regular/'
REGULAR_BLUR_PATH = 'Images/regular_blur/'
REGULAR_NOISE_PATH = 'Images/regular_noise/'

# Cropped Image Paths
REGULAR_CROP_PATH = 'Images/regular_crop/'
REGULAR_CROP_BLUR_PATH = 'Images/regular_crop_blur/'
REGULAR_CROP_NOISE_PATH = 'Images/regular_crop_noise/'

def download_image(obj, size='big'):
    directory = obj['name'].replace(' ', '_').replace('/', '__')
    if size =='big': 
        key = 'image_url'
    elif size == 'small': 
        key = 'image_url_small'
    os.mkdir(directory)
    for x in obj['card_images']:
        card_id = x.get('id')
        url = x.get(key)
        img = Image.open(requests.get(url, stream=True).raw)
        img.save(f'{directory}/{card_id}.jpg')


def blur_image(img_path): 
    img = cv2.imread(img_path)
    img = cv2.blur(img, ksize)
    return img


def noise_image(img_path): 
    img = cv2.imread(img_path)
    img = random_noise(dm, mode='s&p',amount=.5)
    return img


def crop_image(img_path, blur=False, noise=False, offset=(0,0)):
    horizontal_shift, vertical_shift = offset
    x, y = 50, 110
    h, w = 325, 325
    if noise: 
        img = noise_image(img)
    if blur: 
        img = blur_image(img)
    crop_img = img[y:y+h+vertical_shift, x:x+w+horizontal_shift]
    return crop_img

if __name__ == '__main__': 
    ygo_data = requests.get(url)
    result = map(download_image, ygo_data.json()['data'])
    print(list(result))