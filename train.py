import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, metrics, Model
import cv2
from functools import partial
import itertools
import json
import authenticate as auth
import siamese_nn

# Local Imports
import pull_process_image as ppi
import model_references as MR

global environment
environment = 'cloud'

class TrainModel(siamese_nn.SiameseModel, auth.DataStore): 

    def __init__(self, target_shape:tuple=(224,224), pretrained_model:str='RESNET50', to_gray:bool=False, use_crop:bool=True, output_size:int=64): 
        super(siamese_nn.SiameseModel, self).__init__()
        self.target_shape = target_shape
        self.to_gray = to_gray
        self.output_size = output_size
        self.pretrained_model = pretrained_model
        if use_crop:
            self.use_crop = '_crop'
    
        else: 
            self.use_crop = ''

        self.regular_path = f'Images/regular{self.use_crop}/'
        self.blurred_path = f'Images/regular{self.use_crop}_blur/'
        self.noise_path = f'Images/regular{self.use_crop}_noise/'    

    def process_file_data(self): 
        folders = os.listdir(self.regular_path)[1:]
        self.mapping = {}
        self.all_pairs = []
        for x in folders:
            anchor_image_files = os.listdir(self.regular_path + x)
            positive_image_files = [self.blurred_path + x for y in anchor_image_files] 
            positive_image_files.extend([self.regular_path + x for y in anchor_image_files])
            pairs = list(itertools.product(anchor_image_files, positive_image_files))
            self.all_pairs.append(pairs)
            self.mapping[x] = {}
            self.mapping[x]['anchor'] = [(pairs[0][1] + '/'+ pairs[0][0]).replace('regular_crop_blur', 'regular_crop')  for x in pairs] #[regular_path + x + '/' + y for y in  [x[0] for x in pairs]]
            self.mapping[x]['positive'] =  [pairs[1][1] + '/' + pairs[1][0] for x in pairs]
        
        self.sorted_keys = sorted(self.mapping)
        self.all_pairs = []
        # card_image_files = os.listdir(self.regular_path)[1:]
        for card in self.sorted_keys: 
            ls = os.listdir(self.regular_path + '/' + card)
            for num in ls: 
                self.all_pairs.append((card, num))
        self.mapping_copy = self.mapping.copy()
        self.anchor_paths = [f'Images/regular_crop/{x[0]}/{x[1]}' for x in self.all_pairs]

    def preprocess_image(self, filename:str, to_gray:bool=False): 
        
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float16)    
        image = tf.image.resize(image, self.target_shape)
        if to_gray: 
            image = tf.image.rgb_to_grayscale(image)
            image = tf.repeat(image[...,tf.newaxis], 3,-1)
            image = tf.reshape(image, self.target_shape + (3,))
        return image 


    def preprocess_triplets(self, anchor:tf.Variable, positive:tf.Variable, negative:tf.Variable, to_gray:bool=False): 
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """
        return (
                self.preprocess_image(anchor, to_gray),
                self.preprocess_image(positive, to_gray),
                self.preprocess_image(negative, to_gray),
            )
    
    def create_dataset(self):
        self.anchor_images = []
        self.positive_images = []
        for x in self.sorted_keys:

            self.anchor_images.extend(self.mapping[x]['anchor'])
            self.positive_images.extend(self.mapping[x]['positive'])

        # return anchor_images, positive_images
        self.image_count = len(self.anchor_images)
        print(self.image_count)
        anchor_dataset = tf.data.Dataset.from_tensor_slices(self.anchor_images)
        positive_dataset = tf.data.Dataset.from_tensor_slices(self.positive_images)


        # To generate the list of negative images, let's randomize the list of
        # available images and concatenate them together.
        rng = np.random.RandomState(seed=42)
        ai = self.anchor_images.copy()
        pi = self.positive_images.copy()
        rng.shuffle(ai)
        rng.shuffle(pi)
        negative_images = ai + pi
        np.random.RandomState(seed=32).shuffle(negative_images)


        negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
        negative_dataset = negative_dataset.shuffle(buffer_size=24000)

        preprocess_triplets2 = partial(self.preprocess_triplets, to_gray=self.to_gray)
        self.dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        self.dataset = self.dataset.shuffle(buffer_size=12000)
        self.dataset = self.dataset.map(preprocess_triplets2, True)
        # return positive_images

    def generate_dataset(self, size:int=1):

        self.dataset = self.create_dataset(False)
        if size > 1: 
            for i in range(size):
                self.dataset.concatenate(self.create_datset(False))
        self.multiplier = size

    def create_train_val_dataset(self, train_ratio:float=0.8): 
        self.train_dataset = self.dataset.take(round(self.image_count * train_ratio ))
        self.val_dataset = self.dataset.skip(round(self.image_count * train_ratio ))

    def train_model(self, epochs:int=10, save:bool=True, batch_size:int=64): 
        print(self.pretrained_model)
        print(self.output_size)
        print(batch_size)
        if environment == 'cloud':
            checkpoint_path = f'gs://cards-data/yugioh/models/{self.pretrained_model}_{self.output_size}/' + 'cp-{epoch:04d}.cpkt'
        else: 
            checkpoint_path = f'model/{self.pretrained_model}_{self.output_size}/' + 'cp-{epoch:04d}.cpkt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                         save_weights_only=True, save_freq=5*batch_size)
        self.create_model(self.target_shape, model=self.pretrained_model, weights='IMAGENET')
        self.siamese_model.fit(self.train_dataset.batch(batch_size, drop_remainder=False), epochs=epochs, 
                                     validation_data=self.val_dataset.batch(batch_size, drop_remainder=False),
                                     callbacks = [cp_callback], verbose=2)

    def preprocess_and_embed(self, x:str): 
        
        x = tf.Variable([self.preprocess_image(x)])
        x = MR.PREPROCESS[self.pretrained_model].preprocess_input(x)
        return self.embedding_model(x)

    def fetch_anchor_embeddings(self, save:bool=True): 
        self.ANCHOR_EMBEDDINGS = []
        i = 0 
        tmp = tf.data.Dataset.from_tensor_slices(self.anchor_paths)
        ############################
        for img in tmp:
            if i % 1000 == 0 : 
                print(i)
            self.ANCHOR_EMBEDDINGS.append(self.preprocess_and_embed(img))
            i += 1
        ############################
        # pool = mp.Pool(num_cores)
        # ANCHOR_EMBEDDINGS = pool.map(preprocess_and_embed, tmp)
        self.ANCHOR_EMBEDDINGS = [x.numpy().tolist()[0] for x in self.ANCHOR_EMBEDDINGS]
        self.anchor_labels = [str(x.numpy()).split('/')[-2]for x in tmp]
        self.embedding_vec = dict(zip(self.anchor_paths, self.ANCHOR_EMBEDDINGS))
        if save: 
            
            with open(f'embeddings/anchor_embedding_{self.pretrained_model}_{self.output_size}.json', 'w') as outfile:
                json.dump(self.embedding_vec, outfile)
                
            with open(f'gs://cards-data/yugioh/embeddings/anchor_embedding_{self.pretrained_model}_{self.output_size}.json', 'w') as outfile:
                json.dump(self.embedding_vec, outfile)
if __name__ == '__main__':
    print(environment)
    m = TrainModel(pretrained_model='MOBILENETV2')
    m.process_file_data()
    m.create_dataset()
    m.create_train_val_dataset()
    m.create_model()
    m.train_model(epochs=1)
    m.fetch_anchor_embeddings(True)
