import tensorflow as tf 
import numpy as np 
import pandas as pd 
import train
import os
import siamese_nn


class ModelPredicting(siamese_nn.SiameseModel): 

    def __init__(self, pretrained_model:str='RESNET101', target_shape=(200,200), output_size:int=64, environment:str='cloud'): 
        super(siamese_nn.SiameseModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.MODEL = pretrained_model
        self.output_size = output_size
        self.target_shape = target_shape
        self.environment = environment
    
    def initalize_and_load_model(self): 
        
        self.create_model(model=self.pretrained_model, weights='IMAGENET')
        
        if self.environment == 'cloud':
            latest = tf.train.latest_checkpoint(f'gs://cards-data/yugioh/models/{self.pretrained_model}_{self.output_size}')
        else: 
            latest = tf.train.latest_checkpoint(f'model/{self.pretrained_model}_{self.output_size}')
        self.siamese_model.load_weights(latest)
        self.prediction_model = self.siamese_model.layers[0].layers[-2]


