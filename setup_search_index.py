import faiss 
import numpy as np
import tensorflow as tf
import train
import json
import predict

class FaissSearch(predict.ModelPredicting): 
    
    def __init__(self, anchor_labels:list=None, data:np.array=None, output_size:int=64): 
        
        self.output_size = output_size
        self.pretrained_model = pretrained_model
        self.data = data
        self.anchor_labels = anchor_labels
        if not data: 
            self.path = open(f'Images/anchor_embedding_{output_size}_{pretrained_model}.json')
            self.read_embedding_vectors()
            self.data = np.array([np.array(x) for x in list(self.embedding_vectors.values())])
            self.data = np.stack(self.data).astype('float16')

        self.data = data.astype('float16')

    def read_embedding_vectors(self):
        self.embedding_vectors =  json.load(self.path)
        
    def setup_faiss_index(self,  metric:str='L2'):
        
        feature_length = data.shape[1]
        
        if metric=='L2': 
            self.index = faiss.IndexFlatL2(feature_length)
        
        if metric == 'cosine':
            self.index = faiss.index_factory(feature_length, 'Flat',faiss.METRIC_INNER_PRODUCT)
            self.index.ntotal
            faiss.normalize_L2(data)
        self.index.add(data)

    def search_image_from_disk(self, img_path:str, index:int, k:int=5): 
        embed_vector = self.process_and_embed(tf.Variable(img_path))
        self.D, self.I = index.search(embed_vector.numpy(), k)
        
        return [self.anchor_labels[i] for i in  self.I[0].tolist()]
    
    def search_image_from_image(self)