import os
from google.cloud import storage 
import gcsfs

class DataStore: 

    def __init__(self, bucket_name:str='cards-data'):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
