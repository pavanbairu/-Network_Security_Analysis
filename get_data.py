import os
import json
import sys

from dotenv import load_dotenv
import pymongo.mongo_client

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")


import certifi
ca = certifi.where()

import numpy as np
import pandas as pd
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_converter(self, filepath):
        try:
            data = pd.read_csv(filepath)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def pushing_data_to_mongo(self, records, database, collections):
        try:
            self.records= records
            self.database = database
            self.collections = collections

            self.mongo_client = pymongo.MongoClient(MONGODB_URL)
            self.database = self.mongo_client[self.database]
            self.collections = self.database[self.collections]
            self.collections.insert_many(self.records)

            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
if __name__ == "__main__":
    FILE_PATH = r"D:\Data Science\github\Projects\ML\Network_Security_Analysis\datasets\NetworkData.csv"
    DATABSE = "ml-cluster"
    COLLECTION = "NetworkData"

    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_converter(FILE_PATH)
    no_of_records = networkobj.pushing_data_to_mongo(records, DATABSE, COLLECTION)
    print(no_of_records)
