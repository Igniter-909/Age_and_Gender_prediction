import os
import sys
from src.logger import logging
from src.exception import CustomException

import zipfile
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Data_Extraction:
    def __init__(self,zipfile_file_path,extract_to_zipfile):
        self.zipfile_file_path = zipfile_file_path
        self.extract_to_zipfile = extract_to_zipfile

    def data_extraction(self):
        os.makedirs(self.extract_to_zipfile,exist_ok=True)
        logging.info("Files folder is created:")

        try:
            with zipfile.ZipFile(self.zipfile_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_to_zipfile)
                logging.info("Files extracted successfully from the zip file")

        except Exception as e:
            raise CustomException(e,sys)


