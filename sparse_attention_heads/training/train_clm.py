import os
import configparser
from data import WikipediaData, Vocab
from modules import SparseTransformer

CFG_FILE = "train.cfg"

# getting train config
config = configparser.ConfigParser()
abs_path = os.path.join(os.getcwd(), CFG_FILE)
config.read(abs_path)




# initializing modules
dataset = WikipediaData()