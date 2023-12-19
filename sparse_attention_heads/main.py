from training.train import train
import logging
import sys

if __name__ == "__main__":
    
    logFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s (%(asctime)s)")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format("logs", "train"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

    train()