from collections import OrderedDict
import csv
import os

keys = ['T Loss', 'V Top-1']

class Logger_dict():
    def __init__(self, logger, save_path, keys=keys):
        self.dict = OrderedDict()
        self.logger = logger
        self.savepath = save_path
        if os.path.exists(os.path.join(self.savepath, 'log.csv')):
            self.init_csv()
        self.write_csv(keys)
                
    def update(self, key, value):
        self.dict[key] = value
        
    def init_csv(self):
        fileVariable = open(os.path.join(self.savepath, 'log.csv'), 'r+')
        fileVariable.truncate(0)
        fileVariable.close()
        
    def write_csv(self, x):
        with open(os.path.join(self.savepath, 'log.csv'), "a") as outfile:
            csvwriter = outfile
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(x)
        
    def print(self):
        i = 0
        values = []
        for key, value in self.dict.items():
            print(f'{key}' +'\t'+ f'{value}')
            i += 1
            values.append(value)
        self.write_csv(values)
        print()
            
    