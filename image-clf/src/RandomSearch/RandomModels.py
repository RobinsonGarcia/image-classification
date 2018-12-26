import json
import os
import numpy as np
from numpy.random import uniform,choice
#==================INPUT DATA=================================#

class RandomModels:

    def __init__(self,kwargs={'architecture':['vgg16'],
                'img_input_sizes':224,
                'modes' : ['fine_tunning','transfer_learning'],
                'num_classes':10,
                'dense_layer_sizes':[1024,2048],
                'num_layers':[1,2,3],
                'optimizers':['adam'],
                'learning_rate':(-7,-5),
                'momentum':(0.9,0.99),
                'decay':(-5,-8),
                'nesterov':1,
                'epochs':50,
                'batch_size':20,
                'dropout':[0.5],
                'reg_rate':(-20,-5),
                'train_dir':'/home/robinson/Documents/ImageClassification/AssetX_Image3/train',
                'val_dir':'/home/robinson/Documents/ImageClassification/AssetX_Image3/val',
                'test_dir':'/home/robinson/Documents/ImageClassification/AssetX_Image3/test'}):
            for k,v in kwargs.items():
                print(v,type(v))
                setattr(self,k,v)
            self.filenames = []


    def build_configs(self,number_of_models,path='tmp'):
        for i in range(number_of_models):

            config = {
                    'architecture':choice(self.architecture),
                    'dense_layer_size':int(choice(self.dense_layer_sizes)),
                    'mode':choice(self.modes),
                    'num_layers':int(choice(self.num_layers)),
                    'num_classes':self.num_classes,
                    'img_input_size':self.img_input_sizes,

                    'optimizer':choice(self.optimizers),
                    'learning_rate':float(10**uniform(self.learning_rate[0],self.learning_rate[1])),

                    'momentum':float(choice(self.momentum)),
                    'decay':float(10**uniform(self.decay[0],self.decay[1])),
                    'nesterov':self.nesterov,
                    'epochs':self.epochs,
                    'batch_size':self.batch_size,

                    'dropout':float(choice(self.dropout)),
                    'reg_rate':0,#float(10**uniform(self.reg_rate[0],self.reg_rate[1])),

                    'train_dir':self.train_dir,
                    'val_dir':self.val_dir,
                    'test_dir':self.test_dir
                    }
                
            filename = path+'/config'+str(i).zfill(4)+'.json'

            if path not in os.listdir():os.system('mkdir '+path)

            json.dump(config,open(filename,'w'))
            self.filenames.append(filename)
    
    def showfile(self,name=None):
        if name==None:
            print('Choose one file:')
            print('verify filenames as self.filenames')

            return self.filenames
        else:
            f = open(name,"r")
            config = json.loads(f.read())
            for k,v in config.items():
                print(k,v)
            f.close()
            pass
    

