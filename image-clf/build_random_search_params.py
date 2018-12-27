import pickle
import argparse
import os

#if __name__=="__main__":
parser = argparse.ArgumentParser()

parser.add_argument('--save_to',help='Set location to save random_search_params.pickle',default='.')
arguments = vars(parser.parse_args())

args = {'architecture':['vgg16'],
'img_input_sizes':224,
'modes' : ['fine_tunning','transfer_learning'],
'num_classes':10,
'dense_layer_sizes':[1024,2048,4096],
'num_layers':[1,2,3],
'optimizers':['adam'],
'learning_rate':(-7,-3),
'momentum':(0.9,0.99),
'decay':(-5,-8),
'nesterov':1,
'epochs':30,
'batch_size':20,
'dropout':[0.5],
'reg_rate':(-20,-2),
'train_dir':'/home/robinson/Documents/image-classification/dataset/train',
'val_dir':'/home/robinson/Documents/image-classification/dataset/val',
'test_dir':'/home/robinson/Documents/image-classification/dataset/test'}




f = open(os.path.join(arguments['save_to'],'random_search_params.pickle'),'wb')
pickle.dump(args,f)
f.close()
