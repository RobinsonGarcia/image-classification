#https://keras.io/models/sequential/
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#https://keras.io/preprocessing/image/
#https://keras.io/getting-started/sequential-model-guide/
#https://keras.io/applications/#resnet50
#https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
#https://medium.com/@anthony_sarkis/tensorboard-quick-start-in-5-minutes-e3ec69f673af
#https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/ keras saving loading model
#import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.metrics import top_k_categorical_accuracy
from keras import regularizers
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
import os
import sys
import json
import pprint




args=json.loads(open(sys.argv[1]).read())
job_root = sys.argv[2]
print("#=====================RandomModel summary: =============================#\n\n")
pprint.pprint(args)
print("\n\n#========================Start Trainning============================#\n\n")

ptree = os.listdir()
if 'weights' not in ptree: os.system('mkdir '+job_root+'/weights')
if 'graphs' not in ptree: os.system('mkdir '+job_root+'/graphs')
if 'checkpoints' not in ptree: os.system('mkdir '+job_root+ '/checkpoints')
if 'tboard_logs' not in ptree: os.system('mkdir '+job_root+'/tboard_logs')
if 'csv_logs' not in ptree: os.system('mkdir '+job_root+ '/csv_logs')
if 'logs' not in ptree: os.system('mkdir '+job_root+'/logs')
'''
#==================INPUT DATA=================================#
parser = argparse.ArgumentParser(description='Hyperparameters!')

#==================Architecture and mode=======================#
parser.add_argument('-arch','--architecture',help='Select an architecture:"vgg16","vgg19","inceptionV3"',default='vgg16')
parser.add_argument('-dsize','--dense_layer_size',help='This nets use a standard block {Dense+relu+dropout}, choose the size of the D layer',type=int,default=2096)
parser.add_argument('-m','--mode',help='Choose between "fine_tunning", "transfer_learning", or "retrain" a model',default='fine_tuning')
parser.add_argument('-nl','--num_layers',help='Number of blocks {Dense+relu+dropout} in the classifier',type=int,default=1)
parser.add_argument('-nc','--num_classes',help='Set the number of classes',type=int,default=19)
parser.add_argument('-s','--img_input_size',help='Size of the input image',type=int,default=224)

#========================Optimization hyperparameters:==============#
parser.add_argument('-opt','--optimizer',help='Select an optimization strategy:"adam","sgd"',default='sgd')
parser.add_argument('-lr','--learning_rate',help='Select a starting learning rate',type=float,default=1e-4)
parser.add_argument('-mm','--momentum',help='Select a momentum',type=float,default=0.9)
parser.add_argument('-dc','--decay',help='Select a decay rate',type=float,default=1e-6)
parser.add_argument('-nest','--nesterov',help='Select nesterov true(1) or false (0)',type=int,default=1)
parser.add_argument('-e','--epochs',help='Number of epochs to run',type=int,default=30)
parser.add_argument('-b','--batch_size',help='Set a batch_size',type=int,default=20)

#======================regularization==================================#
parser.add_argument('-dp','--dropout',help='Select a dropout rate',type=float,default=0.5)
parser.add_argument('-reg','--reg_rate',help='Select a rate for l2 weight regularization',type=float,default=0)

#================================Input_filea=============================#
parser.add_argument('-train','--train_dir',help='path to train folder',default='/home/robinson/Documents/ImageClassification/AssetX_Image_temp/train')
parser.add_argument('-val','--val_dir',help='path to val folder',default='/home/robinson/Documents/ImageClassification/AssetX_Image_temp/val')
parser.add_argument('-test','--test_dir',help='path to test folder',default='/home/robinson/Documents/ImageClassification/AssetX_Image_temp/test')


args = vars(parser.parse_args())
'''
#===================Helper functions========================#

#_accurcy metrics...
def Top5(y_true,y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def Top3(y_true,y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def Top2(y_true,y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

#_create model unique identifier
def get_id(x):
    id_ = ''
    for key in x.keys():
        if type(x[key])==str:
            id_+=key[:3]+'-'+x[key][:3]+'_'
        else:
            id_+=key[:3]+'-'+str("{:.2e}".format(x[key]))+'_'
    return id_

model_id = get_id(args)
#==================Data Generators=======================#

size = args['img_input_size']
batch_size = args['batch_size']

train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  args['train_dir'],
                  target_size=(size, size),
                  batch_size=batch_size,
                  class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                 args['val_dir'],
                 target_size=(size, size),
                 batch_size=batch_size,
                 class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                 args['test_dir'],
                 target_size=(size, size),
                 batch_size=batch_size,
                 class_mode='categorical')

#====Selects model architecture/base model====#
def get_base_model(architecture):
    if architecture=='vgg16': 
        from keras.applications import VGG16
        model = VGG16(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    if architecture=='vgg19': 
        from keras.applications import VGG19
        model = VGG16(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    if architecture=='inceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        model = inceptionV3(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    return model

#=====Define the mode====#
def set_mode(mode,model):
    if mode == 'fine_tunning':
        for layer in model.layers[:-4]:
            layer.trainable = False
    if mode == 'transfer_learning':
        for layer in model.layers:
            layer.trainable = False
    for layer in model.layers:
        print(layer, layer.trainable)
    return model

#=====Select an optimizer====#
def get_optimizer(optimizer,lr,decay,momentum,nesterov):
    if nesterov==1:
        bool_=True
    else:
        bool_=False

    if optimizer=='sgd': return SGD(lr=lr, decay=decay, momentum=momentum, nesterov=bool_)
    if optimizer=='adam': return Adam(lr=lr,decay=decay)

#get base model

print("#========================== Load Base model and print mode =====================#")
base_model = get_base_model(args['architecture'])
base_model = set_mode(args['mode'],base_model)

base_model.summary()


#create the model
print("============================Model Summary==================================")
model = Sequential()
 
# Add the vgg convolutional base model
model.add(base_model)
  
# Add new blocks/layers
model.add(Flatten())
for n in range(args['num_layers']):
    model.add(Dense(args['dense_layer_size'], activation='relu',
        kernel_regularizer=regularizers.l2(args['reg_rate'])))
    model.add(Dropout(args['dropout']))

model.add(Dense(args['num_classes'], activation='softmax'))
   
# Show a summary of the model. 
model.summary()


#==============Define the optimization procedure====================#
opt = get_optimizer(args['optimizer'],args['learning_rate'],args['decay'],args['momentum'],args['nesterov'])

#=========================Compile=====================================#
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy',Top5,Top3,Top2])

#=================Keras Callbacks================================!
tensorboard = TensorBoard(log_dir=job_root+'/tboard_logs/', histogram_freq=0,
                          write_graph=False, write_images=True)

filepath=job_root+'/checkpoints/model_'+sys.argv[1][20:-5]+"-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#csv_logger = CSVLogger('csv_logs/trainning.log')

class accHistory(keras.callbacks.Callback):
    def __init__(self,model_id,args):
        self.model_id = model_id
        self.args = args

    def on_train_begin(self,logs={}):
        self.accs = {'model_id':self.model_id,'epoch':[],'acc':[],'Top2':[],'Top3':[],'val_acc':[],'val_Top2':[],'val_Top3':[],'status':0.0,'config':self.args}

    def on_epoch_end(self,epoch,logs):
        self.accs['epoch']=epoch
        self.accs['acc'].append(logs.get('acc'))
        self.accs['Top2'].append(logs.get('Top2'))
        self.accs['Top3'].append(logs.get('Top3'))
        
        self.accs['val_acc'].append(logs.get('val_acc'))
        self.accs['val_Top2'].append(logs.get('val_Top2'))
        self.accs['val_Top3'].append(logs.get('val_Top3'))
        self.accs['status'] = round(100*epoch/self.args['epochs'],2)

        f = open(job_root+'/logs/acc_log-'+sys.argv[1][20:-5]+'.json',"w")
        json.dump(self.accs,f)
        f.close()


class stopExplodingLoss(keras.callbacks.Callback):
    def __init__(self,nclasses):
        self.max_loss = -1.5*np.log(1/nclasses)
    def on_epoch_end(self,epoch,logs={}):
        val_loss = logs.get('val_loss')
        if val_loss>self.max_loss:
            print('Val_loss: %f Max_loss: %f'%(val_loss,self.max_loss))
            self.model.stop_trainning=True
            sys.exit("Training interrupted, loss is too high")

stoploss = stopExplodingLoss(args['num_classes'])

history = accHistory(model_id,args)


earlystop = EarlyStopping(monitor='val_loss',min_delta = 1e-3,patience=5,mode='min',restore_best_weights=True)

#=======Compile the model===============================#
batch_size = args['batch_size']

model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n//batch_size,
                    epochs=args['epochs'],
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n//batch_size,
                    callbacks=[tensorboard,checkpoint,history,earlystop,stoploss])

score = model.evaluate_generator(test_generator,steps=2160//batch_size,max_queue_size=6)

model_json = model.to_json()

with open(job_root+'/graphs/model_'+sys.argv[1][20:-5]+'.json',"w") as json_file:
    json_file.write(model_json)

model.save_weights(job_root+'/weights/model_'+sys.argv[1][20:-5]+'.h5')

