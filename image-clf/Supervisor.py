import os
import pickle
#from src.RandomSearch import RandomModels as rm
from search_hyperparams import RandomSearch as rm
import sys


hm = os.getcwd()

try:
    root = sys.argv[1]
except:
    root = input("Enter project's root path:")

root = os.path.join(hm,root)

try:
    n_models = int(sys.argv[2])
except:
    n_models =int(input("Enter a number of jobs:"))

try:
    params_path = sys.argv[3]
except:
    params_path = input("Enter path to params random_search_params.pickle")

params = os.path.join(hm,params_path)

worker =os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepnets_classification.py')

folders = os.listdir()
if root not in folders: os.system('mkdir '+root)
if root+'/tmp' not in folders: os.system('mkdir '+root+'/tmp')
if root+'/jobs'not in folders: os.system('mkdir '+root+'/jobs')
if root+'/jobs/config'not in folders: os.system('mkdir '+root+'jobs/config')

#================================Generate Random Models================================#

f = open(params_path,'rb')
params = pickle.load(f)
f.close()

rmodels = rm.RandomModels(**params)
rmodels.build_configs(n_models,path=root+'/jobs/config')

pipeline_text = open('pipeline.sh','w')
tboard_text = open('tboard_launch.sh','w')

pipeline_text.write('#!/bin/bash \n\n')
tboard_text.write('#!/bin/bash \n\n')

pipeline = ''
tensorboard = 'tensorboard --logdir '
for i in range(n_models):
    os.system('mkdir '+root+'/jobs/job'+str(i).zfill(4))
    os.system('mkdir '+root+'/jobs/job'+str(i).zfill(4)+'/tboard_logs')
    pipeline+='python '+worker+' --params '+root+'/jobs/config/config'+str(i).zfill(4)+'.json --job_root '+root+'/jobs/job'+str(i).zfill(4)+'| tee -a '+root+'/jobs/job'+str(i).zfill(4)+'/pipeline_log.txt -a pipeline_log.txt;\n'
    tensorboard+='job'+str(i).zfill(4)+':'+root+'/jobs/job'+str(i).zfill(4)+'/tboard_logs,' 
tboard_text.write(tensorboard[:-1]+"\n\n")
pipeline_text.write(pipeline)

tboard_text.close()
pipeline_text.close()

os.system('sudo chmod +x pipeline.sh')
os.system('chmod +x tboard_launch.sh')
os.system('tmux new -s hyper_tunning "sh pipeline.sh"')

