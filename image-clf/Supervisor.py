import os
from src.RandomSearch import RandomModels as rm
import sys
worker = 'deepnets_classification.py'

try:
    root = sys.argv[1]
except:
    root = input("Enter project's root path:")

try:
    n_models = sys.argv[2]
except:
    n_models =int(input("Enter a number of jobs:"))

folders = os.listdir()
if root not in folders: os.system('mkdir '+root)
if root+'/tmp' not in folders: os.system('mkdir '+root+'/tmp')
if root+'/jobs'not in folders: os.system('mkdir '+root+'/jobs')
if root+'/jobs/config'not in folders: os.system('mkdir '+root+'jobs/config')

#================================Generate Random Models================================#
rmodels = rm.RandomModels()
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
    pipeline+='python '+worker+' '+root+'/jobs/config/config'+str(i).zfill(4)+'.json '+root+'/jobs/job'+str(i).zfill(4)+'| tee -a '+root+'/jobs/job'+str(i).zfill(4)+'/pipeline_log.txt -a pipeline_log.txt;\n'
    tensorboard+='job'+str(i).zfill(4)+':'+root+'/jobs/job'+str(i).zfill(4)+'/tboard_logs,' 
tboard_text.write(tensorboard[:-1]+"\n\n")
pipeline_text.write(pipeline)

tboard_text.close()
pipeline_text.close()

os.system('sudo chmod +x pipeline.sh')
os.system('chmod +x tboard_launch.sh')
os.system('tmux new -s hyper_tunning "sh pipeline.sh"')

