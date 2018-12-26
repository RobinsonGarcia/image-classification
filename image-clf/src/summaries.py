import os
import json
import pprint
import pandas as pd
import sys

def format_dict(job0):
    tmp = job0['config']
    tmp['acc'] = job0['acc'][-1]
    tmp['val_acc'] = job0['val_acc'][-1]
    tmp['Top2'] = job0['val_Top2'][-1]
    tmp['val_Top2'] = job0['val_Top2'][-1]
    tmp['Top3'] = job0['Top3'][-1]
    tmp['val_Top3'] = job0['val_Top3'][-1]
    return tmp

def get_summary(root,print_summary=False,return_dframe=False):
    job_list = []
    started=0
    not_started=0
    accs={}
    dataframe={}
    for job in os.listdir(os.path.join(root,'jobs')):
        if 'job' in job:
            try:
                filename = os.listdir(os.path.join(root,'jobs',job,'logs'))[0]
                f = open(os.path.join(root,'jobs',job,'logs',filename),"r")
                job0 = json.loads(f.read())
                f.close()
                dataframe[job]=format_dict(job0)

                started+=1
            except:
                not_started+=1
    print('Number of jobs started:%d'%(started))
    print('Number of jobs pending:%d'%(not_started))

    dataframe = pd.DataFrame.from_dict(dataframe)
    dataframe.to_csv('summary.csv')
    if print_summary:
        try: print(dataframe)
        except: print("couldn't load a dataframe")
    if return_dframe: return dataframe.T
    else: pass

if __name__=="__main__":
    try:
        get_summary(sys.argv[1],print_summary=True)
    except:
        root = input("Enter the project's root folder:")
        get_summary(root,print_summary=True)
