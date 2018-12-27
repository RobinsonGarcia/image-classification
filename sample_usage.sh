!#/bin/bash

python image-clf/build_random_search_params.py

python image-clf/Supervisor.py Shiriu 99 random_search_params.pickle 
