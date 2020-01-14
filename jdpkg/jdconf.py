import json
import os

def get_conf(conf_path):
    ret = None
    with open(conf_path) as fp:
        ret = json.load(fp)
    return ret


glconf={}
glconf['dir_data']='/home/tianfu/data'
if os.name=='nt':
    dir_in_server=r'D:\Users\Hetianfu\data'
    dir_local=r'c:\data'
    if os.path.exists(dir_in_server):
        glconf['dir_data']=dir_in_server
    else:
        glconf['dir_data']=dir_local
