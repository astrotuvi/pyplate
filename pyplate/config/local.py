import os
import sys

cfg_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(cfg_dir, '../..'))
sys.path.insert(0, root_dir)

ROOTDIR = root_dir

PGHOST = 
PGPORT = 
PGUSER = 
PGDATABASE = 
PGPASSWD = 


RDBMS = 'pgsql'
FQTN = True

#RDBMS = 'mysql'
#FQTN = False


SCHEMAFILE = cfg_dir +'/dr4_combined.yaml'


BASE_URL = 'https://www.plate-archive.org'
TOKEN    = 

