import os
import sys

cfg_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(cfg_dir, '../..'))
sys.path.insert(0, root_dir)

ROOTDIR = root_dir

#RDBMS = 'pgsql'
PGHOST = 
PGPORT = 5432
PGUSER = 
PGDATABASE = 
PGPASSWD = 


#RDBMS = 'mysql'
MYHOST = 
MYPORT = 3306
MYUSER = 
MYDATABASE = 
MYPASSWD = 


FQTN = True


#SCHEMAFILE = cfg_dir +'/dr4_combined.yaml'
SCHEMAFILE = cfg_dir +'/dr4_combined_tz.yaml'


BASE_URL = 'https://www.plate-archive.org'
TOKEN    = 

