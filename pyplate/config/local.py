import os
import sys

cfg_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(cfg_dir, '../..'))
sys.path.insert(0, root_dir)

ROOTDIR = root_dir

PGHOST = 
PGPORT = '5432'
PGUSER = 
PGDATABASE = 
PGPASSWD = 


#RDBMS = 'pgsql'
#FQTN = True

MYHOST = 
MYPORT = '3306'
MYUSER = 
MYDATABASE = 
MYPASSWD = 
RDBMS = 'mysql'
FQTN = True


SCHEMAFILE = cfg_dir +'/dr4_combined.yaml'
#SCHEMAFILE = cfg_dir +'/dr4_combined_tz.yaml'


BASE_URL = 'https://www.plate-archive.org'
TOKEN    = 

