import os
import sys

cfg_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(cfg_dir, '../..'))
sys.path.insert(0, root_dir)

ROOTDIR = root_dir

RDBMS = 'pgsql'
PGHOST = None
PGPORT = 5432
PGUSER = None
PGDATABASE = None
PGPASSWD = None

#RDBMS = 'mysql'
MYHOST = None
MYPORT = 3306
MYUSER = None
MYDATABASE = None
MYPASSWD = None

FQTN = True

SCHEMAFILE = cfg_dir + '/applause_dr4.yaml'

BASE_URL = 'https://www.plate-archive.org'
TOKEN = None
