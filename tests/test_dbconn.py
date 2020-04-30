import os
import sys
import re
import psycopg2
from collections import OrderedDict

tests_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(tests_dir, '..'))
sys.path.insert(0, root_dir)

from pyplate.config.local import SCHEMAFILE, RDBMS, PGHOST, PGPORT, PGUSER, PGDATABASE, PGPASSWD
from pyplate.db_pgsql import _exprt_scm
from pyplate.db_pgsql import PlateDB 


## main ## 
pdb = PlateDB()
pdb.open_connection(host=PGHOST,port=PGPORT,user=PGUSER,password=PGPASSWD,database=PGDATABASE)
#print(dir(pdb))
print(pdb.dbversion)

tbl='applause_dr4.archive'
sx = '*'
qry = ("SELECT %s FROM %s;" % (sx, tbl)) 
nrow =  pdb.execute_query(qry) 
print(nrow)



# now try insert (only works once, may be not null for timestam,p is too restritctive?
cols="archive_id,archive_name,institute,timestamp_insert,timestamp_update"
#vals= "1003,'test_dr4','aip_test',make_timestamp(2020,5,1,1,2,23.0),make_timestamp(2020,5,1,1,2,23.1)"
vals= "1002,'test_2dr4','aip_test',make_timestamp(2020,5,1,1,2,23.0),make_timestamp(2020,5,1,1,2,23.1)"
qry2 = ("INSERT INTO %s (%s) VALUES(%s);" % (tbl,cols,vals)) 
nrow =  pdb.execute_query(qry2) 

print(nrow)

qry3 = ("SELECT %s FROM %s where archive_id = 1003;" % (sx, tbl)) 
nrow =  pdb.execute_query(qry3)
print(nrow)

pdb.close_connection()

