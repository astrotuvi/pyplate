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


def _precord(rec):
    print("\n")
    for r in rec:
        row = ','.join(['{}'.format(k)
                            for k in r])
        print(row)    
    print("\n")

## main ## 
pdb = PlateDB()
pdb.open_connection(host=PGHOST,port=PGPORT,user=PGUSER,password=PGPASSWD,database=PGDATABASE)
#print(dir(pdb))
print(pdb.database,'\n')

tbl='applause_dr4.archive'
sx = '*'

# try insert (only works once, may be 'not null' for timestamp is too restritctive?
cols="archive_id,archive_name,institute,timestamp_insert,timestamp_update"
vals= [None] * 5
vals[0]= "1000,'test_2dr4','aip_test',make_timestamp(2020,5,1,1,2,23.0),make_timestamp(2020,5,1,1,2,23.1)"
vals[1]= "1001,'test_1dr4','aip_test',make_timestamp(2020,5,2,1,2,23.0),make_timestamp(2020,5,1,2,2,23.1)"
vals[2]= "1002,'test_1dr4','aip_test',make_timestamp(2020,5,2,1,2,23.0),make_timestamp(2020,5,1,2,2,23.1)"
vals[3]= "1003,'test_dr4','aip_test',make_timestamp(2020,5,1,1,2,23.0),make_timestamp(2020,5,1,1,2,23.1)"
vals[4]= "1004,'test_dr4','aip_test',make_timestamp(2020,5,4,1,2,23.0),Null"
for  v in vals:
    qry2 = ("INSERT INTO %s (%s) VALUES(%s);" % (tbl,cols,v)) 
#    nrow =  pdb.execute_query(qry2) 

#
qry3 = ("SELECT %s FROM %s where archive_id > 1000;" % (sx, tbl)) 
nrow =  pdb.execute_query(qry3)
print(nrow)
rec = pdb.cursor.fetchall()
if(rec):
    _precord(rec) 


print(pdb.dbversion,'\n')
pdb.close_connection()

