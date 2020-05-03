import os
import sys
import re
#import MySQLdb
from collections import OrderedDict

tests_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(tests_dir, '..'))
sys.path.insert(0, root_dir)

from pyplate.config.local import SCHEMAFILE, RDBMS, MYHOST, MYPORT, MYUSER, MYDATABASE, MYPASSWD
from pyplate.db_mysql import PlateDB 


def _precord(rec):
    print("\n")
    for r in rec:
        row = ','.join(['{}'.format(k)
                            for k in r])
        print(row)    
    print("\n")

## main ## 
pdb = PlateDB()
pdb.open_connection(host=MYHOST,port=MYPORT,user=MYUSER,password=MYPASSWD,database=MYDATABASE)
print(pdb.database,'\n')

tbl='applause_dr4.archive'
sx = '*'

# try insert (only works once, may be 'not null' for timestamp is too restritctive?
## throws an error if run twice
cols="archive_id,archive_name,institute,timestamp_insert,timestamp_update"
vals= [None] * 5
vals[0]= "1000,'test_2dr4','aip_test','2020-05-01 01:22:23.0','2020-05-01 01:22:20.0'"
vals[1]= "1001,'test_1dr4','aip_test','2020-05-02 01:22:23.0','2020-05-01 01:02:21.0'"
vals[2]= "1002,'test_1dr4','aip_test','2020-05-04 01:22:23.0','2020-05-01 01:02:22.0'"
vals[3]= "1003,'test_dr4','aip_test','2020-05-01 01:24:23.0','2020-05-01 01:02:23.0'"
vals[4]= "1004,'test_dr4','aip_test','2020-05-01 01:02:23.0',Null"
for  v in vals:
    qry2 = ("INSERT INTO %s (%s) VALUES(%s);" % (tbl,cols,v)) 
    nrow =  pdb.execute_query(qry2) 

# try select
qry3 = ("SELECT %s FROM %s where archive_id > 1000;" % (sx, tbl)) 
nrow =  pdb.execute_query(qry3)
print(nrow)
rec = pdb.cursor.fetchall()
if(rec):
    _precord(rec) 


print(pdb.dbversion)
pdb.close_connection()

