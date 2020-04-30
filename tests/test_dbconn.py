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


def pgconnect(PGU,PGH,PGPA,PGDB):

    try:
        connection = psycopg2.connect(user = PGU,
                                      password = PGPA,
                                      host = PGH,
                                      port = "5432",
                                      database = PGDB)
          

        cursor = connection.cursor()
        # Print PostgreSQL Connection properties
            # Print PostgreSQL version
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("You are connected to - ", record,"\n")
#        cursor.close()

        return connection

    except (Exception, psycopg2.Error) as e:
        print(e)

def create_simplequery(action,table,col_list,val_list, where):

    col_str = ','.join(col_list)
    if(action == 'S'):
        frm = "FROM %s %s" % (table,where)
        sql = ('SELECT {} '.format(col_str))
        sql = sql + frm
        return sql
    elif(action == 'I'): 
        stm = ('INSERT INTO %s ' % table )
        val_str = ','.join(['%s'] * len(col_list))
        sql = ('{} ({}) VALUES({})'.format(col_str,val_str))
        sql = stm + sql
        return sql
    elif(action == 'U'): 
        stm = ('UPDATE %s SET {} WHERE %s' % (table,where) )
        val_str = ','.join(['%s'] * len(col_list))
#        sql = ('{} ({}) VALUES({})'.format(col_str,val_str))
        sql = stm.format(col_str) 
        return sql


def pgfin(connection):

    if(connection):
        connection.close()
        print ("PGconn  closed")


def pgselect(pcursor,qry):
    try:
        pcursor.execute(qry)
        record = pcursor.fetchall()
        return record 
    except (Exception, psycopg2.Error) as e:
        print("SQL problem: %s" % qry)
        return []
            

def pginsert(pcursor,qry):
    try:
        pcursor.execute(qry)
        count = pcursor.rowcount
        return count
    except (Exception, psycopg2.Error) as e:
        return e
        
def _precord(rec):
    for r in rec:
        print("\n")
        for v in r:
            print("%s," % v)

        print("\n")

        
## main ## 



pdb = PlateDB()
pdb.open_connection(host=PGHOST,port=PGPORT,user=PGUSER,password=PGPASSWD,database=PGDATABASE)
print(pdb)

tbl='applause_dr4.archive'
sx = '*'
qry = ("SELECT %s FROM %s" % (sx, tbl)) 
nrow =  pdb.execute_query(qry) 

exit()
