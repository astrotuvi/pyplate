import os
import sys
from collections import OrderedDict

tests_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(tests_dir, '..'))
sys.path.insert(0, root_dir)


from pyplate.db_yaml import fetch_ordered_tables, fetch_ordered_indexes, print_schema_mysql, print_schema_pgsql, print_schema_index
from pyplate.config.local import SCHEMAFILE, RDBMS, FQTN
from pyplate.database import _get_schema_old 

"""
very basic tests, enable with changing conditional
"""
yamlf = SCHEMAFILE
fqtn = FQTN 
if(1!=2):
# postgres
    rdbms= RDBMS
    tbldict = OrderedDict()
    tbldict = fetch_ordered_tables(yamlf,rdbms,FQTN)
    print_schema_pgsql(tbldict,True)
    inxdict = OrderedDict()
    inxdict = fetch_ordered_indexes(yamlf,rdbms,FQTN)
    print_schema_index(inxdict,True)

if(1==2):
# mysql
    rdbms='mysql'
    tbldict = OrderedDict()
    tbldict = fetch_ordered_tables(yamlf,rdbms,True)
    print_schema_mysql(tbldict,True)
    inxdict = OrderedDict()
    inxdict = fetch_ordered_indexes(yamlf,rdbms, True)
    print_schema_index(inxdict,False)



"""
Compare _schema from database.py with yaml file

"""
if(1==2):

    rdbms='pgsql'
    tbldict = OrderedDict()
    dbsdict = OrderedDict()


#    dbsdict = fetch_ordered_tables(yamlf,rdbms,False)
#    tbldict =  _get_schema_old()
#    sql_schema = dbsdict.pop('schema')
#    print('Compare schema with yml: %s' % sql_schema)

    tbldict = fetch_ordered_tables(yamlf,rdbms,False)
    dbsdict =  _get_schema_old()
    sql_schema = tbldict.pop('schema')
    print('Compare schema with dbs: %s' % sql_schema)


    for ky in tbldict.keys():
        try:
            tbly = tbldict[ky]
            tbld = dbsdict[ky]
            print('\nTable: %s' % ky)
            try:
                for cl in tbly.keys():
                    xx = tbld[cl]
                    print('%s\t=>\t%s' % (cl,cl))

            except KeyError as e:
                print('   no clm data %s' % e)

        except KeyError as e:
                print('   no table %s in ' % e)

