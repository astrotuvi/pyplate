import os
import sys
from collections import OrderedDict

tests_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(tests_dir, '..'))
sys.path.insert(0, root_dir)


from pyplate.db_yaml import fetch_ordered_tables, fetch_ordered_indexes, creat_schema_mysql, creat_schema_pgsql, creat_schema_index
from pyplate.config.local import SCHEMAFILE, RDBMS, FQTN, PGUSER
from pyplate.database import _get_schema_old 

"""
very basic tests, enable with changing conditional
"""
yamlf = SCHEMAFILE
fqtn = FQTN 


""" 
    this uses the dr4_combined.yaml 
    to create a complete set of sql creation/deletion statements 
"""
if(1!=2):

    rdbms= RDBMS
    tbldict = OrderedDict()
    creat_dict = OrderedDict()
    tbldict = fetch_ordered_tables(yamlf,rdbms,fqtn)
    scm = tbldict['schema']
    if(RDBMS == 'mysql'):
        creat_schema_mysql(tbldict,creat_dict)
    else:
        creat_schema_pgsql(tbldict,creat_dict)

    inxdict = OrderedDict()
    inxdict = fetch_ordered_indexes(yamlf,rdbms,fqtn)
    creat_schema_index(inxdict,creat_dict)
    filname1 = '%s_creat_schema_%s.sql' % (scm,rdbms)
    fo = open(filname1,'w')
    fo.write(creat_dict['create_schema'])
    fo.write(creat_dict['create_indexes'])
    fo.close()

    filname2 = '%s_drop_schema_%s.sql' % (scm,rdbms)
    fo = open(filname2,'w')
    fo.write(creat_dict['drop_schema'])
    fo.write(creat_dict['drop_indexes'])
    fo.close()

    pstm ='If you use the sql files (%s,%s) for creating, dont forget\nto issue after creation:\n'  % (filname1,filname2) 
    pstm = pstm + '  GRANT ALL ON SCHEMA %s TO %s;\n ' % (scm,PGUSER) 
    pstm = pstm + '  GRANT ALL ON ALL TABLES IN SCHEMA %s TO %s; \n' % (scm,PGUSER)
    print(pstm)


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

