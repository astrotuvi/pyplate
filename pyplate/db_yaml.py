import yaml
import re
from collections import OrderedDict

def fetch_ordered_tables(scmfile,rdbms,sig=False):
    """
    Fetch Tables into ordered Dict

    Parameters
    ----------
    scmfile: str
        yaml file with tables definitions
    rdbms: str
        values: pgsql, mysql
    sig: boolean 
    """

    yscm = __get_yamlschema(scmfile)
    odict = __get_tablesdict(yscm,rdbms,sig =False)
    return odict

def fetch_ordered_indexes(scmfile,rdbms, sig):
    """
    Fetch Index Creation Stmts into Ordered Dict

    Parameters
    ----------
    scmfile: str
        yaml file with tables definitions
    rdbms: str
        values: pgsql, mysql
    sig: for using fully qualified table names (schema.table)
        boolean
        
    """

    yscm = __get_yamlschema(scmfile)
    odict = __get_indexdict(yscm,rdbms,sig)
    return odict

def __get_yamlschema(filename):
    """
    Read yaml file (Helper function)

    Parameters
    ----------
    scmfile: str
        yaml file with tables definitions
    """

    f = open(filename)
    local_schema = yaml.load(f.read())
    f.close()

    lsc = local_schema[0]
    return lsc

def __subs_tbklnam(scm,tbs,tv, t_prefix=False):
    """
    Insert Schema name (Helper function)

    Parameters
    ----------
    scm: str
        schema name
    tbs: str
        table name
    tv: str
        Index Statement
    """

    tvn= ''
    tvar = re.split(r' ',tv)
    for wx in tvar:
        if(tbs == wx):
            tvn = tvn + ' ' + scm + '.' + wx
        else:
            tvn = tvn + ' ' + wx

    if(t_prefix):
        sx = 'INDEX %s_' % tbs
        tvn = tv.replace('INDEX ', sx)

    return tvn

def __get_indexdict(lsc, rdbms, t_prefix=False):
    """
    Convert YAML to Index Dict (Helper function)

    Parameters
    ----------
    lsc: yaml structure
        yaml
    rdbms: str
        values: pgsql, mysql
    t_prefix: use tablename in indexname
        values: true, false
    """

    dindx = rdbms + '_pk'

    odic= OrderedDict()
    lsx = lsc['tables']
    scnam = lsc['name']
    if(lsx):
        for tbl in lsx:
            tbs = tbl['name']
            tbn = scnam + '.' + tbs
            if(tbl['indexes'] is None): break
            xdict =OrderedDict()
            for clm in tbl['indexes']:
                try:
                    kn = clm['name']
                    if(re.search('index',kn)):
                        tv = __subs_tbklnam(scnam,tbs,clm['val'],t_prefix)
                        xdict[kn] = tv
                    elif(kn == dindx):
                        tv = __subs_tbklnam(scnam,tbs,clm['val'],t_prefix)
                        xdict[kn] = tv
#                        print('%s => %s ' % (kn,tv))
                    else:
                        pass


                except KeyError as e:
                    print('   no clm data %s' % e)

            odic[tbn] = xdict
    return(odic)

def __get_tablesdict(lsc,rdbms,sig = False):
    """
    Convert YAML to TablesDict (Helper function)

    Parameters
    ----------
    lsc: yaml structure
        yaml
    rdbms: str
        values: pgsql, mysql
    sig: boolean    
        
    """

    dtyp = rdbms + '_datatype'
    odic= OrderedDict()
    lsx = lsc['tables']
    scnam = lsc['name']

    odic['schema'] = scnam
    if(lsx):
        for tbl in lsx:
            tbn = tbl['name']
            if(sig):
                tbn = scnam + '.' + tbl['name']

            if(tbl['columns'] is None): break
            xdict =OrderedDict()
            for clm in tbl['columns']:
                try:
                    kn = clm['name']
                    tv = clm[dtyp]
                    xdict[kn] = tv

                except KeyError as e:
                    print('   no clm data %s' % e)

            odic[tbn] = xdict
    return(odic)


def _get_columns_sql(tdict,table):
    """
    Print table column statements.

    """

    sql = None

    if table in tdict:
        sql_list = ['    {:15s} {}'.format(k, v) 
                    for k,v in tdict[table].items()]
        sql = ',\n'.join(sql_list)

    return sql

def print_schema_mysql(tdict, use_drop=False, engine='Aria'):
    """
    Print table creation SQL queries to standard output.

    """

    sql_schema = ('CREATE schema %s' % tdict.pop('schema'))

    sql_drop = '\n'.join(['DROP TABLE IF EXISTS {};'.format(k) 
                          for k in tdict.keys()])

    sql_list = ['CREATE TABLE {} (\n{}\n) ENGINE={} '
                'CHARACTER SET=utf8 COLLATE=utf8_unicode_ci;\n'
                .format(k, _get_columns_sql(tdict,k), engine)
                for k in tdict.keys()]
    sql = '\n'.join(sql_list)

    if use_drop:
        sql = sql_drop + '\n\n' + sql

    print(sql)



def print_schema_pgsql(tdict, use_drop=False):
    """
    Print table creation SQL queries to standard output.

    """
    scm_name = tdict.pop('schema')
    sql_schema = ('CREATE schema %s' % scm_name)

    sql_drop_schema = ('DROP schema %s CASCADE;' % scm_name)
    sql_drop = '\n'.join(['DROP TABLE IF EXISTS {};'.format(k) 
                          for k in tdict.keys()])

    sql_list = ['CREATE TABLE {} (\n{}\n);\n'
                .format(k, _get_columns_sql(tdict,k))
                for k in tdict.keys()]
    sql = '\n'.join(sql_list)

    sql =  sql_schema + ';\n\n' + sql  

    if use_drop:
        sql =  sql_drop + '\n\n' + sql_drop_schema + '\n\n' + sql


    print(sql)

def print_schema_index(tdict, use_drop=False):
    """
    Print index creation SQL statements to standard output.

    """

    sql = None
    sql_drop = ''
        

    for table in tdict:
        sql_list = ['    {};'.format(v) 
                    for k,v in tdict[table].items()]
        sql = '\n'.join(sql_list)
            
        sql_drop = ''
        for k,v in tdict[table].items():
            if(re.match('CREATE INDEX',v)):
                vxa =  v.split(' ')
                sql_ind = 'DROP INDEX IF EXISTS %s;\n' % vxa[2]  
                sql_drop = sql_drop + sql_ind        

        if use_drop:
            sql = sql_drop + '\n' + sql

        print(sql + '\n')
