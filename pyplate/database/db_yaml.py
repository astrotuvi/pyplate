import yaml
import re
from collections import OrderedDict

def fetch_ordered_tables(scmfile, rdbms, sig, new_name=None):
    """
    Fetch Tables into ordered Dict

    Parameters
    ----------
    scmfile: str
        yaml file with tables definitions
    rdbms: str
        values: pgsql, mysql
    sig: bool
        If True, use fully qualified table names (schema.table)
    new_name: str
        New name for the schema
    """

    yscm = __get_yamlschema(scmfile)

    if new_name is not None:
        yscm['name'] = new_name

    tablesdict = __get_tablesdict(yscm, rdbms, sig)
    triggersdict = __get_triggersdict(yscm, rdbms, sig)

    return tablesdict, triggersdict

def fetch_ordered_indexes(scmfile, rdbms, sig, new_name=None):
    """
    Fetch Index Creation Stmts into Ordered Dict

    Parameters
    ----------
    scmfile: str
        yaml file with tables definitions
    rdbms: str
        values: pgsql, mysql
    sig: bool
        If True, use fully qualified table names (schema.table)
    new_name: str
        New name for the schema
    """

    yscm = __get_yamlschema(scmfile)

    if new_name is not None:
        yscm['name'] = new_name

    odict = __get_indexdict(yscm, rdbms, sig)
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
    local_schema = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()

    lsc = local_schema[0]
    return lsc

def __subs_tbklnam(scm,tbs,tv, t_prefix):
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

    tvn = tv

    if(t_prefix):
        if(re.match(r'CREATE INDEX', tv)):
            sx = 'CREATE INDEX %s_' % tbs
            tvn = tv.replace('CREATE INDEX ', sx)
            sx = 'ON %s.' % scm
            tvn = tvn.replace('ON ', sx)

        if(re.match(r'ALTER TABLE', tv)):
            sx = 'ALTER TABLE %s.' % scm
            tvn = tv.replace('ALTER TABLE ', sx)

        if(re.match(r'CREATE TABLE', tv)):
            sx = 'TABLE %s.' % scm
            tvn = tv.replace('TABLE ', sx)
    return tvn

def __get_indexdict(lsc, rdbms, t_prefix):
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

            if tbl['indexes'] is None:
                break

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

def __get_tablesdict(lsc,rdbms,sig):
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

    if 'functions' in lsc:
        if rdbms in lsc['functions']:
            odic['functions'] = lsc['functions'][rdbms]

    if(lsx):
        for tbl in lsx:
            tbn = tbl['name']

            if sig:
                tbn = scnam + '.' + tbl['name']

            if tbl['columns'] is None:
                break

            xdict = OrderedDict()

            for clm in tbl['columns']:
                try:
                    kn = clm['name']
                    tv = clm[dtyp]
                    xdict[kn] = tv

                except KeyError as e:
                    print('   no clm data %s' % e)

            odic[tbn] = xdict

    return odic

def __get_triggersdict(lsc, rdbms, sig):
    """
    Convert YAML to TriggersDict (Helper function)

    Parameters
    ----------
    lsc: yaml structure
        yaml
    rdbms: str
        values: pgsql, mysql
    sig: bool

    """

    odic = OrderedDict()
    lsx = lsc['tables']
    scnam = lsc['name']

    odic['schema'] = scnam

    if(lsx):
        for tbl in lsx:
            tbn = tbl['name']

            if sig:
                tbn = scnam + '.' + tbl['name']

            if 'triggers' in tbl:
                if rdbms in tbl['triggers']:
                    trlist = []

                    for trdict in tbl['triggers'][rdbms]:
                        trstr = ('CREATE TRIGGER {} BEFORE UPDATE ON {} '
                                 'FOR EACH ROW EXECUTE PROCEDURE {};')
                        trlist.append(trstr.format(trdict['name'], tbn,
                                                   trdict['function']))

                    odic[tbn] = '\n'.join(trlist)

    return odic

def _get_columns_sql(tdict, table):
    """
    Print table column statements.

    """

    sql = None

    if table in tdict and isinstance(tdict[table], dict):
        sql_list = ['    {:20s} {}'.format(k, v) 
                    for k,v in tdict[table].items()]
        sql = ',\n'.join(sql_list)

    return sql

def _get_triggers_sql(tdict, table):
    """
    Print table triggers statements.

    """

    sql = ''

    if table in tdict:
        sql = '{}\n'.format(tdict[table])

    return sql

def creat_schema_mysql(tablesdict, pdict, engine='Aria'):
    """
    Print table creation SQL queries to standard output.

    """

    tdict = tablesdict.copy()

    if 'schema' in tdict:
        scm_name = tdict.pop('schema')
        sql_schema = ('CREATE DATABASE %s;' % scm_name)
        sql_drop_schema = ('DROP DATABASE %s;' % scm_name)
    else:
        sql_schema = ''
        sql_drop_schema = ''

    if 'functions' in tdict:
        del tdict['functions']

    sql_drop = '\n'.join(['DROP TABLE IF EXISTS {};'.format(k) 
                          for k in tdict.keys()])

    sql_list = ['CREATE TABLE {} (\n{}\n) ENGINE={} '
                'CHARACTER SET=utf8 COLLATE=utf8_unicode_ci;\n'
                .format(k, _get_columns_sql(tdict,k), engine)
                for k in tdict.keys()]
    sql = '\n'.join(sql_list)

    pdict['create_schema'] = sql_schema + '\n\n' + sql  
    pdict['drop_schema'] = sql_drop +'\n\n' + sql_drop_schema

def creat_schema_pgsql(tablesdict, triggersdict, pdict):
    """
    Print table creation SQL queries to standard output.

    """

    tdict = tablesdict.copy()

    if 'schema' in tdict:
        scm_name = tdict.pop('schema')
        sql_schema = 'CREATE schema {};'.format(scm_name)
        sql_drop_schema = 'DROP schema {} CASCADE;'.format(scm_name)
    else:
        sql_schema = ''
        sql_drop_schema = ''

    if 'functions' in tdict:
        funcs = tdict.pop('functions')
        sql_schema = '{}\n\n{}'.format(sql_schema, funcs)

    sql_drop = '\n'.join(['DROP TABLE IF EXISTS {};'.format(k) 
                          for k in tdict.keys()])

    sql_list = ['CREATE TABLE {} (\n{}\n);\n{}'
                .format(k, _get_columns_sql(tdict, k),
                        _get_triggers_sql(triggersdict, k))
                for k in tdict.keys()]

    sql = '\n'.join(sql_list)

    pdict['create_schema'] =  '{}\n\n{}\n'.format(sql_schema, sql)
    pdict['drop_schema'] = '{}\n\n{}\n'.format(sql_drop, sql_drop_schema)

def creat_schema_index(tdict, pdict):
    """
    Print index creation SQL statements to standard output.

    """

    sql = None
    sql_drop = ''
    sql_create = ''

    for table in tdict:
        sql_list = ['{};'.format(v) 
                    for k,v in tdict[table].items()]
        sql = '\n'.join(sql_list)

        sql_create = sql_create + '\n\n' + sql 

        for k,v in tdict[table].items():
            if(re.match('CREATE INDEX',v)):
                vxa =  v.split(' ')
                sql_ind = 'DROP INDEX IF EXISTS %s;\n' % vxa[2]  
                sql_drop = sql_drop + sql_ind        

        sql_drop = sql_drop + '\n'

    pdict['create_indexes'] =  sql_create 
    pdict['drop_indexes'] = sql_drop
