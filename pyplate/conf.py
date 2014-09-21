import ConfigParser

def read_conf(conf_file):
    """
    Read configuration file.

    Parameters
    ----------
    conf_file : str
        Configuration file path.

    Returns
    -------
    conf : a ConfigParser object
    
    """

    conf = ConfigParser.ConfigParser()
    conf.read(conf_file)

    if (conf.has_section('Files') and 
        conf.has_option('Files', 'fits_acknowledgements')):
        fn_ack = conf.get('Files', 'fits_acknowledgements')

        with open(fn_ack, 'rb') as f:
            ack = '\n'.join(line.strip() for line in f.readlines())

        if not conf.has_section('Keyword values'):
            conf.add_section('Keyword values')

        conf.set('Keyword values', 'fits_acknowledgements', ack.strip())

    return conf

