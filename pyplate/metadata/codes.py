"""Object-type and method code tables of the WFPDB."""

OBJECT_TYPES = {
    'A1': 'planet',
    'A2': 'Moon',
    'A3': 'Sun',
    'A4': 'asteroid',
    'A5': 'comet',
    'A6': 'meteor',
    'A7': 'artificial satellite',
    'S1': 'star',
    'S2': 'double star or multiple star',
    'S3': 'variable star',
    'S4': 'star cluster',
    'S5': 'HII region',
    'S6': 'nebula',
    'S7': 'planetary nebula',
    'S8': 'supernova + SN remnants',
    'S9': 'fundamental star',
    'SR': 'reference star around a radio source',
    'SA': 'stellar association',
    'SD': 'dark nebula',
    'SH': 'Herbig-Haro object',
    'SM': 'molecular cloud',
    'SP': 'pulsar',
    'G1': 'galaxy',
    'G2': 'QSO',
    'G3': 'group of galaxies',
    'G4': 'cluster of galaxies',
    'G5': 'supercluster',
    'G6': 'void',
    'F': 'field',
    'G7': 'radio galaxy',
    'GR': 'gamma-ray source',
    'XR': 'X-ray source',
    'RS': 'radio source',
    'IR': 'infrared source',
    'U': 'object of unknown nature'}

METHOD_CODES = {
    1: 'direct photograph',
    2: 'direct photograph, multi-exposure',
    3: 'stellar tracks',
    4: 'objective prism',
    5: 'objective prism, multi-exposure',
    6: 'Metcalf\'s method',
    7: 'proper motions',
    8: 'no guiding',
    9: 'out of focus',
    10: 'test plate',
    11: 'Hartmann test',
    12: 'with mask',
    14: 'sub-beam (Pickering) prism',
    13: 'focusing',
    15: 'raster scan/trail',
    24: 'objective grating',
    25: 'objective grating, multi-exposure',
    31: 'objective prism together with a grating'}


def object_type(code: str | None) -> str | None:
    """Object-type text for a code; an unknown code is returned as is."""
    if not code:
        return None

    return OBJECT_TYPES.get(code, code)


def method(code: int | None) -> str | None:
    """Method text for a method code, or None for an unknown code."""
    return METHOD_CODES.get(code) if code else None
