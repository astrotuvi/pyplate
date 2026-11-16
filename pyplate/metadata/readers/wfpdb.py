"""WFPDB maindata, quality, notes and observer files."""
import os

from ...config import WfpdbConfig
from ..timeutils import normalise_tz
from .values import PlateValues


def str_to_num(s: str):
    """Convert a string to int or float if possible; '' becomes None."""
    if s == '':
        return None

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def prefixed(tz: str | None, t: str) -> str:
    """A recorded time with the prefix ``UT`` or ``ST`` of its time zone.

    Times in a local time zone have no prefix: the zone is ``tz_orig``.
    """
    return '{} {}'.format(tz, t) if tz in ('UT', 'ST') else t


def read_file(path: str, encoding: str) -> dict[str, str]:
    """Entries of a WFPDB file by WFPDB ID (``POT015_000317``)."""
    with open(path, encoding=encoding) as f:
        return {line[:14].rstrip().replace(' ', '_'): line.rstrip('\n')
                for line in f}


def read_files(wfpdb_dir: str, filenames: dict[str, str | None],
               encoding: str) -> dict[str, dict[str, str]]:
    """Read the WFPDB files named by kind (``maindata``, ``quality``,
    ``notes``, ``observer``); missing names give empty dictionaries."""
    return {kind: (read_file(os.path.join(wfpdb_dir, fn), encoding)
                   if fn else {})
            for kind, fn in filenames.items()}


def parse_maindata(entry: str, v: PlateValues, cfg: WfpdbConfig) -> None:
    """Values from a maindata entry."""
    plate_num = entry[7:14]
    v['plate_num'] = '{:d}{}'.format(int(plate_num[:6]),
                                     plate_num[6].strip())

    radec = entry[14:27]

    if radec[0:2].strip():
        ra = '{:0>2}:{:0>2}'.format(radec[0:2].strip(), radec[2:4].strip())

        if radec[4:6].strip():
            ra += ':{:0>2}'.format(radec[4:6].strip())

        v['ra_orig'] = [ra]

    if radec[7:8].strip():
        dec = '{}:{:0>2}'.format(radec[6:9], radec[9:11].strip())

        if radec[11:13].strip():
            dec += ':{:0>2}'.format(radec[11:13].strip())

        v['dec_orig'] = [dec]

    v['flag_coord'] = entry[27]
    datetime = entry[28:42]

    if datetime[0:4].strip():
        v['date_orig_start'] = ['{}-{}-{}'.format(datetime[0:4],
                                                  datetime[4:6],
                                                  datetime[6:8])]

    tz = normalise_tz(cfg.maindata_timezone)

    # Default time zone of the file, also without a recorded time
    if tz:
        v['tz_orig'] = tz

    if datetime[8:10].strip():
        t = datetime[8:10]

        if datetime[10:12].strip():
            t += ':{}'.format(datetime[10:12])

            if datetime[12:14].strip():
                t += ':{}'.format(datetime[12:14])

        if tz == 'ST':
            v['st_start_orig'] = [t]
        elif tz == 'UT':
            v['ut_start_orig'] = [t]

        v['time_orig_start'] = [prefixed(tz, t)]

    v['flag_time'] = entry[42].strip()
    v['object_name'] = [entry[43:63].strip()]
    object_type_code = entry[63:65].strip()

    if object_type_code:
        v['object_type_code'] = [object_type_code]

    try:
        v['method_code'] = int(entry[65:67])
    except ValueError:
        v['method_code'] = None

    try:
        v['numexp'] = int(entry[67:69])
    except ValueError:
        v['numexp'] = 1

    multiply = 60. if cfg.exptime_minutes else 1.

    try:
        v['exptime'] = [float(entry[69:75].strip()) * multiply]
    except ValueError:
        v['exptime'] = []

    v['emulsion'] = entry[75:86].strip()
    v['filter'] = entry[86:93].strip()

    for key, pos in [('plate_size1', slice(95, 97)),
                     ('plate_size2', slice(97, 99))]:
        try:
            v[key] = int(entry[pos])
        except ValueError:
            v[key] = None

    if v['plate_size1'] and v['plate_size2']:
        sizes = [v['plate_size1'], v['plate_size2']]
        v['plate_format'] = '{}x{}'.format(min(sizes), max(sizes))


def parse_quality(entry: str, v: PlateValues) -> None:
    """Values from a quality entry."""
    notes = []

    for keyval in entry[15:].split(';'):
        if ':' not in keyval:
            if keyval.strip():
                v['plate_quality'] = keyval.strip()

            continue

        key, val = [s.strip() for s in keyval.split(':', 1)]

        if key.lower() == 'temperature':
            try:
                v['air_temperature'] = float(val)
            except ValueError:
                v['air_temperature'] = None
        elif key.lower() == 'calmness':
            v['sky_calmness'] = val
        elif key.lower() == 'sharpness':
            v['sky_sharpness'] = val
        elif key.lower() == 'transparency':
            v['sky_transparency'] = val
        elif key.lower() == 'sky conditions':
            v['sky_conditions'] = val
        else:
            notes.append('{}: {}'.format(key, val))

    if notes:
        v['notes'] = '; '.join(notes)


def parse_observer(entry: str, v: PlateValues) -> None:
    """Values from an observer entry."""
    if entry[15:].strip():
        v['observer'] = entry[15:].strip()


def _split(val: str) -> list[str]:
    return [x.strip() for x in val.split(',')]


def parse_notes(entry: str, v: PlateValues, cfg: WfpdbConfig) -> None:
    """Values from a notes entry.

    Times in the notes are in ``[WFPDB] notes_timezone``, which then
    becomes the time zone of the plate.
    """
    obsnotes = []
    notes = []

    for note in entry[15:].split(';'):
        if not note.strip():
            continue

        # Default key: notes
        if ':' not in note:
            note = 'notes: {}'.format(note)

        if cfg.notes_timezone:
            v['tz_orig'] = normalise_tz(cfg.notes_timezone)

        key, val = (s.strip() for s in note.split(':', 1))
        k = key.lower()
        tz = v.get('tz_orig')

        if k == 'exposure time':
            v['exptime'] = [str_to_num(x) for x in _split(val)]
        elif k == 'exposure start':
            v['time_orig_start'] = [prefixed(tz, x) for x in _split(val)]

            if tz == 'ST':
                v['st_start_orig'] = _split(val)
            elif tz == 'UT':
                v['ut_start_orig'] = _split(val)
        elif k == 'exposure end':
            v['time_orig_end'] = [prefixed(tz, x) for x in _split(val)]

            if tz == 'ST':
                v['st_end_orig'] = _split(val)
            elif tz == 'UT':
                v['ut_end_orig'] = _split(val)
        elif k == 'ut-start':
            v['ut_start_orig'] = _split(val)
        elif k == 'ut-end':
            v['ut_end_orig'] = _split(val)
        elif k == 'st-start':
            v['st_start_orig'] = _split(val)
        elif k == 'st-end':
            v['st_end_orig'] = _split(val)
        elif k == 'jd-avg':
            # Recorded mid-point of the first exposure
            v['jd_mid_orig'] = [str_to_num(val)]
        elif k in ('object', 'object name'):
            v['object_name'] = _split(val)
        elif k == 'object type':
            v['object_type_code'] = _split(val)
        elif k == 'ra':
            v['ra_orig'] = _split(val)
        elif k == 'dec':
            v['dec_orig'] = _split(val)
        elif k == 'focus':
            v['focus'] = [str_to_num(x) for x in _split(val)]
        elif k == 'emulsion':
            v['emulsion'] = val
        elif k == 'instrument':
            v['instrument'] = val
        elif k == 'prism name':
            v['prism'] = val
        elif k == 'prism angle':
            v['prism_angle'] = val
        elif k == 'sky conditions':
            v['sky_conditions'] = val
        elif k == 'original plate number':
            v['plate_num_orig'] = val
        elif k == 'bibcode':
            v['bibcode'] = val
        elif k[:13] == 'observer note':
            obsnotes.append(val)
        elif k == 'notes':
            notes.append(val)
        else:
            notes.append('{}: {}'.format(key, val))

    v.combine_times()

    if obsnotes:
        v['observer_notes'] = '; '.join(obsnotes)

    if notes:
        v['notes'] = '; '.join(notes)
