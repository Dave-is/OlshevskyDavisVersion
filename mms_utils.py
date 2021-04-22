""" Various utils needed to read and classify MMS observations.

(c) Vyacheslav Olshevsky (2019)

"""
import numpy as np
import matplotlib.pyplot as plt
import cdflib, os, datetime, csv, pickle
from glob import glob, iglob
import re

# 1. First version of possible labels for the dayside ion measurements, i.e., different regions of space based on ions.
# The magnetopause was excluded later because it is a mixed region
#old_classes = {-1: 'Unknown', 0: 'Solar Wind', 1: 'Ion Foreshock', 2: 'Bow Shock', 3: 'Msheath q-par', 4: 'Msheath q-perp', 5: 'Magnetosphere', 6: 'Magnetopause'}
#class_colors = {-1: 'k', 0: 'darkgrey', 1: 'lightcoral', 2: 'maroon', 3: 'gold', 4: 'lawngreen', 5: 'lightseagreen', 6: 'navy'}

# 2. Second version of possible labels. Two magnetosheaths are hardly distinguishable from dis-dist data, therefore they should be merged.
#regions = {-1: 'Unknown', 0: 'Solar Wind', 1: 'Foreshock', 2: 'Msheath q-par', 3: 'Msheath q-perp', 4: 'Magnetosphere'}
#region_colors = {-1: 'k', 0: 'darkgrey',   1: 'firebrick',     2: 'royalblue',     3: 'slateblue',      4: 'seagreen'}

# 3. Third version of regions, with merged sheaths.
regions = {-1: 'Unknown', 0: 'Solar Wind', 1: 'Foreshock', 2: 'Magnetosheath', 3: 'Magnetosphere'}
region_colors = {-1: 'k', 0: 'darkgrey',   1: 'firebrick', 2: 'royalblue',     3: 'seagreen'}

# Man-labelled regions provided by IRFU
irfu_regions = {0: 'Other', 1: 'Solar Wind', 2: 'Magnetosheath'}
irfu_region_colors = {0: 'k', 1: region_colors[0], 2: region_colors[2]}
irfu_to_my_regions = {0: -1, 1: 0, 2: 2}
my_to_irfu_regions = {-1: 0, 0: 1, 1: 0, 2: 2, 3: 2, 4: 0}

# Some constants
J2000 = datetime.datetime(2000, 1, 1, 11, 58, 55, 816000)
# Regular expression for
RE_MMS_FILE_NAME = re.compile(r'mms(?P<mmsId>[1-4])_(?P<instrument>\w*)_(?P<tmMode>(fast|brst))_(?P<datalevel>\w*)_(?P<detector>\w*)-(?P<product>\w*)_(?P<timestamp>\w*)_(?P<ext>.*)')
RE_MMS_DATETIME = re.compile(r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<min>\d{2})(?P<sec>\d{2})')

# Default energies for each channel
default_dis_channels = [2.16, 3.91, 7.07, 10.93, 14.47, 19.16, 25.37, 33.59, 44.48, 58.89, 77.98,
                        103.24, 136.7, 180.99, 239.63, 317.28, 420.09, 556.22, 736.45, 975.08,
                        1291.03, 1709.37, 2263.26, 2996.62, 3967.62, 5253.24, 6955.46, 9209.24,
                        12193.31, 16144.31, 21375.56, 28301.89]
# Indices of energy channels used by Yuri in his plots Feb 2019
channels_yuri = [8, 12, 17, 20, 25, 27]
# Channels for Msheath and magnetosphere
channels_msheath = [12, 17, 20, 25, 27, 30]
# Ranges for the plot corresponding to each channel
subplot_ranges_yuri = np.power(10, [[-23., -20.], [-23., -20.], [-24., -21.], [-25., -22.], [-26., -23.], [-27., -24.]])
# Ranges for magnetosheath and magnetosphere
subplot_ranges_msheath = np.power(10, [[-23., -20.], [-24., -21.], [-25., -22.], [-26., -23.], [-27., -24.], [-29., -26.]])
# Large value used for epoch <--> ID conversions
epoch_id_large_int = np.int64(1e18)
epoch_id_divisor = 1000

# Earth radius in km
earth_radius = 6378.

def normalize_data(X, verbose=True):
    """ Compute logarithm and normalize the data for learning.

    Parameters:
        X - [epoch, Phi, Theta, Energy]

    """

    # Old way
    if verbose:
        print('Normalizing data array', X.shape)
    try:
        min_value = np.min(X[np.nonzero(X)])
    except ValueError:
        print('Warning! All elements of X are zero, returning a zero-array.')
        return X
    if verbose:
        print('Replacing zeros with min...')
    X = np.where(np.isclose(X, 0, rtol=0, atol=1e-30), min_value, X)
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    if verbose:
        print('Subtracting min...')
    X -= X.min()

    '''
    # New way
    min_value = 1e-30
    if verbose:
        print('Replacing negatives with zeros...')
    X = np.where(X > 0, X, min_value)
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    if verbose:
        print('Subtracting min...')
    X += 30
    '''
    if verbose:
        print('Normalizing to 1...')
    X /= X.max()
    if verbose:
        print('Rolling along Phi...')
    X = np.roll(X, 16, axis=X.ndim-2)
    return X

def parse_mms_filename(s):
    """ Parse filenames of the form:
        'mms1_fpi_fast_l2_dis-dist_20171110120000_v3.3.0.cdf'
    You can also feed a full path to a file.

    Returns a dictionary with CDF file information.

    """
    fname = os.path.split(s)[1]
    m = RE_MMS_FILE_NAME.match(fname)
    g = m.groupdict()
    g['mmsId'] = int(g['mmsId'])
    g['datetime'] = datetime.datetime(*[int(i) for i in RE_MMS_DATETIME.match(g['timestamp']).groups()])
    return g

def epoch2time(epoch):
    """ The Epoch in MMS CDF files is the 'Nanoseconds since J2000'.
    This method takes Epoch and converts it to UTC.
    J2000 is January 1, 2000, 11:58:55.816 UTC

    Parameters:
        epoch - a single epoch (integer or float)

    Returns:
        datetime

    """
    try:
        return np.array([J2000 + datetime.timedelta(microseconds=int(e) // 1000) for e in epoch])
    except:
        return J2000 + datetime.timedelta(microseconds=int(epoch) // 1000)

def time2epoch(t):
    """ Convert datetime to the MMS Epoch.
    The MMS epoch is 'Nanoseconds since J2000'
    J2000 is January 1, 2000, 11:58:55.816 UTC.

    Parameters:
        t - datetime or tuple with (year, month, day, hr, min, sec)

    Returns:
        epoch - integer/np.int64

    TODO: 5 leap seconds were added since 2000: [2005, 2008, 2012, 2015, 2016]. Guess, Python takes care of it.
    TODO: we can use cdflib.cdfepoch.compute_tt2000([2016, 11, 9, 0, 0, 0])? Need to be investigated.

    """
    if isinstance(t, tuple):
        return int((datetime.datetime(*t) - J2000).total_seconds() * 1e9)
    else:
        return int((t - J2000).total_seconds() * 1e9)

def get_index(t, epoch, dt=4.5e6):
    """ Get entry index for given time and date

    Parameters:
        t - datetime

    """
    e = time2epoch(t)
    # TODO: optimize and handle exceptions
    i = np.min(np.where(epoch >= e)[0])
    return i

def compute_dist_log(a, atol=1e-30):
    """ Compute log of a distribution which contains zeros.

    Keywords:
        atol - absolute tolerance, our small double.

    """
    min_value = np.min(a[np.nonzero(a)])
    X = np.where(np.isclose(a, 0, rtol=0, atol=1e-30), min_value, a)
    return np.log10(X)

def normalize_energy(a, axis=1):
    """ Normalizes energy per each channel.

    Parameters:
        a - array [#epoch, #energy_channel, Theta, Phi]

    """
    idcs = [slice(None) for i in a.shape]
    for i in range(a.shape[axis]):
        idcs[axis] = i
        tidcs = tuple(idcs)
        # TODO dealing with log10 of distribution function. Shall it be positive?
        a[tidcs] -= a[tidcs].min()
        a[tidcs] /= a[tidcs].max() #np.std(a[tidcs])

def epoch2index(e, epoch):
    """ Find the closest epoch in the array.

    """
    return (np.abs(epoch-e)).argmin()

def epoch2int(e):
    """ Convert to some simpler int format.

    """
    return np.int(1e-9 * e)

def epoch2id(e, mms):
    """ Convert epoch to a unique ID.

    mms can be either int (1-4) or str, e.g., 'mms1'

    """
    mms_index = int(mms[-1]) if isinstance(mms, str) else mms
    return (e // epoch_id_divisor) + np.int64(mms_index) * epoch_id_large_int

def id2epoch(id):
    """ Converts a unique ID into epoch and MMS index.

    """
    return (id % epoch_id_large_int)*epoch_id_divisor, id // epoch_id_large_int

def id2time(id):
    return epoch2time(id2epoch(id)[0])

def time2str(t):
    """ Converts DateTime object into a string like 20171110120000.

    """
    def time2str_atomic(tt):
        """ ISO format looks like '2017-11-10T10:00:00' """
        #return (tt.isoformat()).replace('-', '').replace('-', '').replace('T', '').replace(':', '')
        s = tt.isoformat()
        return s[:4] + s[5:7] + s[8:10] + s[11:13] + s[14:16] + s[17:19]

    # If array-like is provided, return an array
    try:
        return np.array([time2str_atomic(j) for j in t])
    except:
        return time2str_atomic(t)

def get_title_by_id(id):
    """ Construct a title given id and spectrograph name (DES/DIS).
    The title looks like: 'mms1_20161009'

    """
    epoch, mms_id = id2epoch(id)
    time = epoch2time(epoch)
    time_str = time2str(time)

    def title_by_id_atomic(m, s):
        return str(m) + '_' + s

    try:
        return np.array([title_by_id_atomic(mms_id[i], time_str[i]) for i in range(len(epoch))])
    except:
        return title_by_id_atomic(mms_id, time_str)

def read_fpi_cdf(filename, varname='mms1_dis_dist_fast', subtract_err=True, errname='mms1_dis_disterr_fast'):
    """ Simply read the CDF file and extract some information for convenience.

    Keywords:
        varname - name of the variable which contains the distribution.
        errname - name of the variable specifying the background noise.

    TODO: Add a workaround to deal with cdflib swapping axes in some (?) arrays on Linux, e.g.,
    # for a model, trained on Windows, we do swapaxes on Brain
    #dist = dist.swapaxes(1, 3)

    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    print(filename)
    mms_name, spc_name, _, _ = varname.split('_')

    cdf_file = cdflib.cdfread.CDF(filename)
    # 4D array with energy distribution per each bin: [epoch, energy, theta, phi]
    dist = cdf_file.varget(varname)

    # Error adjustment - background removal
    if subtract_err:
        dist_err = cdf_file.varget(errname)
        dist = dist - 1.1 * dist_err

    #info = cdf_file.cdf_info()
    #varinfo = cdf_file.varinq(varname)
    #varattrs = cdf_file.varattsget(variable=varname)
    #phi = cdf_file.varget(variable=mms_name + '_' + spc_name + '_phi_fast')
    #theta = cdf_file.varget(variable=mms_name + '_' + spc_name + '_theta_fast')
    energy = cdf_file.varget(variable=mms_name + '_' + spc_name + '_energy_fast')
    epoch = cdf_file.varget(variable='Epoch')
    cdf_file.close()

    fpi_dict = {'dist': dist, 'epoch': epoch, 'energy': energy, 'file': cdf_file}

    return fpi_dict

def fgm_name_for_fpi(fpi_filename):
    """ Returns the name of the FGM file for given FPI file
    in the standard MMS folder tree.

    """
    # Prepare the filename pattern corresponding to 1 day.
    d = parse_mms_filename(fpi_filename)
    fgm_pattern = '_'.join(['mms' + str(d['mmsId']), 'fgm', 'srvy', d['datalevel'], d['timestamp'][:8]]) + '*.cdf'

    # Find the required directory in the tree
    parts = os.path.normpath(fpi_filename).split('\\')
    fgm_pth = os.path.join(*([parts[0], os.sep] + parts[:-7] + ['fgm', 'srvy', d['datalevel'], d['timestamp'][:4], d['timestamp'][4:6], fgm_pattern]))

    # Search for all files with given pattern. In principle, exactly one must exists for this day.
    files = glob(fgm_pth)
    if len(files) == 0:
        print('Warning! No FGM files found for ', fpi_filename)
        return ''
    elif len(files) > 1:
        print('Warning! Multiple FGM files found for ', fpi_filename)
    return files[0]

def read_fgm_cdf(fpi_filename, fgm_filename):
    """ Reads FGM data corresponsing to given FPI CDF filename.

    If fgm_filename is empty, finds the file corresponding to fpi_filename.

    TODO: this is an ugly method, it should be prettified in accord with read_many_fgm...

    """
    # Find the needed FGM file
    fgm_filename = fgm_name_for_fpi(fpi_filename)
    print(fgm_filename)
    assert(os.path.exists(fgm_filename))

    # Get MMS name
    mms_name = os.path.basename(fpi_filename)[:4]

    print('Reading', fgm_filename)
    fgm_file = cdflib.cdfread.CDF(fgm_filename)
    epoch_fgm = fgm_file.varget('Epoch')

    # Read GSM magnetic field
    B = fgm_file.varget(mms_name + '_fgm_b_gsm_srvy_l2')

    # Distance is on a 30-sec interval, needs separate epoch
    r = fgm_file.varget('mms1_fgm_r_gsm_srvy_l2')
    #epoch_r = epoch_fgm[0] + 30e9 * np.arange(r.shape[0])
    epoch_state = fgm_file.varget('epoch_state')
  

    return {'epoch': epoch_fgm, 'file': fgm_file, mms_name + '_fgm_b_gsm_srvy_l2': B, mms_name + '_fgm_r_gse_srvy_l2': r, 'epoch_state': epoch_state}

def read_many_fgm_cdf(fgm_path, vars=['epoch', 'epoch_state', 'mms1_fgm_r_gse_srvy_l2', 'mms1_fgm_b_gsm_srvy_l2']):
    """ Reads multiple FGM files and extracts the desired vars.

    Parameters:
        path - path to the folder with multiple FGM CDF files
        vars - ['']

    """
    if not(os.path.exists(fgm_path)):
        raise FileExistsError('The specified fgm_path does not exist!')

    # Search for all files in that folder
    fnames = sorted([os.path.split(f)[-1] for f in iglob(os.path.join(fgm_path, r'*_fgm_srvy_l2_*.cdf'))])

    rdict = {v: np.array([]) for v in vars}
    for fname in fnames:
        print('Reading file', fname)
        cdf_file = cdflib.cdfread.CDF(os.path.join(fgm_path, fname))

        for k, v in rdict.items():
            new_var = cdf_file.varget(k)
            if new_var.dtype != v.dtype:
                v = v.astype(new_var.dtype)
            if new_var.ndim != v.ndim:
                v = v.reshape([0,] + list(new_var.shape[1:]))
            rdict[k] = np.concatenate([v, new_var], axis=0)

        cdf_file.close()

    rdict['files'] = fnames
    return rdict

def read_many_fpi_cdf(fpi_path, vars=['epoch', 'mms1_dis_dist_fast']):
    """ Reads multiple FPI files and extracts the desired vars.

    Parameters:
        path - path to the folder with multiple FGM CDF files
        vars - ['']

    """
    if not(os.path.exists(fpi_path)):
        raise FileExistsError('The specified fpi_path does not exist!\n' + fpi_path)

    # Search for all files in that folder
    #fnames = sorted([os.path.split(f)[-1] for f in iglob(os.path.join(fpi_path, mms_name + r'_fpi_fast_l2_dis-dist*.cdf'))])
    fnames = sorted([os.path.split(f)[-1] for f in iglob(os.path.join(fpi_path, r'*.cdf'))])

    rdict = {v: np.array([]) for v in vars}
    for fname in fnames:
        print('Reading file', fname)
        cdf_file = cdflib.cdfread.CDF(os.path.join(fpi_path, fname))

        for k, v in rdict.items():
            new_var = cdf_file.varget(k)
            if new_var.dtype != v.dtype:
                v = v.astype(new_var.dtype)
            if new_var.ndim != v.ndim:
                v = v.reshape([0,] + list(new_var.shape[1:]))
            rdict[k] = np.concatenate([v, new_var], axis=0)

        cdf_file.close()

    rdict['files'] = fnames
    return rdict

def read_many_edp_cdf(path, vars=['mms1_edp_epoch_fast_l2', 'mms1_edp_dce_gse_fast_l2', 'mms1_edp_dce_par_epar_fast_l2']):
    """ Reads multiple FGM files and extracts the desired vars.

    Parameters:
        path - path to the folder with multiple FGM CDF files
        vars - ['']

    """
    if not(os.path.exists(path)):
        raise FileExistsError('The specified fgm_path does not exist!')

    # Search for all files in that folder
    fnames = sorted([os.path.split(f)[-1] for f in iglob(os.path.join(path, r'*_edp_fast_l2_dce*.cdf'))])

    rdict = {v: np.array([]) for v in vars}
    for fname in fnames:
        print('Reading file', fname)
        cdf_file = cdflib.cdfread.CDF(os.path.join(path, fname))

        for k, v in rdict.items():
            new_var = cdf_file.varget(k)
            if new_var.dtype != v.dtype:
                v = v.astype(new_var.dtype)
            if new_var.ndim != v.ndim:
                v = v.reshape([0,] + list(new_var.shape[1:]))
            rdict[k] = np.concatenate([v, new_var], axis=0)

        cdf_file.close()

    # Set invalid values to zero
    for k in ['mms1_edp_dce_gse_fast_l2', 'mms1_edp_dce_par_epar_fast_l2']:
        v = rdict[k]
        rdict[k][np.where(np.abs(v) >= 700)] = 0

    rdict['files'] = fnames
    return rdict

def find_cdf_files(search_path):
    """ Given a single string or a list of strings representing file paths,
    searches for CDF files and returns a sorted list of files found.

    """
    filenames = []
    if isinstance(search_path, list):
        for sp in search_path:
            filenames.extend(glob(sp))
    elif isinstance(search_path, str):
        filenames = glob(os.path.join(search_path, '*.cdf'))
    else:
        raise ValueError('search_path must be a list or a str!')
    filenames.sort()
    return filenames

def combine_cdf_vtrs(search_path, output_path='', varname='mms1_dis_dist_fast', stride=10, annotate_time=True):
    """ Combines energy measurements from multiple CDFs into a series of VTR files.

    Parameters:
        search_path - prefix to search for files with a wildcard, e.g.
            r'C:/Projects/MachineLearningSpace/mms1_fpi_fast_l2_dis-dist_20171114*.cdf'

    """
    filenames = find_cdf_files(search_path)

    # Where to save VTRs
    if not output_path:
        output_path = filenames[0].split('.')[0][:-9]

    print('Saving VTRs to', output_path)

    vtr_basename = os.path.join(output_path, varname)
    time_annotation = {'index': [], 'epoch': [], 'time': []}

    # Read all CDF files and write VTRs
    for filename in filenames:
        print(filename)
        fpi_dict = read_fpi_cdf(filename, varname=varname)
        save_vtrs(vtr_basename, varname, fpi_dict, stride=stride, time_annotation=time_annotation)

    # Write time annotation in CSV
    csv_name = os.path.join(output_path, 'time.csv')
    with open(csv_name, 'w') as csv_file:
        row_names = ['index', 'epoch', 'time']
        writer = csv.writer(csv_file)
        writer.writerow(['index', 'epoch', 'time'])
        for i in range(len(time_annotation['index'])):
            writer.writerow([time_annotation[k][i] for k in row_names])

def read_irfu_regions(filename):
    """ Read region labels produced by IRFU.
    Those are text files where time label corresponds to the _start_ of the region.
    The end of the each region is the start of the next one!

    Example:
        reg_dict = mu.read_irfu_regions(r'C:/Projects/MachineLearningSpace/data/mms/irfu/cal/mms1/edp/sdp/regions/mms1_edp_sdp_regions_20171206_v0.0.0.txt')

    Parameters:
        filename = r'C:/Projects/MachineLearningSpace/data/mms/irfu/cal/mms1/edp/sdp/regions/mms1_edp_sdp_regions_20171206_v0.0.0.txt'

    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip header lines
    lines = [l for l in lines if (not l.startswith('%'))]

    time = [datetime.datetime.strptime(l.split('.')[0], '%Y-%m-%dT%H:%M:%S') for l in lines]
    epoch = np.array([time2epoch(t) for t in time])
    region = np.array([int(l.split()[-1]) for l in lines])

    return {'time': time, 'epoch': epoch, 'region': region}

def map_interval_labels(labels, start_times, selected_moments, dtype=np.int8):
    """ Projects labelled intervals onto fixed moments.

    # Parameters:
        labels - list/array of labels for each interval.
        start_times - starting moment (time, epoch, etc.) for each interval. Sorted!
                  The end of each interval is the beginning of the next one.
        selected_moments - discrete moments in time, epoch, etc., for which labels are needed.

    # Keywords
        dtype=np.int8 - data type of the result

    # Returns:
        np.array with labels for each of selected_moments.

    """
    selected_labels = np.zeros(len(selected_moments), dtype=dtype)
    for i in range(len(labels) - 1):
        selected_labels[np.where((start_times[i] <= selected_moments) & (selected_moments < start_times[i+1]))] = labels[i]
    return selected_labels

def get_label_fpi_suffix(fpi_filepath):
    """ Generate suffix for label variable.

    """
    d = parse_mms_filename(fpi_filepath)
    return r'_' + r'_'.join(['mms' + str(d['mmsId']), d['instrument'], d['tmMode'], d['detector'], d['product'], d['timestamp']])

def get_cdf_dtype(a):
    #cdf_dtypes = {np.int8: 1, np.int64: 33, np.float32: 21}
    if a.dtype == np.int8:
        return 1
    elif a.dtype == np.int64:
        return 33
    elif a.dtype == np.float32:
        return 21
    else:
        raise ValueError('Unsupported data type', a.dtype)

def write_dict_cdf(filename, var_dict, att_dict={}):
    """ Writes a dict to CDF.

    TODO: Add other CDF datatypes.

    """
    cdf_file = cdflib.cdfwrite.CDF(filename)

    for k, v in var_dict.items():
        print('Writing', k)
        var = {'Variable': k, 'Data_Type': get_cdf_dtype(v), 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable',
               'Dim_Sizes': list(v.shape[1:]), 'Num_Dims': 0, 'Sparse': 'No_sparse', 'Compress': 6, 'Pad': np.array([0,], dtype=v.dtype)}

        if k in att_dict.keys():
            attrs = att_dict[k]
        else:
            attrs = {'VAR_NOTES': 'Written by mms_utils.write_cdf_dict().'}

        cdf_file.write_var(var, var_attrs=attrs, var_data=v)
        print('done')

    cdf_file.close()
    print('Written', filename)

def read_dict_cdf(filename, varnames=None):
    """ Reads specified variables from CDF file and returns a dict.

    Keywords:
        varnames - names of the vars to read. If None, reads all zVariables.

    """
    cdf = cdflib.cdfread.CDF(filename)

    if varnames is None:
        info = cdf.cdf_info()
        varnames = list(info['zVariables'])

    res = {}
    for name in varnames:
        res[name] = cdf.varget(name)

    cdf.close()

    return res

def add_labels_CDF(epoch, pred, prob, fpi_filepath, lbl_cdf_file):
    """ Add labels for 1 CDF file to the CDF with labels for 1 month.

    # Parameters:
        epoch - 1D array, epoch of each observation
        pred - same size array of int8 - predictions
        prob - probability of the predictions, same size as pred and epoch
        fpi_filepath - the file being labelled
        lbl_cdf_file - the (opened!) CDF file to which the new data should be added.
                       Typically 1 file is for 1 month of 1 MMS satellite.

    """
    var_suffix = get_label_fpi_suffix(fpi_filepath)

    # Prediction. CDF_BYTE == 41, CDF_INT1 = 1
    vs_pred = {'Variable': 'label' + var_suffix, 'Data_Type': 1, 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable',
               'Dim_Sizes': [], 'Num_Dims': 0, 'Sparse': 'No_sparse', 'Compress': 6, 'Pad': np.array([-1,], dtype=np.int8)}
    attrs_pred = {'VAR_NOTES': 'Predicted label. ' + str(regions), 'filename': fpi_filepath}

    lbl_cdf_file.write_var(vs_pred, var_attrs=attrs_pred, var_data=pred)

    # Prediction probability. CDF_REAL4 = 21
    vs_prob = {'Variable': 'probability' + var_suffix, 'Data_Type': 21, 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable',
               'Dim_Sizes': [len(regions)-1], 'Num_Dims': 0, 'Sparse': 'No_sparse', 'Compress': 6, 'Pad': np.array([0.,], dtype=np.float32)}
    attrs_prob = {'VAR_NOTES': 'Probability of the predicted label.', 'filename': fpi_filepath}

    lbl_cdf_file.write_var(vs_prob, var_attrs=attrs_prob, var_data=prob)

    # Epoch
    # Info for epoch when it has been provided
    epoch_info = {'Variable': 'epoch' + var_suffix, 'Data_Type': 33, 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable',
                  'Dim_Sizes': [], 'Num_Dims': 0, 'Sparse': 'No_sparse', 'Compress': 6, 'Pad': np.array([-9223372036854775807], dtype=np.int64),
                  'Data_Type_Description': 'CDF_TIME_TT2000'} #, 'Dim_Vary': [],  'Block_Factor': 0}

    epoch_attrs = {'CATDESC': 'Nanoseconds since J2000', 'DELTA_PLUS_VAR': 'Epoch_plus_var', 'DELTA_MINUS_VAR': 'Epoch_minus_var', 'FIELDNAM': 'Time tags',
                   'FILLVAL': np.array([-9223372036854775808], dtype=np.int64), 'LABLAXIS': 'Epoch', 'MONOTON': 'INCREASE', 'REFERENCE_POSITION': 'Rotating Earth Geoid',
                   'SCALETYP': 'linear', 'SI_CONVERSION': '1.0e-9>s', 'TIME_BASE': 'J2000', 'TIME_SCALE': 'Terrestrial Time', 'UNITS': 'ns',
                   'VALIDMIN': np.array([-315575942816000000], dtype=np.int64), 'VALIDMAX': np.array([3155716868184000000], dtype=np.int64),
                   'VAR_NOTES': 'MMS1 FPI/DIS Fast Survey data begin-time; derived from packet time.', 'VAR_TYPE': 'support_data'}

    lbl_cdf_file.write_var(epoch_info, var_attrs=epoch_attrs, var_data=epoch)
    print(epoch_info['Variable'])

def convert_labels(reader):
    """ Decorator which allows to switch between 'old' labels, with 2 MSHs and 'new' ones, with only 4 classes.

    """
    def wrapper(*args, **kwargs):
        lbl_dict = reader(*args, **kwargs)
        Y = lbl_dict['label']
        Y[np.where(Y == 3)] = 2
        Y[np.where(Y == 4)] = 3
        describe_dataset(Y)
        lbl_dict['label'] = Y
        return lbl_dict
    return wrapper

#@convert_labels
def read_labels_cdf(filename):
    """ Reads labels for 1 month of MMS observations.
    Labels are saved in such a way that there are multiple variables corresponding
    to different individual files with observations.
    This method concatenates them all together.

    """
    print('Reading labels from', filename)
    lbl_cdf_file = cdflib.cdfread.CDF(filename)
    info = lbl_cdf_file.cdf_info()

    # Select and sort only the label variables (by date), others will follow the order
    base = 'label'
    label_var_names = sorted([i[len(base):] for i in info['zVariables'] if i.startswith(base)])

    # Read data for first label to find out the dimensions
    rdict = {}
    vname = label_var_names[0]
    for prefix in ['label', 'epoch', 'probability']:
        rdict[prefix] = lbl_cdf_file.varget(prefix + vname)
    label_var_names.remove(vname)
    files_read = [vname]

    # Read the rest of data
    for vname in label_var_names:
        for k, v in rdict.items():
            rdict[k] = np.concatenate([v, lbl_cdf_file.varget(k + vname)], axis=0)
        files_read.append(vname)

    lbl_cdf_file.close()
    rdict['files'] = files_read

    describe_dataset(rdict['label'])

    return rdict

def describe_dataset(labels, name=''):
    """ Print how many different classes are there in the dataset.

    Parameters:
        labels - 1D array with labels from -1 to the number of regions.

    """
    print('Total number of examples in the dataset', name, labels.shape[0])
    for k, v in regions.items():
        n = len(np.where(labels == k)[0])
        print(k, v, n, '(%.1f%%)' % (100 * float(n) / labels.shape[0],))

if __name__ == '__main__':
    # Save each n-th snapshot in a VTR file
    #save_vtrs(os.path.join(os.path.splitext(filename)[0].split('.')[0], varname), varname, var, energy, theta, phi, epoch)

    # Combine multiple CDFs into a series of VTRs
    #combine_cdf_vtrs([r'C:/Projects/MachineLearningSpace/data/mms1_fpi_fast_l2_dis-dist_20151207*.cdf', r'C:/Projects/MachineLearningSpace/data/mms1_fpi_fast_l2_dis-dist_20151208*.cdf'], varname=varname, stride=20)
    #combine_cdf_vtrs([r'C:/Projects/MachineLearningSpace/data/mms1_fpi_fast_l2_dis-dist_20170711*.cdf',], varname=varname, stride=10)

    """
    1) 2015 October 7-8 (Bow Shock, PRL Johalander)
    2) 2015 October 16-17 (PRL Egedal)
    3) 2015 December 6-8 (JGR Yuri)
    4) 2015 December 16-17 (Magnetosheath)
    5) 2016 November 23-24 (Magnetosheath)
    6) 2017 July 11-12 (Magnetotail - Torbert Nature)
    One thing I noticed is that in 2017 MMS go to tail also
    and we are not considering it as a class
    """

    """ Notes on 05.03.2019 telecon with Yuri, Pawel, Sven, Stefano, Andrei, Steven and Slavik.

    1. Solar Wind - single beam
    2. Ion Foreshock - single beam + higher energy cloud
    3. Bow shock - At lower energy a potato, at high energy a beam of the SW
    4. Magnetosheath q-parallel - potato with energetic ions
    5. Magnetosheath q-perp - potato, no energetic ions
    6. Magnetosphere - Isotropic.

    You CAN NOT jump from the solar wind directly to the magnetosphere.

    1) Is SW present?

    """
