""" Various plotting utils used for reading and classification of MMS observations.

(c) Vyacheslav Olshevsky (2019)

"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
import mms_utils as mu
from matplotlib import cm
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Import my libs for writing VTK files
import importlib.util

# Import colormap to match the MATLAB's Parula
import colormaps
plt.register_cmap(name='parula', cmap=colormaps.parula)
parula = cm.get_cmap('parula')

# Graphics tuning
params = {'axes.labelsize': 'large',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'font.size': 20,
          'font.family': 'sans-serif',
          'text.usetex': False,
          'mathtext.fontset': 'stixsans',}
plt.rcParams.update(params)

def prepare_array(a, r):
    """ Prepare 'dist' array for plotting:
        Crop array to min/max ranges, compute log10, and roll along Phi.

    Parameters:
        a - ndarray
        r - [min, max] ranges to which the array should be cropped

    """
    a = np.where(a < r[0], r[0], a)
    a = np.where(a > r[1], r[1], a)

    a = np.roll(a, 16, axis=a.ndim-2)
    a = np.log10(a)

    return a

def plot_skymap(title, energy, var, channels=mu.channels_yuri, subplot_ranges=mu.subplot_ranges_yuri, prepare_data=True):
    """ Try to reproduce the plots produced by IRFU's plot_skymap.m.

    Parameters:
        title - str or datetime, title of the plot
        energy - value of energy for each channel (default_dis_channels)
        var - ion energy distribution ('dist') for the given time moment, a 3D array [#channels, #theta, #phi]

    Keywords:
        channels - the indices of the channels to be plotted
        subplot_ranges - the values used to substitute -inf in the log10(distribution)
                         -inf appear because some cells in the distribution are zero
        prepare_data - whether to normalize, roll and log10 the ion energy distribution.

    plot_skymap.m does the following:
        - Takes 'dist' - a PDist object which is a subclass of TSeries, which 'type' == 'skymap'.
          It is essentially a distribution of energy over a sphere, for each of 32 energy intervals.
        - dist.data is an array [#t_index, #Energy, #Azimuth, #Polar] == [t, E, Phi, Theta] == [1600, 32, 16, 32].
          Notice the difference in order between Theta & Phi.
          Phi is a linspace(0, 2pi, 32)
          Theta is a linspace(0, pi, 16)
          x = -r * sin(Theta) * cos(Phi)
          y = -r * sin(Theta) * sin(Phi)
          z = -r * cos(Theta)
        - Computes the desired interval of time and energy: tId, eId.
          Both may be either specific moments or energies, or the closest ones to the given values.
        - Computes the mean of data[tId, eId, :, :] over first axes, tId and eId, excluding NaNs.
        - C = log10(mean([tId, eId, :, :]))
        - change matrix so it corresponds to where the particles are going to, not coming from
          plotC = flipdim([C(:,17:32) C(:,1:16)],1)
          roll by 16 and transpose?
        - Plots surface of plotC, and changes view to: view(ax,[0 0 -1])
          NOTE, actual plot is 3D: surf(ax,PHI*180/pi,THETA*180/pi,THETA*0,plotC)
          where PHI, THETA = meshgrid(phi_edges,theta_edges)

    """
    if isinstance(title, datetime.datetime):
        title = title.isoformat()
    fig = plt.figure(title, figsize=(24, 15))

    for count in range(len(channels)):
        chan = channels[count]
        ax = fig.add_subplot(2, 3, count+1)

        # Prepare image data. Rolling along Phi is needed to put the Solar Wind direction in the center of the map.
        # in plot_skymap.m:
        # % change matrix so it corresponds to where the particles are going to, not coming from
        # plotC = flipdim([C(:,17:32) C(:,1:16)],1)
        if prepare_data:
            imdata = prepare_array(var[:,:,chan], subplot_ranges[count])
        else:
            imdata = var[:,:,chan]

        imdata = imdata.T

        # Phi is the second-to-last dimension
        Xrange = np.linspace(0, 360, num=imdata.shape[-1]+1)
        # Theta is the last dimension
        Yrange = np.linspace(0, 180, num=imdata.shape[-2]+1)
        X, Y = np.meshgrid(Xrange, Yrange)

        # plot
        im = ax.pcolormesh(X, Y, imdata, cmap=parula, vmin=np.log10(subplot_ranges[count][0]), vmax=np.log10(subplot_ranges[count][1]), )
        ax.set_title(title + '\n' + ('Energy=%i eV' % energy[chan]))
        ax.set_xlabel('Azimuthal angle')
        ax.set_ylabel('Polar angle')
        cbar = fig.colorbar(im, ax=ax)
        ax.set_aspect(2.)

    plt.tight_layout()
    #plt.pause(0.001)
    plt.show()

def plot_avg_dist(d, t, ch, ax):
    """ 2D plot of log10(energy distribution) averaged over phi and theta.
    This refers to FPI instriment, DIS/DES.

    Parameters:
        d - energy distribution [#epoch, #channels, #Theta, #Phi]
        t - time for each moments [#epoch]
        ch - energy channel values [#epoch, #channels]
        ax - Axis object in which to plot

    """
    avgd = np.mean(d, axis=(1, 2))
    avgd = np.ma.log10(avgd)

    # Choose the first snapshot's channels only...
    # TODO: Energy channels may be recalibrated though not in the same FPI CDF file.
    #ax.imshow(avgd.T, cmap='jet', origin='lower', aspect='auto', extent=(mdates.date2num(t[0]), mdates.date2num(t[-1]), ch[0, 0], ch[0,-1]))

    XRange = (mdates.date2num(t-datetime.timedelta(seconds=2.5))).tolist() + [mdates.date2num(t[-1] + datetime.timedelta(seconds=2.5))]
    YRange = ch[0].tolist() + [ch[0,-1] + 0.5*(ch[0,-1] - ch[0,-2]),]
    X, Y = np.meshgrid(XRange, YRange)
    im = ax.pcolormesh(X, Y, avgd.T, cmap='jet')

    ax.set_yscale('log')
    #ax.set_xlabel('Epoch')
    ax.xaxis_date()
    ax.set_ylabel('Energy [eV]')

def save_vtrs(vtr_basename, varname, fpi_dict, crop_range=[1e-27, 1e-20], stride=10, time_annotation = {'index': [], 'epoch': [], 'time': []}):
    """ Write a sequence of VTR files

    Parameters:
        vtr_basename = os.path.join(os.path.split(filename)[0], varname)
        varname - name of the variable var
        var - [#epoch, #energy, #theta, #phi] 4D array with measurements. Normally [1600, 32, 16, 32].
        energy - [#energy] energy bin values
        theta - [#theta] polar angle values (degrees)
        phi - [#phi] azimuthal angle (degrees)
        epoch - [#epoch] epoch for each snapshot.

    Keywords:
        crop_range - [min, max] to which var should be cropped.
        stride - every stride'th snapshot will be saved

    """
    spec = importlib.util.spec_from_file_location('module.name', r'C:/SyA/Projects/iPIC/Python/ipic_utils/vtk_utils.py')
    vtk_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vtk_utils)

    os.makedirs(os.path.split(vtr_basename)[0], exist_ok=True)

    dist = fpi_dict['dist']
    epoch = fpi_dict['epoch']
    energy = fpi_dict['energy']

    # Phi is the second-to-last dimension
    Yrange = np.linspace(0, np.pi, num=dist.shape[-2])
    # Theta is the last dimension
    Zrange = np.linspace(0, 2*np.pi, num=dist.shape[-1])

    for i in range(0, dist.shape[0], stride):
        epch = mu.epoch2int(epoch[i])
        # Energy dimension range
        en = np.log10(energy[i, :])
        vtk_coords, coords = vtk_utils.createVTRCoordinatesIrregular([en, Yrange, Zrange])
        vtr_name = vtr_basename + '_' + ('%10.10i' % epch) + '.vtr'

        # Prepare data.
        data = prepare_array(dist[i, :, :, :], crop_range)

        # Write
        vtk_utils.writeVTR2(vtr_name, {varname: data}, '', vtk_coords)

        # Add time annotation
        time_annotation['index'].append(0 if len(time_annotation['index']) == 0 else time_annotation['index'][-1] + 1)
        time_annotation['epoch'].append(epch)
        time_annotation['time'].append(mu.epoch2time(epoch[i]).isoformat())

def add_fake_scatter_legend(ax, colors, labels, order, s=800):
    """ Add labels with enlarged circles for diagrams made with scatter.

    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0, y0 = xlim[0] - 1000, ylim[0] - 1000
    for c in order:
        ax.scatter([x0], [y0], c=colors[c], s=s, label=labels[c])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def scatter_lbl_plot(X, Y, labels, xlabel, ylabel, plot_labels=range(-1, len(mu.regions)-1), sizes=[20,]*len(mu.regions), colors=mu.region_colors, title='Correlations'):
    """ Scatter plot in given X, Y.
    Color of each point corresponds to the label.

    Example:

    """
    fig = plt.figure(title, figsize=(38, 19))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(plot_labels)):
        c = plot_labels[i]
        j = np.where(labels == c)
        ax.scatter(X[j], Y[j], s=sizes[i], marker='o', c=colors[c])

    ax = add_fake_scatter_legend(ax, colors, mu.regions, plot_labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc=0, frameon=False, borderaxespad=0.01, handletextpad=0.2)

    plt.tight_layout()
    plt.show()
    return ax

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Borrowed from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure(title, figsize=(21, 19))
    ax = fig.add_subplot()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

if __name__ == "__main__":
    pass
