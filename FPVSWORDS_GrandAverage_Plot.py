#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Plot FPVSWORDS Grand-Mean data.
==========================================

OH, May 2023
"""

import os
from os import path as op

import numpy as np

import matplotlib
matplotlib.use('Agg')  #  for running graphics on cluster
from matplotlib import pyplot as plt

# needed to run on SLURM
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from mayavi import mlab
mlab.options.offscreen = True

from copy import deepcopy

from importlib import reload

import mne
from mne.report import Report
from mne.source_estimate import SourceEstimate

import config_fpvswords as config
reload(config)

import FPVS_functions as Ff
reload(Ff)

print(mne.__version__)

# sub-directory for figures per subject
figs_dir = 'Figures'

close_fig = 1  # close figures only if close_fig==1

# plt.ion() # interactive plotting

# for some plots of SNRs
unit_scalings = dict(eeg=1., mag=1., grad=1.)

# grey value for some figure backgrounds
grey_value = (0.75, 0.75, 0.75)

# Base frequencies for frequency sweep for words (not faces)
freqs_all = [str(ff) for ff in config.fpvs_freqs]

print(*freqs_all)

# separate filename prefixes for ICAed and non-ICAed data
prefix = ''

subjects_dir = config.subjects_dir

# output directory for figures
# figs_path = op.join(config.grandmean_path, figs_dir)
figs_path = '/imaging/hauk/users/olaf/MEG/GM/Figures'

# conditions
conds = config.do_conds

def grand_average_plot():
    """Plot grand-average PSDs and derivatives."""
    # initialise html report for one subject
    report = Report(subject='GM', title='FPVS PSDs GM')

    # for STC plotting
    subject = 'fsaverage'

    # # get condition names and frequency names from first subject
    # # assumed to be consistent across subjects
    # sss_map_fname = config.sss_map_fnames[1]
    # conds = []  # names of conditions
    # for raw_stem_in in sss_map_fname[1][2:]:

    #     conds.append(raw_stem_in[:4])

    # conds = np.unique(conds)

    # initialise

    # all psd results for evoked and STC
    # individual subjects and GM
    # modals = ['evo', 'stc']
    # gm_modals = ['evo_gm', 'stc_gm']
    modals = ['evo']
    gm_modals = ['evo_gm']

    # types = ['psd', 'psd_z', 'psd_sum_odd', 'psd_sum_base', 'psd_harm_odd',
    #          'psd_harm_base', 'psd_harm_topos_odd', 'psd_harm_topos_base']

    # evo_types = [
    #     'peak_odd', 'z_peak_odd', 'harm_odd_peak_odd', 'harm_base_peak_odd',
    #     'peak_base', 'z_peak_base', 'harm_odd_peak_base',
    #     'harm_base_peak_base', 'peak_harm_topos_odd', 'peak_harm_topos_base']

    # for evoked
    types = ['psd', 'psd_z', 'psd_sum_odd', 'psd_sum_base', 'psd_harm_odd',
             'psd_harm_base', 'psd_harm_topos_odd', 'psd_harm_topos_base']
             # 'psd_sum_base_indiv_topos', 'psd_sum_odd_indiv_topos']

    # only for evoked: data for peak channels per condition
    evo_types = []
    # [
    #     'peak_odd', 'z_peak_odd', 'harm_odd_peak_odd',
    #     'harm_base_peak_odd', 'peak_base', 'z_peak_base', 'harm_odd_peak_base',
    #     'harm_base_peak_base', 'peak_harm_topos_odd', 'peak_harm_topos_base']

    # for STCs
    stc_types = ['psd', 'psd_sum_odd', 'psd_sum_base', 'psd_harm_odd',
                 'psd_harm_base', 'psd_harm_topos_odd', 'psd_harm_topos_base']

    psds = {}

    do_modals = modals + gm_modals

    # extract label amplitudes
    label_amps = {}
    for ss in stc_types:
        label_amps[ss] = {'lh': [], 'rh': []}

    # Initialise
    for modal in do_modals:

        psds[modal] = {}  # type of data

        do_types = types
        if modal[:3] == 'evo':  # add other types

            do_types = do_types + evo_types

        for tt in do_types:

            psds[modal][tt] = {}  # type of processed PSD

            for cond in conds:

                psds[modal][tt][cond] = {}  # sweep frequencies

                for ev_type in config.event_ids[cond]:

                    psds[modal][tt][cond][ev_type] = []  # subjects

    # Read Evoked GM data

    # Path for grand-mean results
    sbj_path = config.grandmean_path

    if 'evo' in modals:

        modal = 'evo'  # do the evoked results here

        do_types = types + evo_types

        for tt in do_types:

            for cond in conds:  # conditions

                print('###\nCondition: %s.\n###' % cond)

                # if all frequencies in one evoked file
                if tt in types:

                    for ev_type in config.event_ids[cond]:

                        fname_evo = op.join(sbj_path, 'AVE', 'GM_%s_%s_%s-ave.fif' %
                                            (tt, cond, ev_type))

                        evokeds = mne.read_evokeds(fname=fname_evo)

                        print(ev_type)

                        psds[modal][tt][cond][ev_type] = evokeds[0]

                elif tt in evo_types:

                    for ev_type in config.event_ids[cond]:

                        fname_evo = op.join(
                            sbj_path, 'AVE', 'GM_%s_%s_%s-ave.fif' %
                            (tt, cond, ev_type))

                        evokeds = mne.read_evokeds(fname=fname_evo)

                        psds[modal][tt][cond][ev_type] = evokeds[0]

                print('Done reading evoked files.')

                # get channel names for MEG channel selections
                channel_ROIs = Ff.get_MEG_ROI_channel_names(
                    config.meg_selections, evokeds[0].info)
                for roi in config.meg_selections:
                    config.channel_ROIs['Mag ' + roi] = channel_ROIs['Mag ' + roi]
                    config.channel_ROIs['Grad ' + roi] = channel_ROIs['Grad ' + roi]

                # # average certain frequencies, to be plotted separately
                # if cond != 'face':  # if a word condition

                #     print('Averaging frequencies: ')
                #     print(*avg_freqs)

                #     evo_freqs = []  # collect evoked across frequencies

                #     for freq in avg_freqs:

                #         # topography for oddball frequency
                #         evoked = psds[modal][tt][cond][freq]

                #         evo_freqs.append(evoked)

                #     psds[modal][tt][cond]['avg'] =\
                #         Ff.grand_average_evoked_arrays(evo_freqs)

        # PLOTTING ############################################################
        print('Plotting.')

        chtypes = ['mag', 'grad', 'eeg']  # for some topo plots

        for cond in conds:
            print('Condition %s.' % cond)

            # # Plot topographies for individuals
            # do_types = ['psd_sum_odd_indiv_topos', 'psd_sum_base_indiv_topos']

            # for tt in do_types:

            #     for ev_type in config.event_ids[cond]:

            #         evoked = psds[modal][tt][cond][ev_type]

            #         print('Scaling topographies per sample.')
            #         evoked = Ff.scale_evoked_per_channel_type(evoked)

            #         for chtype in chtypes:

            #             # scaling to individual maxima per topography
            #             vmin, vmax = 0., 1.

            #             fig = evoked.plot_topomap(times=evoked.times,
            #                                       ch_type=chtype,
            #                                       vmin=vmin, vmax=vmax,
            #                                       scalings=unit_scalings[chtype],
            #                                       units='Z', show=False)

            #             for ax in fig.axes:
            #                 # ax.set_xlabel(fontsize=24)
            #                 # ax.set_ylabel(fontsize=24)
            #                 xtl = ax.get_xticklabels()
            #                 ax.set_xticklabels(labels=xtl,
            #                                    fontdict={'fontsize': 12})
            #                 ytl = ax.get_yticklabels()
            #                 ax.set_yticklabels(labels=ytl,
            #                                    fontdict={'fontsize': 12})

            #             fig_fname = op.join(
            #                 figs_path, '%s_%s_%s_%s.jpg' %
            #                 (tt, cond, ev_type, chtype))

            #             print('Saving individual topographies: %s.' % fig_fname)

            #             fig.savefig(fig_fname)

            # Plot topographies for sum across harmonic for oddball and base
            # frequencies

            do_types = ['psd_sum_odd', 'psd_sum_base']

            for tt in do_types:

                for ev_type in config.event_ids[cond]:

                    # topography
                    evoked = psds[modal][tt][cond][ev_type]

                    times = [0.]

                    sec_label = '%s_%s' % (cond, ev_type)

                    file_label = '%s_%s_%s_%s' % (prefix, cond, tt, ev_type)

                    # Filename stem for figure; channel type to be added later
                    fname_fig = op.join(figs_path, file_label)

                    print('Creating figure %s.' % fname_fig)

                    figs = Ff.plot_evo_topomap(evoked, times, chtypes,
                                               fname_fig)

                    for ax in figs[0].axes:
                        # ax.set_xlabel(fontsize=24)
                        # ax.set_ylabel(fontsize=24)
                        xtl = ax.get_xticklabels()
                        ax.set_xticklabels(labels=xtl,
                                           fontdict={'fontsize': 12})
                        ytl = ax.get_yticklabels()
                        ax.set_yticklabels(labels=ytl,
                                           fontdict={'fontsize': 12})

                    for [fig, chtype] in zip(figs, chtypes):

                        report.add_figure(fig, tt, section=sec_label)

            # plot amplitudes across harmonics for electrode groups

            print('Plotting topographies and amplitudes across harmonics.')

            do_types = ['psd_harm_topos_base', 'psd_harm_topos_odd']

            for tt in do_types:

                for ev_type in config.event_ids[cond]:

                    print('Plot GM evoked for %s %s.' % (cond, ev_type))

                    evoked = psds[modal][tt][cond][ev_type]

                    # # change times for plotting to one sample per "second"
                    # times = evoked.times
                    # evoked.times = np.arange(0., len(times), 1.)

                    # label for condition and base frequency
                    sec_label = '%s_%s' % (cond, ev_type)

                    file_label = '%s_%s_%s_%s' % (prefix, cond, tt, ev_type)

                    # Plot topopraphies for all harmonics

                    # Filename stem for figure; channel type to be added later
                    fname_fig = op.join(figs_path, file_label)

                    print('Creating figure %s.' % fname_fig)

                    times = evoked.times  # all harmonics

                    figs = Ff.plot_evo_topomap(evoked, times, chtypes,
                                               fname_fig)

                    for ax in fig.axes:
                        # ax.set_xlabel(fontsize=24)
                        # ax.set_ylabel(fontsize=24)
                        xtl = ax.get_xticklabels()
                        ax.set_xticklabels(labels=xtl,
                                           fontdict={'fontsize': 12})
                        ytl = ax.get_yticklabels()
                        ax.set_yticklabels(labels=ytl,
                                           fontdict={'fontsize': 12})

                    # # plot spectra for EEG channel selections
                    # for roi in config.channel_ROIs:

                    #     # determine channel type for scaling
                    #     if roi[:3] == 'EEG':
                    #         chtype = 'eeg'
                    #     elif roi[:3] == 'Gra':
                    #         chtype = 'grad'
                    #     elif roi[:3] == 'Mag':
                    #         chtype = 'mag'

                    #     evoked_roi = deepcopy(evoked)

                    #     ch_names = config.channel_ROIs[roi]

                    #     evoked_roi.pick_channels(ch_names)

                    #     n = len(evoked_roi.times)
                    #     evoked_roi.times = np.arange(1., n + 1, 1.)

                    #     # Plot for peak channels without topographies
                    #     # add a bit more space for channel plot
                    #     max_val = evoked_roi.data.max()
                    #     ylim_vals = [0, evoked_roi.data.max() + 0.33 * max_val]
                    #     fig = evoked_roi.plot(
                    #         spatial_colors=True, picks=None,
                    #         scalings=unit_scalings,
                    #         ylim={chtype: ylim_vals},
                    #         gfp=False)

                    #     for ax in fig.axes:
                    #         # ax.set_xlabel(fontsize=24)
                    #         # ax.set_ylabel(fontsize=24)
                    #         xtl = ax.get_xticklabels()
                    #         ax.set_xticklabels(labels=xtl,
                    #                            fontdict={'fontsize': 12})
                    #         ytl = ax.get_yticklabels()
                    #         ax.set_yticklabels(labels=ytl,
                    #                            fontdict={'fontsize': 12})
                    #         # grey background for better line visibility
                    #         ax.set_facecolor(grey_value)

                    #     fname_fig = op.join(figs_path,
                    #                         file_label + '_%s.jpg' % roi)

                    #     print('Creating figure %s.' % fname_fig)

                    #     fig.savefig(fname_fig)

                    #     sec_label = sec_label + ' ' + roi

                    #     report.add_figure(fig, sec_label, section=sec_label)

                    # get singular values per channel type
                    # don't include last harmonic because of MEG artefact
                    idx = np.arange(0, evoked.data.shape[1] - 1, 1)
                    ss = Ff.svd_per_channel_type(evoked, idx)[0]

                    # channel types for SVD
                    ch_types = ['grad', 'mag', 'eeg']

                    # create new pyplot figure, subplots for channel types
                    fig, axs = plt.subplots(len(ch_types), 1)

                    for [ci, cht] in enumerate(ch_types):

                        # turn singular values into variances
                        s = 100. * ss[cht]**2 / (ss[cht]**2).sum()

                        x = np.arange(1, len(s) + 1, 1)

                        # plot singular values to figure
                        axs[ci].plot(x, s)

                        axs[ci].set_title(cht)

                        # axs[ci].set_xlabel(fontsize=24)
                        # axs[ci].set_ylabel(fontsize=24)
                        xtl = axs[ci].get_xticklabels()
                        axs[ci].set_xticklabels(labels=xtl,
                                                fontdict={'fontsize': 12})
                        ytl = axs[ci].get_yticklabels()
                        axs[ci].set_yticklabels(labels=ytl,
                                                fontdict={'fontsize': 12})
                        # grey background for better line visibility
                        ax.set_facecolor(grey_value)

                    fig.tight_layout(pad=1.)

                    fname_fig = op.join(
                        figs_path, file_label + '_svd.jpg')

                    # save figure for this channel type
                    fig.savefig(fname_fig)

                    plt.close('all')  # close pyplot figures

            # plot evoked spectra and topographies (plot_joint())
            do_types = ['psd', 'psd_z']

            for tt in do_types:

                for ev_type in config.event_ids[cond]:

                    print('Plot GM evoked for %s %s.' % (cond, ev_type))

                    evoked = psds[modal][tt][cond][ev_type]

                    # label for condition and base frequency
                    sec_label = '%s_%s' % (cond, ev_type)

                    file_label = '%s_%s_%s_%s' % (prefix, cond, tt, ev_type)

                    if tt == 'psd_z':  # scale z-scores to "significance"
                        ylim = {'mag': [0, 2], 'grad': [0, 2], 'eeg': [0, 2]}
                    else:
                        ylim = None  # scale to extrema

                    figs = Ff.plot_psd_as_evo(evoked, sbj_path, picks=None,
                                              txt_label=file_label,
                                              close_fig=close_fig,
                                              scalings=unit_scalings,
                                              ylim=ylim)

                    for [fig, chtype] in zip(figs, chtypes):

                        report.add_figure(fig, file_label, section=sec_label)

                    # # plot spectra for EEG channel selections
                    # for roi in config.channel_ROIs:

                    #     evoked_roi = deepcopy(evoked)

                    #     ch_names = config.channel_ROIs[roi]

                    #     evoked_roi.pick_channels(ch_names)

                    #     # CROP PSD for display
                    #     evoked_roi.crop(tmin=config.crop_times[0],
                    #                     tmax=config.crop_times[1])

                    #     # Plot for peak channels without topographies
                    #     fig = evoked_roi.plot(spatial_colors=True, picks=None,
                    #                           scalings=unit_scalings,
                    #                           gfp=False, ylim=ylim)

                    #     for ax in fig.axes:
                    #         # ax.set_xlabel(fontsize=24)
                    #         # ax.set_ylabel(fontsize=24)
                    #         xtl = ax.get_xticklabels()
                    #         ax.set_xticklabels(labels=xtl,
                    #                            fontdict={'fontsize': 12})
                    #         ytl = ax.get_yticklabels()
                    #         ax.set_yticklabels(labels=ytl,
                    #                            fontdict={'fontsize': 12})
                    #         # grey background for better line visibility
                    #         ax.set_facecolor(grey_value)

                    #     fname_fig = op.join(figs_path,
                    #                         file_label + '_%s.jpg' % roi)

                    #     print('Creating figure %s.' % fname_fig)

                    #     fig.savefig(fname_fig)

                    #     sec_label = sec_label + ' ' + roi

                    #     report.add_figure(fig, sec_label, section=sec_label)

                    # plt.close('all')

            # # plot evoked spectra for peak channels
            # do_types = ['peak_odd', 'peak_base', 'z_peak_odd', 'z_peak_base']

            # for tt in do_types:

            #     for ev_type in config.event_ids[cond]:

            #         print('Plot GM evoked for %s %s.' % (cond, ev_type))

            #         evoked = psds[modal][tt][cond][ev_type]

            #         # CROP PSD for display
            #         evoked.crop(tmin=config.crop_times[0],
            #                     tmax=config.crop_times[1])

            #         if tt[0] == 'z':
            #             ylim = {'mag': [0, 2], 'grad': [0, 2], 'eeg': [0, 2]}
            #         else:
            #             ylim = None

            #         # Plot for peak channels without topographies
            #         fig = evoked.plot(spatial_colors=True, picks=None,
            #                           scalings=unit_scalings, gfp=False)

            #         for ax in fig.axes:
            #             # ax.set_xlabel(fontsize=24)
            #             # ax.set_ylabel(fontsize=24)
            #             xtl = ax.get_xticklabels()
            #             ax.set_xticklabels(labels=xtl,
            #                                fontdict={'fontsize': 12})
            #             ytl = ax.get_yticklabels()
            #             ax.set_yticklabels(labels=ytl,
            #                                fontdict={'fontsize': 12})
            #             # grey background for better line visibility
            #             ax.set_facecolor(grey_value)

            #         sec_label = '%s_%s' % (cond, ev_type)

            #         file_label = '%s_%s_%s_%s' % (prefix, cond, tt, ev_type)

            #         fname_fig = op.join(figs_path, file_label + '.jpg')

            #         print('Creating figure %s.' % fname_fig)

            #         fig.savefig(fname_fig)

            #         report.add_figure(fig, sec_label, section=sec_label)

            # plt.close('all')

            # # plot amplitudes of harmonics for peak channels
            # do_types = ['peak_harm_topos_odd', 'peak_harm_topos_base']

            # for tt in do_types:

            #     for ev_type in config.event_ids[cond]:

            #         print('Plot GM evoked for %s %s.' % (cond, ev_type))

            #         evoked = psds[modal][tt][cond][ev_type]

            #         times = evoked.times
            #         evoked.times = np.arange(1., len(times) + 1, 1.)

            #         # Plot for peak channels without topographies
            #         fig = evoked.plot(spatial_colors=True, picks=None,
            #                           scalings=unit_scalings, gfp=False,
            #                           sphere=None)

            #         for ax in fig.axes:
            #             # ax.set_xlabel(fontsize=24)
            #             # ax.set_ylabel(fontsize=24)
            #             xtl = ax.get_xticklabels()
            #             ax.set_xticklabels(labels=xtl,
            #                                fontdict={'fontsize': 14})
            #             ytl = ax.get_yticklabels()
            #             ax.set_yticklabels(labels=ytl,
            #                                fontdict={'fontsize': 14})
            #             # grey background for better line visibility
            #             ax.set_facecolor(grey_value)

            #         sec_label = '%s_%s' % (cond, ev_type)

            #         file_label = '%s_%s_%s_%s' % (prefix, cond, tt, ev_type)

            #         fname_fig = op.join(figs_path, file_label + '.jpg')

            #         print('Creating figure %s.' % fname_fig)

            #         fig.savefig(fname_fig)

            #         # also create PDF because some edits may be needed
            #         fname_fig = op.join(figs_path, file_label + '.pdf')

            #         print('Creating figure %s.' % fname_fig)

            #         fig.savefig(fname_fig)

            #         report.add_figure(fig, sec_label, section=sec_label)

            plt.close('all')

            # plot spectra around target frequencies
            do_types = ['psd_harm_odd', 'psd_harm_base']

            for tt in do_types:

                for ev_type in config.event_ids[cond]:

                    print('Plot GM target frequencies for %s %s.' %
                          (cond, ev_type))

                    evoked = psds[modal][tt][cond][ev_type]

                    fig = evoked.plot(spatial_colors=True, picks=None,
                                      scalings=unit_scalings, gfp=False)

                    for ax in fig.axes:
                        # ax.set_xlabel(fontsize=24)
                        # ax.set_ylabel(fontsize=24)
                        xtl = ax.get_xticklabels()
                        ax.set_xticklabels(labels=xtl,
                                           fontdict={'fontsize': 12})
                        ytl = ax.get_yticklabels()
                        ax.set_yticklabels(labels=ytl,
                                           fontdict={'fontsize': 12})
                        # grey background for better line visibility
                        ax.set_facecolor(grey_value)

                    sec_label = '%s_%s' % (cond, ev_type)

                    file_label = '%s_%s_%s_%s' % (prefix, cond, tt, ev_type)

                    fname_fig = op.join(figs_path, file_label + '.jpg')

                    print('Creating figure %s.' % fname_fig)

                    fig.savefig(fname_fig)

                    report.add_figure(fig, sec_label, section=sec_label)

                #     # plot spectra for EEG channel selections
                #     for roi in config.channel_ROIs:

                #         evoked_roi = deepcopy(evoked)

                #         ch_names = config.channel_ROIs[roi]

                #         evoked_roi.pick_channels(ch_names)

                #         # Plot for peak channels without topographies
                #         fig = evoked_roi.plot(spatial_colors=True, picks=None,
                #                               scalings=unit_scalings,
                #                               gfp=False)

                #         for ax in fig.axes:
                #             # fig.axes[0].set_xlabel(fontsize=24)
                #             # fig.axes[0].set_ylabel(fontsize=24)
                #             xtl = ax.get_xticklabels()
                #             ax.set_xticklabels(labels=xtl,
                #                                fontdict={'fontsize': 12})
                #             ytl = ax.get_yticklabels()
                #             ax.set_yticklabels(labels=ytl,
                #                                fontdict={'fontsize': 12})
                #             # grey background for better line visibility
                #             ax.set_facecolor(grey_value)

                #         fname_fig = op.join(figs_path,
                #                             file_label + '_%s.jpg' % roi)

                #         print('Creating figure %s.' % fname_fig)

                #         fig.savefig(fname_fig)

                #         sec_label = sec_label + ' ' + roi

                #         report.add_figure(fig, sec_label, section=sec_label)

                # plt.close('all')

            # # plot spectra around target frequencies for peak channels
            # do_types = ['harm_odd_peak_odd', 'harm_base_peak_odd',
            #             'harm_odd_peak_base', 'harm_base_peak_base']

            # for tt in do_types:

            #     for ev_type in config.event_ids[cond]:

            #         print('Plot GM evoked for %s %s.' % (cond, ev_type))

            #         evoked = psds[modal][tt][cond][ev_type]

            #         # Plotting PSDs across harmonics
            #         fig = evoked.plot(spatial_colors=True, picks=None,
            #                           scalings=unit_scalings, gfp=False)

            #         for ax in fig.axes:
            #             # ax.set_xlabel(fontsize=24)
            #             # ax.set_ylabel(fontsize=24)
            #             xtl = ax.get_xticklabels()
            #             ax.set_xticklabels(labels=xtl,
            #                                fontdict={'fontsize': 12})
            #             ytl = ax.get_yticklabels()
            #             ax.set_yticklabels(labels=ytl,
            #                                fontdict={'fontsize': 12})
            #             # grey background for better line visibility
            #             ax.set_facecolor(grey_value)

            #         sec_label = '%s_%s' % (cond, ev_type)

            #         file_label = '%s_%s_%s_%s' % (prefix, cond, tt,
            #                                       ev_type)

            #         fname_fig = op.join(figs_path, file_label + '.jpg')

            #         print('Creating figure %s.' % fname_fig)

            #         fig.savefig(fname_fig)

            #         report.add_figure(fig, sec_label, section=sec_label)

                plt.close('all')

    # Plot STCs

    if 'stc' in modals:

        modal = 'stc'  # do source estimates here

        for tt in stc_types:

            for cond in conds:  # conditions

                print('###\nCondition: %s.\n###' % cond)

                stc_evs = {}

                for ev_type in config.event_ids[cond]:

                    fname_stc = op.join(
                        config.grandmean_path, 'STC',
                        '%s_%s_%s_%s-lh.stc' % (prefix, tt, cond, ev_type)
                    )

                    print('Reading source estimate from %s.' % fname_stc)

                    stc = mne.read_source_estimate(fname_stc)

                    stc_evs[ev_type] = stc

                for ev_type in config.event_ids[cond]:

                    # use STC for this event type
                    stc = stc_evs[ev_type]

                    time_label = None

                    # index to time point 0, which will be plotted
                    idx0 = np.abs(stc.times).argmin()

                    thresh = np.abs(stc.data[:, idx0]).max()

                    # # get some round numbers for colour bar

                    # if thresh < 10:

                    #     thresh = np.floor(thresh)

                    # elif thresh < 50:

                    #     thresh = 5 * np.floor(thresh / 5.)

                    # else:

                    #     thresh = 10 * np.floor(thresh / 10.)

                    # plot for left and right hemisphere
                    for hemi in ['both']:  # ['lh', 'rh']:

                        # for some reason, 'both' only works for 'ven' but not
                        # for 'lat'
                        for view in ['ven']:

                            brain = stc.plot(
                                subject=subject, initial_time=0.,
                                time_label=time_label,
                                subjects_dir=subjects_dir,
                                clim=dict(kind='value',
                                          lims=[0, thresh / 2., thresh]),
                                hemi=hemi, views=view
                            )

                            fname_fig = op.join(
                                figs_path,
                                '%s_%s_%s_%s_STC_%s_%s.jpg' %
                                (prefix, tt, cond, ev_type, hemi, view)
                            )

                            print('Saving figure to %s.' % fname_fig)

                            mlab.savefig(fname_fig)

                            mlab.close(all=True)

                    # plot for left and right hemisphere
                    for hemi in ['lh', 'rh']:

                        # for some reason, 'both' only works for 'ven' but not
                        # for 'lat'
                        for view in ['lat']:

                            # apparently 'brain' required for saving?
                            brain = stc.plot(
                                subject=subject, initial_time=0.,
                                time_label=time_label,
                                subjects_dir=subjects_dir,
                                clim=dict(kind='value',
                                          lims=[0, thresh / 2., thresh]),
                                hemi=hemi, views=view
                            )

                            fname_fig = op.join(
                                figs_path,
                                '%s_%s_%s_%s_STC_%s_%s.jpg' %
                                (prefix, tt, cond, ev_type, hemi, view)
                            )

                            print('Saving figure to %s.' % fname_fig)

                            mlab.savefig(fname_fig)

                            mlab.close(all=True)

    # Save HTML report
    fname_report = op.join(figs_path, prefix + 'GM_report.html')

    report.save(fname_report, overwrite=True, open_browser=False)

    plt.close('all')

    return

grand_average_plot()
