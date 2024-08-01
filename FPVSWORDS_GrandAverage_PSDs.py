#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Compute grand-average of the outputs of FPVSWORDS_PSD_compute.py.

In order to average across all subjects, don't specify arguments, or specify
a number larger than the largest subject ID (e.g. using SLURM).
For example:
run FPVS_GrandAverage_PSDs
run FPVS_GrandAverage_PSDs 1 2 3 4
run FPVS_GrandAverage_PSDs 99
==========================================

OH, May 2023
"""

# To do:
# data are not in format for easy averaging across subjects
# then plotting needs changes accordingly

# for Evoked data are in one file for all frequencies
# for STC data are in separate files per condition and freq

import sys

from os import path as op

import numpy as np

from scipy.stats import ttest_rel

from matplotlib import pyplot as plt

from copy import deepcopy

from importlib import reload

import mne
from mne.report import Report
from mne.source_estimate import SourceEstimate
from mne.evoked import EvokedArray

# from FPVS_functions import grand_average_evoked_arrays, peak_channels_evoked

import config_fpvswords as config

reload(config)

import FPVS_functions as Ff

reload(Ff)


print(mne.__version__)

# perform TFR of raw data or not
# do_tfr = config.do_tfr

figs_dir = "Figures"

close_fig = 1  # close figures only if close_fig==1

# plt.ion() # interactive plotting

# for some plots of SNRs
unit_scalings = dict(eeg=1.0, mag=1.0, grad=1.0)

# Base frequencies for frequency sweep for words (not faces)
freqs_all = [str(ff) for ff in config.fpvs_freqs]

# separate filename prefixes for ICAed and non-ICAed data
prefix = ""

# Which modalities and results to process

# all psd results for evoked and STC
# individual subjects and GM

modals = ["evo", "stc"]
gm_modals = ["evo_gm", "stc_gm"]
# modals = ["stc"]
# gm_modals = ["stc_gm"]

# for evoked
types = [
    "PSD",
    "PSDZ",
    "psd_sum_odd",
    "psd_sum_base",
    "HarmOdd",
    "HarmBase",
    "psd_harm_topos_odd",
    "psd_harm_topos_base",
]

# only for evoked: data for peak channels per condition
evo_types = [
    "peak_odd",
    "z_peak_odd",
    "harm_odd_peak_odd",
    "harm_base_peak_odd",
    "peak_base",
    "z_peak_base",
    "harm_odd_peak_base",
    "harm_base_peak_base",
    "peak_harm_topos_odd",
    "peak_harm_topos_base",
]

# for STCs
stc_types = [
    # "psd",
    # "psd_sum_odd",
    # "psd_sum_base",
    # "psd_harm_odd",
    # "psd_harm_base",
    "psd_harm_topos_odd",
    "psd_harm_topos_base",
]

# conditions
# conds = ['face', 'pwhf', 'pwlf', 'lfhf']
conds = config.do_conds

# Labels for ROI analysis
subjects_dir = config.subjects_dir
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

labels = mne.read_labels_from_annot(
    "fsaverage", "HCPMMP1", "both", subjects_dir=subjects_dir
)

label_names = {}
# list of list: labels within sub-lists will be combined
# number of items must correpond for 'lh' and 'rh'
label_names["lh"] = [
    ["L_FFC_ROI-lh"],
    ["L_VVC_ROI-lh"],
    ["L_V4_ROI-lh"],
    ["L_VMV3_ROI-lh"],
    ["L_TE2p_ROI-lh"],
    ["L_V1_ROI-lh"],
]
label_names["rh"] = [
    ["R_FFC_ROI-rh"],
    ["R_VVC_ROI-rh"],
    ["R_V4_ROI-rh"],
    ["R_VMV3_ROI-rh"],
    ["R_TE2p_ROI-rh"],
    ["R_V1_ROI-rh"],
]

# get subset of labels specified in labels_ATL
my_labels = {"lh": [], "rh": []}
for hh in ["lh", "rh"]:
    for nn in label_names[hh]:
        tmp = [label for label in labels if label.name == nn[0]][0]
        if len(nn) > 1:
            for n in nn[1:]:
                tmp = tmp + [label for label in labels if label.name == n][0]
        my_labels[hh].append(tmp)

# Read fsaverage source space for labels
src = mne.read_source_spaces(
    config.subjects_dir + "/fsaverage/bem/fsaverage-ico-5-src.fif"
)


def grand_average_psds(sbj_ids_all):
    """Grand-average PSDs and derivatives across sbj_ids."""
    # initialise html report for one subject

    sbj_ids = sbj_ids_all.copy()

    print("Grand-averaging subjects:")
    print(*sbj_ids)

    # report = Report(subject='GM', title='FPVS PSDs GM')

    # initialise

    psds = {}  # individual subjects and GM

    do_modals = modals + gm_modals

    for modal in do_modals:
        print(modal)

        psds[modal] = {}  # type of data

        do_types = types

        if modal == "evo":  # add other types
            do_types = do_types + evo_types

        for tt in do_types:
            psds[modal][tt] = {}  # type of processed PSD

            for cond in conds:
                psds[modal][tt][cond] = {}  # sweep frequencies

                for ev_type in config.event_ids[cond]:
                    psds[modal][tt][cond][ev_type] = []

                # if cond == 'face':  # no frequency sweep for faces

                #     # base frequency for this condition (Hz as string)
                #     freqs = ['6.0']

                # else:  # for all word condition, use all sweep frequencies

                #     # base frequencies for this condition (Hz as string)
                #     freqs = freqs_all

                # for freq in freqs:

                #     psds[modal][tt][cond][freq] = []  # subjects

    # initialise array for electrode ROIs for group statistics
    roi_chans_rms = {}
    for roi in config.channel_ROIs:
        roi_chans_rms[roi] = {}

        for cond in conds:
            roi_chans_rms[roi][cond] = {}

            # if cond == 'face':  # no frequency sweep for faces

            #     freqs = ['6.0']  # base frequency for this condition (Hz)

            # else:  # for all word condition, use all sweep frequencies

            #     # base frequencies for this condition (Hz as string)
            #     freqs = freqs_all

            n_sbjs = len(sbj_ids_all)
            if 'extra' in cond:
                n_sbjs -= len(config.missing_extras)

            # for freq in freqs:
            for ev_type in config.event_ids[cond]:
                roi_chans_rms[roi][cond][ev_type] = {
                    "odd": np.zeros(n_sbjs),
                    "base": np.zeros(n_sbjs),
                }

    # Reading evoked data, getting data for channel groups
    has_nan = {"fname": [], "id": [], "cond": [], "ev_type": []}  # check for NaN

    if "evo" in modals:
        print("Reading evoked data.")

        modal = "evo"

        for cond in conds:  # conditions
            if 'extra' in cond:
                sbj_ids = Ff.remove_subjects_extras(sbj_ids_all.copy())
            else:
                sbj_ids = sbj_ids_all
            for ev_type in config.event_ids[cond]:
                print("Condition: %s, %s." % (cond, ev_type))

                #     if cond == 'face':  # no frequency sweep for faces

                #         freqs = ['6.0']  # base frequency for condition (Hz as str)

                #         freq_odd = 1.2  # oddball frequency for this condition (Hz)

                #     else:  # for all word condition, use all sweep frequencies

                #         # base frequencies for this condition (Hz as string)
                #         freqs = freqs_all

                #         freq_odd = 1.0  # oddball frequency the same for all sweeps

                # for Evoked data are in one file for all frequencies
                # for STC data are in separate files per condition and freq
                for [ss, sbj_id] in enumerate(sbj_ids):  # across all subjects
                    # path to subject's data
                    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

                    print("Reading PSD results from evoked files:")

                    # PSD (raw):
                    # fname_evo = op.join(sbj_path, 'AVE', 'PSD_%s%s' % (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, psd_all)

                    # # PSD (z-scored):
                    # fname_evo = op.join(sbj_path, 'AVE', 'PSDZ_%s%s' % (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, psd_z_all, overwrite=True)

                    # # Sum PSD segments around harmonics of oddball frequency then z-score:
                    # fname_evo = op.join(sbj_path, 'AVE', 'HarmOdd_%s%s' %
                    #                     (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, psd_harm_all, overwrite=True)

                    # # Sum PSD segments around harmonics of base frequency then z-score:
                    # fname_evo = op.join(sbj_path, 'AVE', 'HarmBase_%s%s' %
                    #                     (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, psd_harm_base_all, overwrite=True)

                    # # Oddball topography of z-scored summed harmonics at centre frequency:
                    # fname_evo = op.join(sbj_path, 'AVE', 'SumTopoOdd_%s%s' %
                    #                     (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, sum_harms_odd_all, overwrite=True)

                    # # Base topography of z-scored summed harmonics at centre frequency:
                    # fname_evo = op.join(sbj_path, 'AVE', 'SumTopoBase_%s%s' %
                    #                     (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, sum_harms_base_all, overwrite=True)

                    # # Oddball topographies at centre frequencies for individual harmonics:
                    # fname_evo = op.join(sbj_path, 'AVE', 'SumToposOdd_%s%s' %
                    #                     (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, topos_odd_all, overwrite=True)

                    # # Base topographies at centre frequencies for individual harmonics:
                    # fname_evo = op.join(sbj_path, 'AVE', 'SumToposBase_%s%s' %
                    #                     (cond, '-ave.fif'))
                    # print(fname_evo)
                    # mne.write_evokeds(fname_evo, topos_base_all, overwrite=True)

                    # Read Evoked

                    # PSD (raw):
                    fname_evo = op.join(
                        sbj_path, "AVE", "PSD_%s_%s%s" % (cond, ev_type, "-ave.fif")
                    )
                    print(fname_evo)
                    psd = mne.read_evokeds(fname_evo, 0)
                    print(psd)
                    print("Bads: ")
                    print(psd.info["bads"])

                    # PSD (z-scored):
                    fname_evo = op.join(
                        sbj_path, "AVE", "PSDZ_%s_%s%s" % (cond, ev_type, "-ave.fif")
                    )
                    print(fname_evo)
                    psd_z = mne.read_evokeds(fname_evo, 0)

                    if True in np.isnan(psd_z.data):
                        print("\nNan\n")
                        has_nan["fname"].append(fname_evo)
                        has_nan["id"].append(sbj_id)
                        has_nan["cond"].append(cond)
                        has_nan["ev_type"].append(ev_type)

                    # Sum PSD segments around harmonics of oddball frequency
                    fname_evo = op.join(
                        sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
                    )
                    print(fname_evo)
                    psd_harm_odd = mne.read_evokeds(fname_evo, 0)

                    # Sum PSD segments around harmonics of base frequency
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_harm_base = mne.read_evokeds(fname_evo, 0)

                    # fname_evo = op.join(sbj_path, 'AVE', 'HarmBaseEpos_%s%s%s' %
                    #                     (cond, freq, '-ave.fif'))

                    # Oddball topography of z-scored summed harmonics at centre
                    # frequency:
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "SumTopoOdd_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_sum_odd = mne.read_evokeds(fname_evo, 0)

                    # Base topography of z-scored summed harmonics at centre
                    # frequency:
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "SumTopoBase_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_sum_base = mne.read_evokeds(fname_evo, 0)

                    # Oddball topographies at centre frequencies for individual
                    # harmonics:
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "SumToposOdd_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_harm_topos_odd = mne.read_evokeds(fname_evo, 0)

                    # Base topographies at centre frequencies for individual
                    # harmonics:
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "SumToposBase_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_harm_topos_base = mne.read_evokeds(fname_evo, 0)

                    ### Continue from here
                    #             channel_ROIs = Ff.get_MEG_ROI_channel_names(
                    #                 config.meg_selections, psd[0].info)
                    #             for roi in config.meg_selections:
                    #                 config.channel_ROIs['Mag ' + roi] = channel_ROIs['Mag ' + roi]
                    #                 config.channel_ROIs['Grad ' + roi] = channel_ROIs['Grad ' + roi]

                    psds[modal]["PSD"][cond][ev_type].append(psd)

                    psds[modal]["PSDZ"][cond][ev_type].append(psd_z)

                    psds[modal]["psd_sum_odd"][cond][ev_type].append(psd_sum_odd)

                    psds[modal]["psd_sum_base"][cond][ev_type].append(psd_sum_base)

                    psds[modal]["psd_harm_topos_odd"][cond][ev_type].append(
                        psd_harm_topos_odd
                    )

                    psds[modal]["psd_harm_topos_base"][cond][ev_type].append(
                        psd_harm_topos_base
                    )

                    psds[modal]["HarmOdd"][cond][ev_type].append(psd_harm_odd)

                    psds[modal]["HarmBase"][cond][ev_type].append(psd_harm_base)

    #                 # hack, float-to-string-to-float-again
    #                 # to be consistent with FPVS_PSD_sweep_plot.py
    #                 basefreq = float(freq)

    #                 # Get max channels from z-scored PSD at base frequency
    #                 # not oddball frequency, which would be biased.
    #                 # This evoked is for condition cond, current subject and
    #                 # current frequency freq.
    #                 evoked = deepcopy(psd_z[fi])

    #                 # Find channels with maximum Z-scores per channel type
    #                 # for base frequency
    #                 # "Latency" is frequency in Hz divided by 1000
    #                 peak_times_base = [basefreq]

    #                 peak_ch_types_base = Ff.peak_channels_evoked(
    #                     evoked=evoked, peak_times=peak_times_base,
    #                     ch_types=None, n_chan=config.n_peak)

    #                 print('###\nPeak channels in Z-scored PSD for base'
    #                       'frequency %f: ' % basefreq)

    #                 # turn channel names into one list
    #                 # assume there was only one peak frequency
    #                 peak_ch_names_base = []
    #                 for chtype in peak_ch_types_base[0]:

    #                     peak_ch_names_base = peak_ch_names_base + \
    #                         peak_ch_types_base[0][chtype]

    #                 # Find channels with maximum Z-scores per channel type
    #                 # for oddball frequency
    #                 # "Latency" is frequency in Hz divided by 1000
    #                 peak_times_odd = [freq_odd]
    #                 peak_ch_types_odd = Ff.peak_channels_evoked(
    #                     evoked=evoked, peak_times=peak_times_odd,
    #                     ch_types=None, n_chan=config.n_peak)

    #                 print('###\nPeak channels in Z-scored PSD for oddball frequency %f: '
    #                       % freq_odd)

    #                 # turn channel names into one list
    #                 # assume there was only one peak frequency
    #                 peak_ch_names_odd = []
    #                 for chtype in peak_ch_types_odd[0]:

    #                     peak_ch_names_odd = peak_ch_names_odd + \
    #                         peak_ch_types_odd[0][chtype]

    #                 #

    #                 # Deepcopy because instance of evoked will be modified.
    #                 evoked = deepcopy(psd_z[fi])

    #                 # reduce evoked to peak channels for base frequency
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_base)

    #                 psds[modal]['z_peak_base'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 # Deepcopy because instance of evoked will be modified.
    #                 evoked = deepcopy(psd_z[fi])

    #                 # reduce evoked to peak channels for oddball frequency
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_odd)

    #                 psds[modal]['z_peak_odd'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 evoked = deepcopy(psd[fi])

    #                 # base freq
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_base)

    #                 psds[modal]['peak_base'][cond][freq].append(
    #                     evoked_peak)

    #                 evoked = deepcopy(psd[fi])

    #                 # odd freq
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_odd)

    #                 psds[modal]['peak_odd'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 evoked = deepcopy(psd_harm_odd[fi])

    #                 # base freq
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_base)

    #                 psds[modal]['harm_odd_peak_base'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 evoked = deepcopy(psd_harm_odd[fi])

    #                 # odd freq
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_odd)

    #                 psds[modal]['harm_odd_peak_odd'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 evoked = deepcopy(psd_harm_base[fi])

    #                 # base freq
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_base)

    #                 psds[modal]['harm_base_peak_base'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 evoked = deepcopy(psd_harm_base[fi])

    #                 # odd freq
    #                 evoked_peak = evoked.pick_channels(peak_ch_names_odd)

    #                 psds[modal]['harm_base_peak_odd'][cond][freq].append(
    #                     evoked_peak)

    #                 #
    #                 # use GM peak channels for harmonics amplitudes
    #                 # base freq
    #                 peak_gm_channel_names = (
    #                     config.channel_ROIs['EEG_Peak_Base'] +
    #                     config.channel_ROIs['Mag_Peak_Base'] +
    #                     config.channel_ROIs['Gra_Peak_Base'])

    #                 evoked = deepcopy(psd_harm_topos_base[fi])

    #                 evoked_peak = evoked.pick_channels(peak_gm_channel_names)

    #                 psds[modal]['peak_harm_topos_base'][cond][freq].append(
    #                     evoked_peak)

    #                 #

    #                 evoked = deepcopy(psd_harm_topos_odd[fi])

    #                 # odd freq
    #                 peak_gm_channel_names = (
    #                     config.channel_ROIs['EEG_Peak_Odd'] +
    #                     config.channel_ROIs['Mag_Peak_Odd'] +
    #                     config.channel_ROIs['Gra_Peak_Odd'])

    #                 evoked_peak = evoked.pick_channels(peak_gm_channel_names)

    #                 psds[modal]['peak_harm_topos_odd'][cond][freq].append(
    #                     evoked_peak)

    #                 # Get data for group statistics (e.g. laterality)
    #                 # RMS across electrodes in ROI
    #                 for roi in config.channel_ROIs:

    #                     ch_names = config.channel_ROIs[roi]

    #                     for stim in ['base', 'odd']:

    #                         type_now = 'psd_harm_' + stim

    #                         evoked_roi = deepcopy(
    #                             psds[modal][type_now][cond][freq][-1])

    #                         evoked_roi.pick_channels(ch_names)

    #                         idx0 = evoked_roi.time_as_index(0.)

    #                         rms = np.sqrt((evoked_roi.data[:, idx0]**2).mean())

    #                         roi_chans_rms[roi][cond][freq][stim][ss] = rms

    # Reading source estimate (STC) data
    if "stc" in modals:
        print("Reading source estimates.")

        modal = "stc"

        for cond in conds:  # conditions
            print("###\nCondition: %s.\n###" % cond)
            if 'extra' in cond:
                sbj_ids = Ff.remove_subjects_extras(sbj_ids_all.copy())
            else:
                sbj_ids = sbj_ids_all
            if cond[:4] == "rest":
                task = "rest"
            else:
                task = cond

            for ev_type in config.event_ids[cond]:
                print("Reading PSD results from STC files:")

                # for Evoked data are in one file for all frequencies
                # for STC data are in separate files per condition and freq
                for sbj_id in sbj_ids:  # across all subjects, EDIT ###
                    # path to subject's data
                    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

                    print("Reading PSD results from STC files:")

                    fname_stc = op.join(
                        sbj_path,
                        "STC",
                        "PSDSumTopoOdd_%s_%s_mph-lh.stc" % (cond, ev_type),
                    )
                    print(fname_stc)
                    stc = mne.read_source_estimate(fname_stc)
                    psds[modal]["psd_harm_topos_odd"][cond][ev_type].append(stc)

                    fname_stc = op.join(
                        sbj_path,
                        "STC",
                        "PSDSumTopoBase_%s_%s_mph-lh.stc" % (cond, ev_type),
                    )
                    print(fname_stc)
                    stc = mne.read_source_estimate(fname_stc)
                    psds[modal]["psd_harm_topos_base"][cond][ev_type].append(stc)

                #                 # fname_stc = op.join(
                #                 #     sbj_path, 'STC', '%sPSDTopoZ_%s_%s_mph-lh.stc' %
                #                 #     (prefix, cond, freq)
                #                 # )
                #                 # print(fname_stc)
                #                 # stc = mne.read_source_estimate(fname_stc)
                #                 # psds[modal]['psd_z'][cond][freq].append(stc)

                #                 fname_stc = op.join(
                #                     sbj_path, 'STC', '%sPSDHarm_%s_%s_mph-lh.stc' %
                #                     (prefix, cond, freq)
                #                 )
                #                 print(fname_stc)
                #                 stc = mne.read_source_estimate(fname_stc)
                #                 psds[modal]['psd_harm_odd'][cond][freq].append(stc)

                #                 fname_stc = op.join(
                #                     sbj_path, 'STC', '%sPSDHarmBase_%s_%s_mph-lh.stc' %
                #                     (prefix, cond, freq)
                #                 )
                #                 print(fname_stc)
                #                 stc = mne.read_source_estimate(fname_stc)
                #                 psds[modal]['psd_harm_base'][cond][freq].append(stc)

                #                 fname_stc = op.join(
                #                     sbj_path, 'STC', '%sPSDSumTopoOdd_%s_%s_mph-lh.stc' %
                #                     (prefix, cond, freq)
                #                 )
                #                 print(fname_stc)
                #                 stc = mne.read_source_estimate(fname_stc)
                #                 psds[modal]['psd_sum_odd'][cond][freq].append(stc)

                #                 fname_stc = op.join(
                #                     sbj_path, 'STC', '%sPSDSumTopoBase_%s_%s_mph-lh.stc' %
                #                     (prefix, cond, freq)
                #                 )
                #                 print(fname_stc)
                #                 stc = mne.read_source_estimate(fname_stc)
                #                 psds[modal]['psd_sum_base'][cond][freq].append(stc)

                #                 fname_stc = op.join(
                #                     sbj_path, 'STC', '%sPSDSumToposOdd_%s_%s_mph-lh.stc' %
                #                     (prefix, cond, freq)
                #                 )
                #                 print(fname_stc)
                #                 stc = mne.read_source_estimate(fname_stc)
                #                 psds[modal]['psd_harm_topos_odd'][cond][freq].append(stc)

                #                 fname_stc = op.join(
                #                     sbj_path, 'STC', '%sPSDSumToposBase_%s_%s_mph-lh.stc' %
                #                     (prefix, cond, freq)
                #                 )
                #                 print(fname_stc)
                #                 stc = mne.read_source_estimate(fname_stc)
                #                 psds[modal]['psd_harm_topos_base'][cond][freq].append(stc)

                #             # Grand-average STCs

                #             print('Grand-averaging source estimates.')

                for tt in stc_types:
                    stcs = psds[modal][tt][cond][ev_type]

                    avg_data = np.average([s.data for s in stcs], axis=0)

                    # turn average into source estimate object
                    stc_avg = SourceEstimate(
                        avg_data, stcs[0].vertices, stcs[0].tmin, stcs[0].tstep
                    )

                    fname_stc = op.join(
                        config.grandmean_path,
                        "STC",
                        "%s_%s_%s" % (tt, cond, ev_type),
                    )

                    print("Writing GM to %s." % fname_stc)

                    stc_avg.save(fname_stc, overwrite=True)

    #                 # Extract label amplitudes
    #                 if tt in ['psd_harm_odd', 'psd_harm_base']:
    #                     amps = {'lh': [], 'rh': []}
    #                     idx0 = np.abs(stc.times).argmin()
    #                     for hh in ['lh', 'rh']:
    #                         amps[hh] = {}
    #                         for ll in my_labels[hh]:
    #                             amps[hh][ll.name] = []
    #                             for stc in stcs:
    #                                 aa = mne.source_estimate.extract_label_time_course(
    #                                     stcs=stc, labels=ll, src=src, mode='max')
    #                                 aa = aa[0, idx0]
    #                                 amps[hh][ll.name].append(aa)

    #                     # t-test
    #                     print(tt)
    #                     for [li, ll] in enumerate(my_labels['lh']):
    #                         print(ll.name)
    #                         data1 = amps['lh'][my_labels['lh'][li].name]
    #                         data2 = amps['rh'][my_labels['rh'][li].name]

    #                         alternative = 'less'  # what type of t-test
    #                         stat, pv = ttest_rel(data1, data2,
    #                                              alternative=alternative)

    #                         print('T-test in source space for L-R with'
    #                               'alternative %s, %s | %s (%f vs %f).' %
    #                               (alternative, cond, freq, np.mean(data1),
    #                                np.mean(data2)))
    #                         # p-value
    #                         print('%f, %f\n' % (stat, pv))

    if has_nan["fname"] != []:
        print("\nNaNs!!!")
        for ii in range(len(has_nan["id"])):
            print(
                has_nan["fname"][ii],
                has_nan["id"][ii],
                has_nan["cond"][ii],
                has_nan["ev_type"][ii],
            )

    # Compute Grand-Averages for Evoked data

    # Path for grand-mean results
    sbj_path = config.grandmean_path

    if "evo_gm" in gm_modals:
        print("Grand-averaging evoked data.")

        psd_evo = psds["evo"]

        for cond in conds:  # conditions
            print("###\nCondition: %s.\n###" % cond)

            for tt in types:
                gm_evos = []  # get Evokeds for frequencies as list

                for ev_type in config.event_ids[cond]:
                    # grand-average across subjects
                    evoked = mne.grand_average(
                        psd_evo[tt][cond][ev_type], interpolate_bads=True
                    )

                    if True in np.isnan(evoked.data):
                        print("\nNan\n")

                    evoked.comment = ev_type  # will be used in plotting script

                    # to keep everything
                    psds["evo_gm"][tt][cond][ev_type] = deepcopy(evoked)

                    fname_evo = op.join(
                        sbj_path, "AVE", "%s_%s_%s-ave.fif" % (tt, cond, ev_type)
                    )

                    print("Writing GM to %s." % fname_evo)

                    mne.write_evokeds(fname=fname_evo, evoked=evoked, overwrite=True)

                # put topographies for individual subjects together
            for topotype in ['psd_sum_odd', 'psd_sum_base']:
                evo_indiv = []
                for ev_type in config.event_ids[cond]:

                    evos = psd_evo[topotype][cond][ev_type]

                    data = evos[0].data

                    # numpy array for topographies with shape (# sensors, # subjs)
                    evo_mats = np.zeros((data.shape[0], len(evos)))

                    for (ei, ee) in enumerate(evos):  # across subjects

                        # evoked only has one sample
                        evo_mats[:, ei] = ee.data[:, 0]

                    evoked = EvokedArray(evo_mats, evos[0].info, tmin=0)

                    fname_evo = op.join(
                        sbj_path, 'AVE', 'GM_%s_indiv_topos_%s_%s-ave.fif' %
                        (topotype, cond, ev_type))

                    print('Writing individual topographies to %s.' % fname_evo)

                    mne.write_evokeds(fname_evo, evoked, overwrite=True)

                # # determine peak channels
                # # just put 0s as peak_times for now
                # peak_times = 0.
                # peak_ch_names = Ff.peak_channels_evoked(
                #     evoked, peak_times, ch_types=None, n_chan=10)
                # print('\nGM Peak channels for \n %s | %s | %s' %
                #       (cond, tt, freq))
                # for chn in peak_ch_names:
                #     print(chn)
                # print('\n')

    #         # the following cannot use grand_average() because channel
    #         # names differ across subjects
    #         # channel names can also differ across frequencies
    #         # therefore separate files for frequencies

    #         for tt in evo_types:

    #             for freq in freqs:

    #                 # Evokeds to average
    #                 evokeds = psd_evo[tt][cond][freq]

    #                 # grand-average across subjects
    #                 evoked =\
    #                     Ff.grand_average_evoked_arrays(evokeds)

    #                 fname_evo = op.join(
    #                     sbj_path, 'AVE', 'GM_%s_%s_%s-ave.fif' %
    #                     (tt, cond, freq))

    #                 print('Writing GM to %s.' % fname_evo)

    #                 mne.write_evokeds(fname=fname_evo, evoked=evoked)

    #         # Group Statistics for electrode ROIs
    #         # Channel group pairs to compare:
    #         group_pairs = {'EEG': ['OT_L', 'OT_R'],
    #                        'Grad': ['Grad Left-occipital',
    #                                 'Grad Right-occipital'],
    #                        'Mag': ['Mag Left-occipital',
    #                                'Mag Right-occipital']}
    #         for freq in freqs:

    #             for stim in ['base', 'odd']:

    #                 print('\nLaterality statistics for %s.' % stim)

    #                 # get lateralization values per participant and channel
    #                 # type for correlation among channel types
    #                 lat = {}

    #                 for ct in group_pairs:
    #                     print(ct)

    #                     g1, g2 = group_pairs[ct][0], group_pairs[ct][1]
    #                     data1 = roi_chans_rms[g1][cond][freq][stim]
    #                     data2 = roi_chans_rms[g2][cond][freq][stim]

    #                     # Two-sided t-test
    #                     stat, pv = ttest_rel(data1, data2)

    #                     print('T-test for %s L-R, %s | %s (%f vs %f).' %
    #                           (ct, cond, freq, data1.mean(), data2.mean()))
    #                     # p-value for one-sided test justified here
    #                     print('%f, %f\n' % (stat, pv / 2.))

    #                     lat[ct] = data1 - data2  # lateralization per Ppt

    #                 # Correlation among sensor types
    #                 print('\nCorrelation of lateralization with confidence'
    #                       'intervals:')
    #                 print('EEG vs Mags')
    #                 # corrs = np.corrcoef(lat['EEG'], lat['Mag'])
    #                 # print(corrs)
    #                 r, p, lo, hi = Ff.pearsonr_ci(lat['EEG'], lat['Mag'])
    #                 print('EEG vs Mags: %f, %f, %f\n' % (r, lo, hi))
    #                 r, p, lo, hi = Ff.pearsonr_ci(lat['EEG'], lat['Grad'])
    #                 print('EEG vs Grads: %f, %f, %f\n' % (r, lo, hi))
    #                 r, p, lo, hi = Ff.pearsonr_ci(lat['Grad'], lat['Mag'])
    #                 print('Grads vs Mags: %f, %f, %f\n' % (r, lo, hi))
    #                 # corrs = np.corrcoef(lat['EEG'], lat['Grad'])
    #                 # print(corrs)
    #                 # print('Grads vs Mags')
    #                 # corrs = np.corrcoef(lat['Grad'], lat['Mag'])
    #                 # print(corrs)

    #         # plot peak amplitudes across individual participants
    #         for stim in ['base', 'odd']:

    #             for freq in freqs:

    #                 type_now = 'harm_%s_peak_%s' % (stim, stim)

    #                 evokeds = deepcopy(psd_evo[type_now][cond][freq])

    #                 # get amplitudes at centre frequency per channel type
    #                 amps = get_amps_channel_types(evokeds)

    #                 for ch_type in amps.keys():

    #                     fig, ax = plt.subplots()

    #                     n = len(amps[ch_type])

    #                     x = np.arange(1, n + 1)

    #                     ax.bar(x, amps[ch_type])

    #                     threshold = 1.96
    #                     ax.plot([0., n], [threshold, threshold], "k--")

    #                     # make sure figures is draws before get_yticklabels
    #                     fig.canvas.draw()

    #                     ytl = ax.get_yticklabels()

    #                     # # ax.set_xlabel(fontsize=24)
    #                     # # ax.set_ylabel(fontsize=24)
    #                     ax.set_xticks(x)
    #                     xtl = [str(y) for y in x]
    #                     ax.set_xticklabels(labels=xtl,
    #                                        fontdict={'fontsize': 14})

    #                     ax.set_yticklabels(labels=ytl,
    #                                        fontdict={'fontsize': 14})

    #                     # output directory for figures
    #                     figs_path = op.join(
    #                         config.grandmean_path, 'Figures_ICA')

    #                     fig_fname = op.join(
    #                         figs_path, 'face_amps_indiv_%s_%s.jpg' % (stim, ch_type))

    #                     print('Saving figure to %s.' % fig_fname)

    #                     fig.savefig(fig_fname)

    #                     plt.close(fig)

    #                 # put amplitudes into list of lists for correlation
    #                 amps_list = [amps['eeg'], amps['grad'], amps['mag']]

    #                 print('Condition: %s.' % stim)

    #                 print('Correlations of peak amplitudes between channel'
    #                       ' types across participants:')
    #                 corrs = np.corrcoef(amps_list)
    #                 print(corrs)

    #                 print('Correlation confidence intervals:')
    #                 r, p, lo, hi = Ff.pearsonr_ci(amps['eeg'], amps['grad'])
    #                 print('EEG vs Grads: %f, %f\n' % (lo, hi))
    #                 r, p, lo, hi = Ff.pearsonr_ci(amps['eeg'], amps['mag'])
    #                 print('EEG vs Mags: %f, %f\n' % (lo, hi))
    #                 r, p, lo, hi = Ff.pearsonr_ci(amps['mag'], amps['grad'])
    #                 print('Mags vs Grads: %f, %f\n' % (lo, hi))

    return


def get_amps_channel_types(evokeds):
    """Extract RMS amplitudes across channels per channel type at latency 0.

    Parameters:
    evokeds: list of instances of Evoked
        Evokeds objects to extract amplitudes from, e.g. for peak channels
        per channel type.

    Returns:
    amps: dict of list
        Dictionary ('mag'/'grad'/'eeg') with lists (amplitudes per evoked).
    """

    ch_types = ["mag", "grad", "eeg"]

    amps = {"mag": [], "grad": [], "eeg": []}

    for evoked in evokeds:
        for ch_type in ch_types:
            if ch_type in ["mag", "grad"]:
                meg = ch_type
                eeg = False

            else:
                eeg = True
                meg = False

            evo = deepcopy(evoked)

            evo.pick_types(meg=meg, eeg=eeg, eog=False, ecg=False)

            idx0 = evo.time_as_index(0.0)

            rms = np.sqrt((evo.data[:, idx0] ** 2).mean())

            amps[ch_type].append(rms)

    return amps


def grand_average_conditions_data(data, evokeds, ch_names):
    """Average data arrays (e.g. peak channels) across subjects per condition.

    Parameters:
    data: dictionary of dictionary of lists of lists of numpy 2D arrays
        Dictionary contains conditions (e.g. "faces", "hflf"), then labels
        ('base'|'odd') then lists for all subjects and all sweep
        frequencies. The actual data are in numpy 2D (n_ch x n_t) arrays.
    evokeds: list of instances of Evoked
        Evokeds objects from which data was derived. For example, data may
        contain the data for peak channels extracted from evokeds.
    ch_names: list of str
        The instances of Evoked in evokes will be reduced to the channels
        specified in ch_names. The number of channels must be n_ch.

    Returns:
    gm_evokeds: dictionary of instances of Evoked
        Dictionary with conditions as in data, with data averaged across
        list items. The grand-averages are returned as
        gm_evoked[cond][sbj][freq].data where evoked is copied from last
        instance of evoked in evokeds per condition and frequency.
        Evokeds then contains appropriate info except for channel names (which
        will probably differ across subjects).
    """
    gm_evokeds = {}  # for instances of Evoked

    conds = data.keys()  # conditions in cond_evo

    # Slight complication: need to rearrange data in order to average

    datas = {}  # will collect data per condition, frequency, then subject
    for cond in conds:
        n_sbjs = len(data[cond])  # number of subjects

        # 'base'|'odd' depending on channel groups
        labels = data[cond].keys()

        gm_evokeds[cond] = {}  # Evokeds grand-averaged as Evoked
        datas[cond] = {}

        for lab in labels:
            gm_evokeds[cond][lab] = {}  # Evokeds grand-averaged as Evoked
            datas[cond][lab] = {}  # dict easier to handle below than list

        for ss in np.arange(n_sbjs):
            n_freqs = len(data[cond][list(labels)[0]][ss])

            for ff in np.arange(n_freqs):
                for lab in labels:
                    # data array for one frequency
                    data_freq = data[cond][lab][ss][ff]

                    if ss == 0:  # first time, initialise
                        datas[cond][lab][ff] = []

                    datas[cond][lab][ff].append(data_freq)

        # Grand-averaging across subjects
        for ff in np.arange(n_freqs):
            for lab in labels:
                # use last Evoked as template for this condition
                # the full evoked data with all channels doesn't contain labels
                # for channel groups
                evoked = deepcopy(evokeds[cond][-1][ff])

                # pick channel selection
                evoked.pick_channels(ch_names)

                # use as index, more informative later
                ff_str = evoked.comment

                # mean across list items
                gm_data = np.mean(datas[cond][lab][ff], axis=0)

                gm_evokeds[cond][lab][ff_str] = evoked

                # put peak channel data into Evoked object
                gm_evokeds[cond][lab][ff_str].data = gm_data

    return gm_evokeds


def grand_average_conditions_evo(evos):
    """Average evokeds across subjects per condition.

    Parameters:
    evos: dictionary of dictionary of lists
        Dictionary contains conditions (e.g. "faces", "hflf"), then lists that
        contain data for all subjects.

    Returns:
    gm_evo: dictionary
        Dictionary with conditions as in cond_evo, with data averaged across
        list items.
    """
    gm_evo = {}

    conds = evos.keys()  # conditions in cond_evo

    # Slight complication: need to rearrange Evoked object in order to use
    # mne.grand_average()

    evokeds = {}  # will collect Evoked per condition, frequency, then subject
    for cond in conds:
        n_sbjs = len(evos[cond])  # number of subjects

        evokeds[cond] = {}  # Evokeds per subject
        gm_evo[cond] = {}  # Evokeds grand-averaged

        for ss in np.arange(n_sbjs):
            n_freqs = len(evos[cond][ss])

            for ff in np.arange(n_freqs):
                # Evoked object for one frequency
                evo_freq = evos[cond][ss][ff]

                if ss == 0:  # first time, initialise
                    evokeds[cond][evo_freq.comment] = []

                evokeds[cond][evo_freq.comment].append(evo_freq)

        # Grand-averaging across subjects

        for ff in evokeds[cond].keys():
            gm_evo[cond][ff] = mne.grand_average(
                evokeds[cond][ff], interpolate_bads=True
            )

    return gm_evo


# get all input arguments except first
# if number not in config list, do all of them
if (len(sys.argv) == 1) or (
    int(sys.argv[1]) > np.max(list(config.map_subjects.keys()))
):
    # IDs don't start at 0
    sbj_ids_all = config.do_subjs

else:
    # get list of subjects IDs to process
    sbj_ids_all = [int(aa) for aa in sys.argv[1:]]

# # EDIT
# sbj_ids = sbj_ids[:2]

# requires all subjects to average across
grand_average_psds(sbj_ids_all)
