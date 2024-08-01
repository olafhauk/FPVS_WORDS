#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Group stats via cluster-based permutation tests for summed harmonics
==========================================

OH, Mar 2024
"""

import sys

from os import path as op

import numpy as np

from scipy.stats import ttest_rel

from matplotlib import pyplot as plt

import pandas as pd

from copy import deepcopy

from importlib import reload

from scipy import stats as stats

import mne
from mne.report import Report
from mne.source_estimate import SourceEstimate
from mne.evoked import EvokedArray
from mne.channels import find_ch_adjacency
from mne.stats import (
    permutation_t_test,
    spatio_temporal_cluster_1samp_test,
    summarize_clusters_stc,
)


# from FPVS_functions import grand_average_evoked_arrays, peak_channels_evoked

import config_fpvswords as config

reload(config)

import FPVS_functions as Ff

reload(Ff)


print(mne.__version__)


figs_dir = "Figures"
suffix = 'pdf'  # suffix for figure filenames

# p-treshold for evoked
p_thresh_evo = 0.05

# p-thresholds for sources
p_threshold_stc = 0.00001
cluster_p_thresh_stc = 0.05

# for some plots of SNRs
unit_scalings = dict(eeg=1.0, mag=1.0, grad=1.0)

# conditions
# conds = config.do_conds
conds = {
    # "face": ["face"],
    "english": ["cn", "nw", "pw"],
    # "french": ["cn", "nw", "pw"],
    # "slow": ["cn", "nw", "pw"],
    # "extraslow": ["pw"],
    # "extrafast": ["pw"]
}

resps = ["odd", "base"]  # currently only works if both specified
modals = ["evo", "stc"]  # ["evo", "stc"]

# Labels for ROI analysis
subjects_dir = config.subjects_dir

sens_types = ["eeg", 'grad', 'mag']

subject = "fsaverage"
src_fname = op.join(
    subjects_dir,
    subject,
    "bem",
    subject + "-ico-" + str(config.src_spacing) + "-src.fif",
)
print("Source space from: %s" % src_fname)
src = mne.read_source_spaces(src_fname)


# Read BEHAVIOURAL DATA
dframe_all = pd.read_excel(config.behav_data_fname, '4Pandas')

# choose only good subjects
all_subjs = dframe_all['Subject']


# variables for correlations
# vars_corr = ['IRSTTimeEng', 'BNTTimeEng', 'MillTimeEng', 'IRSTErrEng', 'BNTAccEng', 'MillAccEng', 'LEAPCompEng', 'LEAPSpeakEng', 'LEAPReadEng']
vars_corr = ['VerbFluIndex', 'IRSTIndex', 'BNTIndex', 'MillIndex'] 

def grand_cluster_permutation_psds(sbj_ids_all):
    """Group stats via cluster-based permutation."""

    print("Using subjects:")
    print(*sbj_ids_all)

    # report = Report(subject='GM', title='FPVS PSDs GM')

    n_permutations = 10000

    # adjacency_mag, ch_names_mag = find_ch_adjacency(info, ch_type="mag")
    # adjacency_gra, ch_names_gra = find_ch_adjacency(info, ch_type="grad")
    # adjacency_eeg, ch_names_eeg = find_ch_adjacency(info, ch_type="eeg")

    for cond in conds:
        if 'extra' in cond:
            sbj_ids = Ff.remove_subjects_extras(sbj_ids_all.copy())
        else:
            sbj_ids = sbj_ids_all
        # need to read data each time because of extra condition
        cond_now = {cond: conds[cond]}
        psds, inst_ori = read_subjects_data(sbj_ids, cond_now, resps, modals)
        
        # get relevant behavioural data
        good_subjs = [np.where(x == all_subjs)[0][0] for x in sbj_ids]
        dframe = dframe_all.copy().iloc[good_subjs]

        for ev_type in conds[cond]:
            for resp in resps:
                if "evo" in psds[cond][ev_type][resp].keys():
                    modal = "evo"

                    for sens_type in sens_types:
                        if sens_type == "eeg":
                            meg, eeg = False, True
                        else:
                            meg, eeg = sens_type, False

                        evoked = deepcopy(inst_ori)

                        picks = mne.pick_types(
                            evoked.info,
                            meg=meg,
                            eeg=eeg,
                            stim=False,
                            eog=True,
                            exclude="bads",
                        )
                        evoked.pick(picks=picks)

                        info = evoked.info

                        X = get_data_matrix(psds, cond, ev_type, resp, modal, picks)

                        T0, p_values, H0 = permutation_t_test(
                            X, n_permutations, n_jobs=None
                        )

                        significant_sensors = picks[p_values <= p_thresh_evo]
                        significant_sensors_names = [
                            inst_ori.info["ch_names"][k] for k in significant_sensors
                        ]

                        print(
                            "Number of significant sensors : %d"
                            % len(significant_sensors)
                        )
                        print("Sensors names : %s" % significant_sensors_names)

                        del evoked
                        evoked = mne.grand_average(psds[cond][ev_type][resp][modal])
                        evoked.pick(picks=picks)
                        evoked.crop(tmin=0.0, tmax=0.0)

                        # evoked = mne.EvokedArray(
                        #     -np.log10(p_values)[:, np.newaxis], info, tmin=0.0
                        # )

                        mask = p_values[:, np.newaxis] <= p_thresh_evo

                        print(cond, ev_type, resp, modal)
                        fig = evoked.plot_topomap(
                            ch_type=sens_type,
                            times=[0],
                            scalings=1,
                            time_format=None,
                            cmap="Reds",
                            vlim=(0.0, np.max),
                            units="-log10(p)",
                            cbar_fmt="-%0.1f",
                            mask=mask,
                            size=3,
                            show_names=False,
                            time_unit="s",
                        )

                        fname = "topomap_%s_%s_%s_%s.%s" % (
                            cond,
                            ev_type,
                            resp,
                            sens_type,
                            suffix
                        )
                        fig_fname = op.join(config.grandmean_path, "Figures", fname)
                        plt.savefig(fig_fname, transparent=True)
                        plt.close()

                        # BEHAVIOURAL CORRELATIONS
                        print('Behavioural Correlations.')
                        for beh_var in vars_corr:
                            corr_stats = [stats.pearsonr(X[:,i], dframe[beh_var]) for i in np.arange(X.shape[1])]
                            corrs, p_values = [], []
                            for cc in corr_stats:
                                corrs.append(cc[0])
                                p_values.append(cc[1])

                            evoked.data[:, -1] = np.array(corrs)
                            mask = np.array(p_values)[:, np.newaxis] <= p_thresh_evo

                            fig = evoked.plot_topomap(
                                ch_type=sens_type,
                                times=[0],
                                scalings=1,
                                time_format=None,
                                cmap=None,
                                vlim=(-.5, .5),
                                units=None,
                                cbar_fmt="%0.1f",
                                mask=mask,
                                size=3,
                                show_names=None,
                                time_unit="s",
                            )

                            fname = "corrmap_%s_%s_%s_%s_%s.%s" % (
                                cond,
                                ev_type,
                                resp,
                                sens_type,
                                beh_var,
                                suffix
                            )
                            fig_fname = op.join(config.grandmean_path, "Figures", fname)
                            print('Saving figure to %s.' % fig_fname)
                            plt.savefig(fig_fname, transparent=True)

                            plt.close()


                if "stc" in psds[cond][ev_type][resp].keys():
                    # https://mne.tools/stable/auto_tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.html
                    modal = "stc"

                    X = get_data_matrix(psds, cond, ev_type, resp, modal, picks=None)

                    # must be subjects x time x space
                    X = np.swapaxes(X, 1, 2)

                    adjacency = mne.spatial_src_adjacency(src)

                    n_subjects = len(sbj_ids)                    
                    df = n_subjects - 1  # degrees of freedom for the test
                    # one-tailed t-test against zero
                    t_threshold = stats.distributions.t.ppf(1 - p_threshold_stc, df=df)

                    (
                        T_obs,
                        clusters,
                        cluster_p_values,
                        H0,
                    ) = clu = spatio_temporal_cluster_1samp_test(
                        X,
                        tail=1,
                        n_permutations=n_permutations,
                        adjacency=adjacency,
                        n_jobs=None,
                        threshold=t_threshold,
                        buffer_size=None,
                        verbose=True,
                    )

                    good_clusters_idx = np.where(cluster_p_values < cluster_p_thresh_stc)[0]
                    good_clusters = [clusters[idx] for idx in good_clusters_idx]

                    fsave_vertices = [s["vertno"] for s in src]

                    stc_all_cluster_vis = summarize_clusters_stc(
                        clu, tstep=1, vertices=fsave_vertices, subject="fsaverage"
                    )

                    # for plotting
                    stc_plot = deepcopy(stc_all_cluster_vis)

                    # plot the mean activation or z-scores
                    Xdata = X.squeeze().mean(axis=0)[:, np.newaxis]
                    # apply "mask"
                    stc_plot.data = stc_all_cluster_vis.data * Xdata.data
                    thresh = stc_plot.data.max()

                    title = "%s_%s_%s" % (cond, ev_type, resp)

                    for hemi in ["lh", "rh"]:
                        brain = stc_plot.plot(
                            hemi=hemi,
                            views="lateral",
                            subjects_dir=subjects_dir,
                            time_label="",
                            size=(800, 800),
                            smoothing_steps=5,
                            clim=dict(kind="value", lims=[0, 0.5 * thresh, thresh]),
                            view_layout="horizontal",
                            time_viewer=False,
                            title=title,
                        )

                        fname = "brain_%s_%s_%s_%s.%s" % (cond, ev_type, resp, hemi, suffix)
                        fig_fname = op.join(config.grandmean_path, "Figures", fname)
                        print("Saving brain to %s." % fig_fname)
                        brain.save_image(fig_fname)
                        brain.close()

                    # BEHAVIOURAL CORRELATIONS
                    print('Behavioural correlations.')
                    X = np.squeeze(X)
                    for beh_var in vars_corr:
                        corr_stats = [stats.pearsonr(X[:,i], dframe[beh_var]) for i in np.arange(X.shape[1])]
                        corrs, p_values = [], []
                        for cc in corr_stats:
                            corrs.append(cc[0])
                            p_values.append(cc[1])

                        mask = np.array(p_values)[:, np.newaxis] <= p_thresh_evo
                        stc_plot.data = mask * np.array(corrs)[:, np.newaxis]

                        title = "%s_%s_%s_%s" % (cond, ev_type, resp, beh_var)

                        for hemi in ["lh", "rh"]:
                            brain = stc_plot.plot(
                                hemi=hemi,
                                views="lateral",
                                subjects_dir=subjects_dir,
                                time_label="",
                                size=(800, 800),
                                smoothing_steps=5,
                                clim=dict(kind="value", pos_lims=[0., .25, .5]),
                                view_layout="horizontal",
                                time_viewer=False,
                                title=title,
                            )

                            fname = "brain_%s_%s_%s_%s_%s.%s" % (cond, ev_type, resp, beh_var, hemi, suffix)
                            fig_fname = op.join(config.grandmean_path, "Figures", fname)
                            print("Saving brain to %s." % fig_fname)
                            brain.save_image(fig_fname)
                            brain.close()


def get_data_matrix(psds, cond, ev_type, resp, modal, picks=None):
    """Get data matrix (X) from data objects.

    Parameters:
    psds: dict of dict of dict of dict of list
        Data objects for subjects in sbj_ids.
        Output from read_subjects_data().
        [cond][ev_type]['base'/'odd']['evo'/'stc'][sbj]
    picks: array of int
        Indices of channels to use. Only for Evoked objects.
    cond: string
        The condition (e.g. 'english').
    ev_type: string
        The event type (e.g. 'pw').
    resp: string
        The frequency response type ('base' or 'odd').
    modal: string
        The data modality ('evo' or 'stc').

    Returns:
    X: numpy array
        The data matrix for cluster-based permutation tests.
        Dimension n_subjects x n_channels.
    """
    X = []
    if modal == "evo":
        for evo_ori in psds[cond][ev_type][resp][modal]:
            evo = deepcopy(evo_ori)
            evo.pick(picks=picks)
            # index to time point 0, which will be plotted
            idx0 = np.abs(evo.times).argmin()
            X.append(evo.data[:, idx0])
    elif modal == "stc":
        for stc_ori in psds[cond][ev_type][resp][modal]:
            stc = deepcopy(stc_ori)
            X.append(stc.data)

    X = np.array(X)

    return X


def read_subjects_data(sbj_ids, conds, resps, modals):
    """Read sensor and source space data for summed harmonics.

    Parameters:
    sbj_ids: list of int
        The IDs of subject data to be read.
    conds: dict
        Conditions ('english' etc.) and event types ('pw', 'cn' etc.).
        For example {'english': ['cn', 'nw', 'cn']).
    resps: list of string
        The frequency response types ('base' or 'odd').
    modals: list of string
        The data modalities ('evo' or 'stc').

    Returns:
    psds: dict of dict of dict of dict of list
        Data objects for subjects in sbj_ids.
        [cond][ev_type]['base'/'odd']['evo'/'stc'][sbj]
    inst: instance of data object
        The first data object (for future use of info etc.).
    """
    psds = {}  # dict for all relevant data objects
    for cond in conds:
        psds[cond] = {}
        for ev_type in conds[cond]:
            psds[cond][ev_type] = {}
            for resp in resps:
                psds[cond][ev_type][resp] = {}
                for modal in modals:
                    psds[cond][ev_type][resp][modal] = []

    inst = []
    for cond in conds:  # conditions
        for ev_type in conds[cond]:
            print("Condition: %s, %s." % (cond, ev_type))

            for [ss, sbj_id] in enumerate(sbj_ids):  # across all subjects
                # path to subject's data
                sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

                print("Reading PSD results from evoked files:")

                # fname_evo = op.join(sbj_path, 'AVE', 'HarmBaseEpos_%s%s%s' %
                #                     (cond, freq, '-ave.fif'))

                if "evo" in modals:
                    # Oddball topography of z-scored summed harmonics at centre
                    # frequency:
                    fname_evo = op.join(
                        sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
                    )
                    print(fname_evo)
                    psd_sum_odd = mne.read_evokeds(fname_evo, 0)
                    if inst == []:
                        inst = psd_sum_odd

                    # Base topography of z-scored summed harmonics at centre
                    # frequency:
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_sum_base = mne.read_evokeds(fname_evo, 0)

                    psds[cond][ev_type]["base"]["evo"].append(psd_sum_base)
                    psds[cond][ev_type]["odd"]["evo"].append(psd_sum_odd)

                ### STC
                if "stc" in modals:
                    fname_stc = op.join(
                        sbj_path,
                        "STC",
                        "PSDZSumTopoOdd_%s_%s_mph-lh.stc" % (cond, ev_type),
                    )
                    print(fname_stc)
                    stc_odd = mne.read_source_estimate(fname_stc)

                    if inst == []:
                        inst = stc_odd

                    fname_stc = op.join(
                        sbj_path,
                        "STC",
                        "PSDZSumTopoBase_%s_%s_mph-lh.stc" % (cond, ev_type),
                    )
                    print(fname_stc)
                    stc_base = mne.read_source_estimate(fname_stc)

                    psds[cond][ev_type]["base"]["stc"].append(stc_base)
                    psds[cond][ev_type]["odd"]["stc"].append(stc_odd)

    return psds, inst


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

# requires all subjects to average across
grand_cluster_permutation_psds(sbj_ids_all)
