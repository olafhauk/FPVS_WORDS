# plot some figures for individual subjects for debugging

import mne
from os import path as op

import numpy as np

from scipy import linalg
from scipy.signal import find_peaks

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from importlib import reload

import config_fpvswords as config

reload(config)


# from mne.report import Report

plt.ion()

# sbj_ids = config.do_subjs
# sbj_ids = [99]
# sbj_ids = [99]  # GM

# "bad ones"
# sbj_ids = [1, 10, 7, 12, 13, 20, 23, 24, 29]

# All
sbj_ids = [
    1,
    2,
    3,
    5,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
]

# sbj_ids = [29]

# 99 is GM

fig_path = "/imaging/hauk/users/olaf/FPVS2/MEG/Figures"

# conds = ['english', 'slow']
# ev_types = ['nw', 'nw']

# ATM only does one condition and one event type at a time
conds = ["english"]
ev_types = ["cn"]

# frequency range to plot
# crop = (0, 13)  # Faces
crop = (0, 21)  # Words
# crop = (0, 9)  # Slow

scalings = None  # dict(eeg=1, grad=1, mag=1)

fname = op.join(fig_path, "PSD_%s_%s_indiv.pdf" % (conds[0], ev_types[0]))
with PdfPages(fname) as pdf:

    for [ss, sbj_id] in enumerate(sbj_ids):  # across all subjects
        for [cond, ev_type] in zip(conds, ev_types):
            # path to subject's data
            if sbj_id == 99:  # GM
                subj = "GM"
                sbj_path = config.grandmean_path
            else:
                subj = config.map_subjects[sbj_id][0]
                sbj_path = op.join(config.data_path, subj)

            # PSD (non-normalised):
            fname_evo =\
                op.join(sbj_path, 'AVE', 'PSD_%s_%s%s' %
                        (cond, ev_type, '-ave.fif'))
            print(fname_evo)
            psd = mne.read_evokeds(fname_evo)[0].crop(crop[0], crop[1])

            # # PSD (z-scored):
            # fname_evo =\
            #     op.join(sbj_path, 'AVE', 'PSDZ_%s_%s%s' %
            #             (cond, ev_type, '-ave.fif'))
            # print(fname_evo)
            # psd_z = mne.read_evokeds(fname_evo)[0].crop(crop[0], crop[1])

            # # Summed epochs, Base:
            # fname_evo = op.join(
            #     sbj_path, "AVE", "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif")
            # )
            # print(fname_evo)
            # harmbase = mne.read_evokeds(fname_evo)[0]

            # # Summed epochs, Odd:
            # fname_evo = op.join(
            #     sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
            # )
            # print(fname_evo)
            # harmodd = mne.read_evokeds(fname_evo)[0]

            # # fname_raw =\
            # #     op.join(sbj_path, 'rawavg_%s_%s.fif' % (cond, ev_type))
            # # raw = mne.io.read_raw_fif(fname_raw).pick_types(meg=True, eeg=True, eog=False, ecg=False,
            # #                                                 stim=False, misc=False, chpi=False)

            # # Inverse Operator
            # subject = config.mri_subjects[sbj_id]
            # fname_inv =\
            #     op.join(sbj_path, '%s_EEGMEG-inv.fif' % subject)
            # inv = mne.minimum_norm.read_inverse_operator(fname_inv)

            # fname_fwd =\
            #     op.join(sbj_path, '%s_EEGMEG-fwd.fif' % subject)
            # fwd = mne.read_forward_solution(fname_fwd)

            # # compute inverse kernel
            # K = mne.minimum_norm.resolution_matrix._get_matrix_from_inverse_operator(inv, fwd, method='MNE')

            # Covariance matrix (with info from rest)
            # fname_rest =\
            #     op.join(sbj_path, 'rest1_sss_f_ica_raw.fif')
            # fname_rest =\
            #     op.join(sbj_path, 'rest1_sss_raw.fif')
            # print(fname_rest)
            # rawrest = mne.io.read_raw_fif(fname_rest)

            # fname_cov =\
            #     op.join(sbj_path, 'rest_sss_f_raw_ica-cov.fif')
            # print(fname_cov)
            # cov = mne.read_cov(fname_cov)

            # # SVDs
            # Ui, si, Vi = linalg.svd(K, full_matrices=False)
            # Uf, sf, Vf = linalg.svd(fwd['sol']['data'], full_matrices=False)

            # fig1 = plt.figure()
            # plt.plot(si)[0]
            # fig1.suptitle(str(sbj_id) + ': ' + subj)
            # fig2 = plt.figure()
            # plt.plot(sf)[0]
            # fig2.suptitle(str(sbj_id) + ': ' + subj)

            # print('Largest s forward: %d %f' % (sbj_id, sf[0]))

            # #### INSPECT TRIGGERS
            # fname_raw =\
            #     op.join(sbj_path, 'rawavg_%s_%s%s' % (cond, ev_type, '_0.fif'))
            # rawstim = mne.io.read_raw_fif(fname_raw).pick_types(
            #     meg=False, eeg=False, eog=False, ecg=False, stim=True, misc=False, chpi=False)
            # x = rawstim.get_data()[2, :]  # trigger value 4
            # p = find_peaks(x, height=(0, 7), width=(1,3))[0]  # peaks of certain triggers with certain width
            # diffs = []
            # all_diffs = []
            # for i in np.arange(len(p)-1):
            #     diff = p[i+1]-p[i]
            #     all_diffs.append(diff)
            #     if diff not in [166, 167, 332, 333, 334]:
            #         diffs.append(diff)

            # print('\n\n')
            # print(diffs)
            # print('\n\n')
            # print(np.mean(all_diffs))

            # fig = mne.viz.plot_cov(cov, rawrest.info)
            # for f in fig:
            #     f.suptitle(str(sbj_id) + ': ' + subj)

            # fig = psd.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))
            # fig.suptitle(str(sbj_id) + ': ' + subj)

            # fig = psd_z.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))
            # fig.suptitle(str(sbj_id) + ': ' + subj)

            # fig = harmbase.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))
            # fig.suptitle(str(sbj_id) + ': ' + subj)

            # figodd = harmodd.plot(spatial_colors=True,
            #                       scalings=dict(eeg=1, grad=1, mag=1))
            # figodd.suptitle(str(sbj_id) + ": " + subj)

            # figbase = harmbase.plot(
            #     spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1)
            # )
            # figbase.suptitle(str(sbj_id) + ": " + subj)

            # mne.viz.plot_raw_psd()

            # fig = harmbase.plot_joint(times=[-0.0166, 0.], picks='grad', ts_args=ts_args)

            # psd.crop(tmin=0., tmax=21.)
            psd.crop(tmin=0.5, tmax=21.)

            # # With topo plots
            ts_args = dict(scalings=scalings)
            # fig = psd_z.plot_joint(times=[1., 2., 3., 8.], ts_args=ts_args)
            fig = psd.plot_joint(times=[2., 4., 6., 20.], ts_args=ts_args)
            # fig3 = psd_z.plot_joint(times=[1.2, 2.4, 3.6, 12.], ts_args=ts_args)

            for i in [0, 1, 2]:
                fname = op.join(fig_path, "PSDtopo_%s_%s_%s_%s.jpg" % (str(sbj_id),
                                                                       cond, ev_type, str(i)))
                fig[i].savefig(fname)

            fig = psd.plot(spatial_colors=True,
                           scalings=scalings)
            fig.suptitle('PSD, %s %s %s' % (cond, ev_type, str(
                sbj_id)), fontweight='bold', fontsize=14)

            fname = op.join(fig_path, "PSD_%s_%s_%s.jpg" % (str(sbj_id),
                                                            cond, ev_type))

            fig.savefig(fname)

            pdf.savefig(fig)

            plt.close('all')

            # fname = "/imaging/hauk/users/olaf/FPVS2/MEG/Figures/%s_%s_harmodd_%s.jpg" % (
            #     cond,
            #     ev_type,
            #     str(sbj_id),
            # )
            # figodd.savefig(fname)

            # fname = "/imaging/hauk/users/olaf/FPVS2/MEG/Figures/%s_%s_harmbase_%s.jpg" % (
            #     cond,
            #     ev_type,
            #     str(sbj_id),
            # )
            # figbase.savefig(fname)
            # plt.close(fig)

# # summed topos Base
# harmbase.plot_topomap(times=[0.0], scalings=dict(eeg=1, grad=1, mag=1), ch_type="eeg")
# harmbase.plot_topomap(times=[0.0], scalings=dict(eeg=1, grad=1, mag=1), ch_type="grad")
# harmbase.plot_topomap(times=[0.0], scalings=dict(eeg=1, grad=1, mag=1), ch_type="mag")
# # summed topos Odd
# harmodd.plot_topomap(times=[0.0], scalings=dict(
#     eeg=1, grad=1, mag=1), ch_type="eeg")
# harmodd.plot_topomap(times=[0.0], scalings=dict(
#     eeg=1, grad=1, mag=1), ch_type="grad")
# harmodd.plot_topomap(times=[0.0], scalings=dict(
#     eeg=1, grad=1, mag=1), ch_type="mag")
