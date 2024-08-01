### plot some figures for individual subjects for debugging

from os import path as op

import numpy as np

from scipy import linalg
from scipy.signal import find_peaks

from matplotlib import pyplot as plt

from importlib import reload

import config_fpvswords as config

reload(config)

import mne

# from mne.report import Report

plt.ion()

subj = "GM"
sbj_path = config.grandmean_path

fig_path = "/imaging/hauk/users/olaf/FPVS2/MEG/Figures"

conds = ["english"]
# conds = ["slow", "extraslow", "extrafast"]
ev_types = ["cn", "nw", "pw"]
# ev_types = ["pw"]  # extra only has pw

# for topoplots in plot_joint()
timess = {
    "face": [1.2, 2.4, 3.6, 4.8],  # face, odd,
    "english": [2, 4, 6, 8],  # english, odd
    "slow": [1, 2, 3],  # slow, odd
    "extraslow": [1, 2, 3],  # slow, odd
    "extrafast": [2, 4, 6, 8],  # slow, odd
}

# for base topomap
times_maps = {
    "face": [6],  # face, base
    "english": [10],  # english, base
    "slow": [4],  # slow, base
    "extraslow": [4],
    "extrafast": [10],
}

f_range = (0.5, 11)  # frequency range to plot for PSDs

scalings = dict(eeg=1, grad=1, mag=1)  # change units for plots
ts_args = dict(scalings=scalings)  # for plot_joint()

# type of figure to save
fig_type = 'pdf'

for cond in conds:

    times = timess[cond]
    times_map = times_maps[cond]

    for ev_type in ev_types:

        # PSD (z-scored):
        fname_evo = op.join(sbj_path, "AVE", "PSDZ_%s_%s%s" % (cond, ev_type, "-ave.fif"))
        print(fname_evo)
        psd_z = mne.read_evokeds(fname_evo)[0].crop(f_range[0], f_range[1])

        fig = psd_z.plot_joint(times=times, ts_args=ts_args)

        for [sens, ff] in zip(["eeg", "mag", "grad"], fig):
            fname_fig = op.join(fig_path, "GM_PSDZ_%s_%s_%s.%s" % (cond, ev_type, sens, fig_type))
            ff.savefig(fname_fig, transparent=True)

        # evoked = mne.read_evokeds(
        #     op.join(sbj_path, "AVE", "%s_f_%s%s" % (cond, ev_type, "_nch-ave.fif")), 0
        # )
        # evoked.plot_joint()

        ### Summed spectra

        # Base:
        fname_evo = op.join(sbj_path, "AVE", "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif"))
        harmbase = mne.read_evokeds(fname_evo)[0]

        # Summed epochs, Odd:
        fname_evo = op.join(sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif"))
        harmodd = mne.read_evokeds(fname_evo)[0]

        figbase = harmbase.plot(spatial_colors=True, scalings=scalings)

        fname_fig = op.join(fig_path, "GM_HarmBase_%s_%s.%s" % (cond, ev_type, fig_type))
        figbase.savefig(fname_fig, transparent=True)

        figodd = harmodd.plot(spatial_colors=True, scalings=scalings)
        fname_fig = op.join(fig_path, "GM_HarmOdd_%s_%s.%s" % (cond, ev_type, fig_type))
        figodd.savefig(fname_fig, transparent=True)

        times_map = 0.0
        for sens in ["eeg", "grad", "mag"]:
            fig = harmbase.plot_topomap(times=times_map, scalings=scalings, ch_type=sens)
            fname_fig = op.join(fig_path, "GM_HarmBase_%s_%s_%s.%s" % (cond, ev_type, sens, fig_type))
            fig.savefig(fname_fig, transparent=True)

        for sens in ["eeg", "grad", "mag"]:
            fig = harmodd.plot_topomap(times=times_map, scalings=scalings, ch_type=sens)
            fname_fig = op.join(fig_path, "GM_HarmOdd_%s_%s_%s.%s" % (cond, ev_type, sens, fig_type))
            fig.savefig(fname_fig, transparent=True)


        # individual topomaps
        fname_evo = op.join(sbj_path, "AVE", "GM_psd_sum_base_indiv_topos_%s_%s-ave.fif" % (cond, ev_type))
        print(fname_evo)
        evo_indiv = mne.read_evokeds(fname_evo, 0)
        times_topo = evo_indiv.times
        for sens in ["eeg", "grad", "mag"]:
            fig = evo_indiv.plot_topomap(times=times_topo, scalings=scalings, ch_type=sens, time_format='')
            fname_fig = op.join(fig_path, "GM_psd_sum_base_indiv_topos_%s_%s_%s.%s" % (cond, ev_type, sens, fig_type))
            fig.savefig(fname_fig, transparent=True)

        fname_evo = op.join(sbj_path, "AVE", "GM_psd_sum_odd_indiv_topos_%s_%s-ave.fif" % (cond, ev_type))
        print(fname_evo)
        evo_indiv = mne.read_evokeds(fname_evo, 0)
        times_topo = evo_indiv.times
        for sens in ["eeg", "grad", "mag"]:
            fig = evo_indiv.plot_topomap(times=times_topo, scalings=scalings, ch_type=sens, time_format='')
            fname_fig = op.join(fig_path, "GM_psd_sum_odd_indiv_topos_%s_%s_%s.%s" % (cond, ev_type, sens, fig_type))
            fig.savefig(fname_fig, transparent=True)


# fig = psd_z.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))
# fig.suptitle(str(sbj_id) + ': ' + subj)

# fig = harmbase.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))
# fig.suptitle(str(sbj_id) + ': ' + subj)

# figodd = harmodd.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))
# figodd.suptitle(str(sbj_id) + ": " + subj)

# figbase = harmbase.plot(
#     spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1)
# )
# figbase.suptitle(str(sbj_id) + ": " + subj)

# mne.viz.plot_raw_psd()

# # With topo plots
# ts_args = dict(scalings=dict(eeg=1, grad=1, mag=1))
# # psd_z.plot_joint(times=[1., 2., 3., 8.], ts_args=ts_args)
# # psd_z.plot_joint(times=[2., 4., 6., 20.], ts_args=ts_args)
# # psd_z.plot_joint(times=[1.2, 2.4, 3.6, 12.], ts_args=ts_args)
# fig = harmbase.plot_joint(times=[-0.0166, 0.], picks='grad', ts_args=ts_args)

# fig = psd_z.plot(spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1))

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
# harmodd.plot_topomap(times=[0.0], scalings=dict(eeg=1, grad=1, mag=1), ch_type="eeg")
# harmodd.plot_topomap(times=[0.0], scalings=dict(eeg=1, grad=1, mag=1), ch_type="grad")
# harmodd.plot_topomap(times=[0.0], scalings=dict(eeg=1, grad=1, mag=1), ch_type="mag")
