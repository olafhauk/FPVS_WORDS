#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Plot results of FPVS Frequency Sweep, produced by FPVS_PSD_sweep_compute.py.

Plots all conditions in sensor space, only face condition in source space.
==========================================

OH, October 2019
added source space April 2020
"""

### NOT DONE YET

import sys

import os
from os import path as op

import numpy as np
import scipy.io  # for exporting to Matlab

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from mayavi import mlab

mlab.options.offscreen = True

import matplotlib

matplotlib.use("Agg")  #  for running graphics on cluster ### EDIT

from matplotlib import pyplot as plt

from mayavi import mlab

mlab.options.offscreen = True

from importlib import reload

import mne
from mne.report import Report

import config_fpvswords as config

reload(config)

import FPVS_functions as Ff

reload(Ff)


print(mne.__version__)

# perform TFR of raw data or not
# do_tfr = config.do_tfr

close_fig = 1  # close figures only if close_fig==1

# plt.ion() # interactive plotting

# for some plots of SNRs
unit_scalings = dict(eeg=1.0, mag=1.0, grad=1.0)

# conditions
# conds = ['face', 'pwhf', 'pwlf', 'lfhf']
conds = config.do_conds


def run_PSD_plot(sbj_id):
    """Compute spectra for one subject."""
    # initialise html report for one subject
    report = Report(subject=str(sbj_id), title="FPVS PSDs")

    # # for STC plotting
    # subject = config.mri_subjects[sbj_id]

    # path to subject's data
    sbj_dir = config.map_subjects[sbj_id][0]
    data_path = config.data_path
    sbj_path = op.join(data_path, sbj_dir)

    # path to sub-directory for figures for individual subjects
    figs_path = op.join("/imaging/hauk/users/olaf/FPVS2/MEG/Figures/")

    # raw-filename mappings for this subject
    sss_map_fname = config.sss_map_fnames[sbj_id]

    # # get condition names and frequency names
    # conds = []  # names of conditions
    # for raw_stem_in in sss_map_fname[1][2:]:

    #     conds.append(raw_stem_in[:4])

    # conds = np.unique(conds)

    # initialise sum across harmonics for conditions
    sum_harms = {}
    for cond in conds:
        sum_harms[cond] = {}

    # Go through conditions and frequencies
    # EDIT
    for cond in conds:  # conditions
        # base and oddball frequencies for this condition
        basefreq = config.fpvs_freqs[cond]["base"]
        oddfreq = config.fpvs_freqs[cond]["odd"]

        for ev_type in config.event_ids[cond]:
            # read Evoked objects for all frequencies per condition
            print("Reading results from Evoked files.")

            # separate filename prefixes for ICAed and non-ICAed data
            prefix = ""

            fname_evo = op.join(
                sbj_path, "AVE", "PSD_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            psds_as_evo = mne.read_evokeds(fname_evo)

            fname_evo = op.join(
                sbj_path, "AVE", "PSDZ_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            psds_z_as_evo = mne.read_evokeds(fname_evo)

            fname_evo = op.join(
                sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            psd_harm_as_evo = mne.read_evokeds(fname_evo)

            fname_evo = op.join(
                sbj_path, "AVE", "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            psd_harm_base_as_evo = mne.read_evokeds(fname_evo)

            fname_evo = op.join(
                sbj_path, "AVE", "SumTopoOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            sum_odd_as_evo = mne.read_evokeds(fname_evo)

            fname_evo = op.join(
                sbj_path, "AVE", "SumTopoBase_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            sum_base_as_evo = mne.read_evokeds(fname_evo)

            # Establish channel types present in these data
            chtypes = ["mag", "grad", "eeg"]  # all possible channel types

            for [ci, ch_type] in enumerate(chtypes):
                if not psds_as_evo[0].__contains__(ch_type):
                    del chtypes[chtypes.index[ch_type]]

            # if cond == 'face':  # no frequency sweep for faces

            #     freqs = ['6.0']  # base frequency for this condition (Hz as string)

            #     oddfreq = 1.2  # oddball frequency for this condition (Hz)

            # else:  # for all word condition, use all sweep frequencies

            #     # base frequencies for this condition (Hz as string)
            #     freqs = freqs_all

            #     oddfreq = 1.0  # oddball frequency the same for all sweeps

            # label for condition and base frequency
            label_str = "%s_%s" % (cond, ev_type)

            # Plot PSD as spectrum plus topographies (plot_joint())
            print("Plotting PSDs for %s." % label_str)

            # Plot z-scored PSD
            evoked = psds_z_as_evo[0]

            # Find channels with maximum Z-scores per channel type
            # for base frequency
            # "Latency" is frequency in Hz divided by 1000
            peak_times_base = [basefreq]
            peak_ch_types_base = Ff.peak_channels_evoked(
                evoked=evoked,
                peak_times=peak_times_base,
                ch_types=None,
                n_chan=config.n_peak,
            )

            print(
                "###\nPeak channels in Z-scored PSD for base frequency %f: " % basefreq
            )

            # turn channel names into one list
            # assume there was only one peak frequency
            peak_ch_names_base = []
            for chtype in peak_ch_types_base[0]:
                peak_ch_names_base = peak_ch_names_base + peak_ch_types_base[0][chtype]

            # Find channels with maximum Z-scores per channel type
            # for oddball frequency
            # "Latency" is frequency in Hz divided by 1000
            peak_times_odd = [oddfreq]
            peak_ch_types_odd = Ff.peak_channels_evoked(
                evoked=evoked,
                peak_times=peak_times_odd,
                ch_types=None,
                n_chan=config.n_peak,
            )

            print(
                "\nPeak channels in Z-scored PSD for oddball frequency %f: " % oddfreq
            )

            # turn channel names into one list
            # assume there was only one peak frequency
            peak_ch_names_odd = []
            for chtype in peak_ch_types_odd[0]:
                peak_ch_names_odd = peak_ch_names_odd + peak_ch_types_odd[0][chtype]

            file_label = sbj_dir + "_PSDTopoZ_%s" % label_str
            figs = Ff.plot_psd_as_evo(
                evoked,
                figs_path,
                picks=None,
                txt_label=file_label,
                close_fig=close_fig,
                scalings=unit_scalings,
            )

            for [fig, chtype] in zip(figs, chtypes):
                sec_label = "%s_%s_Z_" % (evoked.comment, chtype)

                report.add_figure(fig, sec_label, section=sec_label)

            # Plot for peak channels (base frequency) without topographies
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_base,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDTopoZPeakbase_%s.jpg" % label_str
            )

            fig.savefig(fname_fig)

            report.add_figure(fig, sec_label, section=sec_label)

            # Plot for peak channels (oddball frequency) without topographies
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_odd,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDTopoZPeakodd_%s.jpg" % label_str
            )

            fig.savefig(fname_fig)

            report.add_figure(fig, sec_label, section=sec_label)

            plt.close("all")

            # Plot PSD across harmonics for oddball frequency
            evoked = psd_harm_as_evo[0]

            # Plotting PSDs across harmonics (base frequency)
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_base,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDHarmOddPeakbase_%s_%s.jpg" % (cond, ev_type)
            )

            fig.savefig(fname_fig)

            sec_label = evoked.comment
            report.add_figure(fig, sec_label, section=sec_label)

            # Plotting PSDs across harmonics (base frequency)
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_odd,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDHarmOddPeakodd_%s_%s.jpg" % (cond, ev_type)
            )

            fig.savefig(fname_fig)

            sec_label = evoked.comment
            report.add_figure(fig, sec_label, section=sec_label)

            plt.close("all")

            # Plot PSD
            evoked = psds_as_evo[0]

            # # Export the raw spectra to Matlab
            # if config.do_export:

            #     export_mat = {'psd': evoked.data, 'freqs': evoked.times,
            #                   'ch_names': evoked.ch_names}

            #     fname = 'PSD_%s_%s_%s.mat' % (config.map_subjects[sbj_id][0][-3:],
            #                                   cond, ev_type)

            #     export_fname = op.join(config.export_path, fname)

            #     print('Exporting to Matlab file %s.' % export_fname)

            #     scipy.io.savemat(export_fname, export_mat)

            file_label = sbj_dir + "_PSDTopo_%s" % label_str

            figs = Ff.plot_psd_as_evo(
                evoked, figs_path, picks=None, txt_label=file_label, close_fig=close_fig
            )

            for [fig, chtype] in zip(figs, chtypes):
                sec_label = "%s_%s_" % (evoked.comment, chtype)

                report.add_figure(fig, sec_label, section=sec_label)

            # Plot for peak channels (base frequency) without topographies
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_base,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDTopoPeakbase_%s.jpg" % label_str
            )

            fig.savefig(fname_fig)

            report.add_figure(fig, sec_label, section=sec_label)

            # Plot for peak channels (oddball frequency) without topographies
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_odd,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDTopoPeakodd_%s.jpg" % label_str
            )

            fig.savefig(fname_fig)

            report.add_figure(fig, sec_label, section=sec_label)

            plt.close(fig)

            # Plot PSD across harmonics for peak channels for base frequency
            evoked = psd_harm_base_as_evo[0]

            # Plotting PSDs across harmonics
            fig = evoked.plot(
                spatial_colors=True,
                picks=peak_ch_names_base,
                scalings=unit_scalings,
                gfp=True,
                time_unit="s",
            )

            fname_fig = op.join(
                figs_path, sbj_dir + "_PSDHarmBasePeakbase_%s_%s.jpg" % (cond, ev_type)
            )

            fig.savefig(fname_fig)

            sec_label = evoked.comment
            report.add_figure(fig, sec_label, section=sec_label)

            plt.close("all")

            # Plot PSD topography across harmonics for oddball frequency
            evoked = sum_odd_as_evo[0]

            # Note: also for oddball frequency the "latency" is the base
            # frequency, because that's our experimental manipulation
            times = [0.0]

            # Filename stem for figure; channel type to be added later
            fname_fig = op.join(figs_path, "PSDSumTopoOdd_%s_%s" % (cond, ev_type))

            # For html report section label
            sec_label = evoked.comment

            figs = Ff.plot_evo_topomap(evoked, times, chtypes, fname_fig)

            for [fig, chtype] in zip(figs, chtypes):
                sec_label = evoked.comment

                report.add_figure(fig, sec_label, section=sec_label)

            # Plot PSD topography across harmonics for base frequency
            evoked = sum_base_as_evo[0]

            times = [0.0]

            # Filename stem for figure; channel type to be added later
            fname_fig = op.join(figs_path, "PSDSumTopoBase_%s_%s" % (cond, ev_type))

            # For html report section label
            sec_label = evoked.comment

            figs = Ff.plot_evo_topomap(evoked, times, chtypes, fname_fig)

            for [fig, chtype] in zip(figs, chtypes):
                sec_label = evoked.comment

                report.add_figure(fig, sec_label, section=sec_label)

        # Save HTML report
        fname_report = op.join(figs_path, sbj_dir + "_report.html")

        report.save(fname_report, overwrite=True, open_browser=False)

        # In case someone was forgotten
        plt.close("all")

    # # Plot the following STC files
    # fstems_stc = ['%sPSDSumTopoBase_%s_%s%s',
    #               '%sPSDSumTopoOdd_%s_%s%s']

    # for fstem_stc in fstems_stc:

    #     fname_stc = op.join(
    #         sbj_path, 'STC', fstem_stc % (prefix, 'face', '6.0', '-lh.stc')
    #     )

    #     print('Reading source estimate from %s.' % fname_stc)
    #     stc = mne.read_source_estimate(fname_stc)

    #     time_label = 'face 6 Hz'

    #     thresh = stc.data.max()

    #     # get some round numbers for colour bar

    #     if thresh < 10:

    #         thresh = np.floor(thresh)

    #     elif thresh < 50:

    #         thresh = 5 * np.floor(thresh / 5.)

    #     else:

    #         thresh = 10 * np.floor(thresh / 10.)

    #     for hemi in ['both']:  #  ['lh', 'rh']:

    #         for view in ['lat', 'ven']:

    #             brain = stc.plot(
    #                 subject=subject, initial_time=0.,
    #                 time_label=time_label, subjects_dir=config.subjects_dir,
    #                 clim=dict(kind='value', lims=[0, thresh / 2., thresh]),
    #                 hemi=hemi, views=view
    #             )

    #             fname_fig = op.join(
    #                 figs_path,
    #                 fstem_stc %
    #                     (prefix, 'face', '6.0', '_STC_%s_%s.jpg' %
    #                         (hemi, view))
    #             )

    #             print('Saving figure to %s.' % fname_fig)

    #             mlab.savefig(fname_fig)

    mlab.close(all=True)

    return


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    # raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
    run_PSD_plot(ss)
