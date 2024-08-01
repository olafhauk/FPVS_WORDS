#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Compute PSD for average raw sensor and source data for FPVS.

Reads average raw data from FPVS_get_sweeps.py.
Plot figures.
Compute z-scores.
==========================================

OH, March 2023
"""

import sys

from os import remove
from os import path as op
import numpy as np

# needed to run on SLURM
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from copy import deepcopy

# from mayavi import mlab
# mlab.options.offscreen = True

# for running graphics on cluster ### EDIT
# required even if not plotting?
# matplotlib.use('Agg')

from importlib import reload

import mne
from mne_bids import BIDSPath

import config_fpvswords as config

reload(config)

import FPVS_functions as Ff

reload(Ff)


print(mne.__version__)


# conditions
# conds = ['face', 'pwhf', 'pwlf', 'lfhf']
conds = config.do_conds


def run_PSD_raw(sbj_id):
    """Compute spectra for one subject."""
    subject = config.mri_subjects[sbj_id]

    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    inv_fname = op.join(sbj_path, subject + "_EEGMEG-inv.fif")

    print("Reading EEG/MEG inverse operator: %s." % inv_fname)
    invop = mne.minimum_norm.read_inverse_operator(inv_fname)

    # raw-filename mappings for this subject
    sss_map_fname = config.sss_map_fnames[sbj_id]

    # initialise sum across harmonics for conditions
    sum_harms_odd = {}  # for oddball frequency
    sum_harms_base = {}  # for base frequencies
    # topographies for harmonics
    topos_harms_odd = {}  # for oddball frequency
    topos_harms_base = {}  # for base frequencies
    for cond in conds:
        sum_harms_odd[cond] = {}
        sum_harms_base[cond] = {}
        topos_harms_odd[cond] = {}
        topos_harms_base[cond] = {}

    # Go through conditions and frequencies
    for cond in conds:  # conditions
        print("###\nCondition: %s.\n###" % cond)
        if cond[:4] == "rest":
            task = "rest"
        else:
            task = cond

        # create list of Evoked objects for all frequencies per condition
        (
            psd_all,
            psd_z_all,
            sum_harms_odd_all,
            sum_harms_base_all,
            topos_odd_all,
            topos_base_all,
            psd_harm_odd_all,
            psd_harm_base_all,
        ) = ([], [], [], [], [], [], [], [])

        # base and oddball frequencies for this condition
        basefreq = config.fpvs_freqs[cond]["base"]
        oddfreq = config.fpvs_freqs[cond]["odd"]

        for ev_type in config.event_ids[cond]:
            # number of bins for z-scores
            snr_bins = config.psd_snr_bins

            # # initialise for this base frequency
            # sum_harms_odd[cond] = []
            # sum_harms_base[cond] = []
            # topos_harms_odd[cond] = []

            # input average raw data; remove dot from frequency string
            fname_raw_in = str(
                BIDSPath(
                    subject=str(sbj_id).zfill(2),
                    processing="avg",
                    session=None,
                    task=task,
                    run=config.conds_runs[cond],
                    suffix=ev_type,
                    extension=".fif",
                    datatype="meg",
                    root=config.bids_derivatives,
                    check=False,
                ).fpath
            )

            print("Reading average raw data from %s:" % fname_raw_in)

            raw = mne.io.read_raw_fif(fname_raw_in, preload=True)

            print("Resample to %s Hz." % config.psd_resample)

            if config.psd_resample is not None:
                raw.resample(sfreq=config.psd_resample)

            # reduce raw data to relevant channels
            raw.pick(picks=["meg", "eeg"])

            print(raw.info["bads"])
            print(raw)

            # compute_source_psd() returns sensor data only for good channels,
            # but for grand-averaging we will require all channels
            raw.interpolate_bads(mode="accurate", reset_bads=True)

            print("Setting EEG reference.")  # apply here
            raw.set_eeg_reference(ref_channels="average", projection=False)

            # Compute PSD for raw data

            # EDIT: find smallest power of 2 larger than number of samples
            # n_fft = 2**(len(raw.times) - 1).bit_length()

            n_fft = len(raw.times)

            print("n_fft: %d" % n_fft)

            fmin = config.psd_fmin
            fmax = config.psd_fmax
            print("###\nComputing psd_welch() from %f to %f Hz." % (fmin, fmax))

            # print('Computing psd_welch() in sensor and source space.')
            stc_psd, evo_psd = mne.minimum_norm.compute_source_psd(
                raw=raw,
                inverse_operator=invop,
                lambda2=1 / 9.0,
                method="MNE",
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                overlap=0.0,
                nave=1,
                bandwidth="hann",
                low_bias=True,
                return_sensor=True,
            )

            # psd_mat, psd_freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, n_fft=n_fft)

            # raw.del_proj()  # remove average reference before using info for PSDs
            # info_raw = raw.info
            # # in order to turn frequencies into latencies
            # sfreq = 1. / (psd_freqs[1] - psd_freqs[0])
            # info_raw['sfreq'] = sfreq

            # # ch_types = 102 * ['grad', 'grad', 'mag'] + 64 * ['eeg']
            # # info_evo = mne.create_info(ch_names=info.ch_names, sfreq=sfreq, ch_types=ch_types)

            # evo_psd = mne.EvokedArray(psd_mat, info_raw, tmin=psd_freqs[0])

            # psd_all.append(evo_psd)

            # turn power to amplitudes
            evo_psd.data = np.sqrt(evo_psd.data)

            stc_psd.data = np.sqrt(stc_psd.data)

            stc_psd.subject = subject

            fname_stc = op.join(sbj_path, "STC", "PSDTopo_%s" % cond)

            stc_psd.save(fname_stc, overwrite=True)

            # frequencies in PSD
            psd_freqs = evo_psd.times

            print("Frequencies from %f to %f." % (psd_freqs[0], psd_freqs[-1]))

            freq_resol = psd_freqs[1] - psd_freqs[0]
            print("Frequency resolution:\n%f.\n###" % freq_resol)

            # Z-score PSDs with neighbouring frequency bins

            print(type(evo_psd))

            print("Computing Z-scores for Evoked.")
            # inputing Evoked object
            psd_z = Ff.psd_z_score(evo_psd, snr_bins, mode="z", n_gap=config.psd_n_gap)

            psd_z.comment = "PSDTopoZ_" + cond + "_" + ev_type

            # ODDBALL FREQUENCY

            print(
                "Summing PSDs across %d harmonics for oddball frequency"
                % config.fpvs_n_harms_odd
            )

            # get PSDs around harmonics
            (
                psd_harm_odd_ori,
                topo,
                topos,
                freqs_harm,
                psd_harm_odd_epos,
            ) = Ff.psds_across_harmonics(
                psds=evo_psd,
                freqs=psd_freqs,
                basefreq=basefreq,
                oddfreq=oddfreq,
                n_harms=config.fpvs_n_harms_odd,
                n_bins=snr_bins,
                n_gap=config.psd_n_gap,
                skip_harm=config.psd_skip_harm,
                method="sum",
            )

            # get PSDs around harmonics for z-scores
            # needed to get z-scored topographies for harmonics
            (
                psd_harm_odd_z,
                topo_z,
                topos_z,
                freqs_harm_z,
                psd_harm_odd_epos_z,
            ) = Ff.psds_across_harmonics(
                psds=psd_z,
                freqs=psd_freqs,
                basefreq=basefreq,
                oddfreq=oddfreq,
                n_harms=config.fpvs_n_harms_odd,
                n_bins=snr_bins,
                n_gap=config.psd_n_gap,
                skip_harm=config.psd_skip_harm,
                method="sum",
            )

            print('Compute z-score after summing')
            psd_harm_odd = Ff.psd_z_score(
                psd_harm_odd_ori, snr_bins, mode="z", n_gap=config.psd_n_gap
            )

            psd_harm_odd.comment = "PSDHarm_" + cond

            # Save epochs around individual harmonics
            fname_evo = op.join(
                sbj_path, "AVE", "HarmOddEpos_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, psd_harm_odd_epos_z, overwrite=True)

            # Topography of z-scored summed harmonics at centre frequency
            topo_evo = deepcopy(psd_harm_odd)
            topo_evo.crop(tmin=0.0, tmax=0.0)
            sum_harms_odd = topo_evo

            # sum_harms_odd_all.append(sum_harms_odd[cond])

            # z-scored topographies for individual harmonics
            topos.comment = " ".join(str(freqs_harm_z))

            topos_harms_odd = topos_z

            # topos_odd_all.append(topos_harms_odd[cond])

            # STCs odd

            psd_harm_odd_stc, topo, topos, freqs_harm, _ = Ff.psds_across_harmonics(
                psds=stc_psd,
                freqs=psd_freqs,
                basefreq=basefreq,
                oddfreq=oddfreq,
                n_harms=config.fpvs_n_harms_odd,
                n_bins=snr_bins,
                n_gap=config.psd_n_gap,
                skip_harm=config.psd_skip_harm,
                method="sum",
            )

            psd_harm_odd_stc.subject = subject

            # compute z-score after summing
            psd_harm_odd_stc_z = Ff.psd_z_score(
                psd_harm_odd_stc, snr_bins, mode="z", n_gap=config.psd_n_gap
            )

            # save non-z-scored STCs
            fname_stc = op.join(sbj_path, "STC", "PSDHarmOdd_%s_%s" % (cond, ev_type))
            print("Writing pPSDHarmOdd to %s.\n" % fname_stc)
            psd_harm_odd_stc.save(fname_stc, overwrite=True)

            # save z-scored STCs
            fname_stc = op.join(sbj_path, "STC", "PSDZHarmOdd_%s_%s" % (cond, ev_type))
            print("Writing PSDZHarmOdd to %s.\n" % fname_stc)
            psd_harm_odd_stc_z.save(fname_stc, overwrite=True)

            # STC of summed harmonics at centre frequency
            topo_stc = deepcopy(psd_harm_odd_stc)
            topo_stc.crop(tmin=0.0, tmax=0.0)
            sum_harms_odd_stc = topo_stc
            sum_harms_odd_stc.subject = subject
            fname_stc = op.join(
                sbj_path, "STC", "PSDSumTopoOdd_%s_%s" % (cond, ev_type)
            )
            print("Writing PSDSumTopoOdd to %s.\n" % fname_stc)
            sum_harms_odd_stc.save(fname_stc, overwrite=True)

            # STC of z-scored summed harmonics at centre frequency
            topo_stc = deepcopy(psd_harm_odd_stc_z)
            topo_stc.crop(tmin=0.0, tmax=0.0)
            sum_harms_odd_stc = topo_stc
            sum_harms_odd_stc.subject = subject
            fname_stc = op.join(
                sbj_path, "STC", "PSDZSumTopoOdd_%s_%s" % (cond, ev_type)
            )
            print("Writing PSDZSumTopoOdd to %s.\n" % fname_stc)
            sum_harms_odd_stc.save(fname_stc, overwrite=True)

            # BASE FREQUENCY

            print(
                "Summing PSDs across %d harmonics for base frequency"
                % config.fpvs_n_harms_base
            )

            # Sanity check - do it for base frequency
            # i.e. basefreq as oddfreq here, for all its harmonics
            (
                psd_harm_base,
                topo,
                topos,
                freqs_harm,
                psd_harm_base_epos,
            ) = Ff.psds_across_harmonics(
                psds=evo_psd,
                freqs=psd_freqs,
                basefreq=None,
                oddfreq=basefreq,
                n_harms=config.fpvs_n_harms_base,
                n_bins=snr_bins,
                n_gap=config.psd_n_gap,
                skip_harm=0,
                method="sum",
            )

            # sum across harmonics for z-scores
            # needed to get z-scored topographies for harmonics
            (
                psd_harm_base_z,
                topo_z,
                topos_z,
                freqs_harm_z,
                psd_harm_base_epos_z,
            ) = Ff.psds_across_harmonics(
                psds=psd_z,
                freqs=psd_freqs,
                basefreq=None,
                oddfreq=basefreq,
                n_harms=config.fpvs_n_harms_base,
                n_bins=snr_bins,
                n_gap=config.psd_n_gap,
                skip_harm=0,
                method="sum",
            )

            # compute z-score after summing
            psd_harm_base = Ff.psd_z_score(
                psd_harm_base, snr_bins, mode="z", n_gap=config.psd_n_gap
            )

            # psd_harm_base_all.append(psd_harm_base)

            # Save epochs around individual harmonics
            fname_evo = op.join(
                sbj_path, "AVE", "HarmBaseEpos_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, psd_harm_base_epos_z, overwrite=True)

            # Topography of z-scored summed harmonics at centre frequency
            topo_evo = deepcopy(psd_harm_base)
            topo_evo.crop(tmin=0.0, tmax=0.0)
            sum_harms_base = topo_evo

            # sum_harms_base_all.append(sum_harms_base[cond])

            # z-scored topographies for individual harmonics
            topos.comment = " ".join(str(freqs_harm))

            topos_harms_base = topos_z

            # topos_base_all.append(topos_z)

            # STCs base

            psd_harm_base_stc, topo, topos, freqs_harm, _ = Ff.psds_across_harmonics(
                psds=stc_psd,
                freqs=psd_freqs,
                basefreq=None,
                oddfreq=basefreq,
                n_harms=config.fpvs_n_harms_base,
                n_bins=snr_bins,
                n_gap=config.psd_n_gap,
                skip_harm=config.psd_skip_harm,
                method="sum",
            )

            psd_harm_base_stc.subject = subject

            # compute z-score after summing
            psd_harm_base_stc_z = Ff.psd_z_score(
                psd_harm_base_stc, snr_bins, mode="z", n_gap=config.psd_n_gap
            )

            # save non-z-scored STCs
            fname_stc = op.join(sbj_path, "STC", "PSDHarmBase_%s_%s" % (cond, ev_type))
            print("Writing PSDHarmBase to %s.\n" % fname_stc)
            psd_harm_base_stc.save(fname_stc, overwrite=True)

            # save z-scored STCs
            fname_stc = op.join(sbj_path, "STC", "PSDZHarmBase_%s_%s" % (cond, ev_type))
            print("Writing PSDZHarmBase to %s.\n" % fname_stc)
            psd_harm_base_stc_z.save(fname_stc, overwrite=True)

            # STC of non-z-scored summed harmonics at centre frequency
            topo_stc = deepcopy(psd_harm_base_stc)
            topo_stc.crop(tmin=0.0, tmax=0.0)
            sum_harms_base_stc = topo_stc
            sum_harms_base_stc.subject = subject
            fname_stc = op.join(
                sbj_path, "STC", "PSDSumTopoBase_%s_%s" % (cond, ev_type)
            )
            print("Writing PSDSumTopoBase to %s.\n" % fname_stc)
            sum_harms_base_stc.save(fname_stc, overwrite=True)

            # STC of z-scored summed harmonics at centre frequency
            topo_stc = deepcopy(psd_harm_base_stc_z)
            topo_stc.crop(tmin=0.0, tmax=0.0)
            sum_harms_base_stc = topo_stc
            sum_harms_base_stc.subject = subject
            fname_stc = op.join(
                sbj_path, "STC", "PSDZSumTopoBase_%s_%s" % (cond, ev_type)
            )
            print("Writing PSDZSumTopoBase to %s.\n" % fname_stc)
            sum_harms_base_stc.save(fname_stc, overwrite=True)

            # Save Evoked objects for later group stats:

            print("Saving PSD results as evoked files:")

            # PSD (raw):
            fname_evo = op.join(
                sbj_path, "AVE", "PSD_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, evo_psd, overwrite=True)

            # PSD (z-scored):
            fname_evo = op.join(
                sbj_path, "AVE", "PSDZ_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, psd_z, overwrite=True)

            # Sum PSD segments around harmonics of oddball frequency then z-score:
            fname_evo = op.join(
                sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, psd_harm_odd, overwrite=True)

            # Sum PSD segments around harmonics of base frequency then z-score:
            fname_evo = op.join(
                sbj_path, "AVE", "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, psd_harm_base, overwrite=True)

            # Oddball topography of z-scored summed harmonics at centre frequency:
            fname_evo = op.join(
                sbj_path, "AVE", "SumTopoOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, sum_harms_odd, overwrite=True)

            # Base topography of z-scored summed harmonics at centre frequency:
            fname_evo = op.join(
                sbj_path, "AVE", "SumTopoBase_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, sum_harms_base, overwrite=True)

            # Oddball topographies at centre frequencies for individual harmonics:
            fname_evo = op.join(
                sbj_path, "AVE", "SumToposOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, topos_harms_odd, overwrite=True)

            # Base topographies at centre frequencies for individual harmonics:
            fname_evo = op.join(
                sbj_path, "AVE", "SumToposBase_%s_%s%s" % (cond, ev_type, "-ave.fif")
            )
            print(fname_evo)
            mne.write_evokeds(fname_evo, topos_harms_base, overwrite=True)

    return evo_psd, psd_harm_odd_ori, psd_harm_odd, freqs_harm


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    evo_psd, psd_harm_odd_ori, psd_harm_odd, freqs_harm = run_PSD_raw(ss)
