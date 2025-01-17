#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Epoch data segments from FPVS sweeps for ERP analysis.

Read raw data, find runs, segment into individual frequency sweeps,
create and save epochs.
Based on FPVS_get_sweeps.py (still has redundant functions)
==========================================

TO DO: add epoching for ERP analysis

"""

import sys

from os import path as op
import numpy as np

from copy import deepcopy

from importlib import reload

import mne

import config_fpvswords as config

reload(config)


print(mne.__version__)

# conditions
conds = config.do_conds


def run_epoch_sweeps(sbj_id):
    """Compute epochs for one subject."""
    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # only use files for correct conditions
    sss_map_fnames = []
    for cond in conds:
        for [fi, ff] in enumerate(tmp_fnames):
            if cond in ff:
                sss_map_fnames.append(ff)

    print(sss_map_fnames)

    # initialise for data at different sweep frequencies
    epochs = {}
    behaviour = {}  # collect response times for catch trials
    # create epochs with and without Notch filter for base frequency
    for do_notch in [0, 1]:
        for raw_stem_in in sss_map_fnames:
            cond = raw_stem_in.split("_")[0]  # condition
            epochs[cond] = {}
            # reponse times and number of catch trials
            behaviour[cond] = {}

            for ev_type in config.event_ids[cond]:
                epochs[cond][ev_type] = []
                behaviour[cond][ev_type] = {"rts": [], "N": 0}

            # for Notch filter: base frequency and harmonics
            freq = config.fpvs_freqs[cond]["base"]
            freqs_notch = np.arange(freq, 50.0, freq)

            raw_fname_in = op.join(sbj_path, raw_stem_in[:-4] + "_f_ica_raw.fif")

            print("\n###\nReading raw file %s." % raw_fname_in)
            raw_ori = mne.io.read_raw_fif(raw_fname_in, preload=True)

            # Filter for ERP analysis
            # low-pass only, high-pass filter was applied earlier
            raw_ori.filter(
                l_freq=None,
                h_freq=40.0,
                method="fir",
                fir_design="firwin",
                filter_length="auto",
                h_trans_bandwidth="auto",
            )

            if do_notch:  # if Notch filter at base frequency requested
                # trans_bandwith 2* from Rossion et al. review, suppl. (0.02Hz)
                raw_ori.notch_filter(
                    freqs=freqs_notch, fir_design="firwin", trans_bandwidth=0.04
                )

                # add to epoch file name
                str_notch = "_nch"

            else:
                str_notch = ""

            raw = deepcopy(raw_ori)  # keep raw_ori for possible TFR analysis

            # event file was written during filtering, already correcting
            # projector stimulus delay
            event_file = op.join(sbj_path, raw_stem_in + "_sss_f_raw-eve.fif")
            print("Reading events from %s." % event_file)
            events = mne.read_events(event_file)

            for ev_type in config.event_ids[cond]:
                # Find indices of good events (onsets of runs without missing frames)
                event_ids = [config.event_ids[cond][ev_type]]
                print(cond)
                print(ev_type)
                print(event_ids)

                # duration of run incl. all sweeps and lead-in time at beginning
                run_duration = 60.0 + config.fpvs_leadin

                # idx_good, idx_bad: lists of indices to onsets of good/bad runs
                idx_good, idx_bad = find_good_events(
                    events,
                    event_ids=event_ids,
                    run_duration=run_duration,
                    sfreq=raw.info["sfreq"],
                )

                print("Good runs:")
                print(idx_good)
                print(events[idx_good, :])

                if len(idx_bad) != 0:
                    print("Bad runs:")
                    print(idx_bad)
                    print(events[idx_bad, :])
                else:
                    print("No bad runs.")

                # go through all indices to good runs
                for idx in idx_good:
                    # onset time (s) for this good run
                    # note: samples don't start at 0, but times do
                    n_leadin = config.fpvs_leadin * raw.info["sfreq"]  # lead-in period
                    onset_time = (
                        events[idx, 0] - raw.first_samp + n_leadin
                    ) / raw.info["sfreq"]
                    print(onset_time)

                    if onset_time + run_duration <= raw.times[-1]:
                        # just get one "sweep"
                        raw_sweep = get_sweeps_from_raw(
                            raw, t0=onset_time, sweep_duration=60.0, n_sweeps=1
                        )

                        # Epoching for ERP analysis
                        print("###\nEpoching.")
                        # Event IDs: 4(standard), 5 (oddball)
                        event_id = config.event_ids["odd"][
                            cond
                        ]  # onset of individual stimuli
                        epos = mne.Epochs(
                            raw=raw_sweep[0],
                            events=events,
                            event_id=event_id,
                            tmin=config.epo_t1,
                            tmax=config.epo_t2,
                            proj=True,
                            baseline=config.epo_baseline,
                            preload=True,
                            reject=config.epo_reject,
                        )

                        # append for each run and event type
                        epochs[cond][ev_type].append(epos)

                # concatenate epochs across runs
                epochs_conc = mne.concatenate_epochs(epochs[cond][ev_type])

                epo_fname = op.join(
                    sbj_path,
                    "EPO",
                    "%s_f_%s%s-epo.fif"
                    % (
                        cond,
                        ev_type,
                        str_notch,
                    ),
                )
                print("Writing epochs to %s." % epo_fname)

                epochs_conc.save(epo_fname, overwrite=True)

                #         else:  # for frequency sweeps
                #             n_sweeps = len(config.fpvs_freqs)

                #             print(
                #                 "ID: %d, idx: %d, onset time: %f."
                #                 % (events[idx, 2], idx, onset_time)
                #             )

                #             # raw_sweeps: list of raw instances per data segments,
                #             # one per frequency
                #             raw_sweeps = get_sweeps_from_raw(
                #                 raw,
                #                 t0=onset_time,
                #                 sweep_duration=config.fpvs_sweep_duration,
                #                 n_sweeps=n_sweeps,
                #             )

                #             for [fi, ff] in enumerate(config.fpvs_freqs):
                #                 # Epoching for ERP analysis
                #                 print("###\nEpoching.")
                #                 # Event IDs: 4(standard), 5 (oddball)
                #                 event_id = 5  # onset of individual stimuli
                #                 epos = mne.Epochs(
                #                     raw=raw_sweeps[fi],
                #                     events=events,
                #                     event_id=event_id,
                #                     tmin=config.epo_t1,
                #                     tmax=config.epo_t2,
                #                     proj=True,
                #                     baseline=config.epo_baseline,
                #                     preload=True,
                #                     reject=config.epo_reject,
                #                 )

                #                 # append for each run and frequency
                #                 epochs[raw_stem_in[:4]][str(ff)].append(epos)

                # Finally... get response times for catch trials
                # events of colour changes
                catch_eves = np.where(
                    [(ee in [config.event_ids["catch"][cond]]) for ee in events[:, -1]]
                )[0]
                behaviour[cond][ev_type]["N"] += len(catch_eves)
                # button press events
                resp_eves = np.where(events[:, -1] > 1000)[0]
                for cc in catch_eves:
                    rt_diffs = (events[resp_eves, 0] - events[cc, 0]) / epos.info[
                        "sfreq"
                    ]
                    # find earliest button press after catch trial event within 3s
                    rt_min = rt_diffs[(rt_diffs > 0) & (rt_diffs < 3.0)]

                    # pick earliest response after response within limit
                    if len(rt_min) != 0:
                        behaviour[cond][ev_type]["rts"].append(rt_min.min())

                rts = behaviour[cond][ev_type]["rts"]
                rt = np.mean(rts)
                print(
                    "Average response time and correct rate for catch trials: "
                    "%f, %f" % (rt, len(rts) / behaviour[cond][ev_type]["N"])
                )

    return


def find_good_events(events, event_ids, run_duration, sfreq):
    """Find the onsets of good runs in raw data.

    Parameters:
    events: nd-array
        Events from raw data.
    event_ids: list of int
        Possible triggers of run onsets
    run_duration: float
        Duration (s) of a run within session
    sfreq: float
        Sampling frequency (Hz)

    Returns:
    idx_good: list of int
        Indices to onsets of good runs.
    idx_bad: list of int
        List of indices to onsets of bad runs
    """
    max_missed = 2  # how many frames turn a run invalid

    idx_good, idx_bad = [], []  # initialise output

    # number of indices for events in this run
    n_idx = int(run_duration * sfreq)

    # find all onsets in events based on event_ids
    onsets = [ee for ee in events if (ee[2] in event_ids)]

    for onset in onsets:
        # find index of this event
        onset_idx = np.where(events[:, 0] == onset[0])[0][0]

        # get all indices for events in this run
        idx_run = np.where(
            (events[:, 0] > onset[0]) & (events[:, 0] < onset[0] + n_idx)
        )[0]

        # get all events for this run
        events_run = events[idx_run, :]

        # check if missed frames present, and how many
        missed_frames = np.where(events_run[:, 2] == 20)[0]

        print("Missed frames:")
        print(missed_frames)

        # if good run found
        if (len(missed_frames) == 0) or (missed_frames.shape[0] < max_missed):
            idx_good.append(onset_idx)

        else:  # if invalid due to missing frames
            idx_bad.append(onset_idx)

    return idx_good, idx_bad


def get_sweeps_from_raw(raw, t0, sweep_duration, n_sweeps):
    """Get segments from raw data for individual frequency sweeps.

    Parameters:
    raw: instance of Raw
        The raw data including frequency sweeps.
    t0: float
        Start time of segment in s.
    sweep_duration: float
        Duration of one sweep at one frequency (s).
    n_sweeps: int
        Number of sweeps (frequencies) per run.

    Returns:
    raw_sweeps: list of raw instances
        Data segments per frequency.
    """
    raw_sweeps = []  # initialise output

    for ss in np.arange(0, n_sweeps):
        # Start and end latencies of one frequency sweep
        tmin = t0 + (ss * sweep_duration)
        tmax = t0 + (ss + 1) * sweep_duration

        raw_cp = raw.copy()

        # Crop out one frequency sweep
        raw_cp.crop(tmin=tmin, tmax=tmax)

        raw_sweeps.append(raw_cp)

    return raw_sweeps


def average_raws(raws):
    """Average data across raw files.

    Parameters:
    raws: list of instances of Raw
        The raw data to average.
        Every item of raws must have same info.

    Returns:
    raw_avg: instance of Raw
        The average of raw data.
    """
    # get data array from first file
    data = raws[0].get_data()

    if len(raws) > 1:
        for raw in raws[1:]:
            data += raw.get_data()

        data = data / len(raws)

    # don't understand 'copy' option, using default
    raw_avg = mne.io.RawArray(data, raws[0].info, first_samp=0, copy="auto")

    return raw_avg


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]

for ss in sbj_ids:
    # raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
    run_epoch_sweeps(ss)
