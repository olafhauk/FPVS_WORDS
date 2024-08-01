#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Get data segments for individual frequency sweeps for FPVS Frequency Sweep.

Read raw data, find runs, segment into individual frequency sweeps,
average sweeps across runs, write the average as raw file
(and ascii file if desired).
Compute TFR if specified.
Needs event file from filtering step.
==========================================

OH, October 2019
"""

import sys

from os import remove
from os import path as op
import numpy as np

from copy import deepcopy

from importlib import reload

import mne
from mne_bids import BIDSPath

import config_fpvswords as config

reload(config)


print(mne.__version__)

# conditions
conds = config.do_conds

close_fig = 1  # close figures only if close_fig==1

# ascii_eeg_edf = 1  # !=0 if EEG also output as ascii and edf files

# Code to save in EDF format
# https://gist.github.com/skjerns/bc660ef59dca0dbd53f00ed38c42f6be
# if ascii_eeg_edf:

#     from save_edf import write_edf

# plt.ion() # interactive plotting


def run_get_blocks(sbj_id):
    """Compute spectra for one subject."""
    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # initialise dict for results
    data_runs = {}

    for cond in conds:
        if cond[:4] == "rest":
            task = "rest"
        else:
            task = cond
        # result file to write
        raw_fname_in = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                processing="ica",
                session=None,
                task=task,
                run=config.conds_runs[cond],
                suffix='meg',
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
            ).fpath
        )

        data_runs[cond] = {}

        print("\n###\nReading raw file %s." % raw_fname_in)
        raw_ori = mne.io.read_raw_fif(raw_fname_in, preload=True)

        raw = deepcopy(raw_ori)  # keep raw_ori for possible TFR analysis

        # event file was written during filtering, already correcting
        # projector stimulus delay
        event_file = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                session=None,
                task=task,
                run=config.conds_runs[cond],
                suffix="events",
                extension=".txt",
                datatype="meg",
                root=config.bids_derivatives,
                check=False,
            ).fpath
        )
        print("Reading events from %s." % event_file)
        events = mne.read_events(event_file)

        for ev_type in config.event_ids[cond]:
            data_runs[cond][ev_type] = []

            event_id = config.event_ids[cond][ev_type]

            print("Event ID: %d\n" % event_id)

            # idx_good, idx_bad: lists of indices to onsets of good/bad runs
            idx_good, idx_bad = find_good_events(
                events, event_ids=[event_id], run_duration=62, sfreq=raw.info["sfreq"]
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
                onset_time = (events[idx, 0] - raw.first_samp + n_leadin) / raw.info[
                    "sfreq"
                ]
                print(onset_time)

                n_sweeps = 1
                sweep_duration = config.sweep_duration  # [s] to get x samples

                print(
                    "ID: %d, idx: %d, onset time: %f."
                    % (events[idx, 2], idx, onset_time)
                )

                if onset_time + sweep_duration + config.fpvs_leadin <= raw.times[-1]:
                    # raw_sweeps: list of raw instances per data segments,
                    # one per frequency
                    raw_sweeps = get_sweeps_from_raw(
                        raw,
                        t0=onset_time,
                        sweep_duration=sweep_duration,
                        n_sweeps=n_sweeps,
                    )

                    # EDIT
                    # raw_avg = average_raws(raw_sweeps)

                    data_runs[cond][ev_type].append(raw_sweeps[0])
                else:
                    print("Incomplete sweep.\n")

    # AVERAGE raw files per condition and frequency across runs
    # write the result as raw fiff-file
    for cond in data_runs.keys():  # conditions
        if cond[:4] == "rest":
            task = "rest"
        else:
            task = cond
        for ev_type in config.event_ids[cond]:
            # average sweeps across runs
            raw_avg = average_raws(data_runs[cond][ev_type])

            # remove dot from frequency string
            # fname = "rawavg_%s_%s.fif" % (cond, ev_type)

            fname_raw_out = str(
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

            print("Writing average raw data to %s:" % fname_raw_out)

            raw_avg.save(fname_raw_out, overwrite=True)

    return data_runs


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
    data_runs = run_get_blocks(ss)
