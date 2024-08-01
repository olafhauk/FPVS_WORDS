#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Filter and clean data from FPVSWORDS experiment.

Get event information, interpolate bad EEG channels, (notch) filter.
==========================================

OH, March 2023
"""

import config_fpvswords as config
from mne_bids import BIDSPath
import mne
from importlib import reload
from matplotlib import pyplot as plt
import sys
from os import remove
from os import path as op
import numpy as np

import matplotlib

matplotlib.use("Agg")  # for running graphics on cluster ### EDIT


reload(config)

print("MNE Version: %s\n\n" % mne.__version__)  # just in case
print(mne)

# whether to show figures on screen or just write to file
show = False

# conditions
conds = config.do_conds

# # whether to write raw data as EDF or not
# ascii_eeg_edf = True

# if ascii_eeg_edf:
#     from save_edf import write_edf


def run_filter_raw(sbj_id):
    """Clean data for one subject."""
    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    bad_eeg = config.bad_channels[sbj_id]["eeg"]  # bad EEG channels

    # only use files for correct conditions
    sss_map_fnames = []
    for cond in conds:
        if cond[:4] == "rest":
            task = "rest"
        else:
            task = cond
        raw_fname_in = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                processing="sss",
                session=None,
                task=task,
                run=config.conds_runs[cond],
                suffix="raw",
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
                check=False,
            ).fpath
        )
        print(raw_fname_in)

        # result file to write
        # raw_fname_out = (
        #     str(
        #         BIDSPath(
        #             subject=str(sbj_id).zfill(2),
        #             processing="filt",
        #             session=None,
        #             task=task,
        #             run=config.conds_runs[cond],
        #             suffix=None,
        #             extension=".fif",
        #             datatype="meg",
        #             root=config.bids_derivatives,
        #             check=False,
        #         ).fpath
        #     )
        # )
        raw_fname_out = BIDSPath(
                    subject=str(sbj_id).zfill(2),
                    processing="filt",
                    session=None,
                    task=task,
                    run=config.conds_runs[cond],
                    suffix='meg',
                    extension=".fif",
                    datatype="meg",
                    root=config.bids_derivatives,
                    check=False,
                )

        print("\n###\nReading raw file %s." % raw_fname_in)

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        print("Fixing coil types.")
        raw.fix_mag_coil_types()

        print("Marking bad EEG channels: %s" % bad_eeg)
        raw.info["bads"] = bad_eeg

        print("Setting EEG reference.")
        raw.set_eeg_reference(ref_channels="average", projection=True)

        # # if ascii and edf data for EEG requested as well
        # if ascii_eeg_edf:

        #     # reduce to EEG only
        #     raw_eeg = raw.copy()

        #     raw_eeg.pick_types(meg=False, eeg=True, eog=True, stim=True)

        #     # EDF
        #     fname = raw_stem_in + '_eeg.edf'
        #     fname_edf_out = op.join(sbj_path, fname)

        #     print('Writing raw EDF data to %s:' % fname_edf_out)

        #     write_edf(raw_eeg, fname_edf_out, overwrite=True)

        print("Applying Notch filter.")

        # filter_length = raw.times[-1] - raw.times[0]
        # raw.notch_filter(np.arange([50, 101, 50]), fir_design='firwin',
        #                  filter_length=filter_length, trans_bandwidth=0.04)
        raw.notch_filter(np.arange(50, 101, 50))

        # str() because of None
        print(
            "Applying band-pass filter %s to %s Hz."
            % (str(config.l_freq), str(config.h_freq))
        )

        # most settings are the MNE-Python defaults (zero-phase FIR)
        # https://mne.tools/dev/auto_tutorials/discussions/plot_background_filtering.html
        raw.filter(
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            method="fir",
            fir_design="firwin",
            filter_length="auto",
            l_trans_bandwidth="auto",
            h_trans_bandwidth="auto",
        )

        print("Saving data to %s." % raw_fname_out)
        raw.save(raw_fname_out, overwrite=True)

        print("Finding events.")
        # note: short event duration
        events = mne.find_events(
            raw,
            stim_channel="STI101",
            consecutive="increasing",
            min_duration=0.002,
            verbose=True,
        )

        # for some subjects and conditions include missing triggers
        if (sbj_id == 7) and ("face" in raw_fname_in):
            events = np.insert(events, 1, [54261 - 167, 0, 18], 0)

        if (sbj_id == 13) and ("face" in raw_fname_in):
            events = np.insert(events, 802, [186225 + 167, 0, 4], 0)

        # correct possible step-wise trigger onsets
        # get trigger channel values from STI101
        stidat = raw.get_data("STI101").squeeze()

        # difference of adjacent trigger values
        stidif = np.diff(stidat)

        # find indices of transitions from zero
        stion = np.where(stidif > 0)[0] + 1

        for ii in stion:
            # if step-wise trigger
            if (stidat[ii] > 0) and (stidat[ii] < stidat[ii + 1]):
                # set first trigger equal to next trigger value
                stidat[ii] = stidat[ii + 1]

        # correct for stimulus presentation delay
        stim_delay = int(config.delay * raw.info["sfreq"])
        events[:, 0] = events[:, 0] + stim_delay

        # event_file = op.join(sbj_path, raw_stem_in + "_sss_f_raw-eve.fif")
        event_file = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                session=None,
                task=task,
                run=config.conds_runs[cond],
                suffix="events",
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
                check=False,
            ).fpath
        )
        print("Saving events to %s." % event_file)
        mne.write_events(event_file, events, overwrite=True)

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
        print("Saving events to %s." % event_file)
        mne.write_events(event_file, events, overwrite=True)

        # plot only if events were found
        if events.size != 0:
            fig = mne.viz.plot_events(events, raw.info["sfreq"], show=show)

            fname_fig = str(
                BIDSPath(
                    subject=str(sbj_id).zfill(2),
                    session=None,
                    task=task,
                    run=config.conds_runs[cond],
                    suffix="events",
                    extension=".jpg",
                    datatype="meg",
                    root=config.bids_derivatives,
                    check=False,
                ).fpath
            )
            print("Saving figure to %s" % fname_fig)

            fig.savefig(fname_fig)

            plt.close(fig)

        else:
            print("No events found in file %s." % raw_fname_in)

        # get response latencies for target stimuli (colour changes)

        # # find target events (colour changes, trigger value 8)
        # targ_eves = np.where(events[:, 2] == 8)[0]
        # n_targs = len(targ_eves)

        # if n_targs > 0:  # only if target events present (not rest)
        #     rts = []  # collect response times for this fiff-file

        #     # find response events (triggers >= 4096, allowing trigger overlap)
        #     resp_eves = np.where(events[:, 2] >= 4096)[0]

        #     # find responses closest following targets, compute time difference
        #     for tt in targ_eves:
        #         rt = []  # rt for this particular target

        #         # find first response that follows target
        #         for rr in resp_eves:
        #             if rr > tt:
        #                 # subtract samples response - target
        #                 rt = events[rr, 0] - events[tt, 0]

        #                 # turn samples to latency (ms)
        #                 rt = 1000.0 * (rt / raw.info["sfreq"])

        #                 # only count if RT below a threshold
        #                 if rt <= 2000.0:
        #                     rts.append(rt)

        #                 break  # leave this loop

        #     if rts == []:  # if no good responses found
        #         print("\nNo good target responses found!\n")

        #     else:
        #         fname =
        #         fig_fname = op.join(
        #                 config.grandmean_path, "Figures", fname)
        #         print("\nResponse times to targets:\n")
        #         print(*rts)
        #         print("Mean: %f.\n" % np.mean(rts))
        #         print("Accuracy: %.2f" % len(rts)/n_targs)

        # else:
        #     print("No target events present.")

    return raw, events


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    [raw, events] = run_filter_raw(ss)
