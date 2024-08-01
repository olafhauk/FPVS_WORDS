#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Filter and clean data from FPVSWORDS experiment.

Get event information, interpolate bad EEG channels, (notch) filter.
==========================================

OH, March 2023
"""


import sys
from os import path as op
import numpy as np
import pandas as pd
from importlib import reload

import mne
from mne_bids import BIDSPath

import config_fpvswords as config

reload(config)

print("MNE Version: %s\n\n" % mne.__version__)  # just in case
print(mne)

# whether to show figures on screen or just write to file
show = False

# conditions
conds = config.do_conds

# File for behavioural results
fname = 'behaviours.xlsx'
beh_fname = op.join(
    config.grandmean_path, "Figures", fname)

behav = {}  # behavioural results


def run_behaviour(sbj_ids):
    """Get behavioural results for all subjects in list."""
    behav = {'meanrt': {}, 'acc': {}}
    for cond in [cc for cc in conds if cc[:4] != 'rest']:
        behav['meanrt'][cond] = []
        behav['acc'][cond] = []

        for sbj_id in sbj_ids:
            # path to subject's data
            sbj_path = op.join(
                config.data_path, config.map_subjects[sbj_id][0])

            task = cond
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
            print("Reading events from %s." % event_file)
            events = mne.read_events(event_file)

            # get response latencies for target stimuli (colour changes)

            # find target events (colour changes, trigger value 8)
            trigval = config.event_ids['catch'][cond]
            targ_eves = np.where(events[:, 2] == trigval)[0]
            n_targs = len(targ_eves)

            if n_targs > 0:  # only if target events present (not rest)
                rts = []  # collect response times for this fiff-file

                # find response events (triggers >= 4096, allowing trigger overlap)
                resp_eves = np.where(events[:, 2] >= 4096)[0]

                # find responses closest following targets, compute time difference
                for tt in targ_eves:
                    rt = []  # rt for this particular target

                    # find first response that follows target
                    for rr in resp_eves:
                        if rr > tt:
                            # subtract samples response - target
                            rt = events[rr, 0] - events[tt, 0]

                            # turn samples to latency (ms)
                            sfreq = 1000.
                            rt = 1000.0 * (rt / sfreq)

                            # only count if RT below a threshold
                            if rt <= 2000.0:
                                rts.append(rt)

                            break  # leave this loop

                meanrt = np.mean(rts)
                acc = len(rts)/n_targs
                print("\nResponse times to targets:\n")
                print(*rts)
                print("Mean: %f.\n" % meanrt)
                print("Accuracy: %.2f" % acc)

            else:
                print("No target events present.")
                meanrt = 0
                acc = -1
                print("\nNo good target responses found!\n")

            behav['meanrt'][cond].append(meanrt)
            behav['acc'][cond].append(acc)

        rows = [str(ii) for ii in sbj_ids]
        print('Writing to %s.\n' % beh_fname)
        with pd.ExcelWriter(beh_fname, mode='a', if_sheet_exists='overlay') as writer:
        # with pd.ExcelWriter(beh_fname, mode='w') as writer:
            for sheet in ['meanrt', 'acc']:
                behav_df = pd.DataFrame(behav[sheet], index=rows)
                behav_df.to_excel(writer, sheet_name=sheet)

    print('GM:\n')
    for cond in [cc for cc in conds if cc[:4] != 'rest']:
        print(cond)
        rts = np.array(behav['meanrt'][cond])
        acc = np.array(behav['acc'][cond])
        acc = np.delete(acc, np.where(acc==0)[0])  # remove no responses
        # may contain NaNs if no responses present
        print('Number of NaNs: %d' % np. where(np.isnan(rts))[0].size)
        print('RT: %f (SD: %f, min: %f, max: %f)' % (np.nanmean(rts), np.nanstd(rts), np.nanmin(rts), np.nanmax(rts)))
        print('Acc: %f (SD: %f, min: %f, max: %f)' % (np.nanmean(acc), np.nanstd(acc), np.nanmin(acc), np.nanmax(acc)))

    return behav


# get all input arguments except first
sbj_ids = config.do_subjs

behav = run_behaviour(sbj_ids)
