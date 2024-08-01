
### check timing between first and last trigger per run
### to determine run length and optimal number of bins

from os import path as op

import numpy as np

from matplotlib import pyplot as plt

from importlib import reload

import config_fpvswords as config
reload(config)

import mne
# from mne.report import Report

plt.ion()

# All
sbj_ids = [1, 2, 3, 4, 5,  7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


# conds = ['english', 'slow']

cond = ['english']

event_id = config.event_ids[cond[0]]['nw']
base_id = config.event_ids[cond[0]]['base']

basefreq = config.fpvs_freqs[cond[0]]['base']

durations = []

for [ss, sbj_id] in enumerate(sbj_ids):  # across all subjects

    subj = config.map_subjects[sbj_id][0]
    sbj_path = op.join(config.data_path, subj)

    fname_eve = op.join(sbj_path, '%s_sss_raw_sss_f_raw-eve.fif' % cond[0])
    events = mne.read_events(fname_eve)

    # find onsets of individual runs
    onss = np.where(events[:, 2] in event_id)[0]

    # find triggers of base stimuli
    tri = np.where(events[:, 2] == base_id)[0]
    sps = events[tri, 0]
    diffs = np.diff(sps)
    # find gaps in trigger chain between runs
    diffmax = np.where(diffs > 1000)[0]
    ofss = tri[diffmax]
    if (onss.shape[0] == ofss.shape[0] + 1):
        ofss = np.append(ofss, tri[-1])  # append last trigger of run
    elif (onss.shape[0] != 1):
        print('\n\nNumber of onsets and offsets do not add up!\n\n')

    for [ons, ofs] in zip(onss, ofss):

        # duration (incl. duration of last stimulus)
        dur = events[ofs, 0] - events[ons, 0] + np.round(1000. / basefreq)
        print(sbj_id)
        print(dur)
        durations.append(dur)
