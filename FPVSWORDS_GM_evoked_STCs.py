#!/imaging/local/software/miniconda/envs/mne1.4.0/bin/python
"""
Average source estimates of evoked data in FPVS2.

Read morphed STCs produced in FPVSWORDS_source_estimation_evoked.py.
Average and plot them.
==========================================

OH, Feb 2024
"""

import sys

import os
from os import path as op
from os import stat

import numpy as np

from copy import deepcopy

from importlib import reload

import mne
from mne.source_estimate import SourceEstimate

import config_fpvswords as config

reload(config)


print(mne.__version__)

# conditions
conds = config.do_conds


def run_average_STCs_evoked():
    sbj_ids = config.do_subjs

    # for evoked created with and without Notch filter for base frequency
    for do_notch in [0, 1]:
        if do_notch:  # if Notch filter at base frequency requested
            # add to epoch file name
            str_notch = "_nch"

        else:
            str_notch = ""

        for cond in conds:  # conditions
            print("###\nCondition: %s.\n###" % cond)
            if cond[:4] == "rest":
                task = "rest"
            else:
                task = cond

            for ev_type in config.event_ids[cond]:
                stcs = []

                for sbj_id in sbj_ids:
                    # path to subject's data
                    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

                    stc_fname = op.join(
                        sbj_path,
                        "STC",
                        "evo_%s_%s%s_mph" % (cond, ev_type, str_notch),
                    )

                    print("Reading source estimate from %s." % stc_fname)

                    stc = mne.read_source_estimate(stc_fname)

                    stcs.append(stc)

                print("Averaging %d STCs." % len(stcs))
                avg_data = np.average([s.data for s in stcs], axis=0)

                stc_avg = deepcopy(stcs[0])
                stc_avg.data = avg_data

                # path to subject's data
                sbj_path = op.join(config.data_path, "GM")

                stc_fname = op.join(
                    sbj_path, "STC", "evo_%s_%s%s_mph" % (cond, ev_type, str_notch)
                )

                print("Writing %s." % stc_fname)

                stc_avg.save(stc_fname, overwrite=True)

    return


# raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
run_average_STCs_evoked()
