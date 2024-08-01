#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Average epochs from FPVS sweeps for ERP analysis.

Read epochs created by FPVS_epochs_sweeps.py and average.
==========================================

"""

import sys

from os import path as op
import numpy as np

from importlib import reload

import mne

import config_fpvswords as config

reload(config)


print(mne.__version__)

# conditions
conds = config.do_conds


def run_average_epochs(sbj_id):
    """Average epochs for one subject."""
    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    # for evoked created with and without Notch filter for base frequency
    for do_notch in [0, 1]:
        if do_notch:  # if Notch filter at base frequency requested
            # add to epoch file name
            str_notch = "_nch"

        else:
            str_notch = ""

        for cond in conds:  # conditions
            for ev_type in config.event_ids[cond]:
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

                print("Reading epochs from %s." % epo_fname)

                epochs = mne.read_epochs(epo_fname)

                # averaging epochs
                evoked = epochs.average()

                # projection necessary for source estimation
                evoked.set_eeg_reference(ref_channels="average", projection=True)

                evo_fname = op.join(
                    sbj_path,
                    "AVE",
                    "%s_f_%s%s-ave.fif" % (cond, ev_type, str_notch),
                )

                print("Writing evoked data to %s." % evo_fname)
                mne.write_evokeds(evo_fname, evoked, overwrite=True)

    return


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    # raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
    data_runs = run_average_epochs(ss)
