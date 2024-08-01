#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Apply inverse operator to evoked data in FPVS2.

Read evoked created by FPVSWORDS_average_epochs.py,.
read and apply inverse operator, write STCs.
==========================================

OH, Feb 2024
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


def run_source_estimation_evoked(sbj_id):
    """Average epochs for one subject."""
    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    # for STC plotting
    subject = config.mri_subjects[sbj_id]

    inv_fname = op.join(sbj_path, subject + "_EEGMEG-inv.fif")

    print("Reading EEG/MEG inverse operator: %s." % inv_fname)
    invop = mne.minimum_norm.read_inverse_operator(inv_fname)

    morph_mat = []  # morphing matrix, one per subject

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
                evo_fname = op.join(
                    sbj_path,
                    "AVE",
                    "%s_f_%s%s-ave.fif" % (cond, ev_type, str_notch),
                )

                print("Reading evoked data from %s." % evo_fname)
                evoked = mne.read_evokeds(evo_fname, 0)

                method = "MNE"
                lambda2 = 1.0 / 3.0**2

                stc = mne.minimum_norm.apply_inverse(
                    evoked, invop, lambda2=lambda2, method=method, pick_ori="normal"
                )

                stc_fname = op.join(
                    sbj_path,
                    "STC",
                    "evo_%s_%s%s" % (cond, ev_type, str_notch),
                )

                print("Saving source estimate to %s." % stc_fname)

                stc.save(stc_fname, overwrite=True)

                if morph_mat == []:
                    print("Computing morphing matrix for %s." % subject)

                    morph_mat = mne.compute_source_morph(
                        src=stc,
                        subject_from=subject,
                        subject_to=config.stc_morph,
                        subjects_dir=config.subjects_dir,
                    )

                stc_mph = morph_mat.apply(stc)

                mph_fname = op.join(
                    sbj_path,
                    "STC",
                    "evo_%s_%s%s_mph" % (cond, ev_type, str_notch),
                )

                print("Saving morphed STC to %s." % mph_fname)

                stc_mph.save(mph_fname, overwrite=True)

    return


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    # raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
    data_runs = run_source_estimation_evoked(ss)
