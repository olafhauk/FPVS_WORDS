#!//imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
=========================================================
Make inverse operator for FPVS.
=========================================================

"""

import sys
from os import path as op

import numpy as np

from importlib import reload

import mne

import config_fpvswords as config

reload(config)

subjects_dir = config.subjects_dir


def run_make_inverse_operator(sbj_id):
    subject = config.mri_subjects[sbj_id]

    if subject == "":
        print("No subject name for MRI specified - doing nothing now.")

        return

    print("Making Inverse Operator for %s." % subject)

    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    # doesn't matter which raw file, as long as transed
    raw_fname = op.join(sbj_path, "rest1_sss_f_ica_raw.fif")

    raw = mne.io.read_raw(raw_fname, preload=True)
    # remove bad channels, as they will have been interpolated
    raw.interpolate_bads(mode="accurate", reset_bads=True)
    print("Setting EEG reference.")  # projection for invop
    raw.del_proj(0)  # remove projection that didn't include bad channels
    raw.set_eeg_reference(ref_channels="average", projection=True)

    fwd_fname = op.join(sbj_path, subject + "_MEG-fwd.fif")
    print("Reading MEG forward solution: %s." % fwd_fname)

    fwd_meg = mne.read_forward_solution(fwd_fname)

    fwd_fname = op.join(sbj_path, subject + "_EEGMEG-fwd.fif")
    print("Reading EEG/MEG forward solution: %s." % fwd_fname)

    fwd_eegmeg = mne.read_forward_solution(fwd_fname)

    fname_cov = op.join(sbj_path, "rest_sss_f_raw_ica-cov.fif")

    print("Reading covariance matrix: %s." % fname_cov)
    noise_cov = mne.cov.read_cov(fname=fname_cov)

    # make inverse operator
    loose = 0.2
    depth = None

    invop_meg = mne.minimum_norm.make_inverse_operator(
        raw.info,
        fwd_meg,
        noise_cov,
        fixed="auto",
        loose=loose,
        depth=depth,
        rank=config.ranks,
    )

    invop_eegmeg = mne.minimum_norm.make_inverse_operator(
        raw.info,
        fwd_eegmeg,
        noise_cov,
        fixed="auto",
        loose=loose,
        depth=depth,
        rank=config.ranks,
    )

    inv_fname = op.join(sbj_path, subject + "_MEG-inv.fif")
    print("Writing MEG inverse operator: %s." % inv_fname)
    mne.minimum_norm.write_inverse_operator(
        fname=inv_fname, inv=invop_meg, overwrite=True
    )

    inv_fname = op.join(sbj_path, subject + "_EEGMEG-inv.fif")
    print("Writing EEG/MEG inverse operator: %s." % inv_fname)
    mne.minimum_norm.write_inverse_operator(
        fname=inv_fname, inv=invop_eegmeg, overwrite=True
    )


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]

for ss in sbj_ids:
    run_make_inverse_operator(ss)

print("Done.")
