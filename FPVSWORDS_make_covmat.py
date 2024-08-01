#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Compute noise covariance matrix for FPVS for whole blocks.
==========================================

OH, March 2023
"""

import sys

from os import path as op
import numpy as np

from importlib import reload

import mne
from mne_bids import BIDSPath

import config_fpvswords as config

reload(config)


print(mne.__version__)


def run_make_covmat(sbj_id):
    """Compute spectra for one subject."""

    # Concatenate raws for two rest runs.
    # Assumes that rest files are the first two files in sss_map_fname.
    raw_all = []  # will contain list of all raw files
    for run in ["01", "07"]:
        raw_fname_in = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                processing="ica",
                session=None,
                task="rest",
                run=run,
                suffix=None,
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
            ).fpath
        )

        # Read raw data info
        print('Reading data for covariance matrix from %s.' % raw_fname_in)
        raw_tmp = mne.io.read_raw_fif(raw_fname_in, preload=True)

        raw_all.append(raw_tmp)
        del raw_tmp

    # concatenate raws
    print("Concatenating %d raw files." % len(raw_all))
    raw = mne.concatenate_raws(raw_all)

    # interpolate, because we will need all channels for grand-average in sensor space
    raw.interpolate_bads(mode="accurate", reset_bads=True)

    print("Setting EEG reference.")  # apply to data here
    raw.set_eeg_reference(ref_channels="average", projection=False)

    cov = mne.cov.compute_raw_covariance(
        raw, reject=config.reject, method="auto", rank="info"
    )

    fname_cov = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            processing=None,
            session=None,
            task=None,
            run=None,
            suffix="cov",
            extension="fif",
            datatype=None,
            root=config.bids_derivatives,
            check=False,
        ).fpath
    )

    print("Writing covariance matrix to: %s." % fname_cov)
    mne.cov.write_cov(fname=fname_cov, cov=cov, overwrite=True)

    return


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    # raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
    data_runs = run_make_covmat(ss)
