#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Apply ICA for FPVSWORDS data.

Decompostion computed in FPVS_Compute_ICA.py
Based on Fiff_Apply_ICA.py.
==========================================
OH, March 2023
"""

import config_fpvswords as config
import sys

from os import remove
from os import path as op
import numpy as np

from importlib import reload

import mne
from mne_bids import BIDSPath

print("MNE Version: %s\n\n" % mne.__version__)  # just in case
print(mne)


reload(config)

# conditions
conds = config.do_conds

###############################################
# Parameters
###############################################


# "emulate" the args from ArgParser in Fiff_Apply_ICA.py
# filenames depend on subject, the rest are variables
class CreateArgs:
    """Parser for input arguments."""

    def __init__(self, FileRawIn, FileICA, FileRawOut):
        self.FileRawIn = FileRawIn
        self.FileICA = FileICA
        self.FileRawOut = FileRawOut


def run_Apply_ICA(sbj_id):
    """Apply previously computed ICA to raw data."""

    FileICA = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            processing=None,
            session=None,
            task=None,
            run=None,
            suffix="ica",
            extension="fif",
            datatype=None,
            root=config.bids_derivatives,
            check=False,
        ).fpath
    )

    # only use files for correct conditions
    for cond in conds:
        if cond[:4] == "rest":
            task = "rest"
        else:
            task = cond
        # result file to write
        FileRawIn = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                processing="filt",
                session=None,
                task=task,
                run=config.conds_runs[cond],
                suffix='meg',
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
            ).fpath
        )

        FileRawOut = str(
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
        print(FileRawIn)
        print(FileRawOut)

        # define variables for the following ICA pipeline
        # this would be from command line arguments of Fiff_Apply_ICA.py
        args = CreateArgs(FileRawIn, FileICA, FileRawOut)

        # Now turn the "fake command line parameters" into variables for the
        # analysis pipeline
        ###
        # ANALAYSIS PARAMETERS
        ###

        # raw-filenames to be subjected to ICA for this subject
        raw_fname_in = FileRawIn

        # save raw with ICA applied and artefacts removed
        if args.FileRawOut == "":
            raw_fname_out = args.FileRawIn

        else:
            raw_fname_out = args.FileRawOut

        # file with ICA decomposition
        if args.FileICA == "":
            ica_fname_in = args.FileRawIn + "-ica.fif"

        else:
            ica_fname_in = args.FileICA

        ###
        # APPLY ICA
        ###

        print("Reading raw file %s" % raw_fname_in)
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        print("Reading ICA file %s" % ica_fname_in)
        ica = mne.preprocessing.read_ica(ica_fname_in)

        print("Applying ICA to raw file")
        ica.apply(raw)

        print("Saving raw file with ICA applied to %s" % raw_fname_out)
        raw.save(raw_fname_out, overwrite=True)

        # check if EEG in raw data
        if not raw.__contains__("eeg"):
            args.ChanTypes = ["meg"]

            print("No EEG found in raw data, continuing with MEG only.")


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    run_Apply_ICA(ss)
