#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Morph individual STCs for FPVS.

==========================================

OH, April 2020
"""

import sys

from os import remove
from os import path as op

import numpy as np

from importlib import reload

import mne

import FPVS_functions as Ff

reload(Ff)

import config_fpvswords as config

reload(config)

print(mne.__version__)

# for some plots of SNRs
unit_scalings = dict(eeg=1.0, mag=1.0, grad=1.0)

subjects_dir = config.subjects_dir

conds = config.do_conds


def morph_stcs(sbj_ids):
    """Morph STCs for sbj_ids."""
    print("Morphing STCs for subjects:")
    print(*sbj_ids)

    # for Evoked data are in one file for all frequencies
    # for STC data are in separate files per condition and freq
    for sbj_id in sbj_ids:  # across all subjects, EDIT ###
        # path to subject's data
        sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

        subject = config.mri_subjects[sbj_id]

        if subject == "":
            print("No subject name for MRI specified - doing nothing now.")

            return

        # create morphing matrix
        tmp = [i for i in config.event_ids[conds[0]].keys()][0]
        fname_stc = op.join(
            sbj_path,
            "STC",
            "PSDSumTopoOdd_%s_%s" % (conds[0], tmp),
        )
        print(tmp, fname_stc)

        stc_from = mne.read_source_estimate(fname_stc)

        print("Computing morphing matrix for %s." % subject)

        morph_mat = mne.compute_source_morph(
            src=stc_from,
            subject_from=subject,
            subject_to=config.stc_morph,
            subjects_dir=subjects_dir,
        )

        for cond in conds:  # conditions
            print("###\nCondition: %s.\n###" % cond)
            if cond[:4] == "rest":
                task = "rest"
            else:
                task = cond

            for ev_type in config.event_ids[cond]:
                print("Reading PSD results from STC files:")

                fname_stc = op.join(
                    sbj_path,
                    "STC",
                    "PSDSumTopoOdd_%s_%s" % (cond, ev_type),
                )
                print(fname_stc)
                stc = mne.read_source_estimate(fname_stc)
                stc_mph = morph_mat.apply(stc)
                fname_mph = op.join(
                    sbj_path,
                    "STC",
                    "PSDSumTopoOdd_%s_%s_mph" % (cond, ev_type),
                )
                print('Saving to %s.' % fname_mph)
                stc_mph.save(fname_mph, overwrite=True)

                fname_stc = op.join(
                    sbj_path,
                    "STC",
                    "PSDSumTopoBase_%s_%s" % (cond, ev_type),
                )
                print(fname_stc)
                stc = mne.read_source_estimate(fname_stc)
                stc_mph = morph_mat.apply(stc)
                fname_mph = op.join(
                    sbj_path,
                    "STC",
                    "PSDSumTopoBase_%s_%s_mph" % (cond, ev_type),
                )
                stc_mph.save(fname_mph, overwrite=True)


                fname_stc = op.join(
                    sbj_path,
                    "STC",
                    "PSDZSumTopoOdd_%s_%s" % (cond, ev_type),
                )
                print(fname_stc)
                stc = mne.read_source_estimate(fname_stc)
                stc_mph = morph_mat.apply(stc)
                fname_mph = op.join(
                    sbj_path,
                    "STC",
                    "PSDZSumTopoOdd_%s_%s_mph" % (cond, ev_type),
                )
                print('Saving to %s.' % fname_mph)
                stc_mph.save(fname_mph, overwrite=True)

                fname_stc = op.join(
                    sbj_path,
                    "STC",
                    "PSDZSumTopoBase_%s_%s" % (cond, ev_type),
                )
                print(fname_stc)
                stc = mne.read_source_estimate(fname_stc)
                stc_mph = morph_mat.apply(stc)
                fname_mph = op.join(
                    sbj_path,
                    "STC",
                    "PSDZSumTopoBase_%s_%s_mph" % (cond, ev_type),
                )
                stc_mph.save(fname_mph, overwrite=True)

                #
                # fname_stc = op.join(
                #     sbj_path, 'STC', '%sPSDTopoZ_%s_%s-lh.stc' %
                #     (prefix, cond, freq)
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path, 'STC', '%sPSDTopoZ_%s_%s_mph-lh.stc' %
                #     (prefix, cond, freq)
                # )
                # stc_mph.save(fname_mph)
                #

                # fname_stc = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDHarm_%s_%s-lh.stc" % (prefix, cond, freq),
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDHarm_%s_%s_mph-lh.stc" % (prefix, cond, freq),
                # )
                # stc_mph.save(fname_mph)
                # #
                # fname_stc = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDHarmBase_%s_%s-lh.stc" % (prefix, cond, freq),
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDHarmBase_%s_%s_mph-lh.stc" % (prefix, cond, freq),
                # )
                # stc_mph.save(fname_mph)
                # #
                # fname_stc = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumTopoOdd_%s_%s-lh.stc" % (prefix, cond, freq),
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumTopoOdd_%s_%s_mph-lh.stc" % (prefix, cond, freq),
                # )
                # stc_mph.save(fname_mph)
                # #
                # fname_stc = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumTopoBase_%s_%s-lh.stc" % (prefix, cond, freq),
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumTopoBase_%s_%s_mph-lh.stc" % (prefix, cond, freq),
                # )
                # stc_mph.save(fname_mph)
                # #
                # fname_stc = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumToposOdd_%s_%s-lh.stc" % (prefix, cond, freq),
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumToposOdd_%s_%s_mph-lh.stc" % (prefix, cond, freq),
                # )
                # stc_mph.save(fname_mph)
                # #
                # fname_stc = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumToposBase_%s_%s-lh.stc" % (prefix, cond, freq),
                # )
                # print(fname_stc)
                # stc = mne.read_source_estimate(fname_stc)
                # stc_mph = morph_mat.apply(stc)
                # fname_mph = op.join(
                #     sbj_path,
                #     "STC",
                #     "%sPSDSumToposBase_%s_%s_mph-lh.stc" % (prefix, cond, freq),
                # )
                # stc_mph.save(fname_mph)

    return


# get all input arguments except first
if (len(sys.argv) == 1) or (
    int(sys.argv[1]) > np.max(list(config.map_subjects.keys()))
):
    # IDs don't start at 0
    sbj_ids = config.do_subjs

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


# requires all subjects to average across
morph_stcs(sbj_ids)
