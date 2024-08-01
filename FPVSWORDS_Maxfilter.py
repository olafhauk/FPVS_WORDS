#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Maxfilter data from FPVS with Frequency Sweep.

=============================================
OH, modified October 2019
"""
import sys
import os
from os import path as op
import numpy as np

from importlib import reload

from mne_bids import BIDSPath

import config_fpvswords as config

reload(config)

MF = config.MF  # Maxfilter parameters

# conditions to process
conds = config.do_conds


def run_maxfilter(sbj_id):
    """Run maxfilter for one subject."""

    # lists with input and corresponding output filenames
    sss_map_fnames = [[], []]

    for cond in conds:
        if cond[:4] == "rest":
            task = "rest"
        else:
            task = cond
        in_name_bids = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                session=None,
                task=task,
                run=config.conds_runs[cond],
                suffix=None,
                extension=".fif",
                datatype="meg",
                root=config.bids_root,
            ).fpath
        )
        out_name_bids = str(
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

        sss_map_fnames[0].append(in_name_bids)
        sss_map_fnames[1].append(out_name_bids)

    print(sss_map_fnames)

    # path to raw data for maxfilter
    map_subject = config.map_subjects[sbj_id]

    # # raw-filename mappings for this subject
    # sss_map_fname = config.sss_map_fnames[sbj_id]

    raw_path = op.join(config.cbu_path, map_subject[0], map_subject[1])

    # use some file for trans
    raw_fname_ref = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            session=None,
            task="english",
            run=config.conds_runs["english"],
            suffix=None,
            extension=".fif",
            datatype="meg",
            root=config.bids_root,
        ).fpath
    )

    # maxfilter option for bad channels
    if config.bad_channels[sbj_id]["meg"] != []:
        # bad MEG channels without 'MEG'
        bad_channels = config.bad_channels[sbj_id]["meg"]

        bads = [chn[3:] for chn in bad_channels]

        bad_cmd = "-bad %s" % " ".join(bads)

    else:
        bad_cmd = ""

    for raw_fname_in, raw_fname_out in zip(sss_map_fnames[0], sss_map_fnames[1]):
        log_fname_out = raw_fname_out[:-4] + ".log"

        if op.exists(raw_fname_out):  # if necessary delete existing MF output file
            os.remove(raw_fname_out)

        if MF["st_duration"] is None:
            st_cmd = ""
        else:
            st_cmd = " -st %s -corr %f" % (
                str(int(MF["st_duration"])),
                MF["st_correlation"],
            )

        origin = MF["origin"]
        ori_cmd = " -origin %.0f %.0f %.0f " % (
            1000 * origin[0],
            1000 * origin[1],
            1000 * origin[2],
        )

        order_cmd = "-in %d  -out %d" % (MF["in"], MF["out"])

        if not (MF["mv"] == ""):
            mv_cmd = "-movecomp %s" % MF["mv"]

        mf_cmd = (
            "  %s \
                    -f %s \
                    -o %s \
                    -trans %s \
                    -cal %s \
                    -ctc %s \
                    -frame %s \
                    -regularize %s \
                    %s \
                    %s \
                    %s \
                    %s \
                    %s \
                    -autobad on \
                    -force \
                    -linefreq 50 \
                    -v \
                    | tee %s"
            % (
                MF["NM_cmd"],
                raw_fname_in,
                raw_fname_out,
                raw_fname_ref,
                MF["cal"],
                MF["ctc"],
                MF["frame"],
                MF["regularize"],
                st_cmd,
                ori_cmd,
                order_cmd,
                mv_cmd,
                bad_cmd,
                log_fname_out,
            )
        )

        print("Maxfilter command: %s" % mf_cmd)

        # Execute maxfilter command
        os.system(mf_cmd)


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    run_maxfilter(ss)
