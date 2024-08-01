#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Split "extra" fiff-files so the two conditions can be
processed separately with existing scripts.
=============================================
OH, June 2024
"""
import sys
import os
from os import path as op
import numpy as np

import mne

from importlib import reload

from mne_bids import BIDSPath

import config_fpvswords as config

reload(config)

def run_split_extra(sbj_id):
    """Duplicate extra for one subject."""

    # lists with input and corresponding output filenames
    sss_map_fnames = [[], []]

    raw_fname_in = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            processing="ica",
            session=None,
            task='extra',
            run=config.conds_runs['extra'],
            suffix='meg',
            extension=".fif",
            datatype="meg",
            root=config.bids_derivatives,
        ).fpath
    )

    raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

    print("Finding events.")
    # note: short event duration
    events = mne.find_events(
        raw,
        stim_channel="STI101",
        consecutive="increasing",
        min_duration=0.002,
        verbose=True,
    )

    extra_eves = {16: [], 17: []}
    raws_new = {16: [], 17: []}
    eves_new = {16: [], 17: []}
    for eve in extra_eves:
        extra_eves[eve] = np.where(events[:, 2] == eve)[0]
        raws_crop = []
        for ee in extra_eves[eve]:
            # crop a bit more than needed
            tmin = (events[ee, 0] - raw.first_samp) / raw.info['sfreq'] - 0.1
            tmax = tmin + 64.
            raws_crop.append(raw.copy().crop(tmin=tmin, tmax=tmax))
        raws_new[eve] = mne.concatenate_raws(raws_crop)
        eves_new[eve] =  mne.find_events(
            raws_new[eve],
            stim_channel="STI101",
            consecutive="increasing",
            min_duration=0.002,
            verbose=True,
        )

    # extra fast
    raw_fname_out = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            processing="ica",
            session=None,
            task='extrafast',
            run='08',
            suffix='meg',
            extension=".fif",
            datatype="meg",
            root=config.bids_derivatives,
        ).fpath
    )

    print('Saving %s.' % raw_fname_out)
    raws_new[16].save(raw_fname_out, overwrite=True)

    event_file = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            session=None,
            task='extrafast',
            run='08',
            suffix="events",
            extension=".txt",
            datatype="meg",
            root=config.bids_derivatives,
            check=False,
        ).fpath
    )
    print("Saving events to %s." % event_file)
    mne.write_events(event_file, eves_new[16], overwrite=True)

    # extra slow
    raw_fname_out = str(
         BIDSPath(
            subject=str(sbj_id).zfill(2),
            processing="ica",
            session=None,
            task='extraslow',
            run='09',
            suffix='meg',
            extension=".fif",
            datatype="meg",
            root=config.bids_derivatives,
        ).fpath
    )

    print('Saving %s.' % raw_fname_out)
    raws_new[17].save(raw_fname_out, overwrite=True)

    event_file = str(
        BIDSPath(
            subject=str(sbj_id).zfill(2),
            session=None,
            task='extraslow',
            run='09',
            suffix="events",
            extension=".txt",
            datatype="meg",
            root=config.bids_derivatives,
            check=False,
        ).fpath
    )
    print("Saving events to %s." % event_file)
    mne.write_events(event_file, eves_new[17], overwrite=True)


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = config.do_subjs

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]

# remove subjects from missing extras
for mm in config.missing_extras:
    if mm in sbj_ids:
        del sbj_ids[sbj_ids.index(mm)]

for ss in sbj_ids:
    run_split_extra(ss)
