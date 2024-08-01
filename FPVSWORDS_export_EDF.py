#!/imaging/local/software/mne_python/mne1.7.1_0/bin/python
"""
Export raw data blocks to EDF format (EEG only).
Needs MNE-Python version 1.2 or higher (run from command line).

==========================================

OH, May 2023
"""

import sys

from os import path as op
import numpy as np

from copy import deepcopy

from importlib import reload

import mne
from mne_bids import BIDSPath
from mne.export import export_raw

import config_fpvswords as config

reload(config)


print(mne.__version__)

# conditions
# conds = config.do_conds
conds = ['extra']
# conds = ["face"]

edf_path = "/imaging/hauk/users/olaf/FPVS2/MEG/EDFs"


def run_get_blocks(sbj_id):
    """Compute spectra for one subject."""
    # path to subject's data
    sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # only use files for correct conditions
    sss_map_fnames = []
    for cond in conds:
        for [fi, ff] in enumerate(tmp_fnames):
            if cond in ff:
                sss_map_fnames.append(ff)

    # initialise dict for results
    data_runs = {}

    for raw_stem_in in sss_map_fnames:
        cond = raw_stem_in.split("_")[0]  # condition
        data_runs[cond] = {}

        raw_fname_in = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                processing="sss",
                session=None,
                task=cond,
                run=config.conds_runs[cond],
                suffix="raw",
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
                check=False,
            ).fpath
        )
        print(raw_fname_in)

        print("\n###\nReading raw file %s." % raw_fname_in)
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        # reduce to EEG only
        raw.pick_types(meg=False, eeg=True, eog=True)

        # # EDF
        fname = "%s_%s_eeg.edf" % (config.map_subjects[sbj_id][0], cond)
        fname_edf_out = op.join(edf_path, fname)

        print("Writing EDF data to %s:" % fname_edf_out)

        export_raw(fname_edf_out, raw, fmt="edf", overwrite=True)

        # # Fiff
        # fname = "%s_%s_eeg_raw.fif" % (config.map_subjects[sbj_id][0], cond)
        # fname_fif_out = op.join(edf_path, fname)

        # print("Writing Fiff data to %s:" % fname_fif_out)

        # raw.save(fname_fif_out, overwrite=True)

        # write digitiser coordinates for EEG
        dig = raw.info["dig"]
        locs = []
        for dd in dig:
            if dd["kind"] == 3:  # if EEG position
                locs.append(dd["r"])

        locs_arr = np.array(locs)

        fname = "%s_%s_eeg_dig.asc" % (config.map_subjects[sbj_id][0], cond)
        fname_dig_out = op.join(edf_path, fname)
        print("Writing EEG digitiser locations to %s.\n" % fname_dig_out)
        np.savetxt(fname_dig_out, locs_arr)

        # Correct latencies of events wrt onset of data file
        # event_file = op.join(sbj_path, raw_stem_in + "_sss_f_raw-eve.fif")
        event_file = str(
            BIDSPath(
                subject=str(sbj_id).zfill(2),
                session=None,
                task='extra',
                run=config.conds_runs['extra'],
                suffix="events",
                extension=".fif",
                datatype="meg",
                root=config.bids_derivatives,
                check=False,
            ).fpath
        )

        print("Reading events from %s." % event_file)
        events = mne.read_events(event_file)
        events[:, 0] -= raw.first_samp  # subtract first sample
        event_file = "%s_%s_eeg_4EDF-eve.txt" % (config.map_subjects[sbj_id][0], cond)
        event_file = op.join(edf_path, event_file)
        print("Saving events to %s." % event_file)
        mne.write_events(event_file, events, overwrite=True)

        # Also write onset latency of raw file as filename - hack for Letswave analysis
        onset_time = raw.first_samp / raw.info['sfreq']
        fname_lat_out = 'raw_%s_eeg_%s.lat' % (cond, str(onset_time))
        print('Writing onset latency to %s.' % fname_lat_out)
        np.savetxt(fname_lat_out, [])

    return


# get all input arguments except first
if len(sys.argv) == 1:
    sbj_ids = config.do_subjs

else:
    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    # raw, psds, psds_as_evo, freqs = run_PSD_raw(ss)
    run_get_blocks(ss)
