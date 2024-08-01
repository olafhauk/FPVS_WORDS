from os import path as op

import numpy as np

from matplotlib import pyplot as plt

from importlib import reload

import config_fpvswords as config

reload(config)

import mne

# from mne.report import Report

plt.ion()

subj = "GM"
sbj_path = config.grandmean_path

### FACE ###
cond = "extrafast"
ev_type = "pw"
comp = "odd"

subjects_dir = "/imaging/hauk/users/olaf/FPVS2/MRI/"

labels = mne.read_labels_from_annot('fsaverage', 'aparc', 'both',
                                    subjects_dir=subjects_dir)

label_names = ['inferiortemporal-lh', 'middletemporal-lh', 'temporalpole-lh', 'superiortemporal-lh',
               'inferiortemporal-rh', 'middletemporal-rh', 'temporalpole-rh', 'superiortemporal-rh']
# get subset of labels specified in labels_ATL
my_labels = []
for j in np.arange(0, len(label_names)):
    my_labels.append([label for label in labels if label.name == label_names[j]][0])

### PSD

stc = mne.read_source_estimate(
    sbj_path + "/STC/psd_harm_topos_%s_%s_%s-lh.stc" % (comp, cond, ev_type)
)

# index to time point 0, which will be plotted
idx0 = np.abs(stc.times).argmin()
thresh = np.abs(stc.data[:, idx0]).max()

# rescale raw MNEs for visualistion
if thresh < 1e-5:
    stc.data *= 1e10
    thresh *= 1e10

lims = [0.5 * thresh, 0.75 * thresh, thresh]
# lims = [0, 0.5 * thresh, thresh]

brain = stc.plot(
    subject="fsaverage",
    initial_time=0.0,
    time_label=None,
    subjects_dir=subjects_dir,
    clim=dict(kind="value", lims=lims),
    hemi="both",
    views="lat",
)

for m in my_labels:
    brain.add_label(m, borders=True, color=m.color)

title_text = "%s %s %s" % (comp, cond, ev_type)
brain.add_text(0.5, 0.9, title_text, "title", font_size=24, justification="center")

### Evoked

# plot_time = 0.2

# stc = mne.read_source_estimate(
#     sbj_path + "/STC/evo_%s_%s_nch_mph-lh.stc" % (cond, ev_type)
# )

# # index to time point 0, which will be plotted
# idx = stc.time_as_index(plot_time)
# thresh = np.abs(stc.data[:, idx]).max()

# print(idx, thresh)

# # rescale raw MNEs for visualistion
# if thresh < 1e-5:
#     stc.data *= 1e10
#     thresh *= 1e10

# pos_lims = [0.5 * thresh, 0.75 * thresh, thresh]
# pos_lims = [0, 0.5 * thresh, thresh]

# brain = stc.plot(
#     subject="fsaverage",
#     initial_time=plot_time,
#     time_label=None,
#     subjects_dir=subjects_dir,
#     clim=dict(kind="value", lims=pos_lims),
#     hemi="both",
#     views="cau",
# )

# title_text = "Evoked %s %s" % (cond, ev_type)
# brain.add_text(0.5, 0.9, title_text, "title", font_size=24, justification="center")
