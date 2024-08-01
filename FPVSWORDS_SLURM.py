#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
==========================================
Submit sbatch jobs for FPVS Frequency Sweep
analysis
SLURM, Python 3
==========================================

OH, modified October 2019
modified by Federica M for more subjects (ERP drive, MEG/FPVS/Scripts_Federica),
then re-adapted by OH Jan 2020
"""

import subprocess
from os import path as op

from importlib import reload

# import study parameters
import config_fpvswords as config

reload(config)

print(__doc__)

# wrapper to run python script via qsub. Python3
fname_wrap = op.join("/", "home", "olaf", "MEG", "FPVS2",
                     "Python", "Python2SLURM.sh")

# indices of subjects to process
subjs = config.do_subjs

job_list = [
    # # Neuromag MAXFILTER
    # {
    #     "N": "F_MF",  # job name
    #     "Py": "FPVSWORDS_Maxfilter",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "16G",  # memory for qsub process
    #     "dep": "",  # name of preceeding process (optional)
    #     "node": "--constraint=maxfilter",
    # },  # node constraint for MF, just picked one
    # # # # PRE-PROCESSING
    # # # ### Filter raw data
    # {
    #     "N": "F_FR",  # job name
    #     "Py": "FPVSWORDS_filter_raw",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "16G",  # memory for qsub process
    #     "dep": "F_MF",
    # },  # name of preceeding process (optional)
    # ## Compute ICA
    # {
    #     "N": "F_CICA",  # job name
    #     "Py": "FPVSWORDS_Compute_ICA",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "32G",  # memory for qsub process
    #     "dep": "F_FR",
    # },  # name of preceeding process (optional)
    # # Apply ICA (change ica_suff in config_sweep.py if necessary)
    # {
    #     "N": "F_AICA",  # job name
    #     "Py": "FPVSWORDS_Apply_ICA",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "F_CICA",
    # },  # name of preceeding process (optional)
    # # Split files for 'extra' run into 'fast' and 'slow'
    # {
    #     "N": "F_Split",  # job name
    #     "Py": "FPVSWORDS_SplitExtra",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "",
    # },  # name of preceeding process (optional)
    # Noise Covariance Matrix
    # {
    #     "N": "F_Cov",  # job name
    #     "Py": "FPVSWORDS_make_covmat",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "4G",  # memory for qsub process
    #     "dep": "",
    # },
    {
        "N": "F_EE",  # job name
        "Py": "FPVSWORDS_export_EDF",  # Python script
        "Ss": subjs,  # subject indices
        "mem": "4G",  # memory for qsub process
        "dep": "",
    },
    # EVOKED ANALYSIS
    # Get epochs from sweeps for ERP analysis
    # lots of epochs, needs enough memory
    # {
    #     "N": "F_EPO",  # job name
    #     "Py": "FPVSWORDS_epoch",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "32G",  # memory for qsub process
    #     "dep": "",
    # },
    # ### Average epochs
    # {
    #     "N": "F_EVO",  # job name
    #     "Py": "FPVSWORDS_average_epochs",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "F_EPO",
    # },
    # Grand-average and plot evoked data (should be run separately)
    # {
    #     "N": "F_GMEvo",  # job name
    #     "Py": "FPVSWORDS_grand_average_evoked",  # Python script
    #     "Ss": [99],  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "",
    # },
    # # # ### Plot evoked curves and topographies
    # # # {'N':   'F_PlEVO',                  # job name
    # # #  'Py':  'FPVS_plot_evoked',          # Python script
    # # #  'Ss':  subjs,                    # subject indices
    # # #  'mem': '2G',                    # memory for qsub process
    # # #  'dep': 'F_EVO'},
    # # #  ### Source estimation for evoked data
    # # # {'N':   'F_MNEEVO',                  # job name
    # # #  'Py':  'FPVS_source_estimation_evoked',          # Python script
    # # #  'Ss':  subjs,                    # subject indices
    # # #  'mem': '2G',                    # memory for qsub process
    # # #  'dep': ''},
    # ## SOURCE ESTIMATION
    # ## Create Source Spaces
    # {'N':   'F_SS',                  # job name
    #  'Py':  'FPVSWORD_make_SourceSpace',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '2G',                    # memory for qsub process
    #  'dep': ''},                      # name of preceeding process (optional)
    # ### Create surfaces via watershed
    # {'N':   'F_WS',                  # job name
    #  'Py':  'FPVSWORD_make_watershed',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '2G',                    # memory for qsub process
    #  'dep': ''},                      # name of preceeding process (optional)
    # ### Create BEM surfaces and model
    # {'N':   'F_BEM',                  # job name
    #  'Py':  'FPVSWORD_make_BEM',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '2G',                    # memory for qsub process
    #  'dep': ''},                      # name of preceeding process (optional)

    # # Create Forward Solution
    # {
    #     "N": "F_Fwd",  # job name
    #     "Py": "FPVSWORD_ForwardSolution",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "F_Cov",
    # },
    # # Create Inverse Operator
    # {
    #     "N": "F_Inv",  # job name
    #     "Py": "FPVSWORD_InverseOperator",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "F_Fwd",
    # },
    # # Create Sensitivity Maps
    # {
    #     "N": "F_SM",  # job name
    #     "Py": "FPVSWORD_SensitivityMaps",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "F_Inv",
    # },

    # ### GRAND-AVERAGE EVOKED SOURCE ESTIMATES and plot (should be run separately)
    # {
    #     "N": "F_ESTC",  # job name
    #     "Py": "FPVSWORDS_source_estimation_evoked",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "",
    # },
    #   ### GRAND-AVERAGE EVOKED SOURCE ESTIMATES and plot (should be run separately)
    # {'N':   'F_GMESTC',                  # job name
    #  'Py':  'FPVS_average_evoked_STCs',          # Python script
    #  'Ss':  [99],                    # subject indices
    #  'mem': '2G',                    # memory for qsub process
    #  'dep': ''},
    # ###
    # # COMPUTE SPECTRA for averaged sweeps and plot (change ica_suff in config_sweep.py if necessary)
    # # Get sweeps from raw data and average(change ica_suff in config_sweep.py if necessary)
    # {
    #     "N": "F_GS",  # job name
    #     "Py": "FPVSWORDS_get_blocks",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "8G",  # memory for qsub process
    #     "dep": "",
    # },
    # # Compute spectra for averaged sweeps
    # {
    #     "N": "F_P_C",  # job name
    #     "Py": "FPVSWORDS_PSD_compute",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "32G",  # memory for qsub process
    #     "dep": "F_GS",
    # },
    # # Plot PSD results
    # {
    #     "N": "F_P_P",  # job name
    #     "Py": "FPVSWORDS_PSD_plot",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "8G",  # memory for qsub process
    #     "dep": "F_P_C",
    # },
    # # MORPH source estimates before averaging
    # {
    #     "N": "F_Mph",  # job name
    #     "Py": "FPVSWORDS_MorphSTC",  # Python script
    #     "Ss": subjs,  # subject indices
    #     "mem": "2G",  # memory for qsub process
    #     "dep": "F_P_C",
    # },
    # GRAND-AVERAGE SPECTRA (only for 1 "subject")
    # cannot be dependent on previous scripts, because they would
    # have to complete for all participants
    # {
    #     "N": "F_GM",  # job name
    #     "Py": "FPVSWORDS_GrandAverage_PSDs",  # Python script
    #     "Ss": [99],  # subject indices
    #     "mem": "1G",  # memory for qsub process
    #     "dep": "",
    # },
    # ### Plot Grand-Mean (only for 1 "subject")
    # {
    #     "N": "F_GMP",  # job name
    #     "Py": "FPVSWORDS_GrandAverage_Plot",  # Python script
    #     "Ss": [99],  # subject indices
    #     "mem": "1G",  # memory for qsub process
    #     "dep": "F_GM",
    # },
]

# Other processing steps
# ### compute band-limited time courses using Hilbert transform
# {'N':   'SR_TFH',                 # job name
#  'Py':  'SR_TFR_Hilbert',         # Python script
#  'Ss':  subjs,                    # subject indices
#  'mem': '4G',                    # memory for qsub process
#  'dep': 'SR_FR'},                      # name of preceeding process (optional)
# # ### Filter raw data FM-> generating also txt event file
# # {'N':   'F_FR',                  # job name
# #  'Py':  'FPVS_filter_raw_sweep_txtfile',          # Python script
# #  'Ss':  subjs,                    # subject indices
# #  'mem': '16G',                    # memory for qsub process
# #  'dep': 'F_FR'},                      # name of preceeding process (optional)


# directory where python scripts are
dir_py = op.join("/", "home", "olaf", "MEG", "FPVS2", "Python")

# directory for qsub output
dir_sbatch = op.join("/", "home", "olaf", "MEG",
                     "FPVS2", "Python", "sbatch_out")

# keep track of qsub Job IDs
Job_IDs = {}

for job in job_list:
    for Ss in job["Ss"]:
        Ss = str(Ss)  # turn into string for filenames etc.

        N = Ss + job["N"]  # add number to front
        Py = op.join(dir_py, job["Py"])
        Cf = ""  # config file not necessary for FPVS
        mem = job["mem"]

        # files for qsub output
        file_out = op.join(
            dir_sbatch, job["N"] + "_" + Cf + "-%s.out" % str(Ss))
        file_err = op.join(
            dir_sbatch, job["N"] + "_" + Cf + "-%s.err" % str(Ss))

        # if job dependent of previous job, get Job ID and produce command
        if "dep" in job:  # if dependency on previous job specified
            if job["dep"] == "":
                dep_str = ""
            else:
                job_id = Job_IDs[Ss + job["dep"], Ss]
                dep_str = "--dependency=afterok:%s" % (job_id)
        else:
            dep_str = ""

        if "node" in job:  # if node constraint present (e.g. Maxfilter)
            node_str = job["node"]
        else:
            node_str = ""

        if "var" in job:  # if variables for python script specified
            var_str = job["var"]
        else:
            var_str = ""

        # sbatch command string to be executed
        sbatch_cmd = (
            'sbatch \
                        -o %s \
                        -e %s \
                        --export=pycmd="%s.py %s",subj_idx=%s,var=%s \
                        --mem=%s -t 1-00:00:00 %s -J %s %s %s'
            % (
                file_out,
                file_err,
                Py,
                Cf,
                Ss,
                var_str,
                mem,
                node_str,
                N,
                dep_str,
                fname_wrap,
            )
        )

        # format string for display
        print_str = sbatch_cmd.replace(" " * 25, "  ")
        print("\n%s\n" % print_str)

        # execute qsub command
        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, shell=True)

        # get linux output
        (out, err) = proc.communicate()

        # keep track of Job IDs from sbatch, for dependencies
        Job_IDs[N, Ss] = str(int(out.split()[-1]))
