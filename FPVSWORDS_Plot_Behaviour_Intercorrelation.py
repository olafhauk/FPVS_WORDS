# Plot intercorrlation matrix for behavioural measures
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import config_fpvswords as config

# file with results from behavioural tests
behav_data_fname = config.behav_data_fname

# file stem for figures
fig_fstem = '/home/olaf/MEG/FPVS2/Behaviour/Behaviour_intercorrelation'

dframe = pd.read_excel(behav_data_fname, '4Pandas')

# choose only good subjects
all_subjs = dframe['Subject']
good_subjs = [np.where(x == all_subjs)[0][0] for x in config.do_subjs]
dframe = dframe.iloc[good_subjs]

# same values for everyone, cannot be used for correlation
dframe = dframe.drop(columns=['Subject', 'TransOddEng'])

# all 'English' variables
vars_sel = {}
vars_sel['all'] = dframe.columns.values
# all 'English' variables
vars_sel['eng'] = [x for x in dframe.columns.values if 'Eng' in x or 'Index' in x]
# all 'French' variables
vars_sel['fre'] = [x for x in dframe.columns.values if 'Fre' in x]

for var in vars_sel:

    for method in ['pearson', 'spearman', 'kendall']:

        vars_now = vars_sel[var]
        dframe_now = dframe[vars_now]
        
        corr = dframe_now.corr(numeric_only=True, method=method)

        fig = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
                          vmin=-1., vmax=1., annot=True, annot_kws={"size": 4}, cmap='bwr')

        fig_fname = '%s_%s_%s.jpg' % (fig_fstem, var, method)
        print('Saving figure to %s.' % fig_fname)
        plt.savefig(fig_fname, bbox_inches='tight')
        plt.close('all')

age = dframe['Age']
print('\nMean age: %f (+/- %s, %d to %d)' % (np.mean(age), np.std(age), age.min(), age.max()))

n_fem = np.where(dframe['Gender'] == 'F')[0].size
print('Number of females: %d (out of %d)' % (n_fem, len(dframe)))