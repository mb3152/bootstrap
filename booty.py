import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
from multiprocessing import Pool  
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-homedir',action='store',dest='homedir',default='/Users/maxwell/Documents/GitHub/bootstrap')
parser.add_argument('-data',action='store',dest='data',default='msc')
parser.add_argument('-parcels',action='store',dest='parcels',default='custom')
r = parser.parse_args()
locals().update(r.__dict__)
globals().update(r.__dict__)

if parcels == 'custom':parcel_str = 'Individual_timecourses'
if parcels == 'gordon':parcel_str = 'GordonGroup_timecourses'
if parcels == 'schaefer':parcel_str = 'MSC10_SchaeferGroup_timecourses'

global subjects
subjects = np.array(['01','02','03','04','05','06','07','08','09','10'])
global all_n_timepoints
all_n_timepoints = [100,200,300,400,500,600,700,800,900,1000,1500,2000,3000,4000,5000,6000]

"""
1. how do error bars scale with the number of datapoints?
2. how does this vary with individualized parcels?
"""


def error_bars(args=[1,1000,1000]):
	subject,n_timepoints,n_boots = args[0],args[1],args[2]
	"""
	we generate error bars on each edge by sampling
	n_timepoints via n_boots
	we then save the stadard dev of each edge
	"""
	data = np.loadtxt('/%s/MSC/MSC%s_%s.txt'%(homedir,subject,parcel_str))
	if data.shape[1] < n_timepoints:
		return None
	
	boot_results = np.zeros((n_boots,data.shape[0],data.shape[0]))

	for boot in np.arange(n_boots):
		# we make a critical choice here to not sample the entire data
		# this is because sampling all the data is across days / sessions
		# so we make this akin to collecting more data, as 100 time points across multiple days 
		# never occurs when people actually collect data
		timepoints = np.random.choice(np.arange(n_timepoints),n_timepoints,replace=True)
		boot_results[boot] = np.corrcoef(data[:,timepoints])
	
	results = np.zeros((2,data.shape[0],data.shape[0]))
	
	results[0] = np.mean(boot_results,axis=0)
	results[1] = np.std(boot_results,axis=0)
	
	np.save('/%s/results/errorbars_%s_%s_%s.npy'%(homedir,subject,n_timepoints,parcels),results)

def multi_error_bars(n_timepoints,n_boots=1000):
	args = []
	for subject in subjects: args.append([subject,n_timepoints,n_boots])
	pool = Pool(8)
	pool.map(error_bars,args)
	del pool

def all_error_bars():
	for n_timepoints in all_n_timepoints:
		print (n_timepoints)
		multi_error_bars(n_timepoints)

def plot_error_bars():
	sns.set(style='white',font='Palatino')
	cols = columns=['subject','timepoints','error']
	plot_df = pd.DataFrame(columns=cols)
	for n_timepoints in all_n_timepoints:
		for subject in subjects:
			try:results = np.load('/%s/results/errorbars_%s_%s_%s.npy'%(homedir,subject,n_timepoints,parcels))[1]
			except:continue
			triu_indices = np.triu_indices(results.shape[0],1)
			df = pd.DataFrame(columns=cols)
			df['error'] = results[triu_indices]
			df['subject'] = subject
			df['timepoints'] = n_timepoints
			plot_df = plot_df.append(df,ignore_index=True)



















