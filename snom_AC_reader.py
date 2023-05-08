"""
@Author: Henrik Bødker Lassen
Don't blame me if it doesn't work... I know nothing of that error.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import os
from mpl_toolkits import mplot3d
from scipy import signal
from matplotlib.widgets import Slider, Button
import math
import glob
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from scipy.optimize import curve_fit
# from lmfit import Model

class ApproachCurveReader:
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data + '\\'
		self.cd_script = os.getcwd() # Get directory containing scrip
		self.load_data()
		self.header_information()
		self.scan_information()
		self.unpack_data()
		self.shift_data()
		self.normalise_data()
		self.fitting_function_scipy()
		self.calculate_r_squared()
		# self.error_propagation()
		self.change_cd_back()

	def load_data(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.txt"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		self.files_name = self.all_files[-1].replace('.txt','')

	def header_information(self):
		with open(self.path_to_data + [s for s in self.all_files if (self.files_name + '.txt') in s][0], 'r') as file:
			header = []
			for s in file:
				if s.startswith('#'):
					header.append(s[2:])
				else:
					break

		cleaned_header = [s.replace('Â\xa0', '') for s in header] # Remove part related to win32
		cleaned_header = [s.replace('Â', '') for s in cleaned_header]
		cleaned_header = [s.replace('\n', '') for s in cleaned_header] # Remove newline command

		self.header = dict()
		for s in cleaned_header:
			if len(s.split(':')) == 2:
				key, val = s.split(':')
				self.header[key] = list(filter(None, val.split('\t')))

	def scan_information(self):
		self._number_pixels_X = int(self.header['Pixel Area (X, Y, Z)'][1])
		self._number_pixels_Y = int(self.header['Pixel Area (X, Y, Z)'][2])
		self._number_pixels_Z = int(self.header['Pixel Area (X, Y, Z)'][3])

	def unpack_data(self):
		self.AC_raw = pd.read_csv(self.all_files[0], sep='\t', comment='#')

		self.AC_raw['Z'] = self.AC_raw['Z'] - min(self.AC_raw['Z']) # Shifts the Z axis so the starting point is zero, and not an arbitrary number
		self.AC_raw['Z'] = self.AC_raw['Z'] * 1e9 # Scales the Z axis to nm instead of m

	def shift_data(self, n=50): # n is the index of which signal drop begins - use if max of M1A does not work
		self.AC = pd.DataFrame()

		if self.AC_raw['M1A'].idxmax() < len(self.AC_raw['M1A'])/3:
			for i in range(1,6):
				self.AC['O'+str(i)+'A'] = self.AC_raw['O'+str(i)+'A'][(self.AC_raw['M1A'].idxmax()+1):]
				self.AC['O'+str(i)+'P'] = self.AC_raw['O'+str(i)+'P'][(self.AC_raw['M1A'].idxmax()+1):]
			
			self.AC['Z'] = self.AC_raw['Z'][(self.AC_raw['M1A'].idxmax()+1):] - self.AC_raw['Z'][(self.AC_raw['M1A'].idxmax()+1)]
			print('Used max(M1A) for data shift.')
		else:
			for i in range(1,6):
				self.AC['O'+str(i)+'A'] = self.AC_raw['O'+str(i)+'A'][n:]
				self.AC['O'+str(i)+'P'] = self.AC_raw['O'+str(i)+'P'][n:]

			self.AC['Z'] = self.AC_raw['Z'][n:] - self.AC_raw['Z'][n]
			print(f'Used n = {n} for data shift.')

	def update_shift_data(self, N):
		self.shift_data(n=N)
		self.normalise_data()
		self.fitting_function_scipy()

	def normalise_data(self):
		self.AC_norm = pd.DataFrame()
		self.AC_raw_norm = pd.DataFrame()
		
		for i in range(1,6):
			self.AC_norm['O'+str(i)+'A'] = self.AC['O'+str(i)+'A'] / max(self.AC['O'+str(i)+'A'])
			self.AC_norm['O'+str(i)+'P'] = self.AC['O'+str(i)+'P'] / max(self.AC['O'+str(i)+'P'])
			self.AC_raw_norm['O'+str(i)+'A'] = self.AC_raw['O'+str(i)+'A'] / max(self.AC_raw['O'+str(i)+'A'])
			self.AC_raw_norm['O'+str(i)+'P'] = self.AC_raw['O'+str(i)+'P'] / max(self.AC_raw['O'+str(i)+'P'])

		self.AC_raw_norm['M1A'] = self.AC_raw['M1A'] / max(self.AC_raw['M1A'])
		self.AC_norm['Z'] = self.AC['Z']
		self.AC_raw_norm['Z'] = self.AC_raw['Z']

	def exponential_decay(self, x, a, b):
		return a * np.exp(-b * x)

	def double_exponential_decay(self, x, a, b, c, d):
		return a * np.exp(-b * x) + c * np.exp(-d * x)

	def fitting_function_scipy(self):
		self.popt = pd.DataFrame()
		self.pcov = pd.DataFrame()

		for i in range(1,6):
			j, k = curve_fit(self.exponential_decay, self.AC_norm['Z'], self.AC_norm['O'+str(i)+'A'].tolist())
			self.popt['O'+str(i)+'A'] = [j]
			self.pcov['O'+str(i)+'A'] = [k]

		self.popt_double = pd.DataFrame()
		self.pcov_double = pd.DataFrame()

		for i in range(1,6):
			j, k = curve_fit(self.double_exponential_decay, self.AC_norm['Z'], self.AC_norm['O'+str(i)+'A'].tolist())
			self.popt_double['O'+str(i)+'A'] = [j]
			self.pcov_double['O'+str(i)+'A'] = [k]

	def calculate_r_squared(self):

		self.exponential_decay_r_squared = pd.DataFrame()
		self.double_exponential_decay_r_squared = pd.DataFrame()

		for i in range(1,6):
			residuals = self.AC_norm['O'+str(i)+'A'].tolist() - self.exponential_decay(self.AC_norm['Z'], *self.popt['O'+str(i)+'A'][0])
			residuals_double = self.AC_norm['O'+str(i)+'A'].tolist() - self.double_exponential_decay(self.AC_norm['Z'], *self.popt_double['O'+str(i)+'A'][0])
			ss_res = np.sum(residuals**2)
			ss_res_double = np.sum(residuals_double**2)

			ss_tot = np.sum((self.AC_norm['O'+str(i)+'A'].tolist() - np.mean(self.AC_norm['O'+str(i)+'A'].tolist()))**2)

			self.exponential_decay_r_squared['O'+str(i)+'A'] = [1 - (ss_res / ss_tot)]
			self.double_exponential_decay_r_squared['O'+str(i)+'A'] = [1 - (ss_res_double / ss_tot)]


	def error_propagation(self):
		a = pd.DataFrame()
		b = pd.DataFrame()
		c = pd.DataFrame()
		d = pd.DataFrame()

		for i in range(1,6):
			a['O'+str(i)+'A'] = self.popt_double['O'+str(i)+'A'][0]
			b['O'+str(i)+'A'] = self.popt_double['O'+str(i)+'A'][1]
			c['O'+str(i)+'A'] = self.popt_double['O'+str(i)+'A'][2]
			d['O'+str(i)+'A'] = self.popt_double['O'+str(i)+'A'][3]



		# self.double_exponential_decay_error_propagation = np.sqrt()

	def plot_approach_curves(self):
		ACfig_raw = plt.figure()
		plt.plot(self.AC_raw_norm['Z'], self.AC_raw_norm['O5A'], label='O5A')
		plt.plot(self.AC_raw_norm['Z'], self.AC_raw_norm['O4A'], label='O4A')
		plt.plot(self.AC_raw_norm['Z'], self.AC_raw_norm['O3A'], label='O3A')
		plt.plot(self.AC_raw_norm['Z'], self.AC_raw_norm['O2A'], label='O2A')
		plt.plot(self.AC_raw_norm['Z'], self.AC_raw_norm['M1A'], label='M1A')
		plt.xlabel('Z [nm]')
		plt.ylabel('Intensity [a.u.]')
		plt.legend(loc='upper right')
		plt.title('Approach Curves raw')

		ACfig = plt.figure()
		plt.plot(self.AC['Z'], self.AC['O5A'], label='O5A')
		plt.plot(self.AC['Z'], self.AC['O4A'], label='O4A')
		plt.plot(self.AC['Z'], self.AC['O3A'], label='O3A')
		plt.plot(self.AC['Z'], self.AC['O2A'], label='O2A')
		# plt.plot(self.AC['Z'], self.AC_norm['M1A'], label='M1A')
		plt.xlabel('Z [nm]')
		plt.ylabel('Intensity [a.u.]')
		plt.legend(loc='upper right')
		plt.title('Approach Curves')

		ACfig_norm = plt.figure()
		plt.plot(self.AC['Z'], self.AC_norm['O5A'], label='O5A')
		plt.plot(self.AC['Z'], self.AC_norm['O4A'], label='O4A')
		plt.plot(self.AC['Z'], self.AC_norm['O3A'], label='O3A')
		plt.plot(self.AC['Z'], self.AC_norm['O2A'], label='O2A')
		# plt.plot(self.AC['Z'], self.AC_norm['M1A'], label='M1A')
		plt.xlabel('Z [nm]')
		plt.ylabel('Intensity [a.u.]')
		plt.title('Normalised Approach Curves')
		plt.legend(loc='upper right')

		# ACfig_shifted = plt.figure()
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted['O5A'], label='O5A')
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted['O4A'], label='O4A')
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted['O3A'], label='O3A')
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted['O2A'], label='O2A')
		# plt.xlabel('Z [nm]')
		# plt.ylabel('Intensity [a.u.]')
		# plt.legend(loc='upper right')
		# plt.title('Approach Curves - shifted')

		# ACfig_shifted_norm = plt.figure()
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O5A'], label='O5A')
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O4A'], label='O4A')
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O3A'], label='O3A')
		# plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O2A'], label='O2A')
		# plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=2, label='1/e')
		# plt.xlabel('Z [nm]')
		# plt.ylabel('Intensity [a.u.]')
		# plt.legend(loc='upper right')
		# plt.title('Normalised Approach Curves - shifted')


	def plot_approach_curve_curve_fit(self, harmonic=2, figsize=(6.4,4.8), normalisation=True, raw_data=False):
		order = 'O' + str(harmonic) + 'A'

		if normalisation == True and raw_data == False:
			Ac = self.AC_norm
		elif normalisation == False and raw_data == False:
			AC = self.AC
		if normalisation == True and raw_data == True:
			Ac = self.AC_raw_norm
		elif normalisation == False and raw_data == True:
			AC = self.AC_raw

		ACfit = plt.figure(figsize=figsize)
		plt.plot(AC['Z'], AC[order], 'b', label=order)
		# plt.plot(AC['Z'], double_exponential_decay(AC['Z'], *))




	def plot_approach_curve_curve_fit(self, harmonic=2):
		if harmonic == 1:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O1A'], 'b', label='O1A')
			plt.plot(self.AC_Z_shifted_nm, exponential_decay(self.AC_Z_shifted_nm, *self.popt1), 'r', label='fit')
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')

			ACfit_double = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O1A'], 'b', label='O1A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O1A.best_fit, 'r', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O1A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')
		if harmonic == 2:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O2A'], 'b', label='O2A')
			plt.plot(self.AC_Z_shifted_nm, exponential_decay(self.AC_Z_shifted_nm, *self.popt2), 'r', label='fit')
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')

			ACfit_double = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O2A'], 'b', label='O2A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O2A.best_fit, 'r', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O2A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')
		if harmonic == 3:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O3A'], 'b', label='O3A')
			plt.plot(self.AC_Z_shifted_nm, exponential_decay(self.AC_Z_shifted_nm, *self.popt3), 'r', label='fit')
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')

			ACfit_double = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O3A'], 'b', label='O3A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O3A.best_fit, 'r', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O3A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')
		if harmonic == 4:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O4A'], 'b', label='O4A')
			plt.plot(self.AC_Z_shifted_nm, exponential_decay(self.AC_Z_shifted_nm, *self.popt4), 'r', label='fit')
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')

			ACfit_double = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O4A'], 'b', label='O4A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O4A.best_fit, 'r', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O4A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')
		if harmonic == 5:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O5A'], label='O5A')
			plt.plot(self.AC_Z_shifted_nm, exponential_decay(self.AC_Z_shifted_nm, *self.popt5), 'r', label='fit')
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')

			ACfit_double = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O5A'], 'b', label='O5A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O5A.best_fit, 'r', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O5A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves')

	def plot_approach_curve_lmfit(self, harmonic=2):
		if harmonic == 1:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O1A'], label='O1A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_exponential_decay_result_O1A.best_fit, 'r', label='fit - $R^2$ = '+str("%.3f" % self.fit_exponential_decay_result_O1A_rsquared))
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O1A.best_fit, 'b', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O1A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves - shifted')
		elif harmonic == 2:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O2A'], label='O2A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_exponential_decay_result_O2A.best_fit, 'r', label='fit - $R^2$ = '+str("%.3f" % self.fit_exponential_decay_result_O2A_rsquared))
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O2A.best_fit, 'b', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O2A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves - shifted')
		elif harmonic == 3:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O3A'], label='O3A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_exponential_decay_result_O3A.best_fit, 'r', label='fit - $R^2$ = '+str("%.3f" % self.fit_exponential_decay_result_O3A_rsquared))
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O3A.best_fit, 'b', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O3A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves - shifted')
		elif harmonic == 4:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O4A'], label='O4A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_exponential_decay_result_O4A.best_fit, 'r', label='fit - $R^2$ = '+str("%.3f" % self.fit_exponential_decay_result_O4A_rsquared))
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O4A.best_fit, 'b', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O4A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves - shifted')
		elif harmonic == 5:
			ACfit = plt.figure()
			plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O5A'], label='O5A')
			plt.plot(self.AC_Z_shifted_nm, self.fit_exponential_decay_result_O5A.best_fit, 'r', label='fit - $R^2$ = '+str("%.3f" % self.fit_exponential_decay_result_O5A_rsquared))
			plt.plot(self.AC_Z_shifted_nm, self.fit_double_exponential_decay_result_O5A.best_fit, 'b', label='fit - double exponential decay - $R^2$ = '+str("%.3f" % self.fit_double_exponential_decay_result_O5A_rsquared))
			plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
			plt.xlabel('Z [nm]')
			plt.ylabel('Intensity [a.u.]')
			plt.legend(loc='upper right')
			plt.title('Normalised Approach Curves - shifted')
		else:
			print('Choose harmonic between 1-5')

	def plot_single_approach_curve(self, harmonic=2,cmap='b'):

		ACfig_shifted_norm = plt.figure()
		plt.plot(self.AC_Z_shifted_nm, self.AC_shifted_norm['O'+str(harmonic)+'A'], c=cmap, label='0'+str(harmonic)+'A')
		plt.plot([self.AC_Z_shifted_nm.iloc[0], self.AC_Z_shifted_nm.iloc[-1]], [1/np.e, 1/np.e], 'k-', linewidth=1.3, label='1/e')
		plt.xlabel('Z [nm]')
		plt.ylabel('Intensity [a.u.]')
		plt.legend(loc='upper right')
		plt.title('Approach Curve')

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located




# def exponential_decay(x,a,b,c,d,e,f,g):
# 	return a * np.exp(-b * x) + c * np.exp(-d * x) + e * np.exp(-f * x) + g


if __name__ == "__main__":

	# AC_folder = r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\THz_Pt_Capres_TRIM_3nm_Purged_Pt3-1-9 v2\2022-01-21 2515\2022-01-21 113855 THz AC AC_on_Pt'

	# data_AC = ApproachCurveReader(path_to_data=AC_folder)

	# AC_tip314_Pt = ApproachCurveReader(r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_resolution_test_Pt_Gr\2022-07-10 2968\2022-07-10 134425 THz AC on Pt15 tip3.14\\')

	data = ApproachCurveReader(r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_resolution_test_Pt_Gr\2022-07-10 2969\2022-07-10 140117 THz AC on Pt15 tip3.14 lowered TA 100ms')


