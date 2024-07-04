"""
@Author: Henrik Bødker Lassen
It is not pretty but it works - But don't blame me if it doesn't work... I know nothing of that error.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import os
import glob
from pathlib import *
from mpl_toolkits import mplot3d
from scipy import signal
from matplotlib.widgets import Slider, Button
import math
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
from multiple_plots_tabs import plotWindow
import re
import sys
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

class NeaspecDataReader():
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data + '\\'
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.header_info()
		self.scan_information()
		self.type_of_scan()
		self.change_cd_back()

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located
		
	def load_data(self):
		os.chdir(self.path_to_data) # Set current directory to the folder containing the files of interest

		self.all_files = [] # Create empty array to contain all txt files in the directory
		for file in glob.glob("*.txt"): # Searches the current directory for all txt files
			self.all_files.append(file) # Appends files found

		self.files_name = self.all_files[-1].replace('.txt','')

		try:
			file_interferograms = [s for s in self.all_files if re.search('.? Interferograms.+', s)]
			self.raw_interferograms = pd.read_csv(file_interferograms[0], sep='\t', comment='#')
			self.raw_interferograms.drop(self.raw_interferograms[self.raw_interferograms.columns[self.raw_interferograms.isna().any()]], axis=1, inplace=True)
		
		except IndexError:
			print('Error: File not found!')
			sys.exit()

		if bool([s for s in self.all_files if re.search('.? Z.+', s)]) == True:
			file_Z = [s for s in self.all_files if re.search('.? Z.+', s)]
			self.raw_Z = pd.read_csv(file_Z[0], sep='\t', comment='#', header=None).transpose()
			self.raw_Z.drop(self.raw_Z[self.raw_Z.columns[self.raw_Z.isna().any()]], axis=1, inplace=True)
 
		return self.raw_interferograms

	def header_info(self):
		with open(self.path_to_data + [s for s in self.all_files if (self.files_name + '.txt') in s][0], 'r') as full_file: # Open the full file with the header
			raw_header = [] 
			for s in full_file:
				if s.startswith("#"): 
					raw_header.append(s[2:]) # If the line starts with "#" append that line into the array 
				else:
					break # Stop when it doesn't start with "#"

		# Cleaning up header
		cleaned_header = [s.replace('Â\xa0', '') for s in raw_header] # Remove part related to win32
		cleaned_header = [s.replace('Â', '') for s in cleaned_header]
		cleaned_header = [s.replace('\n', '') for s in cleaned_header] # Remove newline command

		self.header = dict() # Put the header into a dictionary
		for s in cleaned_header:
			if len(s.split(':')) == 2: # Use the semicolon in the text to seperate key and val
				key, val = s.split(':')
				self.header[key] = list(filter(None, val.split('\t'))) # Use the tab text sign to order the dictionary

	def scan_information(self):
		self._scan_center_ps = float(self.header['Interferometer Center/Distance'][1])
		self._scan_width_ps = float(self.header['Interferometer Center/Distance'][2])
		self.scanner_position_X = float(self.header['Scanner Center Position (X, Y)'][1])
		self.scanner_position_Y = float(self.header['Scanner Center Position (X, Y)'][2])
		self._number_timesteps = int(self.header['Pixel Area (X, Y, Z)'][3])
		self._timestep_ps = self._scan_width_ps / self._number_timesteps
		self._time_start = self._scan_center_ps - (self._scan_width_ps / 2)
		self._time_stop = self._scan_center_ps + (self._scan_width_ps / 2)
		self._frequency_steps = 1 / self._timestep_ps
		self._max_frequency = 0.5 * self._frequency_steps
		self.time = np.linspace(self._time_start, self._time_stop, self._number_timesteps) # Time axis
		self.time_relative = abs(self.time) - min(abs(self.time))
		self.frequency = np.linspace(-self._max_frequency, self._max_frequency, self._number_timesteps) # Frequency axis
		self._number_pixels_X = int(self.header['Pixel Area (X, Y, Z)'][1])
		self._number_pixels_Y = int(self.header['Pixel Area (X, Y, Z)'][2])
		self._averaging = float(self.header['Averaging'][0])
		self._scan_area_X = float(self.header['Scan Area (X, Y, Z)'][1])
		self._scan_area_Y = float(self.header['Scan Area (X, Y, Z)'][2])
		self.spatial_X = np.linspace(0, self._scan_area_X, self._number_pixels_X) # Spatial axis for X
		self.spatial_Y = np.linspace(0, self._scan_area_Y, self._number_pixels_Y) # Spatial axis for Y
		self.spatial_X_relative = np.linspace((self.scanner_position_X - (self._scan_area_X / 2)), (self.scanner_position_X + (self._scan_area_X / 2)), self._number_pixels_X) # Spatial axis for X relavtive to scanner center position
		self.spatial_Y_relative = np.linspace((self.scanner_position_Y - (self._scan_area_Y / 2)), (self.scanner_position_Y + (self._scan_area_Y / 2)), self._number_pixels_Y) # Spatial axis for Y relavtive to scanner center position
		self.pixel_axis = np.linspace(0,self._number_pixels_X-1,self._number_pixels_X)
		self.aspect_ratio = (np.max(self.spatial_X)-np.min(self.spatial_X))/(np.max(self.spatial_Y)-np.min(self.spatial_Y))

	def type_of_scan(self):
	
		if (self._scan_area_Y == 0 and self._scan_area_X != 0) or (self._scan_area_X == 0 and self._scan_area_Y != 0):
			self._scantype = 'Line'
			if self._scan_area_X == 0:
				self._line_number_pixels = self._number_pixels_Y
				self.line_length = self.spatial_Y
			elif self._scan_area_Y == 0:
				self._line_number_pixels = self._number_pixels_X
				self.line_length = self.spatial_X
			self.line_length_pixel_axis = np.arange(self._line_number_pixels)

			self.time_relative = np.flipud(self.time_relative)
		elif (self._scan_area_Y == 0 and self._scan_area_X == 0):
			self._scantype = 'Point'

		elif (self._scan_area_Y != 0 and self._scan_area_X != 0):
			self._scantype = 'Surface'


	def average_runs(self):
		self.averaged_interferograms = self.raw_interferograms.groupby(['Row','Column','Depth'], as_index=False).mean() # Group the data by Row, Column, and Depth, seperating the runs and average them to create one set

		if self._averaging > 1:
			print(f'TDS reader: Number of scan averages: {int(self._averaging)}')

		return self.averaged_interferograms


	def timeWindow(self, i0=10, i1=100, di=10):
		'''
		Function for creating a time window.
		i0 is the point where the window function has first is equal to a half
		i1 is the second point where the window function is equal to a half
		di is the amount of steps used to go from 0 to 1.
		'''
		N = self._number_timesteps
		w = np.zeros(N)
		icos = np.arange(di)
		cos = 0.5*(1+np.cos(icos/di*np.pi))
		w[i0-di//2:i0+di//2] = 1-cos
		w[i1-di//2:i1+di//2] = cos
		w[i0+di//2:i1-di//2] = 1
		self.timewindow = w
		return self.timewindow

	def plot_time_window():
		pass

	def interpolate_spectrum(self, kind='cubic', steps=1000):
		self.interpolated = pd.DataFrame()
		self.frequency_inter1d = np.linspace(-self._max_frequency, self._max_frequency, steps)

		if len(np.shape(self.efield_spectrum['O2'][0])) == 0:
			axis = 0
		elif len(np.shape(self.efield_spectrum['O2'][0])) == 2:
			axis = 1
		elif len(np.shape(self.efield_spectrum['O2'][0])) == 3:
			axis = 2

		for i in range(6):
			k = 'O' + str(i)
			if axis == 0:
				temp = interp1d(self.frequency, self.efield_spectrum[k], axis=axis)
				self.interpolated[k] = temp(self.frequency_inter1d)
			else:
				temp = interp1d(self.frequency, self.efield_spectrum[k][0], axis=axis)
				self.interpolated[k] = [temp(self.frequency_inter1d)]
		
		return self.interpolated

	def zero_pad(self, n=50):
		pass

class point(NeaspecDataReader):
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data + '\\'
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.header_info()
		self.scan_information()
		self.average_runs()
		self.type_of_scan()
		self.change_cd_back()
		self.point_extract_data()
		self.recontruct_efield_point()

	def point_extract_data(self):
		self.signal_amplitude = pd.DataFrame()
		self.signal_phase = pd.DataFrame()

		for i in range(6):
			k = 'O' + str(i) + 'A'
			l = 'O' + str(i) + 'P'

			self.signal_amplitude[k] = self.averaged_interferograms[k]
			self.signal_phase[l] = self.averaged_interferograms[l]

	def recontruct_efield_point(self):

		self.efield_timedomain = pd.DataFrame()
		self.efield_spectrum = pd.DataFrame()

		for k in range(6):

			i = 'O' + str(k) + 'A'
			j = 'O' + str(k) + 'P'
			name = 'O' + str(k)

			phase_offset = self.signal_phase[j][np.argmax(self.signal_amplitude[i])]

			# Reconstruct the field and calculate the spectrum
			self.efield_timedomain[name] = self.signal_amplitude[i] * np.exp(-1j * (self.signal_phase[j] - phase_offset))
			self.efield_spectrum[name] = abs(np.fft.fftshift(np.conj(np.fft.fft(self.efield_timedomain[name]))))

	def plot_point_timetrace(self, harmonic=2):

		i = 'O' + str(harmonic)

		fig = plt.figure()
		plt.plot(self.time_relative, np.real(self.efield_timedomain[i]))
		plt.xlabel('Time [ps]')
		plt.ylabel('Amplitude [a.u.]')
		plt.title(f'Time trace: {i}')

		return fig

	def plot_point_spectrum(self, harmonic=2, show_interpolate=True):

		i = 'O' + str(harmonic)

		interpolated = self.interpolate_spectrum(steps=1000)

		fig = plt.figure()
		plt.plot(self.frequency, self.efield_spectrum[i])
		plt.plot(self.frequency_inter1d, interpolated[i])
		plt.xlabel('Frequency [THz]')
		plt.ylabel('Amplitude [a.u.]')
		plt.xlim(0, 5)
		plt.title(f'Spectrum: {i}')

		return fig


class line(NeaspecDataReader):
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data + '\\'
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.header_info()
		self.scan_information()
		self.average_runs()
		self.type_of_scan()
		self.change_cd_back()
		self.line_extract_data()
		self.recontruct_efield_line()
		self.convert_line_scan_to_WL()

	def line_extract_data(self):

		self.signal_amplitude = pd.DataFrame()
		self.signal_phase = pd.DataFrame()

		for i in range(6):
		# Create string for determining the chosen harmonic
			k = 'O' + str(i) + 'A' 
			j = 'O' + str(i) + 'P'

			temp = np.array(np.split(self.averaged_interferograms[k].to_numpy(), self._line_number_pixels))
			temp2 = np.array(np.split(self.averaged_interferograms[j].to_numpy(), self._line_number_pixels))

			self.signal_amplitude[k] = [temp]
			self.signal_phase[j] = [temp2]

	def recontruct_efield_line(self):

		self.efield_timedomain = pd.DataFrame()
		self.efield_spectrum = pd.DataFrame()

		for k in range(6):

			i = 'O' + str(k) + 'A'
			j = 'O' + str(k) + 'P'
			name = 'O' + str(k)

			signal_amplitude_norm = self.signal_amplitude[i][0] / self.signal_amplitude[i][0].max(axis=1, keepdims=True)

			self.peaks_list = []

			for row in signal_amplitude_norm:
				peaks, _ = find_peaks(row, height=0.85)
				peaks_list.append(peaks[-1])

			self.peak_position = np.argmax(self.signal_amplitude[i][0], axis=1)

			print(self.peak_position)
			print(peaks_list)

			self.phase_offset = self.signal_phase[j][0][np.arange(self.peak_position.size), self.peak_position]
			self.phase_offset_stacked = np.expand_dims(self.phase_offset, axis=1)

			self.efield_timedomain[name] = [np.fliplr(self.signal_amplitude[i][0] * np.exp(-1j * (self.signal_phase[j][0] - self.phase_offset_stacked)))]
			self.efield_spectrum[name] = [abs(np.fft.fftshift(np.conj(np.fft.fft(self.efield_timedomain[name][0])), axes=1))]

		return self.efield_timedomain, self.efield_spectrum

	def check_phase_line(self, pos=0, harmonic=2):

		i = 'O' + str(harmonic) + 'A'
		h = 'O' + str(harmonic) + 'P'
		k = 'O' + str(harmonic)

		fig = plt.figure()
		plt.plot(self.signal_amplitude[i][0][pos], label='real')
		plt.plot(self.signal_phase[h][0][pos], label='imag')
		plt.legend()
		plt.show()

		fig2 = plt.figure()
		plt.plot(np.real(self.efield_timedomain[k][0][pos]), label='real')
		plt.plot(np.imag(self.efield_timedomain[k][0][pos]), label='imag')
		plt.legend()
		plt.show()

		fig3 = plt.figure()
		plt.pcolormesh(np.real(self.efield_timedomain[k][0]), shading='auto')
		plt.title('real')
		plt.show()

		fig4 = plt.figure()
		plt.pcolormesh(np.imag(self.efield_timedomain[k][0]), shading='auto')
		plt.title('imag')
		plt.show()

		fig5 = plt.figure()
		plt.pcolormesh((self.signal_amplitude[i][0]), shading='auto')
		plt.title('real')
		plt.show()

		fig6 = plt.figure()
		plt.pcolormesh((self.signal_phase[h][0]), shading='auto')
		plt.title('imag')
		plt.show()

	def referenced_signal_line(self):
		# Create a reference by averaging the first three scans of the linecans, assuming the linescan is done over an edge of substrate and sample
		if np.sum(abs(self.efield_frequencydomain_windowed_tukey[0])) > np.sum(abs(self.efield_frequencydomain_windowed_tukey[-1])): # Determine the direction of the scan. Is sample or substrate first. 
			self.reference_signal = np.divide((self.efield_frequencydomain_windowed_tukey[-1] + self.efield_frequencydomain_windowed_tukey[-2] + self.efield_frequencydomain_windowed_tukey[-3]), 3)
		else:
			self.reference_signal = np.divide((self.efield_frequencydomain_windowed_tukey[0] + self.efield_frequencydomain_windowed_tukey[1] + self.efield_frequencydomain_windowed_tukey[2]), 3)

		# Create a complex placeholder array
		self.efield_frequencydomain_referenced_window_tukey = np.zeros(np.shape(self.amplitude_sig), dtype=complex)
		# Divide each Fourier transformed signal with the reference signal
		for i in range(len(self.efield_frequencydomain_windowed_tukey)):
			self.efield_frequencydomain_referenced_window_tukey[i] = np.divide(self.efield_frequencydomain_windowed_tukey[i], self.reference_signal)

	def convert_line_scan_to_WL(self):
		self.efield_timedomain_WL = pd.DataFrame()
		
		for i in range(6):
			order = 'O' + str(i)

			self.efield_timedomain_WL[order] = [np.amax(self.efield_timedomain[order][0], axis=1)]

		return self.efield_timedomain_WL

	def convert_line_scan_to_WL_frequency(self, v=1):

		freq_index = (np.abs(self.frequency - v)).argmin()

		self.frequency_WL = self.frequency[freq_index]
		self.efield_spectrum_WL = pd.DataFrame()

		for i in range(6):

			order = 'O' + str(i)
			temp = self.efield_spectrum[order][0]
			self.efield_spectrum_WL[order] = [temp[:, freq_index]]

		return self.efield_spectrum_WL

	def plot_line_spectrum_2Dcontour_slider(self, c_max=None, c_min=0, harmonic=2, cmap='hot', interpolation=True):

		order = 'O' + str(harmonic)

		if c_max: 
			cb_max = c_max
		else:
			cb_max = np.max(self.efield_spectrum[order][0])

		fig2 = plt.subplot(111)
		plt.subplots_adjust(right=0.75)

		if interpolation:
			img = plt.contourf(self.frequency, self.line_length, self.efield_spectrum[order][0], np.linspace(c_min,cb_max,100), cmap=cmap)
		else:
			img = plt.pcolormesh(self.frequency, self.line_length, self.efield_spectrum[order][0], shading='auto', cmap=cmap)
		cb = plt.colorbar(img)
		plt.xlabel('Frequency [THz]')
		plt.ylabel('Scan distance X [$\mu$m]')
		plt.xlim(0, 2.5)

		slider_color = 'lightgoldenrodyellow'
		cbarslider = plt.axes([0.85, 0.25, 0.0225, 0.5], facecolor=slider_color)
		s_cmax = Slider(cbarslider, 'max', c_min, np.max(self.efield_spectrum[order][0]), valinit=cb_max, orientation='vertical')

		def update(val, s=None):
			_cmax = s_cmax.val
			img.set_clim([c_min, _cmax])
			plt.draw()

		s_cmax.on_changed(update)

		plt.show()

	def plot_line_spectrum_2Dcontour(self, cmap='hot', invert_yaxis=None, harmonic=2, interpolation=None):

		order = 'O' + str(harmonic)

		fig = plt.figure()
		plt.pcolormesh(self.frequency, self.line_length, self.efield_spectrum[order][0], cmap=cmap, shading='auto')
		plt.ylabel('Scan distance [$\mu$m]')
		if invert_yaxis==True: plt.gca().invert_yaxis()
		plt.xlabel('Frequency [THz]')
		plt.xlim(0, 2.5)
		plt.title(f'TDS linescan frequency-domain: {order}')
		plt.colorbar(label='Intensity [a.u.]', format='%.2f')

		plt.show()

		return fig

	def plot_line_time_2Dcontour(self, cmap='hot', invert_yaxis=None, harmonic=2):

		order = 'O' + str(harmonic)

		fig = plt.figure()
		plt.pcolormesh(self.time_relative, self.line_length, np.real(self.efield_timedomain[order][0]), cmap=cmap, shading='auto')
		plt.ylabel('Scan distance [$\mu$m]')
		if invert_yaxis==True: plt.gca().invert_yaxis()
		plt.xlabel('Time [ps]')
		plt.title(f'TDS linescan time-domain: {order}')
		plt.colorbar(label='Intensity [a.u.]', format='%.2f')

		plt.show()

		return fig

	def plot_line_time_domain_WL(self, harmonic=2):

		order = 'O' + str(harmonic)

		fig = plt.figure()
		plt.plot(self.line_length, self.efield_timedomain_WL[order][0])
		plt.xlabel('Scan distance [$\mu$m]')
		plt.ylabel('Amplitude [a.u.]')
		plt.title(f'Time-domain WL: {order}')

		return fig

	def plot_line_timetrace_locate_tracenr(self, harmonic=2):

		order = 'O' + str(harmonic)

		figtracenr = plt.figure()
		plt.pcolormesh(self.time_relative, self.line_length_pixel_axis, np.real(self.efield_timedomain[order][0]), shading='auto')
		plt.xlabel('Trace nr.')
		plt.ylabel('Time [ps]')
		plt.colorbar(label='Intensity [a.u.]', format='%.2f')

		plt.show()

		return figtracenr		

	def plot_line_timetrace_and_spectrum(self, tracenr=1, title='', harmonic=2):

		order = 'O' + str(harmonic)

		fig = plt.figure()
		fig.set_size_inches(15, 5)
		plt.suptitle(f'{order}' + title)
		plt.subplot(1,2,1)
		plt.plot(self.time_relative, np.real(self.efield_timedomain[order][0][tracenr]))
		# plt.gca().invert_xaxis()
		plt.xlabel('Time [ps]')
		plt.ylabel('Intensity [a.u.]')
		plt.title('Time trace')
		plt.subplot(1,2,2)
		plt.plot(self.frequency, self.efield_spectrum[order][0][tracenr])
		plt.xlim(0.1,5)
		plt.xlabel('Frequency [THz]')
		plt.ylabel('Intensity [a.u.]')
		plt.title('Spectrum')
		plt.tight_layout()

		plt.show()

		return fig

	def plot_line_phase(self, harmonic=2, tracenr=20, substrate_position=0):

		# Needs to be checked for errors
		# Does not work at the moment

		order = 'O' + str(harmonic)

		fig13 = plt.figure()
		plt.plot(np.angle(self.efield_spectrum[order][0][tracenr]))

		fig14 = plt.figure()
		plt.plot(np.unwrap(np.angle(self.efield_spectrum[order][0][tracenr])))

		fig15 = plt.figure()
		plt.plot(self.frequency, np.unwrap(np.angle(self.efield_spectrum[order][0][tracenr])-np.angle(self.efield_spectrum[order][0][substrate_position])), 
			'r', label='single')
		plt.xlabel('Frequency [THz]')
		plt.legend(loc='upper right')

		plt.show()

class surface(NeaspecDataReader):
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data + '\\'
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.header_info()
		self.scan_information()
		self.average_runs()
		self.type_of_scan()
		self.change_cd_back()
		self.surface_extract_data()
		self.recontruct_efield_surface()
		self.convert_surface_scan_to_WL()

	def surface_extract_data(self):

		self.signal_amplitude = pd.DataFrame()
		self.signal_phase = pd.DataFrame()

		for i in range(6):

			k = 'O' + str(i) + 'A' 
			j = 'O' + str(i) + 'P'

			temp_A = np.array(np.split(self.averaged_interferograms[k].to_numpy(), self._number_pixels_Y))
			temp_A2 = np.array(np.split(temp_A, self._number_pixels_X, axis=1))

			temp_P = np.array(np.split(self.averaged_interferograms[j].to_numpy(), self._number_pixels_Y))
			temp_P2 = np.array(np.split(temp_P, self._number_pixels_X, axis=1))

			self.signal_amplitude[k] = [temp_A2]
			self.signal_phase[j] = [temp_P2]

	def recontruct_efield_surface(self):

		self.efield_timedomain = pd.DataFrame()
		self.efield_spectrum = pd.DataFrame()

		for k in range(6):

			i = 'O' + str(k) + 'A'
			j = 'O' + str(k) + 'P'
			name = 'O' + str(k)

			peak_position = np.argmax(self.signal_amplitude[i][0], axis=2)
			phase_offset = self.signal_phase[j][0][np.arange(peak_position.shape[0])[:,None], np.arange(peak_position.shape[1])[None], peak_position]
			phase_offset_expanded = np.expand_dims(phase_offset, axis=2)

			self.efield_timedomain[name] = [self.signal_amplitude[i][0] * np.exp(-1j * (self.signal_phase[j][0] - phase_offset_expanded))]
			self.efield_spectrum[name] = [abs(np.fft.fftshift(np.conj(np.fft.fft(self.efield_timedomain[name][0])), axes=2))]
			# self.efield_spectrum[name] = [abs(np.conj(np.fft.fft(self.efield_timedomain[name][0])))]

		return self.efield_timedomain, self.efield_spectrum

	def convert_surface_scan_to_WL(self):
		self.efield_timedomain_WL = pd.DataFrame()
		
		for i in range(6):
			order = 'O' + str(i)

			self.efield_timedomain_WL[order] = [np.fliplr(np.rot90(np.amax(self.efield_timedomain[order][0], axis=2), -1))]

		return self.efield_timedomain_WL

	def convert_surface_scan_to_WL_frequency(self, v=1, data=None, reference=None):

		if data:
			spectrum = data
		else:
			spectrum = self.efield_spectrum

		if reference:
			spectrum / reference

		freq_index = (np.abs(self.frequency - v)).argmin()

		self.frequency_WL = self.frequency[freq_index]
		self.efield_spectrum_WL = pd.DataFrame()

		for i in range(6):

			order = 'O' + str(i)
			temp = spectrum[order][0]
			self.efield_spectrum_WL[order] = [np.fliplr(np.rot90(temp[:, :, freq_index], -1))]

		return self.efield_spectrum_WL, self.frequency_WL

	def set_reference_area_WL(self, loc=(0,0), size=(5,5), show_plot=False, v=None, color='k', linewidth=1, cmap='hot', interpolation=False, title=''):
		'''
		Calculate a reference value from a WL map either from timedomain or for a specific frequency.

			Parameters:
				loc=(0,0): Location of the top left corner of the square in real coordinates, e.g., (5,10) is (5 µm , 10 µm).
				size=(5,5): Size of the square in µm.
				show_plot=False: Show a plot with the square plotted to check position.
				v=None: Frequency WL map at that specific frequency.
				WL=None: Use timedomain WL. Both v and WL cannot be specified at the same time.
				color='k': color of the square used in the plot.
				linewidth=1: Linewidth of the square in the plot.
				cmap='hot': Colormap used in the plot.
				interpolation=False: Whether to use contourf which interpolated the data or pcolormesh which does not interpolate for the plot.
				title='': String that will be placed over the square if description of square is needed.

			Returns:
				self.WL_ref_area: Pandas datastruct with a single reference value for each harmonic order.

		'''

		
		WLmap, freq = self.convert_surface_scan_to_WL_frequency(v=v)
		WLmap_time = self.efield_timedomain_WL

		self.WL_ref_area_freq = pd.DataFrame()
		self.WL_ref_area_time = pd.DataFrame()

		loc_index_x = (np.abs(self.spatial_X - loc[0])).argmin()
		loc_index_y = (np.abs(self.spatial_Y - loc[1])).argmin()

		size_x = (np.abs(self.spatial_X - size[0])).argmin()
		size_y = (np.abs(self.spatial_Y - size[1])).argmin()


		for i in range(6):
			k = 'O' + str(i)

			self.WL_ref_area_freq[k] = [np.mean(WLmap[k][0][loc_index_y:(loc_index_y+size_y), loc_index_x:(loc_index_x+size_x)])]
			self.WL_ref_area_time[k] = [np.mean(WLmap_time[k][0][loc_index_y:(loc_index_y+size_y), loc_index_x:(loc_index_x+size_x)])]

		if show_plot == True:
			fig = plt.figure()
			if interpolation == True:
				plt.contourf(self.spatial_X, self.spatial_Y, np.real(self.efield_timedomain_WL['O2'][0]), 
					np.linspace(np.min(self.efield_timedomain_WL['O2'][0]), np.max(self.efield_timedomain_WL['O2'][0]), 100), cmap=cmap)
			else:
				plt.pcolormesh(self.spatial_X, self.spatial_Y, np.real(self.efield_timedomain_WL['O2'][0]), shading='auto', cmap=cmap)
			plt.gca().invert_yaxis()
			plt.gca().add_patch(patches.Rectangle((loc[0], loc[1]), size[0], size[1], 
				linewidth=linewidth, edgecolor=color, facecolor='none'))
			plt.colorbar()
			plt.text((loc[0]-10),(loc[1]-3), title, color=color)
			plt.xlabel('X [$\mu$m]')
			plt.ylabel('Y [$\mu$m]')
			plt.show()
		else:
			pass

		return self.WL_ref_area_time, self.WL_ref_area_freq

	def plot_surface_WL_background_referenced(self, loc=(0,0), size=(5,5), cmap='RdBu_r', interpolation=False, color='k', linewidth=1, harmonic=2):

		reference = self.set_reference_area_WL(loc=loc, size=size, cmap=cmap, WL=True)

		self.efield_timedomain_WL_referenced = pd.DataFrame()

		for i in range(6):
			k = 'O' + str(i)

			self.efield_timedomain_WL_referenced[k] = [self.efield_timedomain_WL[k][0] / reference[k][0]]

		order = 'O' + str(harmonic)

		fig = plt.figure()
		if interpolation == True:
			plt.contourf(self.spatial_X, self.spatial_Y, np.real(self.efield_timedomain_WL_referenced[order][0]), 
				np.linspace(np.min(self.efield_timedomain_WL_referenced[order][0]), np.max(self.efield_timedomain_WL_referenced[order][0]), 100), cmap=cmap)
		else:
			plt.pcolormesh(self.spatial_X, self.spatial_Y, np.real(self.efield_timedomain_WL_referenced[order][0]), shading='auto', cmap=cmap)
		plt.gca().invert_yaxis()
		plt.gca().add_patch(patches.Rectangle((loc[0], loc[1]), size[0], size[1], 
			linewidth=linewidth, edgecolor=color, facecolor='none'))
		plt.colorbar()
		plt.show()

		return fig

	def plot_surface_WL(self, cmap='RdBu_r', interpolation=False, harmonic=2, invert_yaxis=False):

		order = 'O' + str(harmonic)

		fig = plt.figure()
		if interpolation:
			plt.contourf(self.spatial_X_relative, self.spatial_Y_relative, np.real(self.efield_timedomain_WL[order][0]), 
				np.linspace(np.min(np.real(self.efield_timedomain_WL[order][0])),np.max(np.real(self.efield_timedomain_WL[order][0])),100), cmap=cmap)
		else:
			plt.pcolormesh(self.spatial_X_relative, self.spatial_Y_relative, np.real(self.efield_timedomain_WL[order][0]), shading='auto', cmap=cmap)
		plt.colorbar()
		plt.title(f'Surface TDS WL: {order}')
		if invert_yaxis:
			plt.gca().invert_yaxis()
		plt.xlabel('X [$\mu$m]')
		plt.ylabel('Y [$\mu$m]')

		return fig

	def plot_surface_WL_spectrum(self, v=1, harmonic=2, invert_yaxis=False, interpolation=False, cmap='hot'):

		order = 'O' + str(harmonic)

		maps, freq = self.convert_surface_scan_to_WL_frequency(v=v)

		fig = plt.figure()
		if interpolation:
			plt.contourf(self.spatial_X_relative, self.spatial_Y_relative, maps[order][0], np.linspace(np.min(maps[order][0]), np.max(maps[order][0]), 100), cmap=cmap)
		else:
			plt.pcolormesh(self.spatial_X_relative, self.spatial_Y_relative, maps[order][0], shading='auto', cmap=cmap)
		plt.xlabel('X [$\mu$m]')
		plt.ylabel('Y [$\mu$m]')
		plt.title(f'Surface TDS spectrum WL: {round(freq, 2)} THz')
		if invert_yaxis:
			plt.gca().invert_yaxis()
		plt.colorbar()

		plt.show()

		return fig

def TDSreader(path_to_data='./'):
	temp = NeaspecDataReader(path_to_data=path_to_data)
	if temp._scantype == 'Point':
		print('TDS reader: Point scan detected.')
		data = point(path_to_data)
	elif temp._scantype == 'Line':
		print('TDS reader: Line scan detected.')
		data = line(path_to_data)
	elif temp._scantype == 'Surface':
		print('TDS reader: Surface scan detected.')
		data = surface(path_to_data)
	return data


if __name__ == "__main__":

	path_data = r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\THz_Pt_Capres_TRIM_5nm_Purged_Pt5-3-8 v2\2022-01-19 2505\2022-01-19 182218 THz S surface_scan_100x100um_35x35px_40ms'
	# # data = NeaspecDataReader(path_to_data=path_data)

	# path_data = r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\THz_Pt_Capres_TRIM_5nm_Purged_Pt5-3-8 v2\2022-01-19 2505\2022-01-19 210021 THz S linescan_45um_300px_40ms_3avg'
	data = TDSreader(path_to_data=path_data)

	path_data_p = r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\THz_Pt_Capres_TRIM_5nm_Purged_Pt5-3-8 v2\2022-01-19 2506\2022-01-19 225242 THz S pointscan_on_Pt_40ms_10avg'
	data2 = TDSreader(path_to_data=path_data_p)

	# pat = r'C:\Users\hebla\Desktop'
	# test = NeaspecDataReader(path_to_data=pat)

	paths = r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_TMD_nanoribbon\2022-02-14 2584\2022-02-14 141918 THz S line_TDS_MoS2_nano_ribbon_1.64um_160px_5avg'
	data3 = TDSreader(path_to_data=paths)

