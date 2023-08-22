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
import re
import sys
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
from multiple_plots_tabs import plotWindow
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr

class WhitelightScanReader:
	def __init__(self, path_to_data='./'):
		self.path_to_data = path_to_data + '\\'
		self.cd_script = os.getcwd() # Get directory containing script
		self.load_data()
		self.header_information()
		self.scan_information()
		self.structure_data_optical()
		self.structure_data_AFM()
		self.structure_data_mehcanical()
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
		self.scanner_position_X = float(self.header['Scanner Center Position (X, Y)'][1])
		self.scanner_position_Y = float(self.header['Scanner Center Position (X, Y)'][2])
		self._number_pixels_X = int(self.header['Pixel Area (X, Y, Z)'][1])
		self._number_pixels_Y = int(self.header['Pixel Area (X, Y, Z)'][2])
		self._scan_area_X = float(self.header['Scan Area (X, Y, Z)'][1])
		self._scan_area_Y = float(self.header['Scan Area (X, Y, Z)'][2])
		# self.spatial_X = np.linspace((self.scanner_position_X - (self._scan_area_X / 2)), (self.scanner_position_X + (self._scan_area_X / 2)), self._number_pixels_X) # Spatial axis for X
		# self.spatial_Y = np.linspace((self.scanner_position_Y - (self._scan_area_Y / 2)), (self.scanner_position_Y + (self._scan_area_Y / 2)), self._number_pixels_Y) # Spatial axis for Y
		self.spatial_X = np.linspace(0, self._scan_area_X, self._number_pixels_X) # Spatial axis for X
		self.spatial_Y = np.linspace(0, self._scan_area_Y, self._number_pixels_Y) # Spatial axis for Y
		self.aspect_ratio = (np.max(self.spatial_X)-np.min(self.spatial_X))/(np.max(self.spatial_Y)-np.min(self.spatial_Y))
		# self.spatial_Y = np.flipud(self.spatial_Y)

	def structure_data_optical(self):

		# data.WL_O0A.drop(data.WL_O0A.columns[-1], axis=1, inplace=True)

		# Import the optical amplitudes
		self.WL_optical_amplitudes = pd.DataFrame()
		self.WL_optical_phases = pd.DataFrame()

		optical_amplitudes = [s for s in self.all_files if re.search('.? O.A raw.+', s)]
		optical_phases = [s for s in self.all_files if re.search('.? O.P raw.+', s)]

		for i in range(6):
			k = 'O' + str(i) + 'A'
			l = 'O' + str(i) + 'P'

			temp_A = pd.read_csv(optical_amplitudes[i], sep='\t', comment='#', header=None)
			temp_P = pd.read_csv(optical_amplitudes[i], sep='\t', comment='#', header=None)

			temp_A.drop(temp_A[temp_A.columns[temp_A.isna().any()]], axis=1, inplace=True)
			temp_P.drop(temp_P[temp_P.columns[temp_P.isna().any()]], axis=1, inplace=True)

			self.WL_optical_amplitudes[k] = [temp_A.to_numpy()]
			self.WL_optical_phases[l] = [temp_P.to_numpy()]

		return self.WL_optical_amplitudes, self.WL_optical_phases

	def structure_data_AFM(self):

		scale_image_to_nm = 1e9
		scale_iamge_to_um = 1e6

		self.Z_mod_present = bool([s for s in self.all_files if ' Z raw mod' in s])

		# Import AFM image
		if self.Z_mod_present == True:
			self.WL_Z_mod = pd.read_csv([s for s in self.all_files if ' Z raw mod' in s][0], sep='\t', comment='#', header=None)

			for i in range(len(self.WL_Z_mod.columns)):
				if self.WL_Z_mod.isna().any()[i] == True:
					self.WL_Z_mod.drop(self.WL_Z_mod[self.WL_Z_mod.columns[self.WL_Z_mod.isna().any()]], axis=1, inplace=True)
			print('WL reader: Gwyddion modified AFM map present. Use mod=True in plot commands to plot modified AFM map.')
		else:
			self.WL_Z_mod = pd.read_csv([s for s in self.all_files if ' Z raw.txt' in s][0], sep='\t', comment='#', header=None)

			for i in range(len(self.WL_Z_mod.columns)):
				if self.WL_Z_mod.isna().any()[i] == True:
					self.WL_Z_mod.drop(self.WL_Z_mod[self.WL_Z_mod.columns[self.WL_Z_mod.isna().any()]], axis=1, inplace=True)

			self.WL_Z_mod = self.WL_Z_mod * scale_image_to_nm
			# self.WL_Z_mod = self.WL_Z_mod.clip(0,10)
			print('WL reader: No Gwyddion modified AFM map present.')


		self.WL_Z_raw = pd.read_csv([s for s in self.all_files if ' Z raw.txt' in s][0], sep='\t', comment='#', header=None)

		for i in range(len(self.WL_Z_raw.columns)):
			if self.WL_Z_raw.isna().any()[i] == True:
				self.WL_Z_raw.drop(self.WL_Z_raw[self.WL_Z_raw.columns[self.WL_Z_raw.isna().any()]], axis=1, inplace=True)

			
		self.WL_Z_raw = self.WL_Z_raw * scale_image_to_nm
		self.WL_Z_raw = self.WL_Z_raw - self.WL_Z_raw.values.min()
		
		return self.WL_Z_mod

	def structure_data_mehcanical(self):

		# Import mechanical amplitudes and phases

		self.WL_mechanical_amplitudes = pd.DataFrame()
		self.WL_mechanical_phases = pd.DataFrame()

		mechanical_amplitudes = [s for s in self.all_files if re.search('.? M.A raw.+', s)]
		mechanical_phases = [s for s in self.all_files if re.search('.? M.P raw.+', s)]

		for i in range(6):
			k = 'M' + str(i) + 'A'
			l = 'M' + str(i) + 'P'

			temp_A = pd.read_csv(mechanical_amplitudes[i], sep='\t', comment='#', header=None)
			temp_P = pd.read_csv(mechanical_amplitudes[i], sep='\t', comment='#', header=None)

			temp_A.drop(temp_A[temp_A.columns[temp_A.isna().any()]], axis=1, inplace=True)
			temp_P.drop(temp_P[temp_P.columns[temp_P.isna().any()]], axis=1, inplace=True)

			self.WL_mechanical_amplitudes[k] = [temp_A.to_numpy()]
			self.WL_mechanical_phases[l] = [temp_P.to_numpy()]

		return self.WL_mechanical_amplitudes, self.WL_mechanical_phases

	def plot_locate_reference_area(self, loc=[0,0], size=[10,10], harmonic=2, color='b', cmap='hot'):
		"""
		Function for location a reference area on the WL map.
		"loc" shifts the location of the top left corner of the square.
		"size" determines the size of the square in pixels.
		color is the color of the square.
		Inputs: loc_x=[0,0], size=[10,10], harmonic=2, color='b', cmap='hot'.
		"""

		# AT SOME POINT MAKE THIS AN INTERACTIVE PLOT, WHERE YOU CAN CHANGE
		# THE 4 PARAMETERS ON THE PLOT REAL TIME WITHOUT THE NEED FOR REPLOTTING 

		i = 'O' + str(harmonic) + 'A'

		fig = plt.figure()
		plt.contourf(self.WL_optical_amplitudes[i][0], np.linspace(self.WL_optical_amplitudes[i][0].min(), 
			self.WL_optical_amplitudes[i][0].max(), 200), cmap=cmap)
		plt.gca().invert_yaxis()
		plt.gca().add_patch(patches.Rectangle((loc[0],loc[1]), size[0], size[1], linewidth=1, edgecolor=color, facecolor='none'))
		plt.text((loc[0]-20),(loc[1]-3),'Background Area Average', color=color)
		plt.colorbar()
		plt.show()

		return fig

	def plot_WL_place_square(self, loc=[0,0], size=[10,10], linewidth=1, 
		harmonic=2, cmap='afmhot', color='k', title='', averaged=False):
		"""
		Plots a square on the WL map.
		loc is a list that determines the coordinates of the top left corner - in relative coordinates.
		"""

		i = 'O' + str(harmonic) + 'A' 

		if averaged == True:
			WL_map = self.WL_optical_amplitudes_ref_area[i][0]
		elif averaged == False:
			WL_map = self.WL_optical_amplitudes[i][0]

		

		fig = plt.figure()
		ax = plt.axes()
		cm = ax.contourf(self.spatial_X, self.spatial_Y, WL_map, np.linspace(WL_map.min(), WL_map.max(), 200), cmap=cmap)
		ax.invert_yaxis()
		ax.add_patch(patches.Rectangle((loc[0], loc[1]), size[0], size[1], 
			linewidth=linewidth, edgecolor=color, facecolor='none', zorder=2))
		ax.text((loc[0]-10),(loc[1]-3), title, color=color, zorder=2)
		plt.colorbar()
		plt.show()
		print('loc_x = %d' % loc[0])
		print('loc_y = %d' % loc[1])
		print('size_x = %d' % size[0])
		print('size_y = %d' % size[1])

		return fig

	def referenced_signal_to_area(self, loc=[0,0], size=[10,10]):
		"""
		Function for calculating reference signal using an average of an area as reference.
		"loc" is the position of the top left corner of the square.
		"size" determines the dimensions of the square in pixel.
		the function: locate_reference_area() can be used to find optimal parameter values.
		Inputs: loc=[0,0], size=[10,10]
		Returns: Full maps of amplitude and phase referenced by the chosen area.
		"""

		self.WL_optical_amplitudes_ref_area = pd.DataFrame()
		self.WL_optical_phases_ref_area = pd.DataFrame()
		
		for i in range(6):
			k = 'O' + str(i) + 'A'
			l = 'O' + str(i) + 'P'

			self.WL_optical_amplitudes_ref_area[k] = [self.WL_optical_amplitudes[k][0] / 
				self.WL_optical_amplitudes[k][0][loc[1]:(loc[1]+size[1]), loc[0]:(loc[0]+size[0])].mean()]

			self.WL_optical_phases_ref_area[l] = [self.WL_optical_phases[l][0] / 
				self.WL_optical_phases[l][0][loc[1]:(loc[1]+size[1]), loc[0]:(loc[0]+size[0])].mean()]
				
		return self.WL_optical_amplitudes_ref_area, self.WL_optical_phases_ref_area

	def area_average(self, loc=[0,0], size=[10,10]):
		"""
		Function for getting the average of an area on the WL map.
		"loc" is the position of the top left corner of the square.
		"size" determines the dimensions of the square in pixel.
		the function: locate_reference_area() can be used to find optimal parameter values.
		Inputs: loc=[0,0], size=[10,10]
		Returns: One value for the average of the area for both amplitude and phase.
		"""

		area_average_amplitude = pd.DataFrame()
		area_average_phase = pd.DataFrame()
		
		for i in range(6):
			k = 'O' + str(i) + 'A'
			l = 'O' + str(i) + 'P'

			area_average_amplitude[k] = self.WL_optical_amplitudes[k][0][loc[1]:(loc[1]+size[1]), loc[0]:(loc[0]+size[0])].mean()

			area_average_phase[l] = self.WL_optical_phases[l][0][loc[1]:(loc[1]+size[1]), loc[0]:(loc[0]+size[0])].mean()
				
		return area_average_amplitude, area_average_phase


	def referenced_signal_to_minimum(self):
		"""
		References the WL maps to the minimun data point on the map.
		Returns amplitudes and phases in pandas structure.
		"""

		self.WL_optical_amplitudes_ref_min = pd.DataFrame()
		self.WL_optical_phases_ref_min = pd.DataFrame()

		for i in range(6):
			k = 'O' + str(i) + 'A'
			l = 'O' + str(i) + 'P'

			self.WL_optical_amplitudes_ref_min[k] = [self.WL_optical_amplitudes[k][0] / self.WL_optical_amplitudes[k][0].min()]
			self.WL_optical_phases_ref_min[l] = [self.WL_optical_phases[l][0] / self.WL_optical_phases[l][0].min()]

		return self.WL_optical_amplitudes_ref_min, self.WL_optical_phases_ref_min

	def referenced_signal_to_other_harmonic(self):
		"""
		Function that references one harmonic order order to another one.
		Returns: Amplitudes and Phases pandas structure with O3A/O2A, O4A/o3A, O5A/O4A and corresponding phases.
		"""

		self.WL_optical_amplitudes_ref_harmonic = pd.DataFrame()
		self.WL_optical_phases_ref_harmonic = pd.DataFrame()

		for i in range(2,5):
			k = 'O' + str(i) + 'A'
			k1 = 'O' + str(i+1) + 'A'
			l = 'O' + str(i) + 'P'
			l1 = 'O' + str(i+1) + 'P'

			name_O = k1 + '/' + k
			name_P = l1 + '/' + l

			self.WL_optical_amplitudes_ref_harmonic[name_O] = [self.WL_optical_amplitudes[k1][0] / self.WL_optical_amplitudes[k][0]]
			self.WL_optical_phases_ref_harmonic[name_P] = [self.WL_optical_phases[l1][0] / self.WL_optical_phases[l][0]]

		return self.WL_optical_amplitudes_ref_harmonic, self.WL_optical_phases_ref_harmonic

	def plot_all_harmonic_norm_optical_amplitude(self, cmap='copper'):
		"""
		Plots all harmonic normalised optical amplitudes in one window with tabs to swap between them.
		"""

		WL_map_O, WL_map_P = self.referenced_signal_to_other_harmonic()

		pw = plotWindow(windowTitle='Harmonic normalised Optical Amplitudes')

		for i in range(2,5):
			name = 'O' + str(i+1) + 'A/O' + str(i) + 'A'

			fig = plt.figure()
			plt.contourf(self.spatial_X, self.spatial_Y, WL_map_O[name][0], 
				np.linspace(WL_map_O[name][0].min(), WL_map_O[name][0].max(), 200), cmap=cmap)
			plt.colorbar(label='Intensity [a.u.]', format='%.2f')
			plt.xlabel('X [µm]')
			plt.ylabel('Y [µm]')
			# plt.gca().invert_yaxis()
			# plt.tight_layout()
			plt.gca().set_aspect('equal')

			pw.addPlot(name, fig)

		plt.close('all')

		pw.show()

	def plot_all_optical_amplitudes(self, cmap='hot', ref=0, loc=[0,0], size=[10,10]):
		"""
		Plots all optical amplitudes in one window with tabs to swap between them.
		Input "ref" can be called in function if WL maps wants referenced to either an area or the minimum.
		If area is chosen, loc_x, loc_y, size_x, and size_y determies the location and size of the reference area.
		"""

		# matplotlib.use('qt5agg') # If error occurs when calling, then try to either run it with a pylot backend or uncomment this 

		pw = plotWindow(windowTitle='Optical Amplitudes')

		if ref == 'Area' or ref == 'area':
			WL_map_O, WL_map_P = self.referenced_signal_to_area(loc, size)
			print('WL maps referenced to area.')
			print(f'Location = {loc}, and size = {size}.')
		elif ref == 'Minimum' or ref == 'minimum' or ref == 'Min' or ref =='min':
			WL_map_O, WL_map_P = self.referenced_signal_to_minimum()
			print('Minimum on WL map used as reference.')
		else:
			WL_map_O = self.WL_optical_amplitudes
			WL_map_P = self.WL_optical_phases

		for i in range(1,6):
			name = 'O' + str(i) + 'A'

			fig = plt.figure()
			plt.contourf(self.spatial_X, self.spatial_Y, WL_map_O[name][0], 
				np.linspace(WL_map_O[name][0].min(), WL_map_O[name][0].max(), 200), cmap=cmap)
			plt.colorbar(label='Intensity [a.u.]', format='%.2f')
			plt.xlabel('X [µm]')
			plt.ylabel('Y [µm]')
			plt.gca().invert_yaxis()
			# plt.tight_layout()
			plt.gca().set_aspect('equal')

			pw.addPlot(name, fig)

		plt.close('all')

		pw.show()


	def plot_WL_optical_amplitude(self, harmonic=2, cmap='hot', clim=None, ref=None, loc=[0,0], size=[10,10]):

		i = 'O' + str(harmonic) + 'A'

		if ref == 'Area' or ref == 'area':
			WL_map_O, WL_map_P = self.referenced_signal_to_area(loc, size)
			print('WL maps referenced to area.')
			print(f'Location = {loc}, and size = {size_x}.')
		elif ref == 'Minimum' or ref == 'minimum' or ref == 'Min' or ref =='min':
			WL_map_O, WL_map_P = self.referenced_signal_to_minimum()
			print('Minimum on WL map used as reference.')
		else:
			WL_map_O = self.WL_optical_amplitudes
			WL_map_P = self.WL_optical_phases
			print('No reference.')

		fig = plt.figure()
		ax = plt.axes()

		
		cm = ax.contourf(self.spatial_X, self.spatial_Y, WL_map_O[i][0], 
			np.linspace(WL_map_O[i][0].min(), WL_map_O[i][0].max(), 200), cmap=cmap)

		ax.set_aspect('equal')

		the_divider = make_axes_locatable(ax)
		cax = the_divider.append_axes("right", size="5%", pad=0.1)

		cb = plt.colorbar(cm, cax=cax, label='Intensity [a.u.]', format='%.2f')
		ax.set_title(i)
		ax.set_xlabel('X [µm]')
		ax.set_ylabel('Y [µm]')
		cm.set_clim(clim)
		ax.invert_yaxis()
		plt.tight_layout()
		
		plt.show()

		return fig

	def plot_WL_harmonic_norm_optical_amplitude(self, harmonic='O3A/O2A', 
		cmap='RdBu_r', clim=None, ref=None, loc=[0,0], size=[10,10]):

		WL_map_O, WL_map_P = self.referenced_signal_to_other_harmonic()
		name = str(harmonic)

		fig = plt.figure()
		ax = plt.axes()

		if harmonic == 'O3A/O2A' or harmonic == 'O4A/O3A' or harmonic == 'O5A/O4A':
			cm = ax.contourf(self.spatial_X, self.spatial_Y, WL_map_O[harmonic][0], 
				np.linspace(WL_map_O[harmonic][0].min(), WL_map_O[harmonic][0].max(), 200), cmap=cmap)
			ax.set_aspect('equal')

			the_divider = make_axes_locatable(ax)
			cax = the_divider.append_axes("right", size="5%", pad=0.1)

			cb = plt.colorbar(cm, cax=cax, label='Intensity [a.u.]', format='%.2f')
			ax.set_title(harmonic)
			ax.set_xlabel('X [µm]')
			ax.set_ylabel('Y [µm]')
			cm.set_clim(clim)
			ax.invert_yaxis()

			plt.tight_layout()
			plt.show()
		else:
			print('Please choose valid harmonic ratio: O3A/O2A, O4A/O3A, or O5A/O4A. Input as a string.')

		return fig

	def plot_WL_AFM_image(self, clim=None):
	
		fig = plt.figure()
		ax = plt.axes()

		if self.Z_mod_present == True:
			AFM_map = self.WL_Z_mod 
			plt.title('AFM profile - C')
			print('Corrected AFM map.')
		else:
			AFM_map = self.WL_Z_raw
			plt.title('AFM profile')
			print('Raw AFM map - non-corrected.')


		cm = ax.contourf(self.spatial_X, self.spatial_Y, AFM_map, np.linspace(AFM_map.values.min(), AFM_map.values.max(), 500), cmap='gray')

		ax.set_aspect('equal')

		the_divider = make_axes_locatable(ax)
		cax = the_divider.append_axes("right", size="5%", pad=0.1)

		cb = plt.colorbar(cm, cax=cax, label='Z [nm]', format='%.1f')
		ax.set_xlabel('X [µm]')
		ax.set_ylabel('Y [µm]')
		cm.set_clim(clim)
		ax.invert_yaxis()
		plt.tight_layout()
		
		plt.show()

		return fig

	def plot_WL_AFM_image_with_square(self, loc=[0,0], size=[10,10], color='b', clim=None):

		fig = plt.figure()
		ax = plt.axes()

		if self.Z_mod_present == True:
			AFM_map = self.WL_Z_mod 
			plt.title('AFM profile - C')
			print('Corrected AFM map.')
		else:
			AFM_map = self.WL_Z_raw
			plt.title('AFM profile')
			print('Raw AFM map - non-corrected.')

		cm = ax.contourf(self.spatial_X, self.spatial_Y, 
			AFM_map, np.linspace(AFM_map.values.min(), AFM_map.values.max(), 500), cmap='gray')

		ax.add_patch(patches.Rectangle((loc[0], loc[1]), size[0], size[1], linewidth=1, edgecolor=color, facecolor='None', zorder=2))

		ax.set_aspect('equal')
		
		the_divider = make_axes_locatable(ax)
		cax = the_divider.append_axes("right", size="5%", pad=0.1)

		cb = plt.colorbar(cm, cax=cax, label='Z [nm]', format='%.1f')
		ax.set_xlabel('X [µm]')
		ax.set_ylabel('Y [µm]')
		cm.set_clim(clim)

		plt.tight_layout()

		plt.show()

		return fig

	def plot_WL_AFM_image_with_line(self, x=[5,5], y=[6,6], linewidth=2, color='y', scanner_position_X=None, 
		scanner_position_Y=None, length=None, rot=0, clim=None, in_pixel=False):

		fig = plt.figure()
		ax = plt.axes()

		if scanner_position_X and scanner_position_Y and length:
			rotation = rot*np.pi/180

			scanner_center_postion = [scanner_position_X, scanner_position_Y]

			map_center_position = [self._scan_area_X/2, self._scan_area_Y/2]

			new_scanner_center_position = [(scanner_position_X - self.scanner_position_X + map_center_position[0]),
				(scanner_position_Y - self.scanner_position_Y + map_center_position[1])]

			x_shift = np.cos(rotation)*length
			y_shift = np.sin(rotation)*length

			x_start = new_scanner_center_position[0] - x_shift/2
			x_fin = new_scanner_center_position[0] + x_shift/2
			y_start = new_scanner_center_position[1] - y_shift/2
			y_fin = new_scanner_center_position[1] + y_shift/2

			x = np.linspace(x_start, x_fin, 100)
			y = np.linspace(y_start, y_fin, 100)
			plt.plot(x, y, color=color, linewidth=linewidth)
		else:
			if not in_pixel:
				plt.plot([x[0], y[0]], [x[1], y[1]], color=color, linewidth=linewidth)
			else:
				x_pixel = [(x[0]/self._number_pixels_X)*max(self.spatial_X), (x[1]/self._number_pixels_X)*max(self.spatial_X)]
				y_pixel = [(y[0]/self._number_pixels_Y)*max(self.spatial_Y), (Y[1]/self._number_pixels_Y)*max(self.spatial_Y)]

				plt.plot([x_pixel[0], y_pixel[0]], [x_pixel[1], y_pixel[1]], color=color, linewidth=linewidth)


		if self.Z_mod_present == True:
			AFM_map = self.WL_Z_mod 
			plt.title('AFM profile - C')
			print('Corrected AFM map.')
		else:
			AFM_map = self.WL_Z_raw
			plt.title('AFM profile')
			print('Raw AFM map - non-corrected.')
		cm = ax.contourf(self.spatial_X, self.spatial_Y, 
			AFM_map, np.linspace(AFM_map.values.min(), AFM_map.values.max(), 500), cmap='gray')
		ax.set_aspect('equal')
		the_divider = make_axes_locatable(ax)
		cax = the_divider.append_axes("right", size="5%", pad=0.1)

		cb = plt.colorbar(cm, cax=cax, label='Z [nm]', format='%.1f')
		ax.set_xlabel('X [µm]')
		ax.set_ylabel('Y [µm]')
		cm.set_clim(clim)
		ax.invert_yaxis()

		plt.tight_layout()
		plt.show()

		return fig

	def plot_WL_mechanical_amplitude(self, harmonic=1, cmap='gray', clim=None):

		i = 'M' + str(harmonic) + 'A'

		M_map = self.WL_mechanical_amplitudes

		fig = plt.figure()
		ax = plt.axes()
		cm = ax.contourf(self.spatial_X, self.spatial_Y, M_map[i][0], 
			np.linspace(M_map[i][0].min(), M_map[i][0].max(), 200), cmap=cmap)
		ax.set_aspect('equal')
		the_divider = make_axes_locatable(ax)
		cax = the_divider.append_axes("right", size="5%", pad=0.1)

		cb = plt.colorbar(cm, cax=cax, label='Z [nm]', format='%.1f')
		ax.set_xlabel('X [µm]')
		ax.set_ylabel('Y [µm]')
		cm.set_clim(clim)
		ax.invert_yaxis()

		plt.tight_layout()

		plt.show()

		return fig

	def plot_WL_mechanical_phase(self, harmonic=1, cmap='gray', clim=None):

		i = 'M' + str(harmonic) + 'P'

		M_map = self.WL_mechanical_phases

		fig = plt.figure()
		ax = plt.axes()
		cm = ax.contourf(self.spatial_X, self.spatial_Y, M_map[i][0], 
			np.linspace(M_map[i][0].min(), M_map[i][0].max(), 200), cmap=cmap)
		ax.set_aspect('equal')
		the_divider = make_axes_locatable(ax)
		cax = the_divider.append_axes("right", size="5%", pad=0.1)

		cb = plt.colorbar(cm, cax=cax, label='Angle [a.u.]', format='%.1f')
		ax.set_xlabel('X [µm]')
		ax.set_ylabel('Y [µm]')
		cm.set_clim(clim)
		ax.invert_yaxis()

		plt.tight_layout()

		plt.show()

		return fig

	def plot_neaspec_style(self, ref=None, cmap='hot', harmonic=2):

		i = 'O' + str(harmonic) + 'A'

		if ref == 'Area' or ref == 'area':
			WL_map_O, WL_map_P = self.referenced_signal_to_area(loc, size)
			print('WL maps referenced to area.')
			print(f'Location = {loc}, and size = {size_x}.')
		elif ref == 'Minimum' or ref == 'minimum' or ref == 'Min' or ref =='min':
			WL_map_O, WL_map_P = self.referenced_signal_to_minimum()
			print('Minimum on WL map used as reference.')
		else:
			WL_map_O = self.WL_optical_amplitudes
			WL_map_P = self.WL_optical_phases
			print('No reference.')


		fig = plt.figure(figsize=(12,10 ))
		plt.subplot(2,2,1)
		plt.contourf(self.spatial_X, self.spatial_Y, self.WL_Z_mod, 
			np.linspace(self.WL_Z_mod.values.min(), self.WL_Z_mod.values.max(), 500), cmap='gray')
		plt.colorbar(label='Z [nm]', format='%.1f')
		plt.title('AFM')
		plt.xlabel('X [µm]')
		plt.ylabel('Y [µm]')
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.gca().set_aspect('equal')
		plt.subplot(2,2,2)
		plt.contourf(self.spatial_X, self.spatial_Y, self.WL_mechanical_amplitudes['M1A'][0], 
			np.linspace(self.WL_mechanical_amplitudes['M1A'][0].min(), self.WL_mechanical_amplitudes['M1A'][0].max(), 500), cmap='gray')
		plt.colorbar(label='Amplitude [nm]', format='%.1f')
		plt.title('M1A')
		plt.xlabel('X [µm]')
		plt.ylabel('Y [µm]')
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.gca().set_aspect('equal')
		plt.subplot(2,2,3)
		plt.contourf(self.spatial_X, self.spatial_Y, self.WL_mechanical_phases['M1P'][0], 
			np.linspace(self.WL_mechanical_phases['M1P'][0].min(), self.WL_mechanical_phases['M1P'][0].max(), 500), cmap='gray')
		plt.colorbar(label='Angle [a.u.]', format='%.2f')
		plt.title('M1P')
		plt.xlabel('X [µm]')
		plt.ylabel('Y [µm]')
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.gca().set_aspect('equal')
		plt.subplot(2,2,4)
		plt.contourf(self.spatial_X, self.spatial_Y, WL_map_O[i][0], 
			np.linspace(WL_map_O[i][0].min(), WL_map_O[i][0].max(), 200), cmap=cmap)
		plt.colorbar(label='Amplitude Ratio', format='%.2f')
		plt.title('O2A')
		plt.xlabel('X [µm]')
		plt.ylabel('Y [µm]')
		plt.gca().invert_yaxis()
		plt.tight_layout()
		plt.gca().set_aspect('equal')

		return fig

	def change_cd_back(self):
		os.chdir(self.cd_script) # Change directory back to where the script is located

	def save_fig(self, fig, filename=None, path=None, harmonic=2, add_to_filename=''):
		if not path:
			path = './'
		if not filename:
			filename = self.files_name.split('WL ')[1] + '_' + add_to_filename

		fig.savefig(f'{path}/{filename}')


if __name__ == "__main__":


	# folder_5nm = r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\v3\THz_Pt_Capres_5nm_Pt5-3-8 v3\2022-06-29 2930\2022-06-29 184149 THz WL 100x100um_150x150px_40ms\\'
	# folder_15nm = r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\v3\THz_Pt_Capres_15nm_Pt15-5-7 v3\2022-06-29 2919\2022-06-29 120913 THz WL 100x100um_150x150px_40ms\\'
	# folder_3nm = r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\v3\THz_Pt_Capres_3nm_Pt3-1-9 v3\2022-06-29 2934\2022-06-29 215953 THz WL 100x100um_150x150px_40ms\\'
	# folder_10nm = r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\Pt_Capres_TRIM\v3\THz_Pt_Capres_10nm_Pt10-4-9 v3\2022-06-29 2924\2022-06-29 152701 THz WL 100x100um_150x150px_40ms\\'

	# data5 = WhitelightScanReader(path_to_data=folder_5nm)
	# data15 = WhitelightScanReader(path_to_data=folder_15nm)
	# data3 = WhitelightScanReader(path_to_data=folder_3nm)
	# data10 = WhitelightScanReader(path_to_data=folder_10nm)


	# WL5926 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-09 2480\2021-12-09 141825 THz WL Gr_L1511B_5926_WL_10x10um_100x100px\\')
	# WL8026_10x10um = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-09 2481\2021-12-09 143512 THz WL Gr_L1511B_8026_WL_10x10um_100x100px\\')
	# WL8026_5x5um = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-09 2481\2021-12-09 144932 THz WL Gr_L1511B_8026_WL_5x5um_100x100px\\')
	# WL8026_3x3 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-09 2481\2021-12-09 151002 THz WL Gr_L1511B_8026_WL_3x3um_100x100px\\')
	# WL6027_1 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-16 2482\2021-12-16 105604 THz WL Gr_L1511B_6027_WL_10x10um_100x100px\\')
	# WL6027_2 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-16 2483\2021-12-16 120152 THz WL Gr_L1511B_6027_WL_10x10um_150x150px\\')
	# WL7432 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L1511B_Leonid\2021-12-16 2484\2021-12-16 134124 THz WL Gr_L1511B_7432_WL_10x10um_150x150px\\')

	# WL3780 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L0802_Leonid\2022-02-11 2582 L0802D_3780\2022-02-11 131750 THz WL L0802D_3780_50x50um_100x100px\\')
	# WL4504 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Gr_L0802_Leonid\2022-02-09 2552 L0802A_4504\2022-02-09 144739 THz WL L0802A_4504_50x50um_200x200px\\')

	# data = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_TMD_nanoribbon\2022-02-14 2584\2022-02-14 123657 THz WL WL_MoS2_nano_ribbon_9x9um_250x250px\\')

	# data = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Module_CVD_monolayer_Graphene_Nurbek\2021-10-20 2424\2021-10-20 171022 THz WL Scan at hole edge near center of sample again')
	# data2 = WhitelightScanReader(path_to_data=r'C:\Users\hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_Module_CVD_monolayer_Graphene_Nurbek\2021-10-20 2424\2021-10-20 165918 THz WL Scan at hole edge near center of sample')

	# data = WhitelightScanReader(r'C:\Users\h_las\OneDrive\DTU Fotonik\Ph.d\Data\THz_Leonid_Ultrafoil\2022-09-14 3291\2022-09-14 191536 THz WL U5_3432_100x100um_200x200px_40ms_new_tip')
	# data = WhitelightScanReader(r'C:\Users\h_las\OneDrive\DTU Fotonik\Ph.d\Data\THz_TMD_nanoribbon\2022-02-14 2584\2022-02-14 123657 THz WL WL_MoS2_nano_ribbon_9x9um_250x250px')
	# L2710B_1701_2_WL = WhitelightScanReader(
	# 	r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\Make contrast great again\THz_make_contrast_great_again\2022-11-01 3324\2022-11-01 115351 THz WL L2710B_1701_30x30um_150x150px_50ms_tip_507')
	WL_tip516 = WhitelightScanReader(r'C:\Users\Hebla\OneDrive\DTU Fotonik\Ph.d\Data\THz_mtBLG_S4\2023-07-12 3747\2023-07-12 154819 THz WL')
