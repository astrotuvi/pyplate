import sys
import random
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import ConfigParser
from math import factorial
from astropy.io import fits
from math import atan, sin, cos, degrees
from .conf import read_conf


class Spectrum:


	def __init__(self):

		self.spectra_points = []
		self.spectra_points_all = []
		self.area_value = False			
		self.start = (0, 0)
		self.s = False

	def starting_point(self):
			
		data = [elem[1] for elem in self.spectra_points_all]
			
		if(self.orientation == "hor"):
			pixel = [elem[0][0] for elem in self.spectra_points_all]
		else:
			pixel = [elem[0][1] for elem in self.spectra_points_all]

		dox = sorted(zip(pixel,data), key=lambda x : x[0])
				
		sorted_pix = [n[0] for n in dox]
		sorted_data = [n[1] for n in dox]
		smooth_len = len(sorted_pix) + 1 if len(sorted_pix) % 2 == 0 \
					else len(sorted_pix)
		smoothed_data = savitzky(sorted_data, smooth_len, 6)	
		gradient = np.gradient(smoothed_data)
		val_gr = max(zip(sorted_pix, gradient), key = lambda t: t[1])[0] 

		nearest = min(zip(sorted_pix, smoothed_data), key=lambda x: \
					abs(x[1] - self.area_value) if abs(x[0] - val_gr) \
					< 100 else 100000)

			
		if(self.orientation == "hor"):
			start_x = (val_gr + nearest[0]) / 2
			start_y = self.crd_1[1] if abs(start_x - self.crd_1[0]) < \
						abs(start_x - self.crd_2[0]) else self.crd_2[1]
		else:
			start_y = (val_gr+nearest[0]) / 2
			start_x = self.crd_1[0] if abs(start_y - self.crd_1[1]) < \
						abs(start_y - self.crd_2[1]) else self.crd_2[0]

		print "starting_point: (%s,%s)" % (start_x, start_y)
		self.start = (start_x, start_y)

		#self.plot_spec(sorted_pix, sorted_data, smoothed_data, gradient)	


	def set_s(self):
		if(np.std([elem[1] for elem in self.spectra_points]) < 2000):
			self.s = True		

	def plot_spec(self, sorted_pix, sorted_data, 
				 		smoothed_data=[], gradient=[]):

		plt.figure(1)
		plt.plot(sorted_pix, sorted_data, label = "noisy data")
		if smoothed_data:
			plt.plot(sorted_pix, smoothed_data, label = "smoothed data")	
		plt.legend(loc = 0)
	
		if gradient:
			plt.figure(2)	
			plt.plot(sorted_pix, gradient, label = "1st der")	
			plt.legend(loc = 0)
	
		plt.show()
		o = raw_input("continue?")

	 
class SpectraExtractor:

	
	def __init__(self, image_data, conf=None):
		
		self.image_data = image_data
		self.shape = self.image_data.shape
		self.spectra = [] # list filled with Spectrum class objects
		self.corners = []
		self.pix_values = []

			
		# ver - vertical image, hor - horizontal image. / set in get_ori()
		self.orientation = False # set at run_main()
		
		# user defined values
		# crop image
		
		self.sc = 1 / 20.0 # remove len*sc edge pixels
		
		y_height = len(self.image_data)
		self.y_start = int(y_height * self.sc)	# y_start
		self.y_end =  y_height - self.y_start
			
		x_width = len(self.image_data[0])
		self.x_start = int(x_width * self.sc)		
		self.x_end = x_width - self.x_start	


		self.slices = 4 # total slices^2 frames

		# searching for object:
		self.searchStep = 20 # skipping every _value_ while searching for ..
			# spectras align pendicular

		self.jumpStep = 100 # skipping every _value_ pixels while searching for
			# spectras along their axis

		self.sur_percent = 0.3 # size percent of spectra length for adding
				# surrounding pixels
		
		# parsing object:
		self.in_searchStep = 3
		self.in_jumpStep = 10

		self.min_spectra_width = 300 # px
		self.min_spectra_height = 300
		self.max_spectra_width = 1300 
		self.max_spectra_height = 150 # px????????????????????	
		self.img_divide()
	
		
		self.parses = 0
		self.successful_parses = 0
		self.aborted_parses = 0

		if(conf):
			self.assign_conf(conf)



	def assign_conf(self, conf):
		"""
		Parse configuration and set class attributes

		"""
		
		if(isinstance(conf, str)):
			conf = read_conf(conf)
		
		self.conf = conf

		for attr in ['slices','searchStep' ,'jumpStep'
					 'in_searchStep', 'in_jumpStep'
					 'min_spectra_width', 'max_spectra_width'
					 'min_spectra_height', 'max_spectra_height']:
			try:
				setattr(self, attr, conf.getint('SpectraExtractor', attr))
			except ValueError:
				print ('Error in configuration file '
						'([{}], {}'.format('SpectraExtractor',attr))
			except ConfigParser.Error:
				print "confparser error.."
				pass

		for attr in ['sc', 'sur_percent']:
			try:
				setattr(self, attr, conf.getfloat('SpectraExtractor', attr))
			except ValueError:
				print ('Error in configuration file '
						'([{}], {}'.format('SpectraExtractor',attr))
			except ConfigParser.Error:
				print "confparser error.."
				pass
	
	def img_divide(self):

		ylen, xlen = self.shape
		row_vals = [ylen / self.slices * i for i in range(int(self.slices) \
								+ 1)]
		col_vals = [xlen / self.slices * i for i in range(int(self.slices) \
								+ 1)]

		for rowVal, nextRowVal in zip(row_vals, row_vals[1:]):
			for colVal, nextColVal in zip(col_vals, col_vals[1:]):
				mat = image_data[rowVal:nextRowVal, colVal:nextColVal]	
				if(np.std(mat) < 2000): 
					val = np.mean(mat) - 2000
				else:
					val = np.mean(mat) - np.std(mat)
				self.pix_values.append((rowVal, nextRowVal, colVal,
										nextColVal, val))		
	
	def get_area_value(self, (x, y)):

		for area in self.pix_values:
			if(y >= area[0] and y < area[1] and 
			   x >= area[2] and x < area[3]):
				return area[4]
		return False
		
	def val_in_range(self, x, y, minval = True):

		val = self.image_data[y][x]

		if(minval == True):
			min_val = self.get_area_value((x, y))
			if not min_val:
				return False
			return (True if val < min_val else False)

		else:
			if(minval < 20):
				return True
			return False			
	
	def check_size(self, x, y, dx, dy):

		# todo o-p
		#if(abs(dx-x) > self.max_spectra_width or 
		# (abs(dy-y) > 250 and abs(dx-x) < 150)):
		#	self.aborted_parses += 1
		#	return True
		return False

	def validate(self, x, y):

		if (self.free((x, y)) and self.val_in_range(x, y)):				
			return True
		return False


	def run_main(self):
		"""
		main function
		in: image_data NxM dim. pixel matrix

		out: #result (1D arr)
		"""
		# main loop 
		
		print "start main loop"
	
		self.orientation = "hor"

		if(self.orientation == "hor"):
			self.search_horizontally() # searching horizontally
	
		if(len(self.spectra) < 15): # if we found under 15 spectra, also search vertically
			horspectra = copy.deepcopy(self.spectra) # horizontally found spectra
			horpoints = copy.deepcopy(self.corners)
			self.spectra = []
			self.corners = []
			self.orientation = "ver"
			self.search_vertically() # now searching vertically

		if(len(self.spectra) < 5): # if found under 5 spectra searching vertically, stick to spectras found horizontally
			self.spectra = horspectra
			self.corners = horpoints

	def set_orientation(self):

		self.orientation = "hor"		
		self.search_horizontally(True)

		if(len(self.spectra) >= 3):
			self.spectra = []
			self.corners = []
			print "horizontal"
			return			

		self.orientation = "ver"
		print "vertical"
	
	def search_horizontally(self,test = False):

		y = self.y_start
		while y < self.y_end:
			x = self.x_start
			while x < self.x_end:
				found_x = self.find(x, y)	
				if found_x:
					x = found_x	
					if(len(self.spectra) >= 10 and test):
						print "3 spec found"
						return
				else:
					x += self.jumpStep #
			y += self.searchStep # move to next row

	def search_vertically(self):

		x = self.x_start	
		while x < self.x_end:
			y = self.y_start
			while y < self.y_end:
				found_y = self.find(x, y)
				if found_y:
					y = found_y	
				else:
					y += self.jumpStep #
			x += self.searchStep # move to next col


	def find(self, x, y):

		if(self.val_in_range(x, y)): # found pixel in color range
			arr = self.parse_linear((x, y)) # find all similar pixels
			if arr:
				return self.analyze(arr)
		return False
	
	def parse_linear(self, (x, y)):	
		"""
			| <- pix
			| <- pix		|
			| <- pix		|
		...---(dx,dy)-----------(dx+xstep,avg_dy)----...
			| <- pix		|
			| <- pix
			..<- pix with too low value		
		
		upper pixels are saved in res_up
		lower ...		  res_down
		
		avg_dy =  
		avg_pixval = pixel_value for each crd / no. of pixels
		so for each (dx,dy) we have 
		corresponding up/down pixels and avg_pixval
		"""

		self.parses += 1
		jump_step = self.in_jumpStep
		obj = [] 
		
		try:
			dx = x
			dy = y		
			direction = 1
				
			while True:

				temp_ob = self._get_surr((dx, dy))
				# add point (dx,dy) with its average pixel value 
				# and its sub-crds	
				obj.append(temp_ob)
	
				if(self.orientation == "hor"):	
					dx += jump_step
					dy = np.median(temp_ob[2], axis = 0)[1] # average y	
				elif(self.orientation == "ver"):
					dy += jump_step
					dx = np.median(temp_ob[2], axis = 0)[0]
				
				if(self.check_size(x, y, dx, dy)): # too wide or smtin. abort.
					return []
		
				if not (self.validate(dx, dy)):
					if(direction == 1): 
						# go back to beginning and check the other way
						dx = x 
						dy = y
						direction = -1 # set new direction
						jump_step = -jump_step	
					else: 
						# we're done here.	
						break	
		
		except IndexError as e: 
			# out of bounds?
			pass
		
		return obj

	def _get_surr(self, (x, y), minval = True):
		search_step = self.in_searchStep
		pixels_up = self.get_((x, y), search_step, minval)
		pixels_down = self.get_((x, y), -search_step, minval)

		if(pixels_up[1] and pixels_down[1]):
			avg_pixval = (pixels_up[1] + pixels_down[1]) / 2
		elif(pixels_up[1]):
			avg_pixval = pixels_up[1]
		elif(pixels_down[1]):
			avg_pixval = pixels_down[1]
		else:	
			avg_pixval = self.image_data[y][x]

		# return point (dx,dy) with its average pixel value and its sub-crds	
		return ((x, y), avg_pixval, pixels_up[0]+pixels_down[0]) 	
	
	def get_(self, (x, y), step, minval = True):	
		"""
		in: ...

		out: (list of coordinates which match the criteria,
		average of pixel values)
		"""

		obj = [(x, y)]
		dx, dy = (x, y)
		try:
			count = 0
			temp = []
			while self.val_in_range(dx, dy, minval if minval == True \
													else count):	
				obj.append((dx, dy))		
				temp.append(self.image_data[dy][dx])
				count += 1
				if(self.orientation == "hor"):
					dy += step
				elif(self.orientation == "ver"):
					dx += step

		except IndexError:
			pass
		
		avgValSum = False
		if temp:
			avgValSum = np.median(temp)

		return (obj, avgValSum) 

	def free(self, (x, y)):

		for rect in self.spectra: 
			if(x >= rect[0] and x<= rect[1] and 
			   y >= rect[2] and y <= rect[3]):
				# not free
				return False
		return True


	def analyze(self, obj):
		"""
		in: list of tuples
		function for determinating if pixels make up a spectra
		
		obj structure: ((dx,dy),avg_pixval,crds)
		"""
		
		# all coordinates
		res = [elem for sublist in obj for elem in sublist[2]]		
		if(len(res) < 3):
			return False
		
		x1_crd = min(res,key = lambda t: t[0])
		x2_crd = max(res,key = lambda t: t[0])
		y1_crd = min(res, key = lambda t: t[1]) 
		y2_crd = max(res, key = lambda t: t[1])

		x1 = x1_crd[0] 
		x2 = x2_crd[0]
		y1 = y1_crd[1]
		y2 = y2_crd[1]

		len_x = x2-x1 # length of spectra on x axis
		len_y = y2-y1 # .... on y axis

		# right now we assume we found a spectra if 
		# length is over ... px and height is under .. px
	
		if((self.orientation == "hor" and 
				len_x > self.min_spectra_width and 
				len_y > 30) 
			or (self.orientation == "ver" and 
				len_y > self.min_spectra_height and 
				len_x > 30)):

			result = Spectrum()
			result.crd_1 = x1_crd
			result.crd_2 = x2_crd
			result.orientation = self.orientation
			result.average_value = np.average([elem[1] for elem in obj])
			result.spectra_points = copy.deepcopy(obj)
			result.set_s()
			result.area_value = self.get_area_value((int((x1 + x2) / 2)
													,int((y1 + y2) / 2)))

			if(self.orientation == "hor"):
				t = int((len_x * self.sur_percent) / self.in_jumpStep)
				obj.extend(self.get_surr(x1_crd, x2_crd, t))

			else:
				t = int((len_y * self.sur_percent) / self.in_jumpStep)
				obj.extend(self.get_surr(y1_crd, y2_crd, t))

				
			result.spectra_points_all = obj
			
			result.starting_point()
		
			self.successful_parses += 1		
			self.corners.append((x1, x2, y1, y2))
			self.spectra.append(result)
			
			return x2_crd[0] + 5 if self.orientation == "hor" \
									else y2_crd[1] + 5

		return False

	def get_surr(self, p1, p2, t = 5):
		"""
		get surrounding pixel values up to t times
		"""

		obj = []
		x1, y1 = p1
		x2, y2 = p2
		dy = float(y2 - y1)
		dx = float(x2 - x1)
			
		if(dy == 0 or dx == 0):
			angle = 0
		else:	
			angle = atan(dy / dx) if self.orientation == "hor" \
									else atan(dx / dy)

		hyp = float(self.in_jumpStep)
		
		for i in range(t * 2):
			if(i == t):
				hyp = -hyp
 
			if(self.orientation == "hor"):
				dx = cos(angle) * (hyp * ((i % t) + 1))
				dy = sin(angle) * (hyp * ((i % t) + 1))
			else:
				dx = sin(angle) * (hyp * ((i % t) + 1))
				dy = cos(angle) * (hyp * ((i % t) + 1))


			if(i < t):
				xn = x2 + dx 
				yn = y2 + dy

			else:
				xn = x1 + dx
				yn = y1 + dy	

			obj.append(self._get_surr((xn, yn), minval = False))
				
		return obj

	
	def loginfo(self):

		print ("-------")
		print ("Parsed total  of %s " % self.parses)
		print ("Of which actually met spectra criteria: %s " 
					% self.successful_parses)
		print ("Of which were too big, out of bounds: %s " 
					% self.aborted_parses)
		print ("-------")

"""proc = SpectraExtractor(image_data, conf)
proc.spectra - all spectra which were found
"""


def savitzky(y, window_size, order, deriv=0, rate=1):
    try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError, msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order+1)
	half_window = (window_size -1) // 2
	# precompute coefficients
	b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
	m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve( m[::-1], y, mode='valid')

