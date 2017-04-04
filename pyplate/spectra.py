import sys
import random
import time
import copy
import numpy as np
import ConfigParser
from math import factorial
from astropy.io import fits
from math import atan, sin, cos, degrees


try:
    import matplotlib.pyplot as plt
    use_matplotlib = True
except ImportError:
    print "matplotlib not installed, not using"
    

class Spectrum:
    """
    class for one Spectrum
    variables:
        spectrum_points (list) : all spectrum points
        spectrum_points_all (list) : spectrum points with added surrounding
                                     points for getting more precise
                                     starting point
        area_value (float) : local background average pixel value
        starting point (tuple) : x,y coordinates for starting point of this
                                 spectrum
    """

    def __init__(self):

        self.spectrum_points = []
        self.spectrum_points_all = []
        self.area_value = False
        self.average_value = False # average pixel value for this spectrum
        self.start = (0, 0)

    def starting_point(self):
        
        data = [elem[1] for elem in self.spectrum_points_all]

        if(self.orientation == "hor"):
            pixel = [elem[0][0] for elem in self.spectrum_points_all]
        else:
            pixel = [elem[0][1] for elem in self.spectrum_points_all]

        dox = sorted(zip(pixel,data), key=lambda x : x[0])

        sorted_pix = [n[0] for n in dox]
        sorted_data = [n[1] for n in dox]
        smooth_len = len(sorted_pix) + 1 if len(sorted_pix) % 2 == 0 \
                    else len(sorted_pix)
        smoothed_data = self.savitzky(sorted_data, smooth_len, 6)
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

        self.start = (start_x, start_y)

        #self.plot_spec(sorted_pix, sorted_data, smoothed_data, gradient)

    def savitzky(self, y, window_size, order, deriv=0, rate=1):
        """
        algorithm for data smoothing
        """
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
            
        order_range = range(order + 1)
        half_window = (window_size -1) // 2
        
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] \
                                for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        
        return np.convolve( m[::-1], y, mode='valid')   
        
        
    def plot_spec(self, sorted_pix, sorted_data,
                        smoothed_data=[], gradient=[]):
        if(use_matplotlib):
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
    """
    Example of usage:

    image_data = fits.getdata(sys.argv[1])
    proc = SpectraExtractor()
    proc.set_image(fits_image_data_matrix)
    proc.run_main() : main function which fills self.spectra with Spectrum
                      class objects
    proc.spectra : contains all spectra (Spectrum class objects)
                 which were extracted after self.run_main is called
    proc.loginfo() : basic logging
    """


    def __init__(self, conf=None):
        
        self.spectra = [] # list filled with Spectrum class objects
        self.corners = [] # Spectrum corners, to avoid overlapping
        self.area_values = [] # background area values

        # ver - vertical image, hor - horizontal image. /
        self.orientation = None # set at run_main()
        self.image_set = False
        
        # user/conf defined values --------------------------------

        self.sc = 1 / 20.0 # remove len*sc edge pixels
        
        # image division because of different background value regions
        self.slices = 4 # total slices^2 frames

        # searching for object:
        self.searchStep = 20 # skipping every _value_ while searching for ..
            # spectra align pendicular

        self.jumpStep = 100 # skipping every _value_ pixels while searching for
            # spectra along their axis

        self.sur_percent = 0.3 # size percent of spectra length for adding
                # surrounding pixels

        # parsing object:
        self.in_searchStep = 3
        self.in_jumpStep = 10

        self.min_spectra_width = 300
        self.min_spectra_height = 300
        self.max_spectra_width = 1300
        self.max_spectra_height = 150 


        self.parses = 0
        self.successful_parses = 0
        self.aborted_parses = 0

        
        
    def set_image(self, image_data):
        """
        in:
            image_matrix (NxM array) : fits pixel values matrix  
        """
        
        self.image_data = image_data
        self.shape = self.image_data.shape

        y_height = len(self.image_data)
        self.y_start = int(y_height * self.sc) 
        self.y_end =  y_height - self.y_start

        x_width = len(self.image_data[0])
        self.x_start = int(x_width * self.sc)
        self.x_end = x_width - self.x_start    
        
        self.img_divide()
        self.image_set = True
        
    def img_divide(self):
        """
        divide image equally into self.slices^2 frames, because different
        parts of image usually have different average background pixels
        values
        """
        ylen, xlen = self.shape
        row_vals = [ylen / self.slices * i for i in range(int(self.slices) \
                                + 1)]
        col_vals = [xlen / self.slices * i for i in range(int(self.slices) \
                                + 1)]

        for rowVal, nextRowVal in zip(row_vals, row_vals[1:]):
            for colVal, nextColVal in zip(col_vals, col_vals[1:]):
                mat = self.image_data[rowVal:nextRowVal, colVal:nextColVal]
                if(np.std(mat) < 2000):
                    val = np.mean(mat) - 2000
                else:
                    val = np.mean(mat) - np.std(mat)
                self.area_values.append((rowVal, nextRowVal, colVal,
                                        nextColVal, val))

    def get_area_value(self, (x, y)):

        for area in self.area_values:
            """
            area_values consists of five values, of which first four
                are the corners (which were set in self.img_divide()):
                rowVal,
                nextRowVal,
                colVal,
                nextColVal,
                val (actual local background value)
            """

            if(y >= area[0] and y < area[1] and
               x >= area[2] and x < area[3]):
                return area[4]
        return False

    def check_size(self, x, y, dx, dy):
        """
        optional addition:
            while analyzing possible spectrums lengths get too big, we could
            mark it as an error or something non related, save time and skip
        """
        # todo
        return False

    def run_main(self):
        """
        main function
        uses: image_data NxM dim. pixel matrix (read in at class initialization)

        out: self.spectra array filled with Spectrum objects
        """
        # main loop

        print "start main loop"
        if not self.image_set:
            print "image data not set. Use set_image(fits_data)"
            return

        """
        todo implement orientation setting
        """

        self.orientation = "hor"

        if(self.orientation == "ver"):
            self.search_vertically() # searching vertically
        elif(self.orientation == "hor"):
            self.search_horizontally() # searching horizontally

    def search_horizontally(self):

        y = self.y_start
        while y < self.y_end:
            x = self.x_start
            while x < self.x_end:
                found_x = self.find(x, y)
                if found_x:
                    x = found_x
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
            arr = self.parse_coordinate((x, y)) # find all similar pixels
            if arr:
                return self.analyze(arr)
        return False

    def val_in_range(self, x, y, minval = True):

        val = self.image_data[int(y)][int(x)]

        if(minval == True):
            # minval set, so we accept only pixels whose value are below background area's
            # value
            min_val = self.get_area_value((x, y)) # get our area value
            if not min_val:
                return False
            return (True if val < min_val else False)

        else:
            # no minval set, we just take 20 pixels to get a better look at
            # spectrum surroundings
            if(minval < 20):
                return True
            return False
            
    def parse_coordinate(self, (x, y)):
        """
            | <- pix
            | <- pix        |
            | <- pix        |
        ...---(dx,dy)-----------(dx+xstep,avg_dy)----...
            | <- pix        |
            | <- pix
            ..<- pix with too low value

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

                temp_ob = self.add_pixels_to_spine_coordinate((dx, dy))
                # add point (dx,dy) with its average pixel value
                # and its sub-crds
                obj.append(temp_ob)

                if(self.orientation == "hor"):
                    dx += jump_step
                    dy = np.median(temp_ob[2], axis = 0)[1] # average y
                elif(self.orientation == "ver"):
                    dy += jump_step
                    dx = np.median(temp_ob[2], axis = 0)[0]

                #if(self.check_size(x, y, dx, dy)): # too wide or smtin. abort.
                #    return []

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

        except IndexError: 
            # out of bounds
            pass

        return obj

    def add_pixels_to_spine_coordinate(self, (x, y), minval = True):
        """
        adds pixels to spine coordinate point (x,y) (horizontally or vertically)

        in:
            (x, y) (tuple) : spine coordinate
            minval (boolean) : true if only adding points
            below local background value

        """
        # check that we're still in range

        if(0 > x > self.shape[0] or 0 > y > self.shape[1]):
            raise IndexError


        search_step = self.in_searchStep

        pixels_up = self._add_pixels_to_spine_coordinate((x, y), \
                                                        search_step, minval)
        pixels_down = self._add_pixels_to_spine_coordinate((x, y), \
                                                          -search_step, minval)


        if(pixels_up[1] and pixels_down[1]):
            avg_pixval = (pixels_up[1] + pixels_down[1]) / 2
        elif(pixels_up[1]):
            avg_pixval = pixels_up[1]
        elif(pixels_down[1]):
            avg_pixval = pixels_down[1]
        else:
            avg_pixval = self.image_data[int(y)][int(x)]

        # return point (dx,dy) with its average pixel value and its sub-crds
        return ((x, y), avg_pixval, pixels_up[0]+pixels_down[0])

    def _add_pixels_to_spine_coordinate(self, (x, y), step, minval = True):
        """
        in:
        1) starting point pixel coordinates. Checks for pixels matching
        value either up and down or left and right (depending on orientation)
        2) step: how many pixels are skipped after each round
        3) minval (value of surrounding local background value): if set to true,
        keeps searching until a pixel not below our local background value
        is found.

        if set to false, then it does the same process but only 20 times*
        *atm hardcoded in function val_in_range

        purpose: to get surroundings of spectrum to get bigger and wider look
        of spectrums surroundings.

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
                temp.append(self.image_data[int(dy)][int(dx)])
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

    def validate(self, x, y):
        """
        Check if coordinate (x,y) doesn't overlap with any other spectrum
        and is within our value range.
        """
        if (self.free((x, y)) and self.val_in_range(x, y)):
            return True
        return False    
        
    def free(self, (x, y)):
        """
            check whether crd (x,y) overlaps with an existing spectrum area
        """
        for rect in self.corners:
            if(x >= rect[0] and x<= rect[1] and
               y >= rect[2] and y <= rect[3]):
                # not free
                return False
        return True

    def analyze(self, obj):
        """
        in: list of tuples
        function for determinating if pixels make up a spectra

        out:
        adds possible object to self.spectra
        obj structure: object of Spectrum class
        """

        # all coordinates (spectrum spine crds with their local crds)
        res = [elem for sublist in obj for elem in sublist[2]]
        if(len(res) < 3):
            return False

        x1_crd = min(res,key = lambda t: t[0])
        x2_crd = max(res,key = lambda t: t[0])
        y1_crd = min(res, key = lambda t: t[1])
        y2_crd = max(res, key = lambda t: t[1])
        x1 = x1_crd[0] # min x crd of spectrum
        x2 = x2_crd[0] # max x crd of spectrum
        y1 = y1_crd[1] # min y crd of spectrum
        y2 = y2_crd[1] # max y crd of spectrum

        len_x = x2-x1 # length of spectrum on x axis
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

            # vaata yle!
            if(self.orientation == "hor"):
                result.crd_1 = x1_crd
                result.crd_2 = x2_crd
            elif(self.orientation == "ver"):
                result.crd_1 = y1_crd
                result.crd_2 = y2_crd

            result.orientation = self.orientation

            result.average_value = np.average([elem[1] for elem in obj])

            result.spectrum_points = copy.deepcopy(obj)
            result.area_value = self.get_area_value((int((x1 + x2) / 2)
                                                    ,int((y1 + y2) / 2)))

            if(self.orientation == "hor"):
                t = int((len_x * self.sur_percent) / self.in_jumpStep)
                obj.extend(self.get_spectrum_surrounding_pixels(x1_crd, x2_crd, t))

            else:
                t = int((len_y * self.sur_percent) / self.in_jumpStep)
                obj.extend(self.get_spectrum_surrounding_pixels(y1_crd, y2_crd, t))


            result.spectrum_points_all = obj

            result.starting_point()

            self.successful_parses += 1
            self.corners.append((x1, x2, y1, y2))
            self.spectra.append(result)

            # continue searching at the end of spectrum
            return x2_crd[0] + 5 if self.orientation == "hor" \
                                    else y2_crd[1] + 5

        return False

    def get_spectrum_surrounding_pixels(self, p1, p2, t = 5):
        """
        get surrounding pixel values up to t times on both sides of spectrum
        in:
            p1 (tuple) : (x,y) last coordinate of one side of spectrum
            p2 (tuple) : -""- other side
            t (int)    : by how many points we extend the spectrum spine
                         one side

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

            try:
                obj.append(self.add_pixels_to_spine_coordinate((xn, yn), minval = False))
            except IndexError:
                pass

        return obj

    def loginfo(self):

        print ("-------")
        print ("Parsed total  of %s " % self.parses)
        print ("Of which actually met spectrum criteria: %s "
                    % self.successful_parses)
        print ("Of which were too big, out of bounds: %s "
                    % self.aborted_parses)
        print ("-------")
    
    def write_starting_points_to_file(self, filename):
        try:
            print "writing spectrum points to %s" % filename
            f = open(filename,"w")
            for spec in self.spectra:
                f.write("%s %s \n" % (spec.start[0],spec.start[1]))
            f.close()
        except Exception as e:
            print "writing to %s was unsuccessful: " % filename
            print(e)
            
