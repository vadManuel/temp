'''
Author: Manuel Vasquez
Date:   03/19/2019

Python 3.6.5 64-bit (Anaconda3)
'''


import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def gaussian_window(window_size, sigma):
    '''
    gaus
    '''

    half_window = window_size//2
    output_kernel = np.zeros((window_size, window_size))
    output_kernel[half_window, half_window] = 1
    return gaussian_filter(output_kernel, sigma)


def main():
    '''
    main
    '''

    window = gaussian_window(5, 1)
    # cv2.imshow('window 1', out1)

    # while True:
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
