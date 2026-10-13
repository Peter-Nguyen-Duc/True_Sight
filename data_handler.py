
from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file, ignore_logging
import logging
generate_logger_file()


import glob

import tensorflow as tf

import cv2

import os

from os import listdir
from os.path import isfile, join

import numpy as np


def get_file_extension(path):
  filename, file_extension = os.path.splitext(path)

  return file_extension


@decorate_all_class_methods
class video_data_handler:
  """
  This is the video data handler made for the TCC latent anomaly interfacer repurposed

  """

  
  def __init__(self, image_path = None):
    self._image_path = image_path
    self._data = []


  def load_from_self(self, target_image_size = None):
    self.load_images_from_path(self._image_path, target_size=target_image_size)

  @ignore_logging # This method is run many hundreds of times so we ignore it in the logging
  def load_single_image_from_path(self, path, target_size=None):
    """
    SUMMARY: 
      This method takes a path to an image and creates a numpy.ndarray variable of that image
    

    ARGS: 
      path [string]: This variable contains the path where the images are contained.

    RETURNS:
      image_data [bool]: returns numpy.ndarray image data
    """
    
    im = cv2.imread(path)

    if im is None:
      raise Exception("ERROR: no image found at path!")


    # Downscaling logic
    if target_size is not None:
        # cv2.INTER_AREA is the best interpolation method for downscaling (shrinking)
        # source: https://opencv.org/resizing-and-rescaling-images-with-opencv/#h-resizing-with-different-interpolation-methods
        im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)

    return im


  def load_images_from_path(self, path, target_size=None):
    """
    SUMMARY: 
      This method clears self._data then appends the images within the input path

    ARGS: 
      path [string]: This variable contains the path where the images are contained.


    RETURNS:
      information_checked [bool]: This variable describes if the information was successfully extrapolated from the file path
                                  so far it only returns true, because i dont have any exceptions i want to catch.
    """

    files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and (get_file_extension(join(path, f)) == ".png" or get_file_extension(join(path, f)) == ".jpg")] 
   
    # 2. SORT the files lexicographically
    files = sorted(files)

    for file in files:
      im = self.load_single_image_from_path(path=file, target_size=target_size)

      self._data.append(im)

    return True


  

  def get_data(self):
    return self._data


  def check_data(self):
    """
    SUMMARY:
      Checks if the data variable is empty

    RETURNS:
      check [boolean]: Returns true if self._data != none
    """

    return (self._data != None)


  def clear_data(self):
    self._data = []




def main():
  vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"

  vid_obj = video_data_handler(vid_path)

  for image in vid_obj._data:
    print(image)

  

          

if __name__ == '__main__':
  main()
