
from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()


import glob

import tensorflow as tf

import cv2

import os

from os import listdir
from os.path import isfile, join


@log_decorator(logging.INFO)
def get_file_extension(path):
  filename, file_extension = os.path.splitext(path)

  return file_extension


@decorate_all_class_methods()
class video_data_handler:
  """
  This is the video data handler made for the TCC latent anomaly interfacer repurposed

  """

  
  def __init__(self, image_path):

    self._image_path = image_path
    self._data = None
    self.load_images_from_path(self._image_path)



  def load_images_from_path(self, path):
    """
    SUMMARY: 
      This method finds the images in the given path and saved them as numpy arrays within the datavalues variables
    

    ARGS: 
      path [string]: This variable contains the path where the images are contained.


    RETURNS:
      information_checked [bool]: This variable describes if the information was successfully extrapolated from the file path
                                  so far it only returns true, because i dont have any exceptions i want to catch.
    """

    files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and (get_file_extension(join(path, f)) == ".png" or get_file_extension(join(path, f)) == ".jpg")] 
    self._data = []


    for file in files:

      im = cv2.imread(file)
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
    self._data = None




def main():
  vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"

  vid_obj = video_data_handler(vid_path)

  for image in vid_obj._data:
    print(image)
    input

  

          

if __name__ == '__main__':
  main()
