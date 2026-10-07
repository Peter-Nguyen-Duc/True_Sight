
from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()

from data_handler import video_data_handler


decorate_all_class_methods
class True_sight_interfacer:
  """
  This class is center software module for the anomaly detection method named true sight, it combines a video data manipulator, video memory module and an LLM reasoning module 
  into an anomaly detection framework, which can be used for a user to detect when in a video something anomalous happens, which the user has described before hand, or
  by presenting a set of normal data points, where the estimated videos deviate from it
  """

  def __init__(self):
    self._vid_obj = None



  def load_data(self, path):
    self._vid_obj = video_data_handler(path)

    





def main():
  vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"

  vid_obj = video_data_handler(vid_path)

  for image in vid_obj._data:
    print(image)
    input

  

          

if __name__ == '__main__':
  main()
