
from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()


from data_handler import video_data_handler

from True_sight_LLM_reasoning_module import single_agent_anomaly_detection_module

@decorate_all_class_methods()
class True_sight_interfacer:
  """
  This class is center software module for the anomaly detection method named true sight, it combines a video data manipulator, video memory module and an LLM reasoning module 
  into an anomaly detection framework, which can be used for a user to detect when in a video something anomalous happens, which the user has described before hand, or
  by presenting a set of normal data points, where the estimated videos deviate from it
  """

  def __init__(self, reasoning_module):
    self._vid_obj = None

    self._reasoning_module = reasoning_module

    if not(self._reasoning_module.is_alive()):
      raise Exception("ERROR: reasoning module was not initiated properly!")

    print("all system check!")



  def load_data(self, path, image_size = None):
    self._vid_obj = video_data_handler(path)
    self._vid_obj.load_from_self(target_image_size=image_size)



  





def main():
          
  MODEL_NAME = "qwen3.5:4b"

  reasoning_module = single_agent_anomaly_detection_module(model_name=MODEL_NAME)

  anomaly_detector = True_sight_interfacer(reasoning_module=reasoning_module)
  

          

if __name__ == '__main__':
  main()
