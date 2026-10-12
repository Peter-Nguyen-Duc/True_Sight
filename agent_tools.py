from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()

from data_handler import video_data_handler
from RAG import RAG_image_interfacer
import numpy as np

from ollama import ChatResponse


"""
This class lists all tools that the True sight Reasoning module can use to solve the task
"""

def check_arguments_valid(arguments_dict: dict, response: ChatResponse):


  if response.message.tool_calls:
    first_call = response.message.tool_calls[0]

    actual_keys = set(first_call.function.arguments.keys())

    # 3. Compare
    if actual_keys == arguments_dict:
      return True

    else:
      return False


@decorate_all_class_methods
class True_sight_agent_toolbox:
  def __init__(self):
    pass

  def get_current_data(self, vid_data_handler: video_data_handler, RAG: RAG_image_interfacer):
    self._vid_data_handler = vid_data_handler
    self._RAG_obj = RAG



  def TOOL_get_frame_at_index(self, frame_indx: int) -> np.array: 
    
    """
    SUMMARY: Retrieve a single frame from the loaded video data by its index.

    ARGS:
    frame_index [int]:  Zero-based index of the frame to retrieve. Must be within
                        [0, len(self._vid_data_handler)).

    RETURNS [np.ndarray]: The frame at the given index as an image array.
    """

    frame_indx = max(0, min(len(self._vid_data_handler._data) - 1, int(frame_indx)))
    im = self._vid_data_handler._data[frame_indx]
    return im 


  

  def TOOL_determine_image_anomality(self, verdict: bool, anomaly_category: str, reasoning: str) -> tuple:
      """
      SUMMARY:  The reasoning agent will call this tool to finalize the anomaly assessment. 
                This crystallizes the agent's reasoning into a formal verdict and classification.

      ARGS: 
        verdict [bool]: True if the image is anomalous, False if it is normal.
        anomaly_category [str]: The specific type of anomaly. 
                                Valid options: 'Stuck on track', 'Pre-trap rejection', 
                                'Post-trap rejection', 'Other', or 'None' (if normal).
        reasoning [str]: A concise explanation of the visual evidence supporting the verdict.

      RETURNS:
        (bool, str, str): The verdict, category, and reasoning are passed directly through 
                          to the framework for final reporting and logging.
      """
      return verdict, anomaly_category, reasoning