from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()

from data_handler import video_data_handler
from RAG import RAG_image_interfacer
import numpy as np


"""
This class lists all tools that the True sight Reasoning module can use to solve the task
"""

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