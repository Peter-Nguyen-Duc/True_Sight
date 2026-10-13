from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()

from data_handler import video_data_handler
from RAG import RAG_image_interfacer
import numpy as np

from ollama import ChatResponse, chat

import json

"""
This class lists all tools that the True sight Reasoning module can use to solve the task
"""


@log_decorator(logging.INFO)
def check_arguments_valid(arguments_dict: dict, response: ChatResponse):


  if response.message.tool_calls:
    first_call = response.message.tool_calls[0]

    actual_keys = set(first_call.function.arguments.keys())

    # 3. Compare
    if actual_keys == arguments_dict.keys():
      return True

    else:
      return False

@log_decorator(logging.INFO)
def generate_dynamic_tool(variable_dict: dict) -> list:
    """
    Converts a dictionary of values into a JSON tool schema 
    that forces the LLM to use these specific keys.
    """

    # supported variables that can be passed into the LLM model
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object"
    }

    properties = {}
    required_fields = []

    for key, value in variable_dict.items():

        # 1. CLEAN THE KEY: Remove trailing colons or spaces
        clean_key = key.strip().replace(":", "")

        if key!=clean_key:
          raise Exception(f"Key \"{key}\" contained illegal variables! change your key values,")

        # Determine the type of the value to tell the model what to send back
        val_type = value
        json_type = type_map.get(val_type, "string") 
        
        properties[key] = {
            "type": json_type,
            "description": f"Required value for {key}. Current target value: {value}"
        }
        required_fields.append(key)

    # This is the standard format expected by most LLM tool-calling APIs
    return [{
        "type": "function",
        "function": {
            "name": "TOOL_dynamic_variable_constrained_output",
            "description": "This method forces the LLM to give its output using a specific variable defined by the user.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields
            }
        }
    }]

@log_decorator(logging.INFO)
def variable_constrained_prompt(message, variable_dict: dict, model_name: str):

  current_messages = list(message)


  # 2. Create a strict instruction for the model
  type_hints = {k: v.__name__ for k, v in variable_dict.items()}
  constraints_str_keys = json.dumps(type_hints, indent=2)

  instruction = (
      f"You MUST call the tool 'TOOL_dynamic_variable_constrained_output'. "
      f"You are REQUIRED to pass the following exact arguments:\n{constraints_str_keys}"
  )
  current_messages.append({"role": "user", "content": instruction})

  response_rejected = True
  max_tries = 3
  iters = 0

  while response_rejected:
    variable_constrain_tool = generate_dynamic_tool(variable_dict)


    response = chat(
      model=model_name,
      messages=current_messages,
      tools=variable_constrain_tool,
      options={"num_ctx": 32768},
    )


    if not(check_arguments_valid(arguments_dict=variable_dict, response=response)):
      error_msg = (
          f"ERROR: The tool call was rejected because it did not contain the required variables. "
          f"You MUST pass the following arguments to the tool 'TOOL_determine_image_anomality': {variable_dict}"
      )
      current_messages.append({"role": "user", "content": error_msg})

      iters += 1

      if iters > max_tries:
        raise Exception(f"ERROR: Model could not follow the format, after {max_tries} re-evaluations!")
      
    else:
      response_rejected = False


  if response.message.tool_calls:

    fetched_tools = []
    arguments_list = []

    for call in response.message.tool_calls:

      tool_name = call.function.name

      arguments = dict(call.function.arguments or {})

      arguments_list.append(arguments)

    dict_variables = (arguments_list[0])

  return dict_variables


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