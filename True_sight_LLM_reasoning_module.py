

from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()

from urllib import response

import ollama
from ollama import chat

import time

import logging

import os

import numpy as np

import cv2

from data_handler import video_data_handler




@decorate_all_class_methods()
class true_sight_reasoning_framework:
  """
  This is a framework for building agentic systems with multiple agents.
  This system only supports Ollama models currently. 
  The framework allows you to define multiple agent roles, 
  each with its own goal and set of tools. 
  The main agent role is responsible for handling user queries and delegating tasks to other agents as needed.
  """



  def __init__(self, model_name):

    self._model_name = model_name
    
    self._roles = {}

    self._tools_for_agents = {}


    self._goals_for_roles = {}

    self._main_agent_role = None
    
    
  def add_agent(self, role_name, role_description, tool_func_list, goal_for_role, main_agent = False):
    """Add a new agent role to the framework."""

    self._roles[role_name] = role_description

    self._tools_for_agents[role_name] = {}


    for i in range(len(tool_func_list)):
      self._tools_for_agents[role_name][tool_func_list[i].__name__] = tool_func_list[i]


    self._goals_for_roles[role_name] = goal_for_role

    if main_agent:
        self._main_agent_role = role_name



  def run_framework(self, query, image_input = None):
    """Run the framework with the main agent role."""

    if not self._main_agent_role:
        raise ValueError("No main agent role defined. Please set main_agent=True for one of the agents.")


    prompt = f"""
            ### role:
              {self._roles[self._main_agent_role]}

            ### goal:
              {self._goals_for_roles[self._main_agent_role]}

            ### call content:
              {query}
            """


    image_data = []

    if isinstance(image_input, list):
      for image in image_input:
        image_data.append(self._get_image_data_from_input(image))
    else:
      image_data.append(self._get_image_data_from_input(image_input))


      
        
    if image_data == None:
      messages = [{"role": "user", 
                   "content": prompt}]
    else:
      messages = [{
          "role": "user",
          "content": prompt,
          "images": image_data,     # <-- flat, parallel to content
      }]

    tools = self._tools_for_agents[self._main_agent_role].values() or None

    response = chat(
        model=self._model_name,
        messages=messages,
        tools=None,
        options={"num_ctx": 16384},
    )
    if response.done_reason != "stop":
      output = response.message.content + "\n ERROR: Model context limit has been reached!"

    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        tool_name = call.function.name
        arguments = dict(call.function.arguments or {})
                
        # Pass the user's query as an argument to the tool
        result = self._tools_for_agents[self._main_agent_role][tool_name](**arguments)


        with open("database/diet_plan_template_output.txt", "r") as file:
            output_template = file.read()


        function_output_back_prompt = f"""
                  ### role:
                    {self._roles[self._main_agent_role]}
                    
                  ### call content:
                    {result}

                  ### goal:
                    {self._goals_for_roles[self._main_agent_role]}

                  ### output template:
                    {output_template}
                  """

        
        messages = [{"role": "user", "content": function_output_back_prompt}]
        


        
        final_response = chat(model=self._model_name, messages=messages, think=False)
        print(" ----- final response ----- ")
        print(final_response)

        output = final_response.message.content
    else:
      output = response.message.content

    return output

  def _get_image_data_from_input(self, image_input):
    image_data = None

    if image_input is not None:
      if isinstance(image_input, str):
        # It's a path, pass it as is
        image_data = image_input

      elif isinstance(image_input, np.ndarray):
        # It's a numpy array, encode it to JPG bytes
        success, buffer = cv2.imencode('.jpg', image_input)
        if success:
          image_data = buffer.tobytes()
        else:
          raise ValueError("Could not encode numpy array to image bytes.")

      else:
        raise TypeError("image_input must be either a string (path) or a numpy array.")

      return image_data

  def run_agent(self, role_name, query):
    """Run a specific agent role in the framework."""

    if role_name not in self._roles:
        raise ValueError(f"Role '{role_name}' is not defined in the framework.")

    prompt = f"""
            ### role:
              {self._roles[role_name]}

            ### goal:
              {self._goals_for_roles[role_name]}

            ### call content:
              {query}
            """

    
    messages = [{"role": "user", "content": prompt}]


    response = chat(
        model=self._model_name,
        messages=messages,
        tools=self._tools_for_agents[role_name].values(),
        think=False
    )
    output = response.message.content


    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        tool_name = call.function.name
        arguments = dict(call.function.arguments or {})
        

        # overwrite the output with the tool call result
        output = self._tools_for_agents[role_name][tool_name](**arguments)
    return output


  def grant_agent_function_call(self, agent_name, func):

    self._tools_for_agents[agent_name][func.__name__] = func


  def _ensure_model_exists(self):
    """Checks if the model exists and pulls it if it is not found."""
    local_models = ollama.list()
    exists = any(m.model == self._model_name or m.model.startswith(f"{self._model_name}:") 
                for m in local_models.models)
    
    if not(exists):
      print(f"ERROR: Model with name \"{self._model_name}\" does not exist, or has yet to be downloaded.")

    return exists


  def _ensure_main_model_exists(self):
    """ Checks if there is a main agent that runs when running framework """
    main_model_exists = True if not(self._main_agent_role == None) else False

    if not(main_model_exists):
      print("ERROR: Main model does not exist!")

    return main_model_exists


        
  def is_alive(self):
    system_check = []

    system_check.append(self._ensure_model_exists())
    system_check.append(self._ensure_main_model_exists())

    return all(system_check)


     

@decorate_all_class_methods()
class single_agent_anomaly_detection_module(true_sight_reasoning_framework):
  ###
  # This agentic system is used for labeling video as anomalies and highlightinh critical 
  # parts of the video as anomalies for further inspection.
  # Furthermore, the agent is capable of manipulating video data to ensure it has seen enough to 
  # reasonably decide wherther the data is anomalous or not.
  ###

  def __init__(self, model_name):
    """

    This REASONING_STATUS variable is used to ensure that the model has had enough reasoning time before giving i final answer
    in which the reasoning status variable is flicked into "final answer ready"

    The status types for REASONING_STATUS:
     - Awaiting prompt
     - retrieving input
     - Reasoning over data
     - final answer ready

    """
    
    super().__init__(model_name)

    self._REASONING_STATUS = "Awaiting prompt"


    #chat.dk generated reasoning role for the agent 
    reasoning_agent_description =   """

                        you only speak in english
                        You are an expert Video Anomaly Detection Specialist. Your objective is to determine if a video sequence contains anomalous behavior and to precisely isolate the critical segments.

                        Your reasoning process must follow these steps:
                        1. INITIAL ANALYSIS: Examine the provided video frames/metadata to identify patterns that deviate from the expected norm.
                        2. UNCERTAINTY EVALUATION: If the visual evidence is ambiguous (e.g., low resolution, occlusion, or fast motion), you must explicitly state what information is missing.
                        3. ACTIVE DATA MANIPULATION: Use your tools to manipulate the video data—such as requesting higher-resolution crops of specific areas, adjusting frame sampling rates, or analyzing specific temporal windows—to gather the evidence needed for a confident decision.
                        4. FINAL VERDICT: Once sufficient evidence is gathered, label the video as 'Anomalous' or 'Normal'.
                        5. CRITICAL HIGHLIGHTING: For anomalies, provide the exact timestamps and spatial coordinates (bounding boxes) of the critical events.

                        Maintain a rigorous chain-of-thought: Hypothesis -> Evidence Gathering -> Refinement -> Conclusion.
                        """

    # This is the specific goal passed to the agent framework
    anomaly_detection_goal =  """
                              Accurately classify video sequences as anomalous or normal,
                              isolate critical timestamps/regions for inspection, and iteratively
                              manipulate video data to resolve visual uncertainty.
                              """

    
    self.add_agent(
        role_name="reasoning_agent",
        role_description=reasoning_agent_description,
        tool_func_list=[],
        goal_for_role=anomaly_detection_goal,
        main_agent=True
    )





class diet_planner_agentic_agent(true_sight_reasoning_framework):

  ###
  # This agentic framework is built on the blinders framework.
  # It has two agents: a user assistant and a nutritionist. The user assistant 
  # handles communication with the user and calls the nutritionist tool to 
  # generate a diet plan based on the user's requirements. 
  # The nutritionist generates a diet plan based on the user's requirements.
  ###




  def __init__(self, model_name):
    super().__init__(model_name)


    user_assistant_goal = """Inform the user of the decisions made by the underlying agentic system diet app.
                          Only call the nutritionist tool to generate a diet plan based on the user's requirements.
                          Do not make any decisions on your own.

                          Get the nutrionist's output and present it to the user in a nice and readable format.
                          """

    # Example tool
    nutrition_call_tool = {
                  "type": "function",
                  "function": {
                      "name": "call_nutritionist_tool",
                      "description": "Call the nutritionist tool to generate a diet plan based on user requirements.",
                      "parameters": {
                          "type": "object",
                          "properties": {
                              "user_requirements": {
                                  "type": "string",
                                  "description": "The user's requirements for the diet plan (e.g., 'I want to lose weight and eat healthy food')."
                              }
                          },
                          "required": ["user_requirements"]
                      }
                  }
              }

    
    
    self.add_agent(
        role_name="user assistant",
        role_description="This user handles communication with the user and calls and gives the user a 7 day diet plan based on the user requirements:",
        tool_func_list=[],
        tool_descriptions = [],
        goal_for_role=user_assistant_goal,
        main_agent=True
    )


    # self.add_agent(
    #     role_name="nutritionist",
    #     role_description="This user generates a nutrition plan based on the user's requirements",
    #     tool_func_list=[],
    #     tool_descriptions = [],
    #     goal_for_role=nutrition_goal
    # )


  ###
  # Databases and variables used in the true sight framework system
  ###






  ###
  # Tools used in the true sight framework system 
  ###





if __name__ == "__main__":
  vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"

  vid_obj = video_data_handler(vid_path)
  vid_obj.load_from_self(target_image_size=(224, 224))

  image = [vid_obj._data[0], vid_obj._data[100], vid_obj._data[100]]

  vid_obj.load_images_from_path("/home/nguyenlocal/True_Sight/data", target_size=(224, 224))

  image.append(vid_obj._data[0])

  MODEL_NAME = "qwen3.5:4b"

  agentic_system = single_agent_anomaly_detection_module(MODEL_NAME)


  query = "which image is the odd one out, describe the main differences"
  # image_path = "/home/nguyenlocal/True_Sight/data/DCM4_anomalies/normal_data/Video_index_4_viewport_0/Video_index_4_viewport_0_frame_1.png"

  response_val = agentic_system.run_framework(query, image_input=image)

  print(" ----- final response ----- ")
  print(response_val)






