

import copy

from urllib import response

import ollama
from ollama import chat

import time

import logging

import os

import numpy as np

import cv2

from data_handler import video_data_handler


from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
# generate_logger_file()

from agent_tools import check_arguments_valid, variable_constrained_prompt


@decorate_all_class_methods
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

    # This is a copy of the agent tools, so that if additional tools are added or deleted, the standard agent are not cleared.
    # You must manually copy the self._tools_for_agents varible into this variable when granting additional tools from a child class.
    self._standard_agent_tools = {}
    
    
  def add_agent(self, role_name, role_description, tool_func_list, goal_for_role, main_agent = False):
    """Add a new agent role to the framework."""

    self._roles[role_name] = role_description

    self._tools_for_agents[role_name] = {}


    for i in range(len(tool_func_list)):
      self._tools_for_agents[role_name][tool_func_list[i].__name__] = tool_func_list[i]


    self._goals_for_roles[role_name] = goal_for_role

    if main_agent:
      if not(self._main_agent_role == None):
        print("WARNING: overwriting the main agent.")

      self._main_agent_role = role_name




  def preprocess_input(self, query, image_input):


    image_data = []
    messages = []

    prompt = f"""
            ### role:
              {self._roles[self._main_agent_role]}

            ### goal:
              {self._goals_for_roles[self._main_agent_role]}

            ### call content:
              {query}
            """

    # We use a list for 'content' to allow interleaving text and images
    content_blocks = []
    
    # Add the initial text prompt
    content_blocks.append({"type": "text", "text": prompt})


    if isinstance(image_input, dict):
      for group_name, images in image_input.items():
        if not images:
          continue

        image_data_group = {"role": "user",}
        image_data =  []

        for im in images:
          image_data.append(self._get_image_data_from_input(im))

        # Add a text label block before the images of this group
        label = f"\n--- {group_name.upper()} ---\n"

        image_data_group["content"] = label
        image_data_group["images"] = image_data
        messages.append(image_data_group)
    
    # Fallback: If image_groups is still a list, we treat it as one big group
    elif isinstance(image_input, list):
        content_blocks.append({"type": "text", "text": "\n--- IMAGES ---\n"})

        image_data =  []

        for im in image_input:
          image_data.append(self._get_image_data_from_input(im))

        image_data_group = {"role": "user",}

        # Add a text label block before the images of this group
        label = f"\n--- IMAGES ---\n"

        image_data_group["content"] = label
        image_data_group["images"] = image_data
        messages.append(image_data_group)
    
    # Fallback: Single image input
    elif image_input is not None:
        img_data = self._get_image_data_from_input(image_input)

        image_data_group = {"role": "user",}

        # Add a text label block before the images of this group
        label = f"\n--- IMAGE ---\n"

        image_data_group["content"] = label
        image_data_group["images"] = [img_data]
        messages.append(image_data_group)


    # if isinstance(image_input, list):
    #   for image in image_input:
    #     image_data.append(self._get_image_data_from_input(image))
    # else:
    #   image_data.append(self._get_image_data_from_input(image_input))


    prompt_message = {
              "role": "user",
              "content": prompt,    # <-- flat, parallel to content
          }
    messages.append(prompt_message)
    
    return messages


  def run_framework(self, query, image_input = None, allow_tool_calling = False, variable_constrain=None):

    messages = self.preprocess_input(query=query, image_input=image_input)

    response = self.call_main_agent(query=messages, allow_tool_calling=allow_tool_calling, variable_constrain=variable_constrain)

    return response


  def get_agent_tools(self, agent_name):
    tools = self._tools_for_agents[agent_name].values() or None

    return tools


  
  def call_main_agent(self, query, variable_constrain=None):
    """Run the framework with the main agent role."""

    if not self._main_agent_role:
      raise ValueError("No main agent role defined. Please set main_agent=True for one of the agents.")

    if variable_constrain == None:
      response = chat(
        model=self._model_name,
        messages=query,
        tools=self.get_agent_tools(self._main_agent_role),
        options={"num_ctx": 32768},
      )
    else:
      response = variable_constrained_prompt(message=query, variable_dict=variable_constrain, model_name=self._model_name)

    return response




    

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
        options={"num_ctx": 32768},
    )
    output = response.message.content


    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        tool_name = call.function.name
        arguments = dict(call.function.arguments or {})
        

        # overwrite the output with the tool call result
        output = self._tools_for_agents[role_name][tool_name](**arguments)
    return output


  def grant_main_agent_tool(self, func):
    self._tools_for_agents[self._main_agent_role][func.__name__] = func


  def grant_agent_tool(self, agent_name, func):
    self._tools_for_agents[agent_name][func.__name__] = func

  def clear_all_tools(self):
    """
    Clears all tools granted during runtime of the agent, does not clear the standard tools required for the agentic system to run. 
    """
    # default back to the old copied values
    self._tools_for_agents = copy.deepcopy(self._standard_agent_tools)

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


     

@decorate_all_class_methods
class single_agent_anomaly_detection_module(true_sight_reasoning_framework):
  ###
  # This agentic system is used for labeling video as anomalies and highlightinh critical 
  # parts of the video as anomalies for further inspection.
  # Furthermore, the agent is capable of manipulating video data to ensure it has seen enough to 
  # reasonably decide wherther the data is anomalous or not.
  ###

  def __init__(self, model_name="qwen3.5:4b"):
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


     

@decorate_all_class_methods
class self_reflective_single_agent(true_sight_reasoning_framework):
  ###
  # This agentic system is used for labeling video as anomalies and highlightinh critical 
  # parts of the video as anomalies for further inspection.
  # Furthermore, the agent is capable of manipulating video data to ensure it has seen enough to 
  # reasonably decide wherther the data is anomalous or not.
  ###

  def __init__(self, model_name="qwen3.5:4b"):
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
                        You do video analysis using the tools available and afterwards use reasoning to deduce if an image or a video is anomalous or not
                        """

    # This is the specific goal passed to the agent framework
    anomaly_detection_goal =  """
                              Commit to any video analysis task given by the user. 
                              """


    #chat.dk generated reasoning role for the agent 
    reflective_agent = """
                      You are a Critical Supervisory Agent and Quality Auditor. Your sole purpose is to find flaws, 
                      omissions, or formatting errors in the Main Agent's responses. 


                      You operate with a "skeptical" mindset. You do not assume the Main Agent is correct; instead, 
                      you actively look for reasons why the response might be insufficient, misleading, or 
                      incorrectly formatted. You are the final gatekeeper before a response reaches the user.
                      """


    # This is the specific goal passed to the agent framework
    reflective_agent_goal =  """
                            Your goal is to evaluate the interaction between the User and the Main Agent.
                            Keep your verdict short and consice, to avoid a noisy output.

                            1. TOOL VALIDATION: 
                              - Determine if the query required external data or a tool call.
                              - If a tool was required but NOT called, this is a failure.
                              - If a tool WAS called, verify that the Main Agent actually used the information returned by the tool in its final answer, rather than ignoring it or hallucinating.
                              - DEBUGGING ADDITION, when evidence of tool calling is detected, no other output is required to pass the test.

                            2. FINAL VERDICT:
                              - If the response passes, you MUST start your response with the exact word: 'APPROVED'.
                              - If any check fails, DO NOT use 'APPROVED'. Instead, provide a detailed, bulleted list of exactly what is wrong and how the Main Agent should fix it. 
                                Be specific (e.g., "The tool 'get_weather' was called, but you ignored the temperature value in your final answer").
                            """



    self.add_agent(
        role_name="reasoning_agent",
        role_description=reasoning_agent_description,
        tool_func_list=[],
        goal_for_role=anomaly_detection_goal,
        main_agent=True
    )


    self.add_agent(
        role_name="reflective_agent",
        role_description=reflective_agent,
        tool_func_list=[],
        goal_for_role=reflective_agent_goal,
        main_agent=False
    )


    # Update the list of standard tools that the reflective agent is equipped with from the start
    self._standard_agent_tools = copy.deepcopy(self._tools_for_agents)
    


  def run_framework(self, query, image_input=None, max_reflections=3, variable_constrain = None):
    """
    RUN THE FRAMEWORK WITH A REFLECTIVE LOOP
    """
    # 1. Initial preprocessing
    messages = self.preprocess_input(query=query, image_input=image_input)
    
    # 2. Initial call to the main agent
    response = self.call_main_agent(query=messages, variable_constrain=variable_constrain)
    
    # Loop for reflection and regeneration
    for i in range(max_reflections):

      is_approved, reasoning = self.get_reflective_agent_input(message=messages, response=response)

      if is_approved:
        break
          
      feedback_prompt = f"Your previous response was: {response}\n\nCritique: {reasoning}\n\nPlease regenerate the response addressing these points."
      messages.append({"role": "user", "content": feedback_prompt})
      
      response = self.call_main_agent(query=messages, variable_constrain=variable_constrain)
        
    return response


  def get_reflective_agent_input(self, message, response) -> tuple[bool, str]:

    prompt = f"""
            fill out the arguments with the verdict and the reason, such that:
            verdict = True, if the text is APPROVED
            verdict = False, if the text is NOT APPROVED

            reasoning: Contains the arguments of the verdict
    

            ### User Message:
              {message}

            ### Main Agent Response:
              {response}

            ### Instruction:
            If response satisfies established criteria, set boolean "is_approved" to true.
            Otherwise, provide specific, actionable feedback on what is missing or wrong and set "is_approved" = False.
            """

    
    messages = [{"role": "user", 
            "content": prompt}]


    variable_dict = {"is_approved": bool,
                     "reasoning": str,
                      }
    output = variable_constrained_prompt(messages, variable_dict, model_name=self._model_name)

    is_approved = output["is_approved"]
    reasoning = output["reasoning"]

    return is_approved, reasoning


  



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






