

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
import matplotlib.pyplot as plt

from transformers import AutoTokenizer




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


    # 1. Create a mapping from Ollama tags to Hugging Face paths
    model_mapping = {
        "qwen3.5:4b": "Qwen/Qwen3.5-4B", # Closest HF equivalent for tokenizer
    }
    self._tokenizer = AutoTokenizer.from_pretrained(model_mapping[self._model_name])


    # This is a copy of the agent tools, so that if additional tools are added or deleted, the standard agent are not cleared.
    # You must manually copy the self._tools_for_agents varible into this variable when granting additional tools from a child class.
    self._standard_agent_tools = {}
    


  def check_context_limit(self, messages, limit=300000):
    total_tokens = 0
    for msg in messages:
        # Tokenize the content of each message
        tokens = self._tokenizer.encode(msg['content'], add_special_tokens=False)
        total_tokens += len(tokens)
    
    if total_tokens > limit:
        print(f"WARNING: Context limit exceeded!")
        print(f"Total tokens: {total_tokens} | Limit: {limit}")
        print(f"The model will truncate approximately {total_tokens - limit} tokens from the start.")
        return False

    
    return True


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



  def add_images_to_message(self, messages: list, image_input):


    image_data = []


    if isinstance(image_input, dict):
      for group_name, images in image_input.items():
        if not images:
          continue

        image_data_group = {"role": "user",}
        image_data =  []
        # Add a text label block before the images of this group
        label = f"\n--- {group_name.upper()} ---\n"
        # print(label)
        for im in images:
          image_data.append(self._get_image_data_from_input(im))

        image_data_group["content"] = label
        image_data_group["images"] = image_data
        messages.append(image_data_group)
    
    # Fallback: If image_groups is still a list, we treat it as one big group
    elif isinstance(image_input, list):
        
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

    
    return messages


  def add_agent_agenda_to_message(self, message: list, query: str, agent_name: str):

    prompt = f"""
            ### role:
              {self._roles[agent_name]}

            ### goal:
              {self._goals_for_roles[agent_name]}

            ### call content:
              {query}
            """
    
    prompt_message = {
              "role": "user",
              "content": prompt,    # <-- flat, parallel to content
          }
    
    message.append(prompt_message)

    return message
    


  def run_framework(self, query, image_input = None, variable_constrain=None):

    messages = []

    messages = self.add_images_to_message(messages=messages, image_input=image_input)

    messages_main_agent = self.add_agent_agenda_to_message(message=messages, query=query, agent_name=self._main_agent_role)
    response = self.call_main_agent(query=messages_main_agent, variable_constrain=variable_constrain)

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
        options={"num_ctx": 300000},
      )
    else:
      response = variable_constrained_prompt(message=query, variable_dict=variable_constrain, model_name=self._model_name)

    return response





  # Add this method to your class
  def plot_processed_image(self, image_data):
      """
      Visualizes the output of _get_image_data_from_input.
      Handles both file paths (str) and encoded bytes.
      """
      img = None

      if isinstance(image_data, str):
          # Case 1: It's a path, read it from disk
          img = cv2.imread(image_data)
          if img is None:
              print(f"Error: Could not read image from path {image_data}")
              return

      elif isinstance(image_data, bytes):
          # Case 2: It's encoded bytes, decode them back to a numpy array
          nparr = np.frombuffer(image_data, np.uint8)
          img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
          if img is None:
              print("Error: Could not decode image bytes.")
              return

      else:
          print("Error: image_data is neither a path string nor bytes.")
          return

      # OpenCV reads in BGR, Matplotlib expects RGB
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      plt.figure(figsize=(10, 8))
      plt.imshow(img_rgb)
      plt.axis('off') 
      plt.title("Processed Image Preview")
      plt.show()

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


      # self.plot_processed_image(image_data)
      return image_data

  def call_agent(self, role_name, query, variable_constrain=None):
    """Run the framework with the main agent role."""

    if not self._main_agent_role:
      raise ValueError("No main agent role defined. Please set main_agent=True for one of the agents.")

    if variable_constrain == None:
      response = chat(
        model=self._model_name,
        messages=query,
        tools=self.get_agent_tools(role_name),
        options={"num_ctx": 300000},
      )
    else:
      response = variable_constrained_prompt(message=query, variable_dict=variable_constrain, model_name=self._model_name)

    return response



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

    super().__init__(model_name)



    #chat.dk generated reasoning role for the agent 
    reasoning_agent_description =   """
                        You do video analysis using the tools available and afterwards use reasoning to deduce if an image or a video is anomalous or not

                        You have access to, and should infer all your reasoning from, all the following provided data.
                        - RAG retrieved data: images will be provided with the label, " --- RAG images --- " these are not labeled but given
                        - Normal process data: These images are given to the agent also and does not require evidence, These images will be
                                                labeled as " --- sub_sampled_normal_video_process --- "
                        - Input image data:  These are the input data variables that must be used to check for anomalies in these will be within
                                              the message, labeled with " --- input images --- "
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

                      Input data that the main agent will have access to, which does not require evidence from the main agent to refer to.
                      Therefore do not categorize the usage of this data as a hallucination behavior
                       - RAG retrieved data: images will be provided with the label, " --- RAG images --- " these are not labeled but given
                       - Normal process data: These images are given to the agent also and does not require evidence, These images will be
                                              labeled as " --- sub_sampled_normal_video_process --- "
                       - Input image data:  These are the input data variables that must be used to check for anomalies in these will be within
                                            the message, labeled with " --- input images --- "

                      Do not worry about correct toolcalling, as this is handled and guranteed at your level.
                      """


    # This is the specific goal passed to the agent framework
    reflective_agent_goal =  """
                            Your goal is to evaluate the interaction between the User and the Main Agent.
                            Keep your verdict short and consice, to avoid a noisy output.

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
    messages = []
    messages_only_images = self.add_images_to_message(messages=messages, image_input=image_input)
    
    # 2. Initial call to the main agent
    main_agent_messages = self.add_agent_agenda_to_message(copy.deepcopy(messages_only_images), query=query, agent_name=self._main_agent_role)
    response = self.call_main_agent(query=main_agent_messages, variable_constrain=variable_constrain)
    
    # Loop for reflection and regeneration
    for i in range(max_reflections):

      is_approved, reasoning = self.get_reflective_agent_input(message=main_agent_messages, response=response)

      if is_approved:
        break
          
      feedback_prompt = f"""
      Your response has been rejected by the reflective agent. Please regenerate the response, while addressing the critique given

        ### Critique:
              {reasoning}
      """

      query += feedback_prompt

      # Regenerate the main agent message, but add the critique given be the reflective agent.
      main_agent_messages = self.add_agent_agenda_to_message(copy.deepcopy(messages_only_images), query=query, agent_name=self._main_agent_role)

      response = self.call_main_agent(query=main_agent_messages, variable_constrain=variable_constrain)
        
    return response


  def get_reflective_agent_input(self, message, response) -> tuple[bool, str]:


    # 1. Extract the actual text from the response object to add it to the conversation
    if hasattr(response, 'message') and hasattr(response.message, 'content'):
        main_agent_text = response.message.content
    elif isinstance(response, dict) and 'message' in response:
        main_agent_text = response['message'].get('content', '')
    else:
        main_agent_text = str(response)

    # 2. Construct the "Audit Context"
    # We start with the original messages (which include the images)
    audit_history = copy.deepcopy(message)

    # We append the main agent's response as a message so the reflective agent
    # sees it as the last thing said by the assistant.
    audit_history.append({
        "role": "assistant",
        "content": main_agent_text
    })


    query = f"""
            Please audit the previous response from the assistant.
            
            Fill out the arguments with the verdict and the reason:
            - is_approved = True, if the response is APPROVED.
            - is_approved = False, if the response is NOT APPROVED.
            - reasoning: Provide specific, actionable feedback on what is missing, 
              incorrect, or hallucinated based on the provided images.

            Criteria for Approval:
            1. Does the response logically follow from the images provided?
            2. Does it avoid hallucinating details not present in the images?
            3. Is the formatting correct?
            """

    messages_with_reflective_agent = self.add_agent_agenda_to_message(message=audit_history, query=query, agent_name="reflective_agent")
    

    variable_dict = {"is_approved": bool,
                     "reasoning": str,
                      }
    
    self.check_context_limit(messages=messages_with_reflective_agent)
    output = variable_constrained_prompt(messages_with_reflective_agent, variable_dict, model_name=self._model_name)

    is_approved = output["is_approved"]
    reasoning = output["reasoning"]

    return is_approved, reasoning




  
class voting_based_multi_agent(true_sight_reasoning_framework):
  ###
  # This agentic system is used for labeling video as anomalies and highlighting critical parts on a video.
  # The main focus of this method is to split thoughts into a multi agent, independent thinking process.

  # The three main voting agents will be prompted to think in three different directions
  # - Will focus on behavior that resembles trap rejection, a detailed role prompt will also be granted which aids the agent into this thinking direction
  # - Will focus on the movement behavior, and track if there are any irregularities, regarding the direction the part moves, in comparison to the normal data.
  # - Will focus on the normal prompt that was given, and specifically cross list anomaly subjects on the list to check if any anomalies that resembles the
  #   described anomalous behavior has been identified. This agent will also be granted a reflective agent.

  # The task delegator will count up the votes of all agents, read their reasoning and choose the highest vote count, where the anomaly description will be 
  # Inferred based on their reasoning

  ###

  def __init__(self, model_name="qwen3.5:4b"):
    super().__init__(model_name)

    # Role: The specialist focusing on trap rejection behavior
    trap_rejection_agent_role = """
    You are a Specialized Quality Assurance Agent focusing exclusively on 'Trap Rejection' behaviors. 
    Your expertise is in identifying when a part fails to seat correctly in a trap, is pushed back out, 
    or exhibits the specific mechanical signature of a rejection event. 

    Key focus areas Video end deduction, Given that last 2-3 images how can it be 
    concluded that that the video ended? Furthermore, you must also identify
    movement irregularities given that the part might be rejected. Vertical movement to the normal
    movement direction is a clear sign of the start of part rejection
    """

    # Goal: The specific task for the trap rejection agent
    trap_rejection_agent_goal = """
    Analyze the video to determine if a 'Trap Rejection' event occurred. 
    1. Identify the exact moment the part exhibits anomalous behavior according to trap rejection
    2. Determine if the part was successfully accepted or if it was rejected.
    3. Provide a boolean vote of anomality (True if anomalous, False if not) based strictly on trap rejection criteria.
    4. Provide both a reasoning for the vote, and a estimated anomaly category which correlates to the reasoning given
    """


    # Role: The specialist focusing on movement and trajectory
    movement_behavior_agent_role = """
    You are a trajectory Analysis Agent specializing in movement trajectories. 
    Your role is to compare the observed motion of the part against the 'Golden Standard' of normal operation. 
    oscillations, sporadicity, and sudden lags are all acceptable deviations from the normal, but unstable 
    deviation from the normal where the part object leaves the track entirely, is not acceptable.
    
    Furthermore, you must deduce if the object reached the goal at the end, which is on the cliff side of the track.
    If the part never reaches the cliff, and is solely moving around pre-trap and on the trap ramp side, then this is
    also seen as an anomaly.

    """

    # Goal: The specific task for the movement agent
    movement_behavior_agent_goal = """
    Evaluate the movement patterns in the video.
    1. Track the trajectory of the part from start to finish.
    2. Compare the observed path against normal operational data (Does the part get processed within tolerable bounds?)
    3. Provide a boolean vote of anomality (True if anomalous, False if not) based strictly on the reasoning and the end-position of the object
    4. Provide both a reasoning for the vote, and a estimated anomaly category which correlates to the reasoning given
    """


    # Role: The final decision-maker and synthesizer
    task_delegator_role = """
    You are the Chief Synthesis Agent. Your role is to act as the final judge in a 
    voting-based multi-agent system. You do not analyze the video yourself; instead, 
    you analyze the reasoning and votes provided by three specialized agents. 
    Your goal is to extract the most probable truth.
    
    the voting data will be preprocessed and given to you such that you already 
    have a clear idea of the highest score given.

    Ir you find improper or lacking reasoning behind the vote given, you are capable of
    commanding a re-investigation by the specific expert agents, until you find the
    reasoning to the specific vote satisfying.
    """

    # Goal: The final aggregation task
    task_delegator_goal = """
    1: Analyze the reasoning behind each expert voter, and determine the probability and proper reasonability of each vote.
    2: If neccesary reprompt, any agent with critique that can be used by the agents to properly reason their vote
    3: If all reasonings are satisfactory and well correlated with their specific vote, then output the following values
        - Anomaly boolean: True if video contains anomalous movements, false if not
        - Anomaly category: A boiled down single-word category of the anomaly depicted (examples could be, pre-trap rejection or post-trap rejection, none if the video is categorized as normal data)
        - reasoning: Give a reasoning for the score given, that is a boiled down concise version of each agents texts.
    """


    self.add_agent(
        role_name="part_rejection_analysist",
        role_description=trap_rejection_agent_role,
        tool_func_list=[],
        goal_for_role=trap_rejection_agent_goal,
        main_agent=False
    )


    self.add_agent(
        role_name="Movement_analysist",
        role_description=movement_behavior_agent_role,
        tool_func_list=[],
        goal_for_role=movement_behavior_agent_goal,
        main_agent=False
    )

    # Agentic agents cannot be directlty added as agents to a system so just spawn a new one as a variable
    self._reflective_Agentic_agent = self_reflective_single_agent(true_sight_reasoning_framework)

    self.add_agent(
        role_name="task_delegator_judge",
        role_description=task_delegator_role,
        tool_func_list=[],
        goal_for_role=task_delegator_goal,
        main_agent=True
    )


  def run_framework(self, query, image_input=None, max_reflections=3, variable_constrain = None):
    """
    RUN THE FRAMEWORK WITH A REFLECTIVE LOOP
    """
    output_variable_constrain = copy.deepcopy(variable_constrain)
    # 1. Initial preprocessing

  
    messages = []

    messages_images_only = self.add_images_to_message(messages=messages, image_input=image_input)

    expert_committee_variable_constrain = {
      "is_anomaly": bool,
      "anomaly_category": str,
      "reasoning": str,
    }

    message_for_rejection_analysist = self.add_agent_agenda_to_message(messages=copy.deepcopy(messages_images_only), query=query, agent_name="part_rejection_analysist")
    message_for_movement_analysist = self.add_agent_agenda_to_message(messages=copy.deepcopy(messages_images_only), query=query, agent_name="Movement_analysist")

    response_rejection_analysist  = self.call_agent(role_name="part_rejection_analysist", query=message_for_rejection_analysist, variable_constrain=expert_committee_variable_constrain)
    message_for_movement_analysist  = self.call_agent(role_name="Movement_analysist", query=message_for_rejection_analysist, variable_constrain=expert_committee_variable_constrain)
    response_reflective_agent = self._reflective_Agentic_agent.run_framework(query=query, image_input=image_input, variable_constrain=expert_committee_variable_constrain)


    # 1. CALCULATE THE VOTE TALLY
    all_responses = [response_rejection_analysist, message_for_movement_analysist, response_reflective_agent]
    anomaly_votes = sum(1 for r in all_responses if r.get("is_anomaly") is True)
    normal_votes = len(all_responses) - anomaly_votes
    
    majority_verdict = "ANOMALY" if anomaly_votes > normal_votes else "NORMAL"
    tally_string = f"VOTE TALLY: {anomaly_votes} ANOMALY vs {normal_votes} NORMAL | MAJORITY: {majority_verdict}"


    committee_findings = []
    
    findings_map = {
      "Trap Rejection Specialist": response_rejection_analysist,
      "Movement Dynamics Specialist": message_for_movement_analysist,
      "General Reflective Agent": response_reflective_agent
    }

    for name, data in findings_map.items():
      verdict = "ANOMALY" if data.get("is_anomaly") else "NORMAL"
      category = data.get("anomaly_category", "N/A")
      reasoning = data.get("reasoning", "No reasoning provided.")
      
      committee_findings.append(f"--- {name} ---\nVerdict: {verdict}\nCategory: {category}\nReasoning: {reasoning}")

    # Join the detailed reports and prepending the tally at the top
    committee_report = f"### SUMMARY TALLY\n{tally_string}\n\n### DETAILED REPORTS\n" + "\n\n".join(committee_findings)

    
    response = self.call_judge(messages_images_only=messages_images_only, committee_report=committee_report, query=query, variable_constrain=output_variable_constrain)

        
    return response

  def call_judge(self, messages_images_only, committee_report: str, query: str, variable_constrain=None):
    judge_query = f"""
        You are the Final Judge. Your task is to review the original user request and the findings from three specialized agents.
        
        ### Original User Request:
        {query}

        ### Committee Expert Reports:
        {committee_report}

        ### Your Task:
        1. Analyze the conflict between agents (e.g., if one says ANOMALY and others say NORMAL).
        2. If a specialist (like Rejection or Movement) provides strong evidence, weight their vote higher than the general agent.
        3. Provide a final definitive verdict on whether the video is anomalous.
        4. Infer the final anomaly description based on the most convincing reasoning.
        """
    
    messages_for_judge = self.add_agent_agenda_to_message(message=messages_images_only, 
                                                                      query=judge_query, 
                                                                      agent_name="task_delegator_judge")
    
    
    self.check_context_limit(messages=messages_for_judge)
    # 6. Final Call to the Judge
    final_verdict = self.call_agent(
        role_name="task_delegator_judge", 
        query=messages_for_judge, 
        variable_constrain=variable_constrain # Or a specialized judge constrain
    )


    return final_verdict

    

  if __name__ == "__main__":
    vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"

    vid_obj = video_data_handler(vid_path)
    vid_obj.load_from_self(target_image_size=(512, 512))

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






