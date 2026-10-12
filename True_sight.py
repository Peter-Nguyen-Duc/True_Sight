


from RAG import RAG_image_interfacer, ImageVectorManager

from data_handler import video_data_handler

from True_sight_LLM_reasoning_module import single_agent_anomaly_detection_module, true_sight_reasoning_framework, self_reflective_single_agent

from agent_tools import True_sight_agent_toolbox


from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()

import cv2
import numpy as np
import matplotlib.pyplot as plt

from agent_tools import check_arguments_valid


@log_decorator(logging.INFO)
def plt_image(target_image):


    plt.imshow(target_image)

    plt.tight_layout()
    plt.show()




@decorate_all_class_methods
class True_sight_interfacer:
  """
  This class is center software module for the anomaly detection method named true sight, it combines a video data manipulator, video memory module and an LLM reasoning module 
  into an anomaly detection framework, which can be used for a user to detect when in a video something anomalous happens, which the user has described before hand, or
  by presenting a set of normal data points, where the estimated videos deviate from it
  """

  def __init__(self, reasoning_module: true_sight_reasoning_framework, RAG_sentence_transformer_model: RAG_image_interfacer, image_size):
    
    self._image_size = image_size
    
    self._vid_obj = None

    self._reasoning_module = reasoning_module

    if not(self._reasoning_module.is_alive()):
      raise Exception("ERROR: reasoning module was not initiated properly!")

    self._sentence_transformer_name = RAG_sentence_transformer_model

    vector_manager = ImageVectorManager(model_name=self._sentence_transformer_name)
    self._rag_obj = RAG_image_interfacer(image_manager=vector_manager)

    self._agent_tool_box = True_sight_agent_toolbox()


  def load_data(self, path, image_size = None):
    self._vid_obj = video_data_handler(path)
    self._vid_obj.load_from_self(target_image_size=image_size)



  def load_tools_for_reasoning_module(self):
    self._agent_tool_box.get_current_data(vid_data_handler=self._vid_obj, RAG=self._rag_obj)

    self._reasoning_module.grant_main_agent_tool(self._agent_tool_box.TOOL_get_frame_at_index)


  def get_tool_call(self, video_folder_path, query):

    Task_prompt = "TASK: Call the correct tool func given the query"

    prompt =  f"""

              TASK: {Task_prompt}

              Prompt: {query}

              """
    
    self._vid_obj = video_data_handler(video_folder_path)
    self._vid_obj.load_from_self(target_image_size=self._image_size)
    self._vid_obj.load_images_from_path(video_folder_path, target_size=self._image_size)

    self.load_tools_for_reasoning_module()
    
    response = self._reasoning_module.run_framework(query=prompt)


    if response.message.tool_calls:

      fetched_tools = []
      arguments_list = []
      for call in response.message.tool_calls:

        tool_name = call.function.name

        tool = self._reasoning_module._tools_for_agents[self._reasoning_module._main_agent_role][tool_name]
        arguments = dict(call.function.arguments or {})


        fetched_tools.append(tool)
        arguments_list.append(arguments)



    else:
      print("ERROR: No tools was received!")

      raise Exception("ERROR: Reasoning module did not fetch any tools")

    return fetched_tools, arguments_list



  def reasoning_single_image(self, image, query, normal_data, normal_prompt):
    """
    SUMMARY:  This function inputs an image, normal data (both image and text), and a textual command. It 
              It uses these inputs to generate a reasoning prompt, which will be added to the full report
              of the video analyzer later.
    
    ARGS:     image [np.ndarray]: the image that will be tested if is an anomaly or not
              Query [string]: This text is the main command that is followed by the input data
              normal_data [[nd.ndarray]]: A list of images that are semantically close to the estimated 
                                          image, but within the normal data. this data is retrieved
                                          by RAG.
              normal_prompt [string]: this text describes what in the image is focused on in terms of
                                      what makes it normal data, and where to focus.

    RETURNS:
              Image_report [string]:  The output of this method is a string based report, which has the 
                                      following values:

                                       - Image index
                                       - anomaly label
                                       - anomaly description
    """

    # 1. Combine Textual Data
    # We create a structured prompt that clearly separates the task, the user query, 
    # and the RAG context (normal_prompt).
    task_prompt = "TASK: Analyze the provided image for anomalies. Compare the target image against the provided 'Normal' reference images."
        
    combined_images = [image] + list(normal_data)


    full_prompt = f"""
    {task_prompt}

    USER QUERY: 
    {query}

    NORMAL DATA DESCRIPTION:
    {normal_prompt}

    INSTRUCTION: 
    Compare Image 1 against Images 2-{len(combined_images)}. 
    If Image 1 differs significantly from the normal references in the ways described in the context, mark it as an anomaly.
    """

    response_rejected = True

    while response_rejected:
      response = self._reasoning_module.run_framework(query=full_prompt, image_input=combined_images)

      if not(check_arguments_valid(arguments_dict={"verdict", "anomaly_category", "reasoning"}, response=response)):
        full_prompt += "FUNCTION MUST PASS THE FOLLOWING ARGUMENTS TO THE TOOL \"TOOL_determine_image_anomality\" TOOL CALL: \n\"verdict\" [bool], \"anomaly_category\"  [str], \"reasoning\" [str]"
      else:
        response_rejected = False


    return response





@log_decorator(logging.INFO)
def main():

  k_nn = 10
  image_size = (224, 224)
  RAG_sentence_transformer_model = "CLIP-ViT-B-32"


  # reasoning_module = single_agent_anomaly_detection_module()
  reasoning_module = self_reflective_single_agent()

  anomaly_detector = True_sight_interfacer(reasoning_module=reasoning_module, RAG_sentence_transformer_model=RAG_sentence_transformer_model, image_size=image_size)
  
  prompt = "Use the given function call, to give me an anomaly report on the target image."

  normal_data_prompt = """
                            ***Normal process behavior***:
                          This process is a linear vibration feeding simulation, the linear track (large grey rectangular object) 
                          is vibrating at a frequency where the object seems still in space, but the effect is apparent on the 
                          object on the track which is moving forward.

                          The intention is for the object to move nigh unhindered from the left side of the track to the right.
                          The obejct is to pass across the part trap, which is the large ramp seen in the middle of the track.
                          All anomalous and normal behavior is solely determined based on the actions of the part on the track,
                          which in our case is a lego roundplate. The normal process of the roundplate part is as follows: 
                           - move towards the ramp
                           - Climb the ramp unhindered
                           - Fall off the ledge onto the right side of the ramp linear feeder
                           - Move forwards on the linear feeder
                           - As long as the roundplate is still on the track, then no anomalies has occured. 
                           - Movement of the object cannot be used for argumentation, since the images does not capture directional information. 
                          

                          Anomalous behavior could be exemplified as:
                           - The part has been pushed of the linear feeder vertical to the part movement direction, due to the previous movements, when falling of the ramp (post-track rejection)
                           - The part has been pushed of the linear feeder vertical to the part movement direction, prior to falling of the ramp (pre-trap rejection)
                           - trap rejection means that the object has been or is amidst of being pushed off the track entirely, while movement parallel to the track direction is expected.
                           - It is an anomaly if there are any indications that the part is orientented in such a way, that it seems to be falling off the track. 

                           There are more anomalous behaviors than these examples, so watch out for all behavior that does not adhere to the normal process.

                           
                              ***Normal data description***:
                          The following normal data images are RAG retrieved semantically adjacent images of the image depicting the process that is been conducted.
                          The images only represent static information, such as position of part in relation to the linear feeder. Thereefore it is solely the position of the object, 
                          that should be used to determine the anomality of the object 
                           """




  vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"
  input_image_path = "data/DCM4_anomalies/post_trap_rejection/Video_index_15_viewport_0/Video_index_15_viewport_0_frame_173.png"

  vid_obj = video_data_handler(vid_path)
  vid_obj.load_from_self(target_image_size=image_size)


  anomaly_detector.load_data(path=vid_path, image_size=image_size)
  input_image = anomaly_detector._vid_obj.load_single_image_from_path(path=input_image_path, target_size=image_size)

  anomaly_detector._rag_obj.append_video_to_database(anomaly_detector._vid_obj._data)
  anomaly_detector._rag_obj.initialize_KD_tree()

  output = anomaly_detector._rag_obj.run_RAG(input_image, k_nn=k_nn)

  anomaly_detector._reasoning_module.grant_main_agent_tool(anomaly_detector._agent_tool_box.TOOL_determine_image_anomality)
  response = anomaly_detector.reasoning_single_image(image=input_image, query=prompt, normal_data=output["images"], normal_prompt=normal_data_prompt)

  if response.message.tool_calls:

      fetched_tools = []
      arguments_list = []
      for call in response.message.tool_calls:

        tool_name = call.function.name

        tool = anomaly_detector._reasoning_module._tools_for_agents[anomaly_detector._reasoning_module._main_agent_role][tool_name]
        arguments = dict(call.function.arguments or {})

        fetched_tools.append(tool)
        arguments_list.append(arguments)

  verdict, anomaly_category, reasoning = fetched_tools[0](**arguments_list[0])
  
  print(f"\n{'='*60}\n COMMITTING VERDICT\n{'='*60}")
  print(f" Verdict:   {verdict}")
  print(f" Category:  {anomaly_category}")
  print(f" Reasoning: {reasoning}")



if __name__ == '__main__':
  main()


