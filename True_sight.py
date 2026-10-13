


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


  def load_vid_obj(self, vid_obj: video_data_handler):
    self._vid_obj = vid_obj



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




  def single_image_AD(self, vid_path, input_image_path, k_nn, image_size, prompt):

    task_prompt = "TASK: Analyze the provided image for anomalies. Compare the target image against the provided 'Normal' reference images."
        
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

    full_prompt = f"""
    {task_prompt}

    USER QUERY: 
    {prompt}

    NORMAL DATA DESCRIPTION:
    {normal_data_prompt}

    INSTRUCTION: 
    Compare Image 1 against Images 2-{len(combined_images)}. 
    If Image 1 differs significantly from the normal references in the ways described in the context, mark it as an anomaly.
    """


    # Clean up the data storage before running.
    if self._rag_obj != None: self._rag_obj.clear_database()
    if self._vid_obj != None: self._vid_obj.clear_data()
    if self._reasoning_module != None: self._reasoning_module.clear_all_tools()

    # Load normal data, insert into RAG object and create rag database
    vid_obj = video_data_handler(vid_path)
    vid_obj.load_from_self(target_image_size=image_size)
    self.load_vid_obj(vid_obj)
    self._rag_obj.append_to_database(self._vid_obj._data)
    self._rag_obj.initialize_KD_tree()
    
    # Load normal image
    input_image = self._vid_obj.load_single_image_from_path(path=input_image_path, target_size=image_size)

    # Get RAG image retrieval
    output = self._rag_obj.run_RAG(input_image, k_nn=k_nn)

    # Combine input image with normal images
    combined_images = [input_image] + list(output["images"])


    # Generate variable constrain for prompt
    variable_constrain = {"is_anomaly": bool, 
                          "anomaly_category": str,
                          "description": str,}

    
    output = self._reasoning_module.run_framework(query=full_prompt, image_input=combined_images, variable_constrain=variable_constrain)


    print(f"\n{'='*60}\n COMMITTING VERDICT\n{'='*60}")
    print(f" Verdict:   {output["is_anomaly"]}")
    print(f" Category:  {output["anomaly_category"]}")
    print(f" Reasoning: {output["description"]}")


    return output["is_anomaly"], output["anomaly_category"], output["description"]





@log_decorator(logging.INFO)
def main():

  input_images_count = 10
  RAG_k_nn = 1
  image_size = (224, 224)
  RAG_sentence_transformer_model = "CLIP-ViT-B-32"


  # reasoning_module = single_agent_anomaly_detection_module()
  reasoning_module = self_reflective_single_agent()

  anomaly_detector = True_sight_interfacer(reasoning_module=reasoning_module, RAG_sentence_transformer_model=RAG_sentence_transformer_model, image_size=image_size)
  

  prompt = f"Use the given function call, to give me an anomaly report on the video sequence represented by the {input_images_count} first images on image input."

  task_prompt = "TASK: Analyze the provided image for anomalies. Compare the target images against the provided 'Normal' reference images."
      
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

  full_prompt = f"""
  {task_prompt}

  USER QUERY: 
  {prompt}

  NORMAL DATA DESCRIPTION:
  {normal_data_prompt}

  INSTRUCTION: 
  Compare video represented by Images 1-{input_images_count} with the normal data images with indexes {input_images_count + 1} - {int(input_images_count + input_images_count*RAG_k_nn)}. 
  If Image 1 differs significantly from the normal references in the ways described in the context, mark it as an anomaly.
  """


  normal_data_path_list = ["data/DCM4_anomalies/normal_data/Video_index_0_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_1_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_2_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_3_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_4_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_5_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_6_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_7_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_8_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_9_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_11_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_12_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_13_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_14_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_15_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_16_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_17_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_18_viewport_0",
                      "data/DCM4_anomalies/normal_data/Video_index_19_viewport_0",]
  
  video_input_path = "data/DCM4_anomalies/normal_data/Video_index_10_viewport_0"

  # Clean up the data storage before running.
  if anomaly_detector._rag_obj != None: anomaly_detector._rag_obj.clear_database()
  if anomaly_detector._vid_obj != None: anomaly_detector._vid_obj.clear_data()
  if anomaly_detector._reasoning_module != None: anomaly_detector._reasoning_module.clear_all_tools()

  vid_obj = video_data_handler()

  # # Load normal data into vid object

  # for normal_data_path in normal_data_path_list:
  #   vid_obj.load_images_from_path(path=normal_data_path, target_size=image_size)

  # # give normal data to rag_obj to create the RAG database
  # anomaly_detector._rag_obj.append_to_database(vid_obj._data)
  # anomaly_detector._rag_obj.initialize_KD_tree()
  # anomaly_detector._rag_obj.save_database_to_path("RAG_data/normal_data.pkl", save_raw_images=True)

  anomaly_detector._rag_obj.load_database_from_path("RAG_data/normal_data.pkl")
  # Load input video
  vid_obj.clear_data()
  vid_obj.load_images_from_path(video_input_path)
  video_images = vid_obj._data # list of video_images

  target_input_images = []

  target_input_images.append(video_images[0])

  indx_skip = len(video_images)//input_images_count

  start_index = 0 + int(indx_skip//1.5) # Start in the middle of the video

  for i in range(input_images_count - 2):
    target_input_images.append(video_images[start_index])
    start_index += indx_skip

  target_input_images.append(video_images[len(video_images) - 1 ])


  # Get RAG image retrieval
  RAG_images_list = []
  for im in target_input_images:
    output = anomaly_detector._rag_obj.run_RAG(im, k_nn=RAG_k_nn)
    RAG_images_list.append(output["images"][0])

  # Combine input image with normal images
  combined_images = target_input_images + RAG_images_list



  # Generate variable constrain for prompt
  variable_constrain = {"is_anomaly": bool, 
                        "anomaly_category": str,
                        "description": str,}

  output = anomaly_detector._reasoning_module.run_framework(query=full_prompt, image_input=combined_images, variable_constrain=variable_constrain)


  print(f"\n{'='*60}\n COMMITTING VERDICT\n{'='*60}")
  print(f" Verdict:   {output["is_anomaly"]}")
  print(f" Category:  {output["anomaly_category"]}")
  print(f" Reasoning: {output["description"]}")


  return output["is_anomaly"], output["anomaly_category"], output["description"]




if __name__ == '__main__':
  main()


