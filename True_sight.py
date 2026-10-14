


from RAG import RAG_image_interfacer, ImageVectorManager

from data_handler import video_data_handler

from True_sight_LLM_reasoning_module import single_agent_anomaly_detection_module, true_sight_reasoning_framework, self_reflective_single_agent

from agent_tools import True_sight_agent_toolbox


from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
# generate_logger_file()

import cv2
import numpy as np
import matplotlib.pyplot as plt

from agent_tools import check_arguments_valid
import copy
import os 
import json

@log_decorator(logging.INFO)
def plt_image(target_image):


    plt.imshow(target_image)

    plt.tight_layout()
    plt.show()


def plot_images(images, title_name):
  for i, im in enumerate(images):
    if i < 7:
      continue
    # 2. Get the image we JUST appended (the last one in the list)
    img = np.asarray(im)

    # 3. Normalize the image for visualization
    # If it's a float array (0.0 to 1.0), convert it to 0-255 uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.1: # Check if it's in 0-1 range
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # 4. USE imshow instead of imread
    plt.figure(figsize=(5, 5)) # Create a window for each image
    plt.imshow(img)            # Display the numpy array
    plt.title(f"{title_name} --- Frame Index: {i}")
    plt.axis('off')           # Hide the axis numbers for a cleaner look
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



  def generate_normal_database(self, normal_data_path_list, image_size, save_path="RAG_data/normal_data.pkl"):
    # Load normal data into vid object, pass vid_object data into the RAG tool to create a vectorization of the images and save it 

    # normal_data_path_list = ["data/DCM4_anomalies/normal_data/Video_index_0_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_1_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_2_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_3_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_4_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_5_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_6_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_7_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_8_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_9_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_11_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_12_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_13_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_14_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_15_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_16_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_17_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_18_viewport_0",
    #                     "data/DCM4_anomalies/normal_data/Video_index_19_viewport_0",]

    
    vid_obj = video_data_handler()

    for normal_data_path in normal_data_path_list:
      vid_obj.load_images_from_path(path=normal_data_path, target_size=image_size)

    # give normal data to rag_obj to create the RAG database
    self._rag_obj.append_to_database(vid_obj._data)
    self._rag_obj.initialize_KD_tree()
    self._rag_obj.save_database_to_path(save_path, save_raw_images=True)

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





  def video_AD(self, video_input_path, image_size, subsambled_input_images, normal_data_path="RAG_data/normal_data.pkl"):



    input_images_count = subsambled_input_images
    normal_data_process_images_count = 10
    RAG_k_nn = 1

    normal_video_demonstration = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"
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
                              -  To mark the object with anomaly post_trap rejection, the roundsplate lego part must clearly be off the linear track boundary, in such a way, that some of 
                                  the object pixels are non-overlapping with the linear feeder. Post-track rejection in constrast to pre-trap rejection stems from visual indication that the 
                                  part was pushed off the track on the right side of the ramp of the linear feeder.
                            - The part has been pushed of the linear feeder vertical to the part movement direction, prior to falling of the ramp (pre-trap rejection)
                              -  As with post trap rejection but in this case, the part must have visual indication that the part was pushed off, before entering the trap ramp.
                            - trap rejection means that the object has been or is amidst of being pushed off the track entirely, while movement parallel to the track direction is expected.
                            - It is an anomaly if there are any indications that the part is orientented in such a way, that it seems to be falling off the track. 

                            There are more anomalous behaviors than these examples, so watch out for all behavior that does not adhere to the normal process.

                          
                                ***Normal data description***:
                            Two sets of normal data will be given. The first set of normal images are RAG retrieved normal data instances. They dont have any specific order, and are 
                            simply semantically adjacent to the input images. The next set of images is a single normal video sequence, which follows the normal procedure.
                            The following normal data images are RAG retrieved semantically adjacent images of the image depicting the process that is been conducted.
                            The images only represent static information, such as position of part in relation to the linear feeder. Thereefore it is solely the position of the object, 
                            that should be used to determine the anomality of the object.
                            """

    full_prompt = f"""
    {task_prompt}

    USER QUERY: 
    {prompt}

    NORMAL DATA DESCRIPTION:
    {normal_data_prompt}

    INSTRUCTION: 
    Compare video represented with labels --- input images --- with the RAG retrieved normal data images labeled --- RAG ---. Note, 
    that the RAG retrieved images are not following any particular order, but are simply images that are semantically close to the input images. 
    The images under --- sub_sampled_normal_video_process --- represents aa normally behaving process, which must also be taken into account when 
    determining the normality of the input video sequence If Images with the labels --- input images --- differs significantly from the normal 
    references in the ways described in the context, mark it as an anomaly.
    """


    # Clean up the data storage before running.
    if self._rag_obj != None: self._rag_obj.clear_database()
    if self._vid_obj != None: self._vid_obj.clear_data()
    if self._reasoning_module != None: self._reasoning_module.clear_all_tools()



    self._rag_obj.load_database_from_path(normal_data_path)
    
    
    # Load normal video process sequence
    normal_sequence_video_path = normal_video_demonstration
    normal_video_process_subsambled = []

    vid_obj = video_data_handler()
    vid_obj.load_images_from_path(normal_sequence_video_path, target_size=image_size)
    normal_process_video_images = vid_obj._data # list of video_images

    normal_video_process_subsambled.append(normal_process_video_images[0])

    indx_skip = len(normal_process_video_images)//normal_data_process_images_count

    start_index = 0 + int(indx_skip) # Start in the middle of the video

    for i in range(normal_data_process_images_count - 2):
      normal_video_process_subsambled.append(normal_process_video_images[start_index])
      start_index += indx_skip

    normal_video_process_subsambled.append(normal_process_video_images[len(normal_process_video_images) - 1 ])

    
    # Load input video

    vid_obj.clear_data()
    vid_obj.load_images_from_path(video_input_path, target_size=image_size)
    video_images = copy.deepcopy(vid_obj._data) # list of video_images
    video_images = video_images[:-1]
    target_input_images = []

    target_input_images.append(video_images[0])

    indx_skip = len(video_images)//input_images_count

    start_index = 0 + int(indx_skip//1.5) # Start in the middle of the video

    for i in range(input_images_count - 2):

      target_input_images.append(video_images[start_index])
      start_index += indx_skip

    target_input_images.append(video_images[len(video_images) - 1 ])

    #plot_images(target_input_images, "new normal data")

    # Get RAG image retrieval
    RAG_images_list = []
    for im in target_input_images:
      output = self._rag_obj.run_RAG(im, k_nn=RAG_k_nn)
      RAG_images_list.append(output["images"][0])

    # Combine input image, RAG retrieved normal images, and a static normal image video process
    combined_images = {}
    target_input_images, RAG_images_list, normal_video_process_subsambled

    combined_images["input images"] = target_input_images
    combined_images["RAG"] = RAG_images_list
    combined_images["sub_sampled_normal_video_process"] = normal_video_process_subsambled

    # plot_images(combined_images["input images"], "input_images")
    # plot_images(combined_images["RAG"], "RAG")
    # plot_images(combined_images["sub_sampled_normal_video_process"], "sub_sampled_normal_video_process")
    




    # Generate variable constrain for prompt
    variable_constrain = {"is_anomaly": bool, 
                          "anomaly_category": str,
                          "description": str,}

    output = self._reasoning_module.run_framework(query=full_prompt, image_input=combined_images, variable_constrain=variable_constrain)


    return output #output["is_anomaly"], output["anomaly_category"], output["description"]




@log_decorator(logging.INFO)
def main():

  json_dataset_path = "output_data"
  image_size = (224, 224)
  RAG_sentence_transformer_model = "CLIP-ViT-B-32"
  # reasoning_module = single_agent_anomaly_detection_module()
  reasoning_module = self_reflective_single_agent()

  anomaly_detector = True_sight_interfacer(reasoning_module=reasoning_module, RAG_sentence_transformer_model=RAG_sentence_transformer_model, image_size=image_size)

  get_list_of_datasets = "test_dataset.txt"

  with open(get_list_of_datasets, 'r') as f:
    paths = sorted([x.strip() for x in f.readlines()])


  for path in paths:
    # Iterate through every item inside the parent folder
    results_list = []

    for subfolder in os.listdir(path):
      full_subfolder_path = os.path.join(path, subfolder)

      # Ensure we are only processing directories (ignoring files like .DS_Store or .txt)

      if os.path.isdir(full_subfolder_path):

        try:
          results = anomaly_detector.video_AD(
              video_input_path=full_subfolder_path, 
              subsambled_input_images=10, 
              image_size=image_size
              )

        except Exception as e:
            
            del anomaly_detector
            print("Re-initializing anomaly_detector...")
            
            # Re-setup the detector as requested
            image_size = (224, 224)
            RAG_sentence_transformer_model = "CLIP-ViT-B-32"
            reasoning_module = self_reflective_single_agent()
            
            anomaly_detector = True_sight_interfacer(
                reasoning_module=reasoning_module, 
                RAG_sentence_transformer_model=RAG_sentence_transformer_model, 
                image_size=image_size
            )



        results["video_path"] = full_subfolder_path


        print(f"\n{'='*60}\n COMMITTING VERDICT\n{'='*60}")
        print(f" video datapath: {results["video_path"]}")
        print(f" anomaly detected:   {results["is_anomaly"]}")
        print(f" Category:  {results["anomaly_category"]}")
        print(f" Reasoning: {results["description"]}")


        results_list.append(results)


    # 2. Create the JSON filename based on the folder name (last part of the path)
    json_filename = f"{os.path.basename(path)}.json"

    # 3. Save the results to the JSON file
    with open(json_dataset_path + "/" + json_filename, 'w') as json_file:
      json.dump(results_list, json_file, indent=4)

    print(f"Saved results to {json_filename}")




if __name__ == '__main__':
  main()


