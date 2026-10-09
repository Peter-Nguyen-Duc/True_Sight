


from RAG import RAG_image_interfacer, ImageVectorManager

from data_handler import video_data_handler

from True_sight_LLM_reasoning_module import single_agent_anomaly_detection_module, true_sight_reasoning_framework, self_reflective_single_agent

from agent_tools import True_sight_agent_toolbox


from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()



@decorate_all_class_methods()
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







@log_decorator(logging.INFO)
def main():

  image_size = (224, 224)
  RAG_sentence_transformer_model = "CLIP-ViT-B-32"


  # reasoning_module = single_agent_anomaly_detection_module()
  reasoning_module = self_reflective_single_agent()

  anomaly_detector = True_sight_interfacer(reasoning_module=reasoning_module, RAG_sentence_transformer_model=RAG_sentence_transformer_model, image_size=image_size)
  
  prompt = "give me the tool with the arguments required to retrieve the images of the video from index 7,8 and 4"

  video_data_folder = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"

  val1, val2 = anomaly_detector.get_tool_call(video_folder_path=video_data_folder, query=prompt)

  print(f"Gained functions:  {val1}")
  print(f"With arguments: {val2}")





if __name__ == '__main__':
  main()


