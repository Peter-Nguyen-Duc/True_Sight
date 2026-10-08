# You will need: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import cv2
import os

from sklearn.neighbors import KDTree

from data_handler import video_data_handler

from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file
import logging
generate_logger_file()



@decorate_all_class_methods()
class ImageVectorManager:
	def __init__(self, model_name, RAG_data_path = "RAG_data"):
		# This loads a small, fast CLIP model specifically for vectorization

		if os.path.isdir(RAG_data_path + "/" + model_name):
			self.model = SentenceTransformer(RAG_data_path + "/" + model_name)
		else:
			print("downloading model for later use...")
			self.model = SentenceTransformer(model_name)

			os.makedirs(RAG_data_path, exist_ok=True)
			os.makedirs(RAG_data_path + "/" + model_name, exist_ok=True)

			self.model.save(RAG_data_path + "/" + model_name )


	def vectorize_image(self, image_np):
		"""
		Converts a numpy array (OpenCV format) to a normalized CLIP vector.
		"""
		# 1. Convert OpenCV BGR to RGB (CLIP expects RGB)
		image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		
		# 2. Convert numpy array to PIL Image (CLIP requirement)
		pil_img = Image.fromarray(image_rgb)
		
		# 3. Encode image to a vector (embedding)
		vector = self.model.encode(pil_img, show_progress_bar=False)

		normalized_vector = vector / np.linalg.norm(vector)

		return normalized_vector



	def calculate_similarity(self, vec1, vec2):
		"""
		Calculates cosine similarity between two vectors.
		1.0 = Identical, 0.0 = Completely different.
		"""
		return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


@decorate_all_class_methods()
class RAG_image_interfacer:

	def __init__(self, image_manager: ImageVectorManager):
		self._image_manager = image_manager
		self._normal_data_list = []
		self._kdtree = None

	def append_img_to_database(self, image):

		vectorized_image = self._image_manager.vectorize_image(image)

		self._normal_data_list.append(vectorized_image)


	def append_video_to_database(self, image_list):
		"""
		SUMMARY: 
								This method uses a list og images in lexicographical order to represent the images
								All vectorized images are then stored into the database variable sefl._normal_data_list
		
		image_list [list[numpy.ndarray]]: List of images for vectorization
		"""


		for image in image_list:
			self.append_img_to_database(image=image)


	def clear_database(self):
		self._normal_data_list = np.array([])


	def load_database_from_path(self, path):
		self._normal_data_list = np.load(file=path)

		self.initialize_KD_tree(data = self._normal_data_list)


	def save_database_to_path(self, path):
		np.save(file = path, arr=self._normal_data_list)


	def initialize_KD_tree(self, data=None):

		if data==None:
			self._kdtree = KDTree(self._normal_data_list)
		else:
			self._kdtree = KDTree(data)




	def run_RAG(self, query, k_nn):

		if self._kdtree == None:
			raise Exception("ERROR: kdtree was not initialized yet!")

		vectorized_query = self._image_manager.vectorize_image(query)

		distances, indices = self._kdtree.query([vectorized_query], k=k_nn)
		distances = distances[0]
		indices = indices[0]

		output = {}

		output["images"] = np.array(self._normal_data_list)[indices]
		output["cosine_sims"] = self.convert_distances_to_cosine_sims(distances=distances)

		return output


	def convert_distances_to_cosine_sims(self, distances):
		cosine_sims = []

		for dist in distances:
			cosine_sims.append(1 - (dist*dist) / 2)

		return cosine_sims


def image_manager_tester():
	anomaly_image_path = "data/high_visual_anomaly.png"

	normal_data_1 = "data/DCM4_anomalies/normal_data/Video_index_2_viewport_0/Video_index_2_viewport_0_frame_3.png"

	normal_data_2 = "data/DCM4_anomalies/normal_data/Video_index_13_viewport_0/Video_index_13_viewport_0_frame_4.png"


	model_name = "CLIP-ViT-B-32"
	# --- Integration into your workflow ---
	vector_manager = ImageVectorManager(model_name=model_name)

	# 1. Create baseline vector
	baseline_img = cv2.imread(normal_data_1)
	baseline_vec = vector_manager.vectorize_image(baseline_img)

	# 2. Check a new image
	new_img = cv2.imread(normal_data_2)
	new_vec = vector_manager.vectorize_image(new_img)


	# 2. Check anomalous image
	new_img = cv2.imread(anomaly_image_path)
	new_vec_2 = vector_manager.vectorize_image(new_img)


	similarity = vector_manager.calculate_similarity(baseline_vec, new_vec)

	similarity_2 = vector_manager.calculate_similarity(baseline_vec, new_vec_2)

	print("similarity embeddings [0]:", similarity)
	print("similarity embeddings [1]:", similarity_2)


@log_decorator(logging.INFO)
def main():

	
	model_name = "CLIP-ViT-B-32"
	# --- Integration into your workflow ---
	vector_manager = ImageVectorManager(model_name=model_name)
	rag_obj = RAG_image_interfacer(image_manager=vector_manager)


	vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"
	anomaly_image_path = "data/high_visual_anomaly.png"
	normal_data_1 = "data/DCM4_anomalies/normal_data/Video_index_2_viewport_0/Video_index_2_viewport_0_frame_3.png"


	vid_obj = video_data_handler(vid_path)
	vid_obj.load_from_self(target_image_size=(224, 224))

	rag_obj.append_video_to_database(vid_obj._data)
	rag_obj.initialize_KD_tree()


	anomalous_im = vid_obj.load_single_image_from_path(anomaly_image_path)

	
	normal_data = vid_obj.load_single_image_from_path(normal_data_1)


	print("Normal data output: ")
	print(rag_obj.run_RAG(normal_data, k_nn=10)["cosine_sims"])


	print("="*25)
	print(rag_obj.run_RAG(anomalous_im, k_nn=10)["cosine_sims"])



	

if __name__ == "__main__":
	main()
	