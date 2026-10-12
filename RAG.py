# You will need: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import cv2
import os

from sklearn.neighbors import KDTree

from data_handler import video_data_handler

from logging_decorator.decorator_logging import decorate_all_class_methods, log_decorator, generate_logger_file, ignore_logging
import logging
generate_logger_file()


import matplotlib.pyplot as plt
import numpy as np
import pickle


@log_decorator(logging.INFO)
def plot_rag_results(output, target_image, k_nn=10, figsize_per_col=2.5):
    """
	Le chat generated plotting

    Sequentially plot the retrieved (estimated) images next to the target image,
    annotated with their cosine similarity.

    Parameters
    ----------
    output : dict
        The dict returned by rag_obj.run_RAG(...), expected to contain:
        - "cosine_sims": sequence of k similarity scores
        - "images": sequence of k images (arrays) or paths
    target_image : np.ndarray or str
        The query/target image (array) or a path to load it from.
    k_nn : int
        Number of neighbors actually retrieved.
    """

    cosine_sims = output["cosine_sims"][:k_nn]
    images = output["images"][:k_nn]

    # --- handle target image ---
    target_img = (plt.imread(target_image) if isinstance(target_image, str)
                  else target_image)
    if target_img.dtype != np.uint8 and target_img.max() > 1.5:
        target_img = (target_img * 255).astype(np.uint8)

    n = len(images)
    fig, axes = plt.subplots(1, n + 1, figsize=(figsize_per_col * (n + 1), 3.5))

    if n + 1 == 1:
        axes = np.array([axes])

    # --- target image ---
    axes[0].imshow(target_img)
    axes[0].set_title("Target", fontsize=16)
    axes[0].axis("off")

    # --- retrieved images with cosine sim ---
    for j, (img, sim) in enumerate(zip(images, cosine_sims), start=1):
        if isinstance(img, str):
            img = plt.imread(img)
        img = np.asarray(img)
        if img.dtype != np.uint8 and img.max() > 1.5:
            img = (img * 255).astype(np.uint8)

        axes[j].imshow(img)
        axes[j].set_title(f"{float(sim):.4f}", fontsize=16)
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()



@decorate_all_class_methods
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

	@ignore_logging # This method is run many hundreds of times so we ignore it in the logging
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


@decorate_all_class_methods
class RAG_image_interfacer:

	def __init__(self, image_manager: ImageVectorManager):
		self._image_manager = image_manager
		self._vectorized_normal_image_list = []
		self._normal_image_list = []
		self._kdtree = None

	@ignore_logging
	def append_img_to_database(self, image):

		vectorized_image = self._image_manager.vectorize_image(image)

		self._vectorized_normal_image_list.append(vectorized_image)
		self._normal_image_list.append(image)


	def append_video_to_database(self, image_list):
		"""
		SUMMARY: 
								This method uses a list og images in lexicographical order to represent the images
								All vectorized images are then stored into the database variable self._vectorized_normal_image_list
		
		image_list [list[numpy.ndarray]]: List of images for vectorization
		"""


		for image in image_list:
			self.append_img_to_database(image=image)


	def clear_database(self):
		self._vectorized_normal_image_list = []
		self._normal_image_list = []


	def load_database_from_path(self, path):

		# Load the model from the file
		with open(path, 'rb') as f:
			dict_output = pickle.load(f)

		self._vectorized_normal_image_list = dict_output["vectorized_images"]
		self._normal_image_list = dict_output["images"]

		self.initialize_KD_tree(data = self._vectorized_normal_image_list)

		#self._vectorized_normal_image_list = np.load(file=path)



	def save_database_to_path(self, path, save_raw_images=False):

		data = {}

		data["vectorized_images"] = self._vectorized_normal_image_list

		if save_raw_images:
			data["images"] = self._normal_image_list
		else:
			data["images"] = None
		
		try:
			with open(path, 'wb') as file:
				pickle.dump(data, file)

		except:
			print("ERROR: EXCEPTION OCCURED")
			return False


		# np.save(file = path, arr=self._vectorized_normal_image_list)

		return True

	def initialize_KD_tree(self, data=None):

		if data==None:
			self._kdtree = KDTree(self._vectorized_normal_image_list)
		else:
			self._kdtree = KDTree(data)




	def run_RAG(self, query, k_nn):
		"""
		SUMMARY:  This function performs a Retrieval-Augmented Generation (RAG) lookup to find the k most 
				  semantically similar images from the normal dataset. It vectorizes the input query 
				  image and uses a k-d tree for efficient nearest-neighbor search to retrieve reference 
				  images and their similarity scores.

		ARGS:     query [np.ndarray]: The target image used as the search query to find similar normal images.
				  k_nn [int]: The number of nearest neighbors (top-k) to retrieve from the database.

		RETURNS:    output [dict]: A dictionary containing the results of the RAG retrieval, consisting of:
					- images [np.ndarray]: An array of the retrieved normal images.
					- cosine_sims [np.ndarray]: The calculated cosine similarity scores between the 
					  query image and the retrieved references.
		"""
		if self._kdtree == None:
			raise Exception("ERROR: kdtree was not initialized yet!")

		vectorized_query = self._image_manager.vectorize_image(query)

		distances, indices = self._kdtree.query([vectorized_query], k=k_nn)
		distances = distances[0]
		indices = indices[0]

		output = {}
		if self._normal_image_list == None:
			output["images"] = None
		else:
			output["images"] = np.array(self._normal_image_list)[indices]


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
	image_size = (512, 512)
	k_nn = 10
	model_name = "CLIP-ViT-B-32"
	# --- Integration into your workflow ---
	vector_manager = ImageVectorManager(model_name=model_name)
	rag_obj = RAG_image_interfacer(image_manager=vector_manager)


	vid_path = "data/DCM4_anomalies/normal_data/Video_index_0_viewport_0"
	high_anomaly_data = "data/high_visual_anomaly.png"
	normal_data_1 = "data/DCM4_anomalies/post_trap_rejection/Video_index_5_viewport_0/Video_index_5_viewport_0_frame_200.png"


	vid_obj = video_data_handler(vid_path)
	vid_obj.load_from_self(target_image_size=image_size)

	rag_obj.append_video_to_database(vid_obj._data)
	rag_obj.initialize_KD_tree()



	normal_data = vid_obj.load_single_image_from_path(normal_data_1, target_size=image_size)


	print("Normal data output: ")
	output = rag_obj.run_RAG(normal_data, k_nn=k_nn)

	plot_rag_results(output=output, target_image=normal_data, k_nn=k_nn)


	

if __name__ == "__main__":
	main()
	