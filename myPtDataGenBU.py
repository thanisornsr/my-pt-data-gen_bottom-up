import os
import json
import math
import numpy as np 
import matplotlib.image as mpimg
import tensorflow as tf
from skimage.transform import resize
import random

class Pt_datagen_bu:

	def __init__(self,data_dir,anno_dir,model_input_shape,model_output_shape,batch_size_select,data_for):

		self.kps_and_valid = []
		self.kps = []
		self.valids = []

		self.track_ids = []
		self.img_ids = []
		self.id_to_file_dict = {}
		self.vid_to_id_dict = {}

		self.start_idx = []
		self.end_idx = []

		self.input_shape = model_input_shape
		self.output_shape = model_output_shape

		self.n_imgs = None
		self.batch_size = batch_size_select
		self.n_batchs = None

		self.data_dir = data_dir
		self.anno_dir = anno_dir
		self.data_for = data_for

		self.get_data_from_dir()

	def get_data_from_dir(self):
		temp_image_id_with_label = []
		temp_file_name_with_label = []
		temp_anno_bbox_list = []
		temp_anno_track_id_list = []
		temp_anno_id_list = []
		temp_anno_kp_list = []
		temp_vid_to_id_dict = {}

		temp_anno_dir = self.anno_dir + self.data_for + '/'
		for anno_file in os.listdir(temp_anno_dir):
			current_image_id_with_label = []
			if anno_file.endswith('.json'):
				temp = temp_anno_dir + anno_file
				with open(temp) as f:
					data = json.load(f)
				data_images = data['images']
				data_annotations = data['annotations']
			for temp_image in data_images:
				if temp_image['is_labeled']:
					temp_image_id_with_label.append(temp_image['id'])
					current_image_id_with_label.append(temp_image['id'])
					temp_file_name_with_label.append(temp_image['file_name'])
			temp_vid_to_id_dict[anno_file] = current_image_id_with_label

			
			for anno in data_annotations:
				temp_keys = list(anno.keys())
				to_check_keys = ['image_id','bbox','track_id','image_id','keypoints']
				if all(item in temp_keys for item in to_check_keys):
					if anno['image_id'] in temp_image_id_with_label:
						bbox_temp = anno['bbox']
						if bbox_temp[2] > 0 and bbox_temp[3] > 0:
							if bbox_temp[0] >= 0 and bbox_temp[1] >= 0:
								temp_anno_bbox_list.append(anno['bbox'])
								temp_anno_track_id_list.append(anno['track_id'])
								temp_anno_id_list.append(anno['image_id'])
								temp_anno_kp_list.append(anno['keypoints'])
		temp_id_to_file_dict = {temp_image_id_with_label[i]:temp_file_name_with_label[i] for i in range(len(temp_image_id_with_label))}

		self.n_imgs = len(temp_anno_bbox_list)
		self.bbox = temp_anno_bbox_list
		self.kps_and_valid = temp_anno_kp_list
		self.track_ids = temp_anno_track_id_list
		self.img_ids = temp_anno_id_list
		self.id_to_file_dict = temp_id_to_file_dict
		self.vid_to_id_dict = temp_vid_to_id_dict
