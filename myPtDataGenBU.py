import os
import json
import math
import numpy as np 
import matplotlib.image as mpimg
from skimage import io
import tensorflow as tf
from skimage.transform import resize
import random

class Pt_datagen_bu:

	def __init__(self,data_dir,anno_dir,model_input_shape,model_output_shape,batch_size_select,data_for):


		self.id_to_file_dict = {}
		self.vid_to_id_dict = {}
		self.id_to_track_id = {}
		self.id_to_kpv = {}
		self.id_to_kp = {}
		self.id_to_valid = {}
		self.id_to_wh = {}

		self.pair_dict = {}

		self.start_idx = []
		self.end_idx = []

		self.img_ids = []

		self.input_shape = model_input_shape
		self.output_shape = model_output_shape

		self.n_imgs = None
		self.batch_size = batch_size_select
		self.n_batchs = None

		self.data_dir = data_dir
		self.anno_dir = anno_dir
		self.data_for = data_for

		self.get_data_from_dir()
		self.get_pair_dict()
		# split kps and valids
		# get wh dict
		# 

	def get_data_from_dir(self):
		temp_image_id_with_label = []
		temp_file_name_with_label = []
		temp_anno_bbox_list = []
		temp_anno_track_id_list = []
		temp_anno_id_list = []
		temp_anno_kp_list = []
		temp_vid_to_id_dict = {}
		temp_id_to_track_id = {}
		temp_id_to_kpv = {}

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
				current_track_id = []
				current_image_id = []
				current_kpv = []
				temp_keys = list(anno.keys())
				to_check_keys = ['image_id','bbox','track_id','keypoints']
				if all(item in temp_keys for item in to_check_keys):
					if anno['image_id'] in temp_image_id_with_label:
						bbox_temp = anno['bbox']
						if bbox_temp[2] > 0 and bbox_temp[3] > 0:
							if bbox_temp[0] >= 0 and bbox_temp[1] >= 0:
								# temp_anno_bbox_list.append(anno['bbox'])
								current_track_id.append(anno['track_id'])
								current_image_id.append(anno['image_id'])
								current_kpv.append(anno['keypoints'])

				# create dict
				current_unique_image_id = list(set(current_image_id))
				for cid in current_unique_image_id:
					cidx = [ x for x in range(len(current_image_id)) if current_image_id[x] == cid]
					c_track_id = [current_track_id[x] for x in cidx]
					c_kpv = [current_kpv[x] for x in cidx]

					temp_id_to_track_id[cid] = c_track_id
					temp_id_to_kpv[cid] = c_kpv


		temp_id_to_file_dict = {temp_image_id_with_label[i]:temp_file_name_with_label[i] for i in range(len(temp_image_id_with_label))}

		# self.bbox = temp_anno_bbox_list
		# self.kps_and_valid = temp_anno_kp_list
		# self.track_ids = temp_anno_track_id_list
		# self.img_ids = temp_anno_id_list
		self.id_to_file_dict = temp_id_to_file_dict
		self.vid_to_id_dict = temp_vid_to_id_dict
		self.id_to_track_id = temp_id_to_track_id
		self.id_to_kpv = temp_id_to_kpv

	def get_pair_dict(self):
		temp_vid_to_id_dict = self.vid_to_id_dict
		temp_valid_keys = self.id_to_kpv
		temp_pair_dict = {}
		temp_img_ids = []
		temp_wh_dict = {}
		for vid,i_ids in temp_vid_to_id_dict.items():
			len_imgs_in_vid = len(i_ids)
			temp_first_img = i_ids[0]
			temp_img = io.imread(self.data_dir + self.id_to_file_dict[temp_first_img])
			img_h = temp_img[0]
			img_w = temp_img[1]
			for i in range(len_imgs_in_vid-1):
				f_0_id = i_ids[i]
				f_1_id = i_ids[i+1]
				if f_0_id in temp_valid_keys and f_1_id in temp_valid_keys:
					temp_pair_dict[f_0_id] = f_1_id
					temp_wh_dict[f_0_id] = (img_w,img_h)
					temp_img_ids.append(f_0_id)

		self.pair_dict = temp_pair_dict
		self.id_to_wh = temp_wh_dict
		self.img_ids = temp_img_ids
		self.n_imgs = len(temp_img_ids)

