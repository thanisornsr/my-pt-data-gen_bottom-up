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
		self.start_idx, self.end_idx, self.n_batchs = self.get_start_end_idx()
		self.split_kp_and_v()

		self.limb_list = [(0,1),(2,0),(0,3),(0,4),(3,5),(4,6),(5,7),(6,8),(4,3),(3,9),(4,10),(10,9),(9,11),(10,12),(11,13),(12,14)]
		self.n_keypoints = 15
		self.n_limbs = 16
		print('Create datagen_{}: Done ...'.format(self.data_for))


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

			
			current_track_id = []
			current_image_id = []
			current_kpv = []
			for anno in data_annotations:
				
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
			img_h = temp_img.shape[0]
			img_w = temp_img.shape[1]
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

	def get_start_end_idx(self):
		max_idx = self.n_imgs
		temp_batch_size = self.batch_size
		l = list(range(max_idx))
		temp_start_idx = l[0::temp_batch_size]
		def add_batch_size(num,max_id=max_idx,bz=temp_batch_size):
			return min(num+bz,max_id)
		temp_end_idx = list(map(add_batch_size,temp_start_idx))
		temp_n_batchs = len(temp_start_idx)

		return temp_start_idx, temp_end_idx, temp_n_batchs

	def get_target_valid_joint(self,input_kav):
		splited_kps = []
		splited_valids = []
		for temp_anno_kp in input_kav:
			temp_x = np.array(temp_anno_kp[0::3])
			temp_y = np.array(temp_anno_kp[1::3])
			temp_valid = np.array(temp_anno_kp[2::3])
			temp_valid = temp_valid > 0
			temp_valid = temp_valid.astype('float32')
			temp_target_coord = np.stack([temp_x,temp_y],axis=1)
			temp_target_coord = temp_target_coord.astype('float32')

			splited_kps.append(temp_target_coord)
			splited_valids.append(temp_valid)

		return splited_kps,splited_valids

	def split_kp_and_v(self):
		temp_id_to_kp = {}
		temp_id_to_valid = {}
		temp_id_to_kpv = self.id_to_kpv
		for i_id,kpv in temp_id_to_kpv.items():
			t_ks,t_vs = self.get_target_valid_joint(kpv)
			for i in range(len(t_ks)):
				temp_k = t_ks[i]
				temp_v = t_ks[i]

				t_ks[i] = np.delete(temp_k,[1,2],0)
				t_vs[i] = np.delete(temp_v,[1,2])

			temp_id_to_kp[i_id] = t_ks
			temp_id_to_valid[i_id] = t_vs

		self.id_to_kp = temp_id_to_kp
		self.id_to_valid = temp_id_to_valid



