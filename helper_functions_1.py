# ===================================================================================================

import time
import sympy
from sympy.ntheory import factorint
from scipy.special import comb
import copy
from collections import OrderedDict
from itertools import combinations
import random
import numpy as np
import os
from fractions import Fraction
import math
import networkx
from networkx.algorithms.approximation import steinertree

import tvm
from tvm import te
from tvm import topi
from tvm import relay
import tvm.relay.testing

# 3rd party to get DNNs
import mxnet
import gluoncv

# autoTVM
from tvm import autotvm
import logging
import tempfile
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity, ConfigEntity, SplitSpace, ReorderSpace, AnnotateSpace, OtherOptionSpace

# Ansor
from tvm import auto_scheduler


# os.environ['CUDA_VISIBLE_DEVICES']='1' # select the GPU we want to use






def get_delta_changes(diff, cand_features):
	'''
	Get all combinations of changes on each can_feature diff can bring.
	INPUT:
		diff:
			int. The value difference (in terms of multiplication on some axis in the loop nest)
		cand_features:
			list of strings. The parallel setting features related to the cost model.
	OUTPUT:
		dictionary of list storing all possible changes on cand_features by diff.
	'''
	fact_dict = factorint(int(diff))
	# use itertools.combinations() to generate all combinations of delta changes of each feature
	# The combinations would not be too many, because if the reuse cost is larger than the tune cost (limited by early stop number)
	# we would directly tune it.
	last_dict = dict()
	for feature in cand_features:
		last_dict[feature] = list()
		last_dict[feature].append(1)
	# 
	for factor in fact_dict.keys():
		delta_dict = dict()
		for feature in cand_features:
			delta_dict[feature] = list()
		positions = list(range( 1, fact_dict[factor] + len(cand_features) + 1 ))
		for select_psts in combinations(positions, len(cand_features)):
			amounts = sorted(select_psts)
			for i in range( len( last_dict[cand_features[0]] ) ):
				# iterate over all partial combinations 
				for f_i in range(len(cand_features)):
					feature = cand_features[f_i]
					if f_i == 0:
						amount = amounts[f_i] - 1
					else:
						amount = amounts[f_i] - amounts[f_i - 1] - 1
					delta_dict[feature].append( last_dict[feature][i] * (factor ** amount ) )
		last_dict = delta_dict
	# Now last_dict stores all the possible delta changes on the cand_features
	return last_dict






def get_combinations(tot, groups):
	'''
	Get all combinations to divide tot into groups so that the product of values assigned to groups is tot.
	INPUT:
		num:
			int. The tot number of things to be dicided into groups
		groups:
			list of strings. The groups to partition tot into.
	OUTPUT:
		dictionary of list storing all possible value in groups.
	'''
	fact_dict = factorint(int(tot))
	# use itertools.combinations() to generate all combinations
	last_dict = dict()
	for group in groups:
		last_dict[group] = list()
		last_dict[group].append(1)
	# 
	for factor in fact_dict.keys():
		delta_dict = dict()
		for group in groups:
			delta_dict[group] = list()
		positions = list(range( 1, fact_dict[factor] + len(groups) ))
		for select_psts in combinations(positions, len(groups) - 1):
			amounts = sorted(select_psts)
			for i in range( len( last_dict[groups[0]] ) ):
				# print(fact_dict, fact, f_i)
				# iterate over all partial combinations 
				for f_i in range(len(groups)):
					group = groups[f_i]
					if f_i == 0:
						amount = amounts[f_i] - 1
					elif (f_i < len(groups) - 1):
						amount = amounts[f_i] - amounts[f_i - 1] - 1
					else:
						amount = fact_dict[factor] + len(groups) - amounts[f_i - 1] - 1
					delta_dict[group].append( last_dict[group][i] * (factor ** amount ) )
		last_dict = delta_dict
	# Now last_dict stores all the possible delta changes on the cand_features
	return last_dict










def get_product(elements):
	'''
		Get the product of the elements. 
		INPUT:	elements: list of ints.
	'''
	product = 1
	for i in elements:
		product = product * i
	return product





def get_input_len_for_conv(out_len, stride, dilation, reduc_len):
	'''
		A helper founction to compute the input length (1d) 
		if we want to get the given #out_length points in convolution.
		INPUT:
			out_len: int. The number of output points we want to computed.
			stride: int. The stride value.
			dilation: int. The dilation value.
			reduc_len: int. The reduction length. 
				For example, kernel width is 3, and we only read 2 into shared mem. So the reduc_len = 2.
		OUTPUT:
			in_len: int.
	'''
	return ((out_len - 1) * stride + (reduc_len - 1) * dilation + 1)






def conv_inputIdex(out_idx, reduc_idx, stride, dilation):
	'''
		A helper function to compute the index in the input we need to compute the given output point.
		RMK: The index starts from 0.
		INPUT:
			out_idx:	int. The index of the output point we want to compute.
			reduc_idx:	int. The index of the reduction element we want to compute.
			stride:		int. The stride value.
			dilation:	int. The dilation value.
		OUTPUT:
			in_idx:		int.
	'''
	return (out_idx * stride + reduc_idx * dilation)






def cal_32B_trans(blk_shape, op_para, simplify = False):
	'''
	Calculate a block needs how many memory transactions to load data from global memory to shared memory.
	According to the documatation, P100 cache global memory in L2 cache, and conducts 32-byte memory transactions,
	so we count the number of 32B transactions a given block shape and thread size would request.
	Every time, a thread would request a word, which is 4 byte (4B) for float32 number.
	# 
	INPUT:
		blk_shape:
			list of int. The block shape.
		# thrd_size:
		# 	int. how many threads are there in this block.
		op_para:
			a dictionary of parameters, which are necessary for us to compute the amount of memory transactions, also INCLUDES op_type and op loop structure.
		simplify:
			bool. If True, then we use one block's #(cache_line request) to approximate the actual amount; else we count the sum over all blocks.
	# 
	RMK:
		We only calculate the number of cache lines assuming perfect vectorization of memory load
	'''
	def xy2idx(coordinates, glob_shape):
		''' 
			transform the coordinates in global mem to the unique 1 dimension id in the global memory env 
			coordinates: the coordinates in the global env, list of ints.
			glob_shape:	the extent of the global data space axes corr. to the coordinates, list of ints. THE EXTENTS are listed from outer axes to inner axes.
		'''
		idx = coordinates[0]
		for i in range(1, len(coordinates)):
			idx = idx * glob_shape[i] + coordinates[i]
		return idx
	# 
	def update_ncl(cl_size, offset, xyranges, coordinates, glob_shape, results):
		idx = xy2idx(coordinates, glob_shape) + offset
		if idx > results[1]:
			# belong to a new cache line
			results[0] = results[0] + 1
			results[1] = (idx // cl_size + 1) * cl_size - 1
		# 
		flag = False
		for i in range(len(coordinates)-1, -1, -1):
			# iter from inner axes
			if coordinates[i] < xyranges[i][1]:
				flag = True
				coordinates[i] = coordinates[i] + 1
				for j in range(i+1, len(coordinates)):
					coordinates[j] = xyranges[j][0]
				break
		if flag:
			return coordinates # update_ncl(cl_size, offset, xyranges, coordinates, glob_shape, results)
		else:
			return None
	# 
	def ncacheline_fast(cl_size, offset, xyranges, glob_shape):
		''' 
			get the number of cache lines given the coordinate ranges from outer to inner axes, and the initial offset of the first data.
			This methods try to calculate the result directly to some degree, so that more efficient.
			The dictionary below: 
				KEY: ((1) the OFFSET in a cache line of the first point of a consecutive segment, (2) the cache line NO increase of the last point based on the 1st point, 
				(3) the OFFSET in a cache line of the last point, (4) the cache line NO increase of the 1st point of the next segment based on the 1st point, 
				(5) the OFFSET in a cache line of the 1st point of the next segment).
		''' 
		def key_transit(key, up_off):
			'''Compute the new key based on the given key and the up_off to the first point in the segment corresponding to the key'''
			idx1 = key[0] + up_off
			idx2 = key[1] * cl_size + key[2] + up_off
			idx3 = key[3] * cl_size + key[4] + up_off
			key = (idx1 % cl_size, (idx2//cl_size) - (idx1//cl_size), idx2%cl_size, (idx3//cl_size) - (idx1//cl_size), idx3%cl_size)
			return key
		# 
		cl_types = dict()
		last_key = None
		last_idx = None
		for i in range(len(xyranges) - 1, -1, -1):
			# compute from the last (innermost) axis to the 1st (outermost) axis 
			if i == len(xyranges) - 1:
				xy1 = [xyranges[i][0] for i in range(len(xyranges))]
				xy2 = copy.deepcopy(xy1)
				xy2[-1] = xyranges[-1][1]
				idx1 = xy2idx(xy1, glob_shape) + offset
				idx2 = xy2idx(xy2, glob_shape) + offset
				key = (idx1 % cl_size, (idx2//cl_size) - (idx1//cl_size), idx2%cl_size, None, None)
				cl_types[key] = 1
				last_key = key
				last_idx = idx1
			else:
				if xyranges[i][0] == xyranges[i][1]:
					continue
				new_cl_types = dict()
				up_off = None
				for v in range(xyranges[i][0], xyranges[i][1] + 1):
					# iter all values in i-th range
					if v == xyranges[i][0]:
						# we need to update last_key
						del cl_types[last_key]
						xy_next = [xyranges[i][0] for i in range(len(xyranges))]
						xy_next[i] = xyranges[i][0] + 1
						idx_next = xy2idx(xy_next, glob_shape) + offset
						last_key = (last_key[0], last_key[1], last_key[2], (idx_next//cl_size) - (last_idx//cl_size), idx_next%cl_size)
						if last_key in cl_types.keys():
							cl_types[last_key] = cl_types[last_key] + 1
						else:	
							cl_types[last_key] = 1
						# copy cl_types
						for key, value in cl_types.items():
							new_cl_types[key] = value
					else:
						# update new_cl_types
						up_off = v - xyranges[i][0]
						for dt in glob_shape[i+1:]:
							up_off = up_off * dt
						for key, value in cl_types.items():
							key_t = key_transit(key, up_off)
							if key_t in new_cl_types.keys():
								new_cl_types[key_t] = new_cl_types[key_t] + value
							else:
								new_cl_types[key_t] = value
						# last_key = key_transit(last_key, up_off) # update last_key
				# update last_key and last_idx
				last_key = key_transit(last_key, up_off) 
				new_cl_types[last_key] = new_cl_types[last_key] - 1
				if new_cl_types[last_key] == 0:
					del new_cl_types[last_key]
				xy1 = [xyranges[i][0] for i in range(i)] + [xyranges[i][1] for i in range(i, len(xyranges)-1)] + [xyranges[-1][0]]
				# xy2 = copy.deepcopy(xy1)
				# xy2[-1] = xyranges[-1][1]
				idx1 = xy2idx(xy1, glob_shape) + offset
				# idx2 = xy2idx(xy2, glob_shape) + offset
				last_key = (last_key[0], last_key[1], last_key[2], None, None)
				# last_key = (idx1 % cl_size, (idx2//cl_size) - (idx1//cl_size), idx2%cl_size, None, None)
				last_idx = idx1 
				new_cl_types[last_key] = 1
				# assert idx1 % cl_size == last_key[0], "\n check correctness fail \n"
				cl_types = new_cl_types
		# 
		num_cl = 0
		for key, value in cl_types.items():
			cls = key[1] + 1
			if (key[3] != None) and (key[3] == key[1]):
				cls -= 1
			num_cl = num_cl + cls * value
		return num_cl
	# 
	def ncacheline(cl_size, offset, xyranges, glob_shape):
		''' 
			get the number of cache lines given the coordinate ranges from outer to inner axes, and the initial offset of the first data.
		'''
		num_cl = 0
		last_end = -1
		results = [num_cl, last_end]
		coordinates = [xyranges[i][0] for i in range(len(xyranges))]
		while coordinates != None:
			coordinates = update_ncl(cl_size, offset, xyranges, coordinates, glob_shape, results)
		return results[0]
	# 
	# we first get the number of points a thread need to load
	cl_size = 8 # the cache line size
	tot_n_cl = 0 # the total number of cache lines required
	op_type = op_para['op_type']
	if op_type == "conv2d":
		# get parameters in need
		n, f, y, x = blk_shape
		loop, kh, kw, stride, dilation = op_para['loop'], op_para['kh'], op_para['kw'], op_para['stride'], op_para['dilation']
		glob_shape_d = [loop[0], loop[4], 
						get_input_len_for_conv(loop[2], stride[0], dilation[0], kh), 
						get_input_len_for_conv(loop[3], stride[1], dilation[1], kw)]
		glob_shape_k = [loop[1], loop[4], kh, kw]
		offset_d = 0
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		xyranges_d = [list() for i in range(4)]
		xyranges_d[1] = [0, loop[4] - 1]
		xyranges_k = [list() for i in range(4)]
		xyranges_k[1] = [0, loop[4] - 1]
		xyranges_k[2] = [0, kh - 1]
		xyranges_k[3] = [0, kw - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]
			for bfi in range(loop[1] // f):
				xyranges_k[0] = [bfi * f, bfi * f + f - 1]
				for byi in range(loop[2] // y):
					# xyranges_d[2] = [ byi * y * stride[0], (byi * y + y - 1) * stride[0] + kh - 1 ] 
					xyranges_d[2] = [conv_inputIdex(byi * y, 0, stride[0], dilation[0]), conv_inputIdex(byi * y + y - 1, kh - 1, stride[0], dilation[0])]
					for bxi in range(loop[3] // x):
						# xyranges_d[3] = [ bxi * x * stride[1], (bxi * x + x - 1) * stride[1] + kw - 1 ]
						xyranges_d[3] = [conv_inputIdex(bxi * x, 0, stride[1], dilation[1]), conv_inputIdex(bxi * x + x - 1, kw - 1, stride[1], dilation[1])]
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
						# 
						# tttt = ncacheline(cl_size, offset_d, xyranges_d, glob_shape_d)
						# if (tttt != ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)):
						# 	print((cl_size, offset_d, xyranges_d, glob_shape_d), tttt, ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d))
						# 	assert False
						# 
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
						# 
						# tttt = ncacheline(cl_size, offset_k, xyranges_k, glob_shape_k)
						# if (tttt != ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)):
						# 	print((cl_size, offset_k, xyranges_k, glob_shape_k), tttt, ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k))
						# 	assert False
						if simplify:
							return tot_n_cl
	# 
	elif op_type == "group_conv2d":
		# get parameters in need
		n, f, y, x = blk_shape
		loop, kh, kw, stride, dilation = op_para['loop'], op_para['kh'], op_para['kw'], op_para['stride'], op_para['dilation']
		glob_shape_d = [loop[0], op_para['in_channel'], 
						get_input_len_for_conv(loop[2], stride[0], dilation[0], kh), 
						get_input_len_for_conv(loop[3], stride[1], dilation[1], kw)]
		glob_shape_k = [loop[1], loop[4], kh, kw]
		offset_d = 0
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		filter_a_group = op_para['num_filter'] // op_para['groups']
		inchnl_a_group = op_para['in_channel'] // op_para['groups']
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		xyranges_k[1] = [0, loop[4] - 1]
		xyranges_k[2] = [0, kh - 1]
		xyranges_k[3] = [0, kw - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]
			for bfi in range(loop[1] // f):
				xyranges_d[1] = [(bfi * f) // filter_a_group * inchnl_a_group, 
								(bfi * f + f - 1) // filter_a_group * inchnl_a_group + loop[4] - 1]
				xyranges_k[0] = [bfi * f, bfi * f + f - 1]
				for byi in range(loop[2] // y):
					xyranges_d[2] = [conv_inputIdex(byi * y, 0, stride[0], dilation[0]), conv_inputIdex(byi * y + y - 1, kh - 1, stride[0], dilation[0])]
					for bxi in range(loop[3] // x):
						xyranges_d[3] = [conv_inputIdex(bxi * x, 0, stride[1], dilation[1]), conv_inputIdex(bxi * x + x - 1, kw - 1, stride[1], dilation[1])]
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
						if simplify:
							return tot_n_cl
		#
	elif op_type == "depthwise_conv2d":
		# get parameters in need
		n, f, y, x = blk_shape
		loop, kh, kw, stride, dilation = op_para['loop'], op_para['kh'], op_para['kw'], op_para['stride'], op_para['dilation']
		glob_shape_d = [loop[0], op_para['in_channel'], 
						get_input_len_for_conv(loop[2], stride[0], dilation[0], kh), 
						get_input_len_for_conv(loop[3], stride[1], dilation[1], kw)]
		# we change the shape of kernel here to make the computation easier (from 4d to 3d)
		glob_shape_k = [op_para['in_channel'] * op_para['channel_multiplier'], kh, kw]
		offset_d = 0
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(3)] # we use 3d shape here
		xyranges_k[1] = [0, kh - 1]
		xyranges_k[2] = [0, kw - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]
			for bfi in range(loop[1] // f):
				xyranges_d[1] = [(bfi * f) // op_para['channel_multiplier'], 
								(bfi * f + f - 1) // op_para['channel_multiplier']]
				xyranges_k[0] = [bfi * f, bfi * f + f - 1] # we merge the 1st and 2nd axes of kernel together
				for byi in range(loop[2] // y):
					xyranges_d[2] = [conv_inputIdex(byi * y, 0, stride[0], dilation[0]), conv_inputIdex(byi * y + y - 1, kh - 1, stride[0], dilation[0])]
					for bxi in range(loop[3] // x):
						xyranges_d[3] = [conv_inputIdex(bxi * x, 0, stride[1], dilation[1]), conv_inputIdex(bxi * x + x - 1, kw - 1, stride[1], dilation[1])]
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
						if simplify:
							return tot_n_cl
		#
	elif op_type == "transpose_conv2d":
		# get parameters in need
		n, f, y, x = blk_shape
		loop, kh, kw = op_para['loop'], op_para['kh'], op_para['kw']
		glob_shape_d = [loop[0], loop[4], 
						get_input_len_for_conv(loop[2], 1, 1, kh), 
						get_input_len_for_conv(loop[3], 1, 1, kw)]
		glob_shape_k = [loop[1], loop[4], kh, kw]
		offset_d = 0
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		xyranges_d = [list() for i in range(4)]
		xyranges_d[1] = [0, loop[4] - 1]
		xyranges_k = [list() for i in range(4)]
		xyranges_k[1] = [0, loop[4] - 1]
		xyranges_k[2] = [0, kh - 1]
		xyranges_k[3] = [0, kw - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]
			for bfi in range(loop[1] // f):
				xyranges_k[0] = [bfi * f, bfi * f + f - 1]
				for byi in range(loop[2] // y):
					xyranges_d[2] = [conv_inputIdex(byi * y, 0, 1, 1), conv_inputIdex(byi * y + y - 1, kh - 1, 1, 1)]
					for bxi in range(loop[3] // x):
						xyranges_d[3] = [conv_inputIdex(bxi * x, 0, 1, 1), conv_inputIdex(bxi * x + x - 1, kw - 1, 1, 1)]
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
						if simplify:
							return tot_n_cl
		#
	elif op_type == "conv2d_capsule":
		# get parameters in need
		n, y, x, cap_i, cap_j, f = blk_shape # 6D
		loop, kh, kw, stride = op_para['loop'], op_para['kh'], op_para['kw'], op_para['stride']
		glob_shape_d = [loop[0], 
						get_input_len_for_conv(loop[1], stride, 1, kh),
						get_input_len_for_conv(loop[2], stride, 1, kw),
						loop[3], loop[8], loop[9]]
		glob_shape_k = [loop[6], loop[7], loop[8], loop[4], loop[9], loop[5]]
		offset_d = 0
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		xyranges_d = [list() for i in range(6)]
		xyranges_d[4] = [0, loop[8] - 1]
		xyranges_d[5] = [0, loop[9] - 1]
		xyranges_k = [list() for i in range(6)]
		xyranges_k[0] = [0, loop[6] - 1]
		xyranges_k[1] = [0, loop[7] - 1]
		xyranges_k[2] = [0, loop[8] - 1]
		xyranges_k[4] = [0, loop[9] - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]			
			for byi in range(loop[1] // y):
				xyranges_d[1] = [conv_inputIdex(byi * y, 0, stride, 1), conv_inputIdex(byi * y + y - 1, kh - 1, stride, 1)]
				for bxi in range(loop[2] // x):
					xyranges_d[2] = [conv_inputIdex(bxi * x, 0, stride, 1), conv_inputIdex(bxi * x + x - 1, kw - 1, stride, 1)]
					for bcapi_i in range(loop[3] // cap_i):
						xyranges_d[3] = [bcapi_i * cap_i, bcapi_i * cap_i + cap_i - 1]
						for bcapj_i in range(loop[4] // cap_j):
							xyranges_k[3] = [bcapj_i * cap_j, bcapj_i * cap_j + cap_j - 1]
							for bfi in range(loop[5] // f):
								xyranges_k[5] = [bfi * f, bfi * f + f - 1]
								tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
								tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
								if simplify:
									return tot_n_cl
		# 
	elif op_type == "conv1d":
		# get parameters in need
		n, f, x = blk_shape
		loop, kw, stride, dilation = op_para['loop'], op_para['kw'], op_para['stride'], op_para['dilation']
		glob_shape_d = [loop[0], loop[3], get_input_len_for_conv(loop[2], stride[0], dilation[0], kw)] # n, inchannel, w
		glob_shape_k = [loop[1], loop[3], kw]
		offset_d = 0 
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		xyranges_d = [list() for i in range(3)]
		xyranges_d[1] = [0, loop[3] - 1]
		xyranges_k = [list() for i in range(3)]
		xyranges_k[1] = [0, loop[3] - 1]
		xyranges_k[2] = [0, kw - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]
			for bfi in range(loop[1] // f):
				xyranges_k[0] = [bfi * f, bfi * f + f - 1]
				for bxi in range(loop[2] // x):
					xyranges_d[2] = [conv_inputIdex(bxi * x, 0, stride[0], dilation[0]), conv_inputIdex(bxi * x + x - 1, kw - 1, stride[0], dilation[0])]
					tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
					tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
					if simplify:
						return tot_n_cl
		# 
	elif op_type == "conv3d":
		# get parameters in need
		n, f, z, y, x = blk_shape
		loop, stride, dilation = op_para['loop'], op_para['stride'], op_para['dilation']
		glob_shape_d = [loop[0], loop[5], 
						get_input_len_for_conv(loop[2], stride[0], dilation[0], loop[6]), 
						get_input_len_for_conv(loop[3], stride[1], dilation[1], loop[7]), 
						get_input_len_for_conv(loop[4], stride[2], dilation[2], loop[8])]
		glob_shape_k = [loop[1], loop[5], loop[6], loop[7], loop[8]]
		offset_d = 0
		offset_k = get_product(glob_shape_d)
		# calculate the cache line number for each block
		xyranges_d = [list() for i in range(5)]
		xyranges_d[1] = [0, loop[5] - 1]
		xyranges_k = [list() for i in range(5)]
		xyranges_k[1] = [0, loop[5] - 1]
		xyranges_k[2] = [0, loop[6] - 1]
		xyranges_k[3] = [0, loop[7] - 1]
		xyranges_k[4] = [0, loop[8] - 1]
		for bni in range(loop[0] // n):
			xyranges_d[0] = [bni * n, bni * n + n - 1]
			for bfi in range(loop[1] // f):
				xyranges_k[0] = [bfi * f, bfi * f + f - 1]
				for bzi in range(loop[2] // z):
					xyranges_d[2] = [conv_inputIdex(bzi * z, 0, stride[0], dilation[0]), conv_inputIdex(bzi * z + z - 1, loop[6] - 1, stride[0], dilation[0])]
					for byi in range(loop[3] // y):
						xyranges_d[3] = [conv_inputIdex(byi * y, 0, stride[1], dilation[1]), conv_inputIdex(byi * y + y - 1, loop[7] - 1, stride[1], dilation[1])]
						for bxi in range(loop[4] // x):
							xyranges_d[4] = [conv_inputIdex(bxi * x, 0, stride[2], dilation[2]), conv_inputIdex(bxi * x + x - 1, loop[8] - 1, stride[2], dilation[2])]
							tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_d, xyranges_d, glob_shape_d)
							tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_k, xyranges_k, glob_shape_k)
							if simplify:
								return tot_n_cl
		# 		
	elif op_type == "batch_matmul":
		# get parameters in need
		n, y, x = blk_shape
		loop = op_para['loop']
		offset_A = 0
		offset_B = get_product(op_para['X_shape']) # the position of the first element in B (the second input matrix)
		glob_shape_A = op_para['X_shape']
		glob_shape_B = op_para['Y_shape']
		# calculate the cache line number for each block
		xyranges_A = [list() for i in range(len(op_para['X_shape']))]
		xyranges_A[2] = [0, loop[3] - 1]
		xyranges_B = [list() for i in range(len(op_para['Y_shape']))]
		xyranges_B[2] = [0, loop[3] - 1]
		for bni in range(loop[0] // n):
			xyranges_A[0] = [bni * n, bni * n + n - 1] if (op_para['X_shape'][0] != 1) else [0, 0]
			xyranges_B[0] = [bni * n, bni * n + n - 1] if (op_para['Y_shape'][0] != 1) else [0, 0]
			for byi in range(loop[1] // y):
				xyranges_A[1] = [byi * y, byi * y + y - 1]
				for bxi in range(loop[2] // x):
					xyranges_B[1] = [bxi * x, bxi * x + x - 1]
					tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_A, xyranges_A, glob_shape_A)
					tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_B, xyranges_B, glob_shape_B)
					if simplify:
						return tot_n_cl
		# 
	elif op_type == "conv2d_winograd":
		# get parameters in need
		n, y, x, z = blk_shape
		loop = op_para['loop']
		offset_A = 0
		offset_B = get_product(op_para['X_shape']) # the position of the first element in B (the second input matrix)
		glob_shape_A = op_para['X_shape']
		glob_shape_B = op_para['Y_shape']
		# calculate the cache line number for each block
		xyranges_A = [list() for i in range(len(op_para['X_shape']))]
		xyranges_A[3] = [0, loop[4] - 1]
		xyranges_B = [list() for i in range(len(op_para['Y_shape']))]
		xyranges_B[3] = [0, loop[4] - 1]
		for bni in range(loop[0] // n):
			xyranges_A[0] = [bni * n, bni * n + n - 1]
			xyranges_B[0] = [bni * n, bni * n + n - 1]
			for byi in range(loop[1] // y):
				xyranges_A[1] = [byi * y, byi * y + y - 1]
				xyranges_B[1] = [byi * y, byi * y + y - 1]
				for bxi in range(loop[2] // x):
					xyranges_A[2] = [bxi * x, bxi * x + x - 1]
					for bzi in range(loop[3] // z):
						xyranges_B[2] = [bzi * z, bzi * z + z -1]
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_A, xyranges_A, glob_shape_A)
						tot_n_cl = tot_n_cl + ncacheline_fast(cl_size, offset_B, xyranges_B, glob_shape_B)
						if simplify:
							return tot_n_cl
	return tot_n_cl





def cal_bk_cflct(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para):
	'''
	Calculate how many bank conflicts are there when loading data from shared memory to registers/local memory.
	According to the documentation, P100 has 32 banks, each has 4B bandwidth per clock cycle.
	In a warp, there are 32 threads, where 2 threads accessing any address within the same 32-bit word (even though the two addresses fall in the same bank) would not cause bank conflicts.
	# 
	INPUT:
		blk_shape:
			list of int. The output shape the gpu block is responsible for;
		thrd_shape:
			list of int. The thread shape in a block;
		vthrd_shape:
			list of int. The virtual thread shape in a block;
		reduc_shape:
			list of int. For 2d convolution ops, reduc_shape is the shape of reduction axes 
							each time a block needs to load.
		op_para:
			a dictionary of parameters, which are necessary for us to compute the amount of memory transactions, also INCLUDES op_type and op loop structure.
	OUTPUT:
		list of conflict-free shared_mem access for loading one data from each input. 
		The order is (#conflict-free access in data, ~ in kernel) for convolution or (~ in X, ~ in Y) for batch matmul.
	'''
	def xy2idx(coordinates, space_shape):
		''' 
			transform the coordinates to the unique 1 dimension id in the given space.
			coordinates: the coordinates in the global env, list of ints.
			space_shape: the extent of the space axes corr. to the coordinates, list of ints. THE EXTENTS are listed from outer axes to inner axes.
		'''
		idx = coordinates[0]
		for i in range(1, len(coordinates)):
			idx = idx * space_shape[i] + coordinates[i]
		return idx
	def idx2xy(space_shape, idx):
		'''
			values are listed from outer axes to inner axes. 
			transform the index of a point to the coordinates.
		'''
		coordinates = [0 for i in range(len(space_shape))]
		for i in range(len(space_shape)-1, -1, -1):
			coordinates[i] = idx % space_shape[i]
			idx = idx // space_shape[i]
		return coordinates
	# 
	def nmem_req(coordinate_list, space_shape, bkn, bkwidth = 32):
		'''
			get the cost of warps loading once, i.e., the memory request number: the number of warps + max numer of bank conflicts on one bank (in terms of a warp) when all threads in the block are loading data.
			coordinate_list:
				list of list of coordinates, each coordinate is a list of ints, from outer to inner axes, in the space; each coordinate list is the address requests in a warp.
			space_shape:
				the shape of the space where coordinates are in, extents are listed from outer to inner axes.
			bkn:
				the number of banks in the shared memory. 32 for Nvidia P100.
			bkwidth:
				the bank width, 32-bits for P100.
			RMK:
				When compute the number of bank conflict, consider all warps, but for each warp, we only consider the the first data point the threads need to load:
				Because (1) bank conflict only happens among threads in a warp;
				(2), memory request should be served as a whole, so every time only one request can be served? (NOT SURE, JUST GUESS).
		'''
		# the total number of shared memory requests
		n_req = 0
		for warp_cords in coordinate_list:
			# store the different address requested by a warp in each bank
			req_dict = dict()
			for i in range(bkn):
				req_dict[i] = 0
			# we calculate the number of bank conflicts in each warp
			checked_idx = list()
			# accessed_bk = list()
			for xy in warp_cords:
				idx = xy2idx(xy, space_shape)
				if idx not in checked_idx:
					checked_idx.append(idx)
					bkNO = idx % bkn
					req_dict[bkNO] = req_dict[bkNO] + 1
					# if bkNO in accessed_bk:
					# 	nconflict = nconflict + 1
					# else:
					# 	accessed_bk.append(bkNO)
			n_req = n_req + max(req_dict.values())
		return n_req
	# 
	# get the total number of threads in a gpu block
	warp_size = 32 # the number of threads in a warp
	bank_num = 32 # the number of banks in shared memory
	thrd_num = 1
	for i in thrd_shape:
		thrd_num = thrd_num * i
	# 
	tot_nbkcflct = None
	op_type = op_para['op_type']
	if op_type == "conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[2])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2]]
		# offset_d = 0
		# offset_k = blk_shape[0] *  reduc * ((blk_shape[2] - 1) * stride[0] + kh) * ((blk_shape[3] - 1) * stride[1] + kw)
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "group_conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		filter_a_group = op_para['num_filter'] // op_para['groups']
		inchnl_a_group = op_para['in_channel'] // op_para['groups']
		# we use the first block as the standard to compute shape_d[1]
		shape_d = [blk_shape[0], 
					 ((blk_shape[1] - 1) // filter_a_group + 1) * reduc_shape[0],
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[2])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "depthwise_conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		# we use the first block as the standard to compute shape_d[1]
		shape_d = [blk_shape[0], 
					(blk_shape[1] - 1) // op_para['channel_multiplier'] + 1, 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[0]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[1])]
		# we change the shape of shape_k to 3D from 4D to make the computation easier
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 
						thrd_cord[1] * iter_shape[1] // op_para['channel_multiplier'], 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "transpose_conv2d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], 1, 1, reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], 1, 1, reduc_shape[2])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, 1, 1),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, 1, 1)]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "conv2d_capsule":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride = op_para['stride']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0],  
					get_input_len_for_conv(blk_shape[1], stride, 1, reduc_shape[0]), 
					get_input_len_for_conv(blk_shape[2], stride, 1, reduc_shape[1]), 
					blk_shape[3], reduc_shape[2], reduc_shape[3]]
		shape_k = [reduc_shape[0], reduc_shape[1], reduc_shape[2], blk_shape[4], reduc_shape[3], blk_shape[5]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 
						conv_inputIdex(thrd_cord[1] * iter_shape[1], 0, stride, 1),
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride, 1),
						thrd_cord[3] * iter_shape[3], 0, 0]
			cord_k = [0, 0, 0, thrd_cord[4] * iter_shape[4], 0, thrd_cord[5] * iter_shape[5]]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "conv1d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]
	elif op_type == "conv3d":
		# the list of the first coordinates threads in warps want to access
		cord_list_d = list() 
		cord_list_k = list()
		tot_nbkcflct = 0
		stride, dilation = op_para['stride'], op_para['dilation']
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_d = [blk_shape[0], reduc_shape[0], 
					get_input_len_for_conv(blk_shape[2], stride[0], dilation[0], reduc_shape[1]), 
					get_input_len_for_conv(blk_shape[3], stride[1], dilation[1], reduc_shape[2]), 
					get_input_len_for_conv(blk_shape[4], stride[2], dilation[2], reduc_shape[3])]
		shape_k = [blk_shape[1], reduc_shape[0], reduc_shape[1], reduc_shape[2], reduc_shape[3]]
		# we need to get the coordinates every warp requests
		warp_cords_d = None
		warp_cords_k = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_d != None:
					cord_list_d.append(warp_cords_d)
				if warp_cords_k != None:
					cord_list_k.append(warp_cords_k)
				# new a list to store the coordinates a warp want to access
				warp_cords_d = list()
				warp_cords_k = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_d = [thrd_cord[0] * iter_shape[0], 0, 
						conv_inputIdex(thrd_cord[2] * iter_shape[2], 0, stride[0], dilation[0]),
						conv_inputIdex(thrd_cord[3] * iter_shape[3], 0, stride[1], dilation[1]),
						conv_inputIdex(thrd_cord[4] * iter_shape[4], 0, stride[2], dilation[2])]
			cord_k = [thrd_cord[1] * iter_shape[1], 0, 0, 0, 0]
			warp_cords_d.append(cord_d)
			warp_cords_k.append(cord_k)
		cord_list_d.append(warp_cords_d)
		cord_list_k.append(warp_cords_k)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_d, shape_d, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_k, shape_k, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_d, shape_d, bank_num), \
			nmem_req(cord_list_k, shape_k, bank_num)]		
	elif op_type == "batch_matmul":
		cord_list_A = list() 
		cord_list_B = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_A = [blk_shape[0] if op_para['X_shape'][0] != 1 else 1, blk_shape[1], reduc_shape[0]]
		shape_B = [blk_shape[0] if op_para['Y_shape'][0] != 1 else 1, blk_shape[2], reduc_shape[0]]
		# we need to get the coordinates every warp requests
		warp_cords_A = None
		warp_cords_B = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_A != None:
					cord_list_A.append(warp_cords_A)
				if warp_cords_B != None:
					cord_list_B.append(warp_cords_B)
				# new a list to store the coordinates a warp want to access
				warp_cords_A = list()
				warp_cords_B = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_A = [(thrd_cord[0] * iter_shape[0]) if (op_para['X_shape'] != 1) else 0, thrd_cord[1] * iter_shape[1], 0]
			cord_B = [(thrd_cord[0] * iter_shape[0]) if (op_para['Y_shape'] != 1) else 0, thrd_cord[2] * iter_shape[2], 0]
			warp_cords_A.append(cord_A)
			warp_cords_B.append(cord_B)
		cord_list_A.append(warp_cords_A)
		cord_list_B.append(warp_cords_B)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_A, shape_A, bank_num), \
			nmem_req(cord_list_B, shape_B, bank_num)]		
	elif op_type == "conv2d_winograd":
		cord_list_A = list() 
		cord_list_B = list()
		tot_nbkcflct = 0
		# get the consecutive block a thread is responsible for
		iter_shape = list()
		for i in range(len(blk_shape)):
			iter_shape.append(blk_shape[i] // (vthrd_shape[i] * thrd_shape[i]))
		# get the data and kernel shape in shared memory
		shape_A = [blk_shape[0], blk_shape[1], blk_shape[2], reduc_shape[0]]
		shape_B = [blk_shape[0], blk_shape[1], blk_shape[3], reduc_shape[0]]
		# we need to get the coordinates every warp requests
		warp_cords_A = None
		warp_cords_B = None
		for ti in range(thrd_num):
			if ti % warp_size == 0:
				# store the old warp request addresses
				if warp_cords_A != None:
					cord_list_A.append(warp_cords_A)
				if warp_cords_B != None:
					cord_list_B.append(warp_cords_B)
				# new a list to store the coordinates a warp want to access
				warp_cords_A = list()
				warp_cords_B = list()
			thrd_cord = idx2xy(thrd_shape, ti)
			cord_A = [thrd_cord[0] * iter_shape[0], thrd_cord[1] * iter_shape[1], thrd_cord[2] * iter_shape[2], 0]
			cord_B = [thrd_cord[0] * iter_shape[0], thrd_cord[1] * iter_shape[1], thrd_cord[3] * iter_shape[3], 0]
			warp_cords_A.append(cord_A)
			warp_cords_B.append(cord_B)
		cord_list_A.append(warp_cords_A)
		cord_list_B.append(warp_cords_B)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_A, shape_A, bank_num)
		# tot_nbkcflct = tot_nbkcflct + nmem_req(cord_list_B, shape_B, bank_num)
		tot_nbkcflct = [nmem_req(cord_list_A, shape_A, bank_num), \
			nmem_req(cord_list_B, shape_B, bank_num)]	
	# 
	return tot_nbkcflct




###############################################################################################################
###############################################################################################################


def dict2list(from_d, keys):
	''' 
		Transform dictionary of list to list of list. E.g., transform {'a':[1, 2], 'b':[3, 4]} to [[1, 3], [2, 4]], where keys = ['a', 'b']
		keys:
			list of strings, the data in different keys are listed in the order of keys.
		RMK:
			Every key must have a list value of the same length.
	'''
	size = None
	for key in keys:
		if size == None:
			size = len(from_d[key])
		else:
			if size != len(from_d[key]):
				assert False, "the dictionary does not have the same length on each key"
	ret = list()
	for i in range(size):
		tmp = list()
		for j in range(len(keys)):
			tmp.append(from_d[keys[j]][i])
		ret.append(tmp)
	return ret



def can_divide(shapes, space_shape):
	'''
		Check whether the given list of shapes are subblocks of the given space shape.
		INPUT:
			shapes:
				list of lists of ints. Each list of ints is a shape to be checked.
			space_shape:
				list of ints. The big space shape.
		OUTPUT:
			list of valid shapes, which are subblocks of space_shape.
	'''
	ret = list()
	for shape in shapes:
		flag = True
		for i in range(len(space_shape)):
			tot = space_shape[i]
			sub = shape[i]
			if tot % sub != 0:
				flag = False
				break
		if flag:
			ret.append(shape) 
	return ret



def update_cost_dict(cost, item, costs_dict):
	'''Add item to costs_dict according to the cost'''
	if cost not in costs_dict.keys():
		costs_dict[cost] = [item]
	else:
		costs_dict[cost].append(item)




def topk_items(costs_dict, topk, subkey = None):
	'''
		Get the top k items with the minimum costs from the costs_dict.
		INPUT:
			subkey: a function (e.g., lambda expression) for comparing items with the same cost
	'''
	ret = list()
	count = 0
	all_costs = list(costs_dict.keys())
	all_costs.sort()
	for c_i in range(len(all_costs)):
		tmp_items = costs_dict[all_costs[c_i]]
		tmp_items.sort(key = subkey)
		for s_i in range(len(tmp_items)):
			if count < topk:
				ret.append(tmp_items[s_i])
				count = count + 1
			else:
				return ret
	# if the total number of valid shapes <= topk, we still need return them
	return ret







def get_best_blk_shapes(blk_size, op_para, obj_func, topk, simplify = False):
	'''
	Get the block size in each spatial axis according to the total block size requirement to minimize the the objective function, which is usually the estimated amount of data to be loaded.
	Here we assume that the tiling on reduction axes would not make the same data to be loaded by the same block repeatedly.
	INPUT:
		blk_size:
			int. The total required block size.
		op_para:
			dictionary. The dictionary of parameters of the op which are needed for computing the objective function.
		obj_func:
			objective function which computes the cost of a block shape, and we want to minimize the objective function value.
		top_k:
			int. How many best block shapes we want to get.
		simplify:
			bool. If True, we use one block's #cache_line to approximate the total num for all blocks; else not.
	OUTPUT:
		list of [ns, fs, ys, xs]:
			the top k best block shapes with regard to the objective function.
	'''
	def get_cost_dict(shapes, op_para, obj_func, simplify):
		''' Get the cost for each shape and store them in a dict using costs as keys '''
		costs = dict()
		for shape in shapes:
			c = obj_func(shape, op_para, simplify)
			update_cost_dict(c, shape, costs)
			# if c not in costs.keys():
			# 	costs[c] = [shape]
			# else:
			# 	costs[c].append(shape)
		return costs
	# 
	def get_valid_blk_shape_group_conv2d(shapes, op_para):
		valid_shapes = list()
		group_len_fAxis = op_para['num_filter'] // op_para['groups']
		for shape in shapes:
			if shape[op_para['tile_by_group_knob_idx']] % group_len_fAxis == 0:
				valid_shapes.append(shape)
		return valid_shapes
	#  
	blk_shape_dict = get_combinations(blk_size, op_para['space_tile_knobs'])
	blk_shapes =  dict2list(blk_shape_dict, op_para['space_tile_knobs'])
	space_shape = [op_para['loop'][i] for i in op_para['space_iters']]
	blk_shapes = can_divide(blk_shapes, space_shape)
	# for group_conv2d, we need to add additional constraint
	if op_para['op_type'] == "group_conv2d":
		blk_shapes = get_valid_blk_shape_group_conv2d(blk_shapes, op_para)
	costs_dict = get_cost_dict(blk_shapes, op_para, obj_func, simplify)
	# get the topk cost shapes
	return topk_items(costs_dict, topk)
	# ret = list()
	# count = 0
	# all_costs = list(costs_dict.keys())
	# all_costs.sort()
	# for c_i in range(len(all_costs)):
	# 	tmp_shapes = costs_dict[all_costs[c_i]]
	# 	for s_i in range(len(tmp_shapes)):
	# 		if count < topk:
	# 			ret.append(tmp_shapes[s_i])
	# 			count = count + 1
	# 		else:
	# 			return ret
	# # if the total number of valid shapes <= topk, we still need return them
	# return ret



def all_factors(tot):
	'''Get all factors of tot'''
	# print("call get all factors of ", tot)
	i = 1
	factors = list()
	while i <= tot:
		if tot % i == 0:
			factors.append(i)
		i = i + 1
	return factors






def load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para, load_shape = None):
	'''	
		Get the number of data a thread needs to load from shared memory to its registers. 
		RMK:
			The value in any shapes are listed from outer to inner axes.
			We calculate for the thread with id 0 in the thread block.
			If not given, we assume every time, the reduc axes we load, i.e., load_shape = reduc_shape
		OUTPUT:
			return the list of the number of data to load for each input. (RMK!)
	'''
	def get_conv_InputIdxSet(out_idx_range, reduc_idx_range, stride, dilation):
		'''
			RMK: The index starts from 0.
			Get the indice in one dimension (y or x) in conv op that a thread needs as input.
			INPUT:
				out_idx_range:	[int, int). The start and end index (not included) of the output point we want to compute.
				reduc_idx:		[int, int). The start and end index (not included) of the reduction element we want to compute for each output point.
				stride:			int. The stride value.
				dilation:		int. The dilation value.
		'''
		# print(out_idx_range, reduc_idx_range, stride, dilation)
		indice = set()
		for i in range(out_idx_range[0], out_idx_range[1]):
			for r in range(reduc_idx_range[0], reduc_idx_range[1]):
				# print(i, r, type(i), type(r), type(stride), type(dilation))
				indice.add(conv_inputIdex(i, r, stride, dilation))
		# print(indice)
		return indice
	# 
	def area_size(xyranges):
		''' 
			get the size of the given xyranges, which is a list of ranges on each axis. each axis has a list of ranges, which may overlap each other. 
			RMK:
				We assume the ranges for an axis are listed in an increasding order.
				The range can be list of [lower bound, upper bound], Or the ranges for an axis is just a set of actual indice.
		'''
		tot_size = 1
		for ranges in xyranges: # ranges are for one axis
			if isinstance(ranges, set):
				tot_size = tot_size * len(ranges)
				continue
			# else, the ranges are a list of [lower, upper] bounds
			size = 0
			end = -1
			for i in range(len(ranges)):
				rng = ranges[i]
				if rng[0] > end:
					size = size + rng[1] - rng[0] + 1
					end = rng[1]
				else:
					if rng[1] > end:
						size = size + rng[1] - end
						end = rng[1]
			tot_size = tot_size * size
		return tot_size
	# 
	if load_shape == None:
		load_shape = reduc_shape
	ret = None
	if op_para['op_type'] == "conv2d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], stride[1], dilation[1]))
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "group_conv2d":
		stride, dilation = op_para['stride'], op_para['dilation']
		filter_a_group = op_para['num_filter'] // op_para['groups']
		inchnl_a_group = op_para['in_channel'] // op_para['groups']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[1].append([(vi * vthrd_range) // filter_a_group * reduc_shape[0], 
			# 					(vi * vthrd_range + iter_shape[axis] - 1) // filter_a_group * reduc_shape[0] + reduc_shape[0] - 1])
			for start_d1 in range((vi * vthrd_range) // filter_a_group, (vi * vthrd_range + iter_shape[axis] - 1) // filter_a_group + 1):
				xyranges_d[1].append([start_d1 * load_shape[0], start_d1 * load_shape[0] + reduc_shape[0] - 1])
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], stride[1], dilation[1]))
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "depthwise_conv2d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(3)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[1].append([(vi * vthrd_range) // op_para['channel_multiplier'], 
								(vi * vthrd_range + iter_shape[axis] - 1) // op_para['channel_multiplier']])
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[0] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[0]], stride[0], dilation[0]))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[1], dilation[1]))
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "transpose_conv2d":
		# stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(4)]
		xyranges_k = [list() for i in range(4)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # y
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, 1, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, 1, 1)])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], 1, 1))
		axis = 3 # x
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, 1, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, 1, 1)])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], 1, 1))
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "conv2d_capsule":
		stride = op_para['stride']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(6)]
		xyranges_k = [list() for i in range(6)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # y
		xyranges_d[1] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[1].append([conv_inputIdex(vi * vthrd_range, 0, stride, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[0] - 1, stride, 1)])
			xyranges_d[1].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[0]], stride, 1))
		axis = 2 # x
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride, 1), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride, 1)])			
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride, 1))
		axis = 3 # cap_i 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[3].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 4 # cap_j 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[3].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 5 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[5].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		xyranges_d[4].append([0, reduc_shape[2] - 1])
		xyranges_d[5].append([0, reduc_shape[3] - 1])
		xyranges_k[0].append([0, reduc_shape[0]-1])
		xyranges_k[1].append([0, reduc_shape[1]-1])
		xyranges_k[2].append([0, reduc_shape[2]-1])
		xyranges_k[4].append([0, reduc_shape[3]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]		
	elif op_para['op_type'] == "conv1d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(3)]
		xyranges_k = [list() for i in range(3)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # x
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])	
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "conv3d":
		stride, dilation = op_para['stride'], op_para['dilation']
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_d = [list() for i in range(5)]
		xyranges_k = [list() for i in range(5)]
		axis = 0 # n 
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_d[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 1 # f
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			xyranges_k[0].append([vi * vthrd_range, vi * vthrd_range + iter_shape[axis] - 1 ])
		axis = 2 # z
		xyranges_d[2] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[2].append([conv_inputIdex(vi * vthrd_range, 0, stride[0], dilation[0]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[1] - 1, stride[0], dilation[0])])
			xyranges_d[2].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[1]], stride[0], dilation[0]))
		axis = 3 # y
		xyranges_d[3] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[3].append([conv_inputIdex(vi * vthrd_range, 0, stride[1], dilation[1]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[2] - 1, stride[1], dilation[1])])			
			xyranges_d[3].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[2]], stride[1], dilation[1]))
		axis = 4 # x
		xyranges_d[4] = set()
		vthrd_range = iter_shape[axis] * thrd_shape[axis]
		for vi in range(vthrd_shape[axis]):
			# xyranges_d[4].append([conv_inputIdex(vi * vthrd_range, 0, stride[2], dilation[2]), 
			# 	conv_inputIdex(vi * vthrd_range + iter_shape[axis] - 1, reduc_shape[3] - 1, stride[2], dilation[2])])		
			xyranges_d[4].update(get_conv_InputIdxSet([vi * vthrd_range, vi * vthrd_range + iter_shape[axis]], \
				[0, reduc_shape[3]], stride[2], dilation[2]))	
		xyranges_d[1].append([0, reduc_shape[0] - 1])
		xyranges_k[1].append([0, reduc_shape[0]-1])
		xyranges_k[2].append([0, reduc_shape[1]-1])
		xyranges_k[3].append([0, reduc_shape[2]-1])
		xyranges_k[4].append([0, reduc_shape[3]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_d) + area_size(xyranges_k)
		ret = [area_size(xyranges_d), area_size(xyranges_k)]
	elif op_para['op_type'] == "batch_matmul":
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_A = [list() for i in range(3)]
		xyranges_B = [list() for i in range(3)]
		axis = 0 # n 
		for vi in range(vthrd_shape[axis]):
			xyranges_A[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ] if (op_para['X_shape'][0] != 1) else [0, 0])
			xyranges_B[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ] if (op_para['Y_shape'][0] != 1) else [0, 0])
		axis = 1 # y
		for vi in range(vthrd_shape[axis]):
			xyranges_A[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 2 # x
		for vi in range(vthrd_shape[axis]):
			xyranges_B[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		xyranges_A[2].append([0, reduc_shape[0]-1])
		xyranges_B[2].append([0, reduc_shape[0]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_A) + area_size(xyranges_B)
		ret = [area_size(xyranges_A), area_size(xyranges_B)]
	elif op_para['op_type'] == "conv2d_winograd":
		iter_shape = [blk_shape[i] // (thrd_shape[i] * vthrd_shape[i]) for i in range(len(blk_shape))]
		xyranges_A = [list() for i in range(4)]
		xyranges_B = [list() for i in range(4)]
		axis = 0 # n 
		for vi in range(vthrd_shape[axis]):
			xyranges_A[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
			xyranges_B[0].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 1 # y
		for vi in range(vthrd_shape[axis]):
			xyranges_A[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
			xyranges_B[1].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 2 # x
		for vi in range(vthrd_shape[axis]):
			xyranges_A[2].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		axis = 3 # z
		for vi in range(vthrd_shape[axis]):
			xyranges_B[2].append([vi * iter_shape[axis] * thrd_shape[axis], vi * iter_shape[axis] * thrd_shape[axis] + iter_shape[axis] - 1 ])
		xyranges_A[3].append([0, reduc_shape[0]-1])
		xyranges_B[3].append([0, reduc_shape[0]-1])
		# get the amount of data to load
		# ret = area_size(xyranges_A) + area_size(xyranges_B)
		ret = [area_size(xyranges_A), area_size(xyranges_B)]
	# 
	return ret









def get_vthrd_costs(blk_shape, thrd_shape, reduc_shape, op_para, obj_func, costs_dict):
	'''
	Get the (amount_of_data_to_load * #bank conflicts loading the data once) of all possible (blk_shape, thrd_shape, vthrd_shape)s given the thread shape and block shape.
	INPUT:
		blk_shape:
			list of ints. The block shape (on space axes) is listed from outer to inner axes.
		thrd_shape:
			(each thrd_shapes is a list of ints and is the thread shape in a block).
		reduc:
			list of int. For 2d convolution ops, reduc_shape is the [rc, ry, rx] length each time a block needs to load.
		op_para:
			a dictionary of parameters, which are necessary for us to compute the amount of memory transactions, also INCLUDES op_type and op loop structure.
		obj_func:
			objective function which computes the cost of a block shape, and we want to minimize the objective function value.
		# topk:
		# 	int. How many best (blk_shape, thrd_shape, vthrd_shape)s we want to get.
		costs_dict:
			dictionary. KEY: the cost   VALUE: the list of (blk_shape, thrd_shape, vthrd_shape) pairs.
	OUTPUT:
		no output, the function would update costs_dict.
		# list of virtual thread shapes, each of which is a list of ints, listed from outer to inner axes.
	'''
	# costs_dict = dict()
	def get_tot_conflictFree_accessNum(conflicts_per_load, nums_load):
		'''
			Compute the total number of conflict free access in shared_mem.
			INPUT:
				conflicts_per_load: list of ints. The number of conflict-free access for one data in each input.
				nums_load: list of ints. The number of data to load in each input.
			OUTPUT:
				The total number of conflict-free acess for all threads in one block.
			RMK: the i-th values in conflicts_per_load and nums_load are corresponding to the same input.
		'''
		ret = 0
		for i in range(len(conflicts_per_load)):
			ret = ret + (conflicts_per_load[i] * nums_load[i])
		return ret
	# 
	blk_shape_dim_num = len(blk_shape)
	if blk_shape_dim_num == 4:
		vthrd_shape = [0 for i in range(len(blk_shape))]
		# find possible virtual thread shapes for each thrd_shape
		for vtn in all_factors(blk_shape[0] // thrd_shape[0]):
			vthrd_shape[0] = vtn
			for vtf in all_factors(blk_shape[1] // thrd_shape[1]):
				vthrd_shape[1] = vtf
				for vty in all_factors(blk_shape[2] // thrd_shape[2]):
					vthrd_shape[2] = vty
					for vtx in all_factors(blk_shape[3] // thrd_shape[3]):
						vthrd_shape[3] = vtx
						# cost = obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para) * load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
						cost = get_tot_conflictFree_accessNum(obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para), 
							load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para))						
						update_cost_dict(cost, (blk_shape, thrd_shape, copy.deepcopy(vthrd_shape)), costs_dict)
	elif blk_shape_dim_num == 3:
		vthrd_shape = [0 for i in range(len(blk_shape))]
		# find possible virtual thread shapes for each thrd_shape
		for vtn in all_factors(blk_shape[0] // thrd_shape[0]):
			vthrd_shape[0] = vtn
			for vty in all_factors(blk_shape[1] // thrd_shape[1]):
				vthrd_shape[1] = vty
				for vtx in all_factors(blk_shape[2] // thrd_shape[2]):
					vthrd_shape[2] = vtx
					# cost = obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para) * load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
					cost = get_tot_conflictFree_accessNum(obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para), 
						load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para))
					update_cost_dict(cost, (blk_shape, thrd_shape, copy.deepcopy(vthrd_shape)), costs_dict)
	elif blk_shape_dim_num == 5:
		vthrd_shape = [0 for i in range(len(blk_shape))]
		# find possible virtual thread shapes for each thrd_shape
		for vtn in all_factors(blk_shape[0] // thrd_shape[0]):
			vthrd_shape[0] = vtn
			for vtf in all_factors(blk_shape[1] // thrd_shape[1]):
				vthrd_shape[1] = vtf
				for vty in all_factors(blk_shape[2] // thrd_shape[2]):
					vthrd_shape[2] = vty
					for vtx in all_factors(blk_shape[3] // thrd_shape[3]):
						vthrd_shape[3] = vtx
						for vtz in all_factors(blk_shape[4] // thrd_shape[4]):
							vthrd_shape[4] = vtz
							# cost = obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para) * load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
							cost = get_tot_conflictFree_accessNum(obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para), 
								load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para))
							update_cost_dict(cost, (blk_shape, thrd_shape, copy.deepcopy(vthrd_shape)), costs_dict)
	elif blk_shape_dim_num == 6:
		vthrd_shape = [0 for i in range(len(blk_shape))]
		# find possible virtual thread shapes for each thrd_shape
		for vtn in all_factors(blk_shape[0] // thrd_shape[0]):
			vthrd_shape[0] = vtn
			for vtf in all_factors(blk_shape[1] // thrd_shape[1]):
				vthrd_shape[1] = vtf
				for vty in all_factors(blk_shape[2] // thrd_shape[2]):
					vthrd_shape[2] = vty
					for vtx in all_factors(blk_shape[3] // thrd_shape[3]):
						vthrd_shape[3] = vtx
						for vtz in all_factors(blk_shape[4] // thrd_shape[4]):
							vthrd_shape[4] = vtz
							for vtp in all_factors(blk_shape[5] // thrd_shape[5]):
								vthrd_shape[5] = vtp
								# cost = obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para) * load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para)
								cost = get_tot_conflictFree_accessNum(obj_func(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para), 
									load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para))
								update_cost_dict(cost, (blk_shape, thrd_shape, copy.deepcopy(vthrd_shape)), costs_dict)
	# 
	# original code is as below 
	# if op_para['op_type'] == "conv2d":
	# 	vthrd_shape = [0 for i in range(len(blk_shape))]
	# 	# find possible virtual thread shapes for each thrd_shape
	# 	for vtn in all_factors(blk_shape[0] // thrd_shape[0]):
	# 		vthrd_shape[0] = vtn
	# 		for vtf in all_factors(blk_shape[1] // thrd_shape[1]):
	# 			vthrd_shape[1] = vtf
	# 			for vty in all_factors(blk_shape[2] // thrd_shape[2]):
	# 				vthrd_shape[2] = vty
	# 				for vtx in all_factors(blk_shape[3] // thrd_shape[3]):
	# 					vthrd_shape[3] = vtx
	# 					cost = obj_func(blk_shape, thrd_shape, vthrd_shape, reduc, op_para) * load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc, op_para)
	# 					update_cost_dict(cost, (blk_shape, thrd_shape, copy.deepcopy(vthrd_shape)), costs_dict)
	# elif op_para['op_type'] == "batch_matmul":
	# 	vthrd_shape = [0 for i in range(len(blk_shape))]
	# 	# find possible virtual thread shapes for each thrd_shape
	# 	for vtn in all_factors(blk_shape[0] // thrd_shape[0]):
	# 		vthrd_shape[0] = vtn
	# 		for vty in all_factors(blk_shape[1] // thrd_shape[1]):
	# 			vthrd_shape[1] = vty
	# 			for vtx in all_factors(blk_shape[2] // thrd_shape[2]):
	# 				vthrd_shape[2] = vtx
	# 				cost = obj_func(blk_shape, thrd_shape, vthrd_shape, reduc, op_para) * load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc, op_para)
	# 				update_cost_dict(cost, (blk_shape, thrd_shape, copy.deepcopy(vthrd_shape)), costs_dict)
	# 
	# return topk_items(costs_dict, topk)




def get_best_thrd_vthrd_shapes(blk_shapes, thrd_size, reduc_shape, op_para, obj_func, topk):
	'''
	Get the thread shape and virtual thread shape (in each spatial axis) given block shapes and thread size (the total number of threads) 
	to minimize the the objective function, which is usually the estimated amount of data to be loaded and bank conflicts.
	Here we assume that the tiling on reduction axes would not make the same data to be loaded by the same block repeatedly.
	INPUT:
		blk_shapes:
			list of list of ints. The block shape (on space axes) is listed from outer to inner axes.
		thrd_size:
			int. The total required thread number.
		reduc_shape:
			list of int. For 2d convolution ops, reduc_shape is the [rc, ry, rx] length each time a block needs to load.
		op_para:
			dictionary. The dictionary of parameters of the op which are needed for computing the objective function.
		obj_func:
			objective function which computes the cost of a block shape, and we want to minimize the objective function value.
		topk:
			int. How many best block shapes we want to get.
	OUTPUT:
		list of ([nb, fb, yb, xb], [nt, ft, yt, xt], [nvt, fvt, yvt, xvt]):
			the top k best (blockshape, thread shape, virtual thread shape)s with regard to the objective function.
	'''	
	thrd_shape_dict = get_combinations(thrd_size, op_para['space_tile_knobs'])
	thrd_shapes =  dict2list(thrd_shape_dict, op_para['space_tile_knobs'])
	# 
	# ret = dict()
	# thrd_shapes_list = [can_divide(thrd_shapes, blk_shapes[i]) for i in range(len(blk_shapes))]
	# ret = get_vthrd_costs(blk_shapes, thrd_shapes_list, reduc, op_para, obj_func, topk)
	costs_dict = dict()
	for blk_shape in blk_shapes:
		valid_thrd_shapes = can_divide(thrd_shapes, blk_shape)
		for thrd_shape in valid_thrd_shapes:
			get_vthrd_costs(blk_shape, thrd_shape, reduc_shape, op_para, obj_func, costs_dict)
	return topk_items(costs_dict, topk, subkey=lambda tile_shape:get_product(tile_shape[-1]))






def tile_shapes_2_knobs(tile_shapes, enu_knobs, op_para, tuned_knobs):
	'''
		Transform the tile shapes to the corresponding knob values.
		INPUT:
			tile_shapes:
				list of tuples, each tile is (block shape, thread shape, virtual shape), where each xx shape is a list of ints, listed from outer to inner axes.
			enu_knobs:
				dictionary. KEY: the knob name 	VALUE: list of all possible values of the knob which we want to enumerate. E.g., we want to enumerate the value of the knob "explicit_loop_unroll".
			op_para:
				dictionary. The dictionary of parameters of the op which are needed for computing the objective function.
			tuned_knobs:
				dictionary. KEY: knob names 	VALUE: list of values of the corresponding knob, s.t., (tuned_knobs[knob_1][i], ..., tuned_knobs[knob_n][i]) is the i-th tuned knob_combination.
		OUTPUT:
			NO OUTPUT, tuned_knobs would be updated.
		RMK: WE ASSUME tuned_knobs already have keys set.
	'''
	def idx2xy(space_shape, idx):
		'''
			values are listed from outer axes to inner axes. 
			transform the index of a point to the coordinates.
		'''
		coordinates = [0 for i in range(len(space_shape))]
		for i in range(len(space_shape)-1, -1, -1):
			coordinates[i] = idx % space_shape[i]
			idx = idx // space_shape[i]
		return coordinates
	# 
	if (len(tile_shapes) == 0):
		return
	# we first get tile settings for each axis being tiled
	shape_dims = len(tile_shapes[0][0]) # the dimension number of an xx shape
	tile_knob_values = list()
	for tile_shape in tile_shapes:
		# iter over all tile shape combinations
		blk_shape, thrd_shape, vthrd_shape = tile_shape
		tile_kvalue = list()
		for a_i in range(shape_dims):
			# get the tile setting on the axis a_i
			tmp = [-1, vthrd_shape[a_i], thrd_shape[a_i], blk_shape[a_i] // ( vthrd_shape[a_i] * thrd_shape[a_i] )]
			tile_kvalue.append(tmp)
		tile_knob_values.append(tile_kvalue)
	# 
	# we store the tile settings to tuned_knobs
	enu_num = 1 # the number of preset knob value combinations
	for v in enu_knobs.values():
		enu_num = enu_num * len(v)
	enu_knames = list(enu_knobs.keys())
	enu_k_shape = [len(enu_knobs[enu_knames[i]]) for i in range(len(enu_knames))]
	# 
	for tile_kvalue in tile_knob_values:
		# the tile settings are listed in the order: n, f, y, x (axis)
		for enu_i in range(enu_num):
			enu_xy = idx2xy(enu_k_shape, enu_i)
			for i in range(len(enu_knames)):
				kname = enu_knames[i]
				tuned_knobs[kname].append(enu_knobs[kname][enu_xy[i]])
			for i in range(len(tile_kvalue)):
				# assign value for every space tile knob
				tuned_knobs[op_para['space_tile_knobs'][i]].append(tile_kvalue[i])
	# BELOW IS ORIGINAL CODE
	# if op_para['op_type'] == "conv2d":
	# 	# knobs = ['tile_n', 'tile_f', 'tile_y', 'tile_x'] # these knobs can get values from tile_shapes
	# 	# knobs = knobs + list(enu_knobs.keys())
	# 	# for k in knobs:
	# 	# 	if k not in tuned_knobs.keys():
	# 	# 		tuned_knobs[k] = list()
	# 	for tile_kvalue in tile_knob_values:
	# 		# the tile settings are listed in the order: n, f, y, x (axis)
	# 		for enu_i in range(enu_num):
	# 			enu_xy = idx2xy(enu_k_shape, enu_i)
	# 			for i in range(len(enu_knames)):
	# 				kname = enu_knames[i]
	# 				tuned_knobs[kname].append(enu_knobs[kname][enu_xy[i]])
	# 			tuned_knobs['tile_n'].append(tile_kvalue[0])
	# 			tuned_knobs['tile_f'].append(tile_kvalue[1])
	# 			tuned_knobs['tile_y'].append(tile_kvalue[2])
	# 			tuned_knobs['tile_x'].append(tile_kvalue[3])
	# elif op_para['op_type'] == "batch_matmul":
	# 	# knobs = ['tile_n', 'tile_y', 'tile_x'] # these knobs can get values from tile_shapes
	# 	# knobs = knobs + list(enu_knobs.keys())
	# 	for tile_kvalue in tile_knob_values:
	# 		# the tile settings are listed in the order: n, f, y, x (axis)
	# 		for enu_i in range(enu_num):
	# 			enu_xy = idx2xy(enu_k_shape, enu_i)
	# 			for i in range(len(enu_knames)):
	# 				kname = enu_knames[i]
	# 				tuned_knobs[kname].append(enu_knobs[kname][enu_xy[i]])
	# 			tuned_knobs['tile_n'].append(tile_kvalue[0])
	# 			tuned_knobs['tile_y'].append(tile_kvalue[1])
	# 			tuned_knobs['tile_x'].append(tile_kvalue[2])
	# return tuned_knobs














def update_config_num_dict(config_num_dict, del_config_num, int_op_pair):
	''' Increase the original config num in the dict by del_config_num. '''
	if int_op_pair in config_num_dict.keys():
		config_num_dict[int_op_pair] = config_num_dict[int_op_pair] + del_config_num
	else:
		config_num_dict[int_op_pair] = del_config_num








def get_factors(tot, fact_dict_refer):
	'''
		Compute all factors of tot.
		fact_dict_refer: dictionary. [tot]: the factor list of tot.
	'''
	if fact_dict_refer != None and tot in fact_dict_refer.keys():
		return fact_dict_refer[tot]
	else:
		factors = get_delta_changes(tot, ['factor', ])
		factors = factors['factor']
		if fact_dict_refer != None:
			fact_dict_refer[tot] = factors
		return factors





def all_dividers(lower, upper, tot, fact_dict_refer = None):
	'''
		Get all dividers in [lower, upper], and the min divider > upper (if the max divider in [lower, upper] != upper and is farther then the one > upper) which can divide tot.
		INPUT:
			lower, upper, tot: int.
			fact_dict_refer: dictionary. [tot]: the factor list of tot.
		OUTPUT:
			dividers: list of ints.
		RMK: sometimes, upper may < lower, then we need do the reverse calculation.
	'''
	# print("all divider: ", lower, upper, tot)
	# define the sign value to deal with the case where upper < lower
	sign = 1
	if upper < lower:
		sign = -1
	# 
	# dvd_dict = get_delta_changes(tot, ['dvd', ])
	dvd_list = get_factors(tot, fact_dict_refer)
	dividers = list()
	extra = list()
	for i in dvd_list: #dvd_dict['dvd']:
		if ((sign*i) >= (sign*lower)) and ((sign*i) <= (sign*upper)):
			dividers.append(i)
		elif (sign*i) > (sign*upper):
			extra.append(i)
	# if no extra values to add, return directly
	if len(extra) == 0:
		return dividers
	#
	EPSILON = 0.000001 
	if len(dividers) == 0: # if no value is within the range
		# large1 = lower
		# if sign == 1:
		# 	large2 = min(extra)
		# else:
		# 	large2 = max(extra)
		# NO MATTER HOW FAR THE END VALUE IN extra IS, WE RETURN IT
		if sign == 1:
			return [min(extra), ]
		else:
			return [max(extra), ]
	else:
		if sign == 1:
			large1 = max(dividers)	
			large2 = min(extra)	
		else:
			large1 = min(dividers)	
			large2 = max(extra)	
	# 
	if (sign * large1) < (sign * upper):
		if (sign * large2 / upper) <= (sign * upper / large1 + EPSILON):
			dividers.append(large2)
	return dividers








def all_dividers_newRangeDef(lower, upper, tot, fact_dict_refer = None):
	'''
		Get all dividers in [lower, upper].
		RMK: 	This is an updated version of function: all_dividers.
				The difference is that we search dividers within the range following the rule that:
					if only <=1 divider is exactly in [lower, upper], 
						we would extend the range by adding a extra divider outside it in both directions;
					else we would choose to extent the range if the extra values are closer to the lower or upper bound
						than the closest dividers in [lower, upper].
				! The work of selected the closest divider to some bound is done by another function !
		INPUT:
			lower, upper, tot: int.
			fact_dict_refer: dictionary: {[tot]: the factor list of tot}.
		OUTPUT:
			dividers: list of ints.
		RMK: sometimes, upper may < lower, then we need do the reverse calculation.
	'''
	# reverse upper and lower if need
	if lower > upper:
		tmp = upper
		upper = lower
		lower = tmp
	# 
	dvd_list = get_factors(tot, fact_dict_refer)
	dividers = list()
	upper_extra = list()
	lower_extra = list()
	for i in dvd_list: 
		if (i >= lower) and (i <= upper): # within range
			dividers.append(i)
		elif i > upper:
			upper_extra.append(i)
		else:
			lower_extra.append(i)
	# 
	if len(dividers) <= 1:
		# we must extend range
		if len(upper_extra) > 0:
			dividers.append(min(upper_extra))
		if len(lower_extra) > 0:
			dividers.append(max(lower_extra))
	else:
		EPSILON = 0.000001 
		# we extend range if extra dividers are closer. (There must be some dividers within range.)
		if len(upper_extra) > 0:
			base_d = max(dividers)
			cand_d = min(upper_extra)
			if (cand_d / upper) <= (upper / base_d + EPSILON):
				dividers.append(cand_d)
		if len(lower_extra) > 0:
			base_d = min(dividers)
			cand_d = max(lower_extra)
			if (cand_d / lower) >= (lower / base_d - EPSILON):
				dividers.append(cand_d)
	return dividers





def get_closest_value(options, base_value):
	'''
		Get the value in options that is closest to the base_value in terms of relative ratio.
		INPUT:
			options: list of values.
			base_value: the value as baseline.
		OUTPUT:
			The closest value.
	'''
	EPSILON = 0.000001 
	options_s = sorted(options) # sorted from small to large
	for i in range(len(options_s)):
		if options_s[i] > base_value:
			if i > 0:
				cand_l = options_s[i-1]
				cand_r = options_s[i]
				if (cand_r / base_value) <= (base_value / cand_l + EPSILON):
					return cand_r
				else:
					return cand_l
			else:
				return options_s[i]
	# if reach here, it means all option value is <= base_value, so we return the largest one.
	return options_s[-1]










def getValidThrdNumPerBlk(thrd_sizes):
	'''
		Delete invalid thread sizes per block from given values.
		INPUT:
			thrd_sizes:		list of ints. Possible thread sizes.
		OUTPUT:
			valid thread sizes per block, satisfying the hardware constraint.
	'''
	myhardwareAnaluser = MyHardwareAnalyser()
	ret = [i for i in thrd_sizes if i < myhardwareAnaluser.max_thread_perBlk]
	return ret








def bf_avg_config_esitimator_decompose_allops_largeRange_V3(
	search_tasks, loops, 
	to_tune_idx, reuse_from_idx,
	blk_topk=1, thrd_topk=1, sample_n = -1, fact_dict_refer = None, loop_id = None):
	'''
		RMK: THIS IS FOR THE FUNC: gen_and_tune_hierarchy_nonfactor_decompose_3TuneRound_nonblk_allops_largeRange_ANSOR_V2.
			BUT WITH ORIGINAL "all_dividers()".
		Compute the average possible #config as the estimator of the the #config to measure for an op reuse pair.
		INPUT:
			search_tasks, loops: the list of the search tasks and the corresponding loop list.
			to_tune_idx, reuse_from_idx: int. The index of the task and loop to tune in the list; the index of the task and loop to reuse
			blk_topk, thrd_topk: the number of blk shapes and thrd shapes we would choose given a blk size and (blk_size, thrd_size, #load), respectively.
			fact_dict_refer: dictionary. [tot]: the factor list of tot.
			sample_n: int. The number of samples to select so that the estimator efficienct can be improved.
			loop_id: int. Which loop nest after optimization we deal with in this function. Only for multi-stage ops, like conv2d_winograd.
		OUTPUT:
			emt: int, the avg #configs to measure when to_tune_loop reuses reuse_from_loop.
		RMK:
			we only consider the features: blk size, thrd_size, #load_times in the method. (also in the hrc tuning method)
			This estimator is used for the hierarchy fine tune method.
			We assume the block size, thread num, #load_times in the reused config to be distributed uniformly.
		RMK:
			!This method computes the estimator for the fine tune method which decomposes the three features (blk size, thrd size, #load_times).
	'''
	def get_samples(seq, sample_n, weight = None):
		# print(sample_n)
		if (sample_n == -1):# or (sample_n > len(seq)):
			return seq
		else:
			return np.random.choice(seq, size=sample_n, replace=True, p = weight) # sample with replacement
			# return random.sample(seq, k=sample_n)
	# 
	# 
	EPSILON = 0.000001 
	max_unroll_options_num = len([0, 16, 64, 512, 1024])
	vectorize_options_num = len([2, 4])
	op_para = get_op_para_ansor(search_tasks[to_tune_idx], \
		loops[to_tune_idx])
	op_para_reuse = get_op_para_ansor(search_tasks[reuse_from_idx], \
		loops[reuse_from_idx])
	# need to do some preparation for conv2d_winograd
	if op_para['op_type'] == "conv2d_winograd":
		set_stage_to_tune_for_MultiStages(op_para, loop_id) # we tune stage 1 `bgemm` first, with other stages using default knobs.
		set_stage_to_tune_for_MultiStages(op_para_reuse, loop_id) # op_para_reuse also needs to be updated
	to_tune_loop = op_para['loop']
	reuse_from_loop = op_para_reuse['loop']
	# to_tune_loop = loops[to_tune_idx]
	# reuse_from_loop = loops[reuse_from_idx]
	# 
	size_t = get_product(to_tune_loop)
	size_r = get_product(reuse_from_loop)
	space_size_t = get_product([to_tune_loop[i] \
		for i in op_para['space_iters']])
	space_size_r = get_product([reuse_from_loop[i] \
		for i in op_para_reuse['space_iters']])
	inc_diff = size_t / size_r
	dt_output_space = space_size_t / space_size_r
	dt_reduc_space = inc_diff / dt_output_space
	# 
	blk_sizes_r = get_factors(space_size_r, fact_dict_refer)
	load_nums_r = get_factors(reuse_from_loop[op_para_reuse['load_step_iter']], \
		fact_dict_refer) if op_para_reuse['load_step_iter'] != None else None
	# sample the blk_size_r we would use for estimating expirical avg #config
	# we need sample with weight here, because blk_size_r itself is not uniformly distributed
	p_blk_sizes_r = [len(get_factors(blk_size_r, fact_dict_refer)) for blk_size_r in blk_sizes_r]
	sum_p = sum(p_blk_sizes_r)
	blk_sizes_r_sampled = get_samples(blk_sizes_r, sample_n, \
		[p/sum_p for p in p_blk_sizes_r])
	tot_r_possible = 0
	avg_num_config = 0
	for blk_size_r in blk_sizes_r_sampled:
		if inc_diff > 1:
			if (dt_reduc_space >= 1 - EPSILON) and (dt_reduc_space <= 1 + EPSILON):
				blk_sizes_t = all_dividers(blk_size_r / dt_reduc_space - 1, blk_size_r * inc_diff + 1, space_size_t, fact_dict_refer)
				extra_blk_sizes = all_dividers(blk_size_r / dt_reduc_space - 2, 1, space_size_t, fact_dict_refer)
				if len(extra_blk_sizes) > 0:
					blk_sizes_t.append(max(extra_blk_sizes))				
			else:
				blk_sizes_t = all_dividers(blk_size_r / dt_reduc_space - 1, blk_size_r * inc_diff + 1, space_size_t, fact_dict_refer)
		else:
			if (dt_reduc_space >= 1 - EPSILON) and (dt_reduc_space <= 1 + EPSILON):
				blk_sizes_t = all_dividers(blk_size_r * inc_diff - 1, blk_size_r / dt_reduc_space + 1, space_size_t, fact_dict_refer)
				if len(blk_sizes_t) > 0:
					extra_blk_sizes = all_dividers(max(blk_sizes_t) + 1, space_size_t, space_size_t, fact_dict_refer)
				else:
					extra_blk_sizes = all_dividers(blk_size_r / dt_reduc_space + 2, space_size_t, space_size_t, fact_dict_refer)
				if len(extra_blk_sizes) > 0:
					blk_sizes_t.append(min(extra_blk_sizes))								
			else:
				blk_sizes_t = all_dividers(blk_size_r * inc_diff - 1, blk_size_r / dt_reduc_space + 1, space_size_t, fact_dict_refer)
		# 
		# blk_sizes_t = all_dividers(blk_size_r, blk_size_r * diff, space_size_t)
		# thrd_sizes_r = get_delta_changes(blk_size_r, ['thrd_size', ])
		# thrd_sizes_r = thrd_sizes_r['thrd_size']
		thrd_sizes_r = get_factors(blk_size_r, fact_dict_refer)
		thrd_sizes_r_sampled = get_samples(thrd_sizes_r, 1)
		# 
		for thrd_size_r in thrd_sizes_r_sampled:
			load_nums_r_sampled = get_samples(load_nums_r, 1) if load_nums_r != None else [None]
			tot_r_possible = tot_r_possible + len(load_nums_r_sampled)
			# 
			for load_num_r in load_nums_r_sampled: # FOR EACH REUSED CONFIG
				# num_config = 0
				for blk_size_t in blk_sizes_t:
					dt_blk_size = blk_size_t/blk_size_r
					if inc_diff >= 1:
						thrd_sizes_t = all_dividers(thrd_size_r / (dt_output_space / dt_blk_size) - 1, \
							thrd_size_r * inc_diff / (dt_output_space / dt_blk_size) + 1, blk_size_t, fact_dict_refer)
					else:
						thrd_sizes_t = all_dividers(thrd_size_r / (dt_output_space / dt_blk_size) + 1, \
							thrd_size_r * inc_diff / (dt_output_space / dt_blk_size) - 1, blk_size_t, fact_dict_refer)
					thrd_sizes_t = getValidThrdNumPerBlk(thrd_sizes_t)
					# 
					# thrd_sizes_t = all_dividers(thrd_size_r, thrd_size_r * dt_blk_size, blk_size_t)
					# load_nums_t = all_dividers(load_num_r, \
					# 	load_num_r * dt_blk_size * Fraction(get_product(to_tune_loop[4:]), get_product(reuse_from_loop[4:])), to_tune_loop[4])
					if load_num_r != None:
						load_size = op_para['loop'][op_para['load_step_iter']]
						load_diff = dt_blk_size * get_product([to_tune_loop[i] for i in op_para['reduc_iters']])\
							/get_product([reuse_from_loop[i] for i in op_para_reuse['reduc_iters']])
						if inc_diff >= 1:
							load_nums_t = all_dividers(load_num_r * load_diff * (dt_output_space / dt_blk_size) / inc_diff - 1,\
								load_num_r * load_diff * (dt_output_space / dt_blk_size) + 1, \
								load_size, fact_dict_refer)
						else:
							load_nums_t = all_dividers(load_num_r * load_diff * (dt_output_space / dt_blk_size) / inc_diff + 1,\
								load_num_r * load_diff * (dt_output_space / dt_blk_size) - 1, \
								load_size, fact_dict_refer)
					else:
						load_nums_t = [None]
					# 
					# num_config = num_config + len(thrd_sizes_t) * len(load_nums_t)
					# we decompose thrd size and load num here, make them independent factors.
					avg_num_config = avg_num_config + blk_topk*(len(thrd_sizes_t)*thrd_topk + len(load_nums_t)-1 + \
						max_unroll_options_num-1 + vectorize_options_num * 2) # ASSUME THERE ARE ONLY 2 INPUTS
				# avg_num_config += num_config
	# avg_num_config = avg_num_config * blk_topk * thrd_topk * 2 * Fraction(1, tot_r_possible * len(load_nums_r)) # 2 for 2 possible choices of explict_unroll
	avg_num_config = avg_num_config/tot_r_possible # * len(load_nums_r)) # 2 for 2 possible choices of explict_unroll
	print("TOT BLK SIZES:", tot_r_possible)
	return float(avg_num_config)











def bf_avg_config_esitimator_CrossThrdReduc_decompose_V3(
	search_tasks, loops, 
	to_tune_idx, reuse_from_idx,
	# op_para_t, op_para_r, to_tune_loop, reuse_from_loop, 
	blk_topk, thrd_topk, sample_n = -1, fact_dict_refer = None):
	'''
		RMK: this method estimates the #config to measure for ops adopting the sketch tiled using rule of CrossThreadReduction.
		RMK: This uses the original "all_dividers" and "getValidThreadNum".
		Compute the average possible #config as the estimator of the the #config to measure for an op reuse pair.
		INPUT:
			op_para_t, op_para_r: the dictionary storing necessary parameters for op_to_tune and op_reuse_from respectively.
			to_tune_loop, reuse_from_loop: list of ints. The extents of each axis in the loop.
			blk_topk, thrd_topk: the number of blk shapes and thrd shapes we would choose given a blk size and (blk_size, thrd_size, #load), respectively.
			fact_dict_refer: dictionary. [tot]: the factor list of tot.
			sample_n: int. The number of samples to select so that the estimator efficienct can be improved.
		OUTPUT:
			emt: int, the avg #configs to measure when to_tune_loop reuses reuse_from_loop.
		RMK:
			we only consider the features: thrd_size.
			This estimator is used for the hierarchy fine tune method.
			We assume the thread num in the reused config to be distributed uniformly.
		RMK:
			!This method computes the estimator for the fine tune method which decomposes the three features (blk size, thrd size, #load).
	'''
	def get_samples(seq, sample_n):
		# print(sample_n)
		if (sample_n == -1):# or (sample_n > len(seq)):
			return seq
		else:
			return np.random.choice(seq, size=sample_n, replace=True) # sample with replacement
			# return random.sample(seq, k=sample_n)
	# 
	op_para_t = get_op_para_ansor(search_tasks[to_tune_idx], \
		loops[to_tune_idx])
	op_para_r = get_op_para_ansor(search_tasks[reuse_from_idx], \
		loops[reuse_from_idx])
	# need to do some preparation for conv2d_winograd
	if op_para_t['op_type'] == "conv2d_winograd":
		set_stage_to_tune_for_MultiStages(op_para_t, loop_id) # we tune stage 1 `bgemm` first, with other stages using default knobs.
		set_stage_to_tune_for_MultiStages(op_para_r, loop_id) # op_para_reuse also needs to be updated
	to_tune_loop = op_para_t['loop']
	reuse_from_loop = op_para_r['loop']	
	# to_tune_loop = loops[to_tune_idx]
	# reuse_from_loop = loops[reuse_from_idx]
	# 
	size_t = get_product(to_tune_loop)
	size_r = get_product(reuse_from_loop)
	space_size_t = get_product([to_tune_loop[i] for i in op_para_t['space_iters']])
	space_size_r = get_product([reuse_from_loop[i] for i in op_para_r['space_iters']])
	reduc_size_t = get_product([to_tune_loop[i] for i in op_para_t['reduc_iters']])
	reduc_size_r = get_product([reuse_from_loop[i] for i in op_para_r['reduc_iters']])
	diff = size_t / size_r
	diff_space = space_size_t / space_size_r
	diff_reduc = reduc_size_t / reduc_size_r
	# 
	avg_num_config = 0
	# in our experiments, the reduc_size_r of the conv2d in ResNeSt is larger than 32, so previous cost is correct.
	# here 32 is to deal with op tuned by Ansor. For op tuned by other methods, this should be different. 
	# TODO: this part needs check [this part in Ansor seems to be very complex]
	thrd_sizes_r = get_factors(max(reduc_size_r, 32), fact_dict_refer) 
	if reduc_size_r == 1:
		thrd_sizes_r = [32] # this is computed according to Ansor's rule.
	# 
	thrd_sizes_r_sampled = get_samples(thrd_sizes_r, sample_n)
	for thrd_size_r in thrd_sizes_r_sampled:
		if size_t > size_r:
			thrd_sizes_t = all_dividers(thrd_size_r / diff_space - 1, thrd_size_r * diff_reduc + 1, reduc_size_t, fact_dict_refer)
		else:
			thrd_sizes_t = all_dividers(thrd_size_r / diff_space + 1, thrd_size_r * diff_reduc - 1, reduc_size_t, fact_dict_refer)
		thrd_sizes_t = getValidThrdNumPerBlk(thrd_sizes_t)
		avg_num_config = avg_num_config + len(thrd_sizes_t)
	avg_num_config = avg_num_config * Fraction(1, len(thrd_sizes_r_sampled))
	print("TOT BLK SIZES:", len(thrd_sizes_r_sampled))
	return float(avg_num_config)






def bf_avg_config_esitimator_CrossThrdReduc_frobenius_decompose_V3(
	search_tasks, loops, 
	to_tune_idx, reuse_from_idx,
	# op_para_t, op_para_r, to_tune_loop, reuse_from_loop, 
	blk_topk, thrd_topk, sample_n = -1, fact_dict_refer = None):
	'''
		RMK: this method can also compute the cost for frobenius norm.
		RMK: this method estimates the #config to measure for ops adopting the sketch tiled using rule of CrossThreadReduction.
		RMK: This uses the original "all_dividers" and "getValidThreadNum".
		Compute the average possible #config as the estimator of the the #config to measure for an op reuse pair.
		INPUT:
			op_para_t, op_para_r: the dictionary storing necessary parameters for op_to_tune and op_reuse_from respectively.
			to_tune_loop, reuse_from_loop: list of ints. The extents of each axis in the loop.
			blk_topk, thrd_topk: the number of blk shapes and thrd shapes we would choose given a blk size and (blk_size, thrd_size, #load), respectively.
			fact_dict_refer: dictionary. [tot]: the factor list of tot.
			sample_n: int. The number of samples to select so that the estimator efficienct can be improved.
		OUTPUT:
			emt: int, the avg #configs to measure when to_tune_loop reuses reuse_from_loop.
		RMK:
			we only consider the features: thrd_size.
			This estimator is used for the hierarchy fine tune method.
			We assume the thread num in the reused config to be distributed uniformly.
		RMK:
			!This method computes the estimator for the fine tune method which decomposes the three features (blk size, thrd size, #load).
	'''
	def get_samples(seq, sample_n):
		# print(sample_n)
		if (sample_n == -1):# or (sample_n > len(seq)):
			return seq
		else:
			return np.random.choice(seq, size=sample_n, replace=True) # sample with replacement
			# return random.sample(seq, k=sample_n)
	# 
	op_para_t = get_op_para_ansor(search_tasks[to_tune_idx], \
		loops[to_tune_idx])
	op_para_r = get_op_para_ansor(search_tasks[reuse_from_idx], \
		loops[reuse_from_idx])
	to_tune_loop = op_para_t['loop']
	reuse_from_loop = op_para_r['loop']	
	# 
	size_t = get_product(to_tune_loop)
	size_r = get_product(reuse_from_loop)
	space_size_t = get_product([to_tune_loop[i] for i in op_para_t['space_iters']])
	space_size_r = get_product([reuse_from_loop[i] for i in op_para_r['space_iters']])
	reduc_size_t = get_product([to_tune_loop[i] for i in op_para_t['reduc_iters']])
	reduc_size_r = get_product([reuse_from_loop[i] for i in op_para_r['reduc_iters']])
	diff = size_t / size_r
	diff_space = space_size_t / space_size_r
	diff_reduc = reduc_size_t / reduc_size_r
	# 
	avg_num_config = 0
	# in our experiments, the reduc_size_r of the conv2d in ResNeSt is larger than 32, so previous cost is correct.
	# here 32 is to deal with op tuned by Ansor. For op tuned by other methods, this should be different. 
	# !!!TODO: this part needs more details about Ansor. here we only deal with the cases in the experiments
	# thrd_sizes_r = get_factors(max(reduc_size_r, 32), fact_dict_refer) 
	thrd_sizes_r = get_factors(32 if reduc_size_r <= 32 else 64, fact_dict_refer) 
	if reduc_size_r == 1:
		thrd_sizes_r = [32] # this is computed according to Ansor's rule.
	# 
	thrd_sizes_r_sampled = get_samples(thrd_sizes_r, sample_n)
	thrd_sizes = all_dividers(0, reduc_size_t+1, reduc_size_t)
	sign_tmp = 1 if diff > 1 else -1
	for thrd_size_r in thrd_sizes_r_sampled:
		# we only selected the thrd_sizes such that dt(blk_num*thrd_size) is within dt(whole_space_size)
		thrd_sizes_t = [thrd_size for thrd_size in thrd_sizes \
			if ((math.ceil(space_size_t / thrd_size) * thrd_size * sign_tmp \
				<= math.ceil(space_size_r / thrd_size_r) * thrd_size_r * diff * sign_tmp) and \
				(math.ceil(space_size_t / thrd_size) * thrd_size * sign_tmp\
				>= math.ceil(space_size_r / thrd_size_r) * thrd_size_r * sign_tmp))]
		thrd_sizes_t = getValidThrdNumPerBlk(thrd_sizes_t)
		avg_num_config = avg_num_config + len(thrd_sizes_t)
	avg_num_config = avg_num_config * Fraction(1, len(thrd_sizes_r_sampled))
	print("TOT BLK SIZES:", len(thrd_sizes_r_sampled))
	return float(avg_num_config)



def bf_avg_config_esitimator_2LevelTile_decompose_V3(
	search_tasks, loops, 
	to_tune_idx, reuse_from_idx,
	# op_para_t, op_para_r, to_tune_loop, reuse_from_loop, 
	blk_topk, thrd_topk, sample_n = -1, fact_dict_refer = None, loop_id = None):
	'''
		RMK: this method estimates the #config to measure for ops adopting the sketch tiled using the naive 2 level space axes tiling.
		RMK: This uses the original "all_dividers" and "getValidThreadNum".
		Compute the average possible #config as the estimator of the the #config to measure for an op reuse pair.
		INPUT:
			op_para_t, op_para_r: the dictionary storing necessary parameters for op_to_tune and op_reuse_from respectively.
			to_tune_loop, reuse_from_loop: list of ints. The extents of each axis in the loop.
			blk_topk, thrd_topk: the number of blk shapes and thrd shapes we would choose given a blk size and (blk_size, thrd_size, #load), respectively.
			fact_dict_refer: dictionary. [tot]: the factor list of tot.
			sample_n: int. The number of samples to select so that the estimator efficienct can be improved.
			loop_id: int. The id of the loop nest after optimization we compute the cost estimation in this function.
		OUTPUT:
			emt: int, the avg #configs to measure when to_tune_loop reuses reuse_from_loop.
		RMK:
			we only consider the features: thrd_size.
			This estimator is used for the hierarchy fine tune method.
			We assume the thread num in the reused config to be distributed uniformly.
		RMK:
			!This method computes the estimator for the fine tune method which decomposes the three features (blk size, thrd size, #load).
	'''
	def get_samples(seq, sample_n):
		# print(sample_n)
		if (sample_n == -1):# or (sample_n > len(seq)):
			return seq
		else:
			return np.random.choice(seq, size=sample_n, replace=True) # sample with replacement
			# return random.sample(seq, k=sample_n)
	# 
	op_para_t = get_op_para_ansor(search_tasks[to_tune_idx], \
		loops[to_tune_idx])
	op_para_r = get_op_para_ansor(search_tasks[reuse_from_idx], \
		loops[reuse_from_idx])
	# need to do some preparation for conv2d_winograd
	if op_para_t['op_type'] == "conv2d_winograd":
		set_stage_to_tune_for_MultiStages(op_para_t, loop_id) # we tune stage 1 `bgemm` first, with other stages using default knobs.
		set_stage_to_tune_for_MultiStages(op_para_r, loop_id) # op_para_reuse also needs to be updated
	to_tune_loop = op_para_t['loop']
	reuse_from_loop = op_para_r['loop']	
	# to_tune_loop = loops[to_tune_idx]
	# reuse_from_loop = loops[reuse_from_idx]
	# 
	size_t = get_product(to_tune_loop)
	size_r = get_product(reuse_from_loop)
	space_size_t = get_product([to_tune_loop[i] for i in op_para_t['space_iters']])
	space_size_r = get_product([reuse_from_loop[i] for i in op_para_r['space_iters']])
	reduc_size_t = get_product([to_tune_loop[i] for i in op_para_t['reduc_iters']])
	reduc_size_r = get_product([reuse_from_loop[i] for i in op_para_r['reduc_iters']])
	diff = size_t / size_r
	diff_space = space_size_t / space_size_r
	diff_reduc = reduc_size_t / reduc_size_r
	# 
	avg_num_config = 0
	# here we assume space_size_r > 32 (otherwise, the thrd_size_r can only be space_size_r), which is the case for winograd conv2ds in our experiments
	thrd_sizes_r = get_factors(space_size_r, fact_dict_refer)
	# TODO: this part needs recheck
	if space_size_r <= 32:
		# in fact, in this case, the reused op would not be 2-level tiled.
		thrd_sizes_r = [space_size_r]
	thrd_sizes_r_sampled = get_samples(thrd_sizes_r, sample_n)
	for thrd_size_r in thrd_sizes_r_sampled:
		thrd_sizes_t = all_dividers(thrd_size_r / diff_reduc - 1, thrd_size_r * diff + 1, space_size_t)
		if reduc_size_t == reduc_size_r:
			if diff > 1:
				extra_thrd_size = all_dividers(thrd_size_r - 1, 1, space_size_t)
				if len(extra_thrd_size) > 0:
					thrd_sizes_t.append(max(extra_thrd_size))
			else:
				extra_thrd_size = all_dividers(thrd_size_r + 1, space_size_t, space_size_t)
				if len(extra_thrd_size) > 0:
					thrd_sizes_t.append(min(extra_thrd_size))
		thrd_sizes_t = getValidThrdNumPerBlk(thrd_sizes_t)
		avg_num_config = avg_num_config + len(thrd_sizes_t)
	avg_num_config = avg_num_config * Fraction(1, len(thrd_sizes_r_sampled))
	print("TOT BLK SIZES:", len(thrd_sizes_r_sampled))
	return float(avg_num_config)








def bf_avg_config_esitimator_MultiStages_decompose_V3(
	search_tasks, loops, 
	to_tune_idx, reuse_from_idx,
	blk_topk, thrd_topk, sample_n = -1, fact_dict_refer = None):
	''' 
		This function computes the estimated cost for search tasks with more than one loop nest after optimization!
		It will call other concrete cost estimation function inside.
	''' 
	op_para_t = get_op_para_ansor(search_tasks[to_tune_idx], \
		loops[to_tune_idx])
	cost = 0
	if op_para_t['op_type'] == "conv2d_winograd":
		stages = ["ConstTensor", "MultiLevelTile", "ConstTensor", "2LevelTiling"]
		for i in range(len(stages)):
			stage = stages[i]
			if stage in ["2LevelTiling", "ConstTensor"]:
				sub_cost = bf_avg_config_esitimator_2LevelTile_decompose_V3(
						search_tasks, loops, 
						to_tune_idx, reuse_from_idx,
						# op_para_t, op_para_r, to_tune_loop, reuse_from_loop, 
						blk_topk=1, thrd_topk=1, sample_n = sample_n, fact_dict_refer = fact_dict_refer, loop_id = i)
				cost = cost + sub_cost
			elif stage == "MultiLevelTile":
				sub_cost = bf_avg_config_esitimator_decompose_allops_largeRange_V3(
						search_tasks, loops, 
						to_tune_idx, reuse_from_idx,
						blk_topk=1, thrd_topk=1, sample_n = sample_n, fact_dict_refer = fact_dict_refer, loop_id = i)
				cost = cost + sub_cost
			else:
				assert False, "Do not support other stages cost estimation function."
	else:
		assert False, "We currently only support multi-stage cost estimation for conv2d_winograd search_tasks!"
	return cost








def update_reuse_result_dict(key, value, best_reuse_results):
	'''
		key: the op reuse pair.
		value: the measure pair.
		If the cost in value is smaller than the result in the best_reuse_results, we do the update.
		RMK: can also deal with the format for Ansor, where only stores cost as the measure_result.
	'''
	if key in best_reuse_results.keys():
		if type(value[1]) == float:
			value = (value[0], value[1] / 1e9)
			if best_reuse_results[key][1] < value[1]: # stores flops
				best_reuse_results[key] = value
		else:
			if best_reuse_results[key][1].costs[0] > value[1].costs[0]:
				best_reuse_results[key] = value
	else:
		if type(value[1]) == float:
			value = (value[0], value[1] / 1e9)
		best_reuse_results[key] = value












# ############################################################################################
# ############################################################################################
def get_optimized_flops(best_measure_pair, si_prefix = "G"):
	'''
	Get the running flops for the tuned task based on the best measure pair.
	INPUT:
		best_measure_pair:
			(MeasureInput, MeasureResult). The same as in autoTVM.
	OUTPUT:
		best_flops.
			float. unit: GFLOPS  
	'''
	inp, res = best_measure_pair
	flops = inp.task.flop / np.mean(res.costs)
	return autotvm.util.format_si_prefix(flops, si_prefix)


def get_optimized_flops_Ansor(measure_pair, si_prefix = "G"):
	'''
	Get the running flops for the tuned task based on the measure pair in Ansor.
	INPUT:
		measure_pair:
			(MeasureInput, MeasureResult). The same as in Ansor.
	OUTPUT:
		best_flops.
			float. unit: GFLOPS  
	'''
	inp, res = measure_pair
	flops = inp.task.compute_dag.flop_ct / (res.costs[0].value) 
	return autotvm.utils.format_si_prefix(flops, si_prefix)




def get_optimized_flops(best_measure_pair, task = None, si_prefix = "G"):
	'''
	Get the running flops for the tuned task based on the best measure pair.
	INPUT:
		best_measure_pair:
			(MeasureInput, MeasureResult). The same as in autoTVM.
		task:
			the corresponding task. In case that flops is not stored in measure pair.
	OUTPUT:
		best_flops.
			float. unit: GFLOPS  
	'''
	inp, res = best_measure_pair
	if task != None:
		flops = task.flop / np.mean(res.costs)
	else:
		flops = inp.task.flop / np.mean(res.costs)
	return autotvm.util.format_si_prefix(flops, si_prefix)






def load_best_configs(tasks, best_measure_pairs, file_name):
	'''
	Load best measure pairs for tasks from records stored in files during previous tuning process, 
	and store them in best_measure_pairs.
	INPUT:
		tasks:
			list of Tasks. The tasks we want to load the best best measure pairs for.
		best_measure_pairs:
			dictionary. key: Task    value: the best measure pair
	RMK:
		This function would change the value of best_measure_pair.
	'''
	records = autotvm.record.load_from_file(file_name)
	for inp, res in records:
		for task in tasks:
			if inp.task.workload == task.workload:
				best_measure_pairs[task] = (inp, res)



def load_configs_from_string(best_measure_pairs, line):
	'''
	Store the measure pair from line in best_measure_pairs.
	INPUT:
		best_measure_pairs:
			dictionary. key: Task    value: the best measure pair
		line:
			string. the line containing (MeasureInput, MeasureResult)
	OUTPUT:
		return 
			True if successfully find the corresponding task in best_measure_pairs and store results;
			False otherwise.
	RMK:
		This function would change the value of best_measure_pairs.
		This function requires the task already exists in best_measure_pairs.keys().
	'''
	ret = autotvm.record.decode(line)
	if ret is None:
		return False
	else:
		inp, res = ret
		tasks = best_measure_pairs.keys()
		for task in tasks:
			if inp.task.workload == task.workload:
				best_measure_pairs[task] = ret
				return True
		return False






def get_network(name, batch_size):
	"""Get the symbol definition and random weight of a network"""
	input_shape = (batch_size, 3, 224, 224)
	output_shape = (batch_size, 1000)
	dtype = "float32"
	if "resnet" in name:
		n_layer = int(name.split("-")[1])
		mod, params = relay.testing.resnet.get_workload(
			num_layers=n_layer, batch_size=batch_size, dtype=dtype
		)
	elif "vgg" in name:
		n_layer = int(name.split("-")[1])
		mod, params = relay.testing.vgg.get_workload(
			num_layers=n_layer, batch_size=batch_size, dtype=dtype
	    )
	elif name == "mobilenet":
		mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
	elif name == "squeezenet_v1.1":
		mod, params = relay.testing.squeezenet.get_workload(
			batch_size=batch_size, version="1.1", dtype=dtype
	    )
	elif name == "inception_v3":
		input_shape = (1, 3, 299, 299)
		mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
	elif name == "mxnet":
		# an example for mxnet model
		from mxnet.gluon.model_zoo.vision import get_model
	# 
		block = get_model("resnet18_v1", pretrained=True)
		mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
		net = mod["main"]
		net = relay.Function(
			net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
		)
		mod = tvm.IRModule.from_expr(net)
	else:
		raise ValueError("Unsupported network: " + name)
	# 
	return mod, params, input_shape, output_shape



def get_tasks(nets, batch_sizes, task_names=['conv2d_nchw.cuda', 'depthwise_conv2d_nchw.cuda']):
	'''
	Extract tasks of specific type from given nets.
	INPUT:
		nets:
			list of strings. list of the net names
		batch_sizes:
			list of ints. list of batch sizes.
		task_names:
			list of strings. list of the task op name.
	OUTPUT:
		tot_tasks:
			list of Tasks.
	'''
	print("Extract tasks...")
	tot_tasks = list()
	tot_workloads = list()
	# 
	# ops = list()
	# for ttype in task_types:
	# 	ops.append(relay.op.get(ttype))
	# ops = tuple(ops)
	target = tvm.target.cuda()
	# 
	for batch_size in batch_sizes:
		for network in nets:
			# extract workloads from relay program
			mod, params, input_shape, out_shape = get_network(network, batch_size)
			tasks = autotvm.task.extract_from_program(
				mod["main"], target=target, params=params
			)
			for task in tasks:
				if task.name in task_names:
					if task.workload not in tot_workloads:
						# remove redundant tasks
						tot_workloads.append(task.workload)
						tot_tasks.append(task)
	return tot_tasks














def get_tasks_from_3rdparty_dnn_lib(dnn_name, third_party_name, task_names=[], batch_size = 1, input_shape = [3, 224, 224], **kwargs):
	'''
		Extract tasks for DNNs from 3rd party library, e.g., MXnet.
		INPUT:
			dnn_name: 		string. Each string is the name of a DNN.
			third_party_name:	string. The name of the 3rd party library to get networks from.
			task_names:		list of strings. Names of the ops that we want to tune later.
		OUTPUT:
			tot_tasks:
				list of Tasks.
	'''
	mod, params = None, None
	print(dnn_name, third_party_name)
	if third_party_name == "mxnet":
		from gluoncv.model_zoo import get_model
		# 
		block = get_model(dnn_name, pretrained=True, **kwargs)
		shape_dict = {"data": tuple([batch_size] + input_shape)}
		mod, params = relay.frontend.from_mxnet(block, shape_dict)
	# 
	print("Extract tasks...")
	tot_tasks = list()
	tot_workloads = list()
	# 
	target = tvm.target.cuda()
	tasks = autotvm.task.extract_from_program(
		mod["main"], target=target, params=params
	)
	extracted_task_names = list()
	for task in tasks:
		if task.name not in extracted_task_names:
			extracted_task_names.append(task.name)
	print(extracted_task_names)
	if len(task_names) == 0:
		return tasks
	else:
		for task in tasks:
			if task.name in task_names:
				if task.workload not in tot_workloads:
					# remove redundant tasks
					tot_workloads.append(task.workload)
					tot_tasks.append(task)
		return tot_tasks








def get_network_from_3rdparty_dnn_lib(dnn_name, third_party_name, task_names=[], batch_size = 1, input_shape = [3, 224, 224], **kwargs):
	'''
		Extract tasks for DNNs from 3rd party library, e.g., MXnet.
		INPUT:
			dnn_name: 		string. Each string is the name of a DNN.
			third_party_name:	string. The name of the 3rd party library to get networks from.
			task_names:		list of strings. Names of the ops that we want to tune later.
		OUTPUT:
			tot_tasks:
				list of Tasks.
	'''
	mod, params = None, None
	print(dnn_name, third_party_name)
	if third_party_name == "mxnet":
		from gluoncv.model_zoo import get_model
		# 
		block = get_model(dnn_name, pretrained=True, **kwargs)
		shape_dict = {"data": tuple([batch_size] + input_shape)}
		mod, params = relay.frontend.from_mxnet(block, shape_dict)
		return mod, params





@auto_scheduler.register_workload
def conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
	data = te.placeholder(data_shape)#, name="data")
	kernel = te.placeholder(kernel_shape)#, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	if (conv_type == 'conv2d_nchw.cuda'):
		conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	elif (conv_type == 'depthwise_conv2d_nchw.cuda'):
		conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]




@auto_scheduler.register_workload
# (Input, Filter, stride, padding, dilation, groups, out_dtype=None):
def group_conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, groups, out_dtype, conv_type):
	data = te.placeholder(data_shape)#, name="data")
	kernel = te.placeholder(kernel_shape)#, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	if (conv_type == 'group_conv2d_nchw.cuda'):
		conv = topi.nn.group_conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, groups = groups, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]




# depthwise_conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
@auto_scheduler.register_workload
def depthwise_conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
	data = te.placeholder(data_shape)#, name="data")
	kernel = te.placeholder(kernel_shape)#, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	if (conv_type == 'depthwise_conv2d_nchw.cuda'):
		conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]



# conv2d_transpose_nchw(Input, Filter, strides, padding, out_dtype, output_padding):
@auto_scheduler.register_workload
def conv2d_transpose_nchw(data_shape, kernel_shape, stride, padding, out_dtype, output_padding, conv_type):
	data = te.placeholder(data_shape)#, name="data")
	kernel = te.placeholder(kernel_shape)#, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	if (conv_type == 'conv2d_transpose_nchw.cuda'):
		conv = topi.nn.conv2d_transpose_nchw(data, kernel, stride, padding, out_dtype=out_dtype, output_padding=output_padding)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]






@auto_scheduler.register_workload
def conv1d_ncw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
	data = te.placeholder(data_shape)#, name="data")
	kernel = te.placeholder(kernel_shape)#, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	conv = None
	if (conv_type == 'conv1d_ncw.cuda'):
		conv = topi.nn.conv1d_ncw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]











@auto_scheduler.register_workload
def conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4):
    inputs = te.placeholder((N, H, W, capsule_size, capsule_size, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, capsule_size, capsule_size, CI, CO), name='weight')
    batch_size, in_h, in_w, _, _, in_channel = inputs.shape
    k_h, k_w, _, _, _, out_channel = weight.shape
    # 
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1
    # 
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    cap_k = te.reduce_axis((0, capsule_size), name='cap_k')
    rc = te.reduce_axis((0, in_channel), name="rc")
    # 
    padded = topi.nn.pad(inputs, [0, padding, padding, 0, 0, 0])
    output = te.compute(
        (batch_size, out_h, out_w, capsule_size, capsule_size, out_channel),
        lambda n, h, w, cap_i, cap_j, co: te.sum(
            (padded[n, h * stride + rh, w * stride + rw, cap_i, cap_k, rc]
             * weight[rh, rw, cap_k, cap_j, rc, co]), axis=[rh, rw, cap_k, rc]
        ),
        name='conv2d_capsule_nhwijc'
    )
    return [inputs, weight, output]






@auto_scheduler.register_workload
def conv3d_ncdhw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
	data = te.placeholder(data_shape)#, name="data")
	kernel = te.placeholder(kernel_shape)#, name="kernel")
	# bias = te.placeholder((1, CO, 1, 1), name="bias")
	if (conv_type == 'conv3d_ncdhw.cuda'):
		conv = topi.nn.conv3d_ncdhw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# elif (conv_type == 'depthwise_conv2d_nchw.cuda'):
	# 	conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=dilation_val, out_dtype=out_dtype)
	# out = topi.nn.relu(conv + bias)
	return [data, kernel, conv]






@auto_scheduler.register_workload
def batch_matmul(X_shape, Y_shape, oshape = None, op_name = None):
	X = te.placeholder(X_shape)#, name="X")
	Y = te.placeholder(Y_shape)#, name="Y")
	out = topi.nn.batch_matmul(X, Y, oshape=None, auto_scheduler_rewritten_layout="")
	return [X, Y, out]






@auto_scheduler.register_workload
def frobenius_norm(N, H, W):
    inputs = te.placeholder((N, H, W), name='inputs')
    batch_size, in_h, in_w = inputs.shape
    # 
    rh = te.reduce_axis((0, in_h), name="rh")
    rw = te.reduce_axis((0, in_w), name="rw")
    # 
    sqr_sum = te.compute(
        (batch_size, ),
        lambda n: te.sum(
            (inputs[n, rh, rw] * inputs[n, rh, rw]), axis=[rh, rw]
        ),
        name='frobenius_norm'
    )
    output = topi.math.sqrt(sqr_sum)
    return [inputs, output]





# we would not test this op (subgraph)
@auto_scheduler.register_workload
def TBS(X_shape, Y_shape, oshape = None, op_name = None):
	X = te.placeholder(X_shape, name="X")
	Y = te.placeholder(Y_shape, name="Y")
	out = topi.nn.batch_matmul(X, Y, oshape=None, auto_scheduler_rewritten_layout="")
	out = topi.nn.softmax(out, axis=-1)
	return [X, Y, out]








def get_loops(tasks, func):
	'''
	Get the loop extents of the given tasks using Ansor.
	INPUT:
		tasks:
			list of Tasks. 
	OUTPUT:
		loops:
			list of loops. each loop is a list of its iterator extents.
		search_tasks:
			list of SearchTasks in Ansor.
	'''
	count = 1
	target = tvm.target.Target("cuda")
	loops = list()
	search_tasks = list()
	for i in range(len(tasks)):
		task = tasks[i]
		# 
		paras = list()
		for arg_i in range(len(task.args)):
			arg = task.args[arg_i]
			if (type(arg) == tuple) and (arg[0] == 'TENSOR'):
				paras.append(arg[1])
			else:
				paras.append(arg)
		paras.append(task.name)
		# 
		search_task = auto_scheduler.SearchTask(
    					func=func, args=tuple(paras), target=target)
		search_tasks.append(search_task)
		# 
		# ansor_task = auto_scheduler.create_task(func, tuple(paras), target)
		nodes = func(*paras)
		# Inspect the computational graph
		dag = auto_scheduler.ComputeDAG(nodes)
		init_state = dag.get_init_state()
		# 
		print('*'*50)
		print('TASK NO: ', count)
		print(task)
		count = count + 1
		print('*'*50)
		print(dag)
		print(init_state)
		print(search_task.workload_key)
		# 
		loop = list()
		iters = init_state[nodes[-1]].iters
		for iter_i in iters:
			loop.append(int(iter_i.range.extent))
		loops.append(loop)
	return loops, search_tasks






def get_loops_ansor(tasks):
	'''
	Get the loop extents of the given tasks using Ansor.
	INPUT:
		tasks:
			list of SearchTasks in Ansor. 
	OUTPUT:
		loops:
			list of loops. each loop is a list of its iterator extents.
	'''
	def find_target_stage_id(tstage, stages):
		''' find the index of the corresponding stage with name `tstage` in a list of stages `stages`. '''
		target_stage_id = None
		for stage_id in range(len(stages)):
			if tstage in stages[stage_id].__str__():
				target_stage_id = stage_id
				break
		assert target_stage_id!=None, "error when extracting loops for winograd conv2d!"
		return target_stage_id
	# 
	count = 1
	loops = list()
	search_tasks = list()
	for i in range(len(tasks)):
		task = tasks[i]
		init_state = task.compute_dag.get_init_state()
		# 
		print('*'*50)
		print('TASK NO: ', count)
		count = count + 1
		print('*'*50)
		print(task.compute_dag)
		print(init_state)
		# 
		# deal with winograd
		if "conv2d_winograd" in init_state.stages[-1].__str__():
			loop = list()
			# there 4 loops in winograd conv2d
			target_stages = ["data_pack", "bgemm", "inverse", "conv2d_winograd"]
			for tstage in target_stages:
				tstage_id = find_target_stage_id(tstage, init_state.stages)
				iters = init_state.stages[tstage_id].iters
				loop.append([int(iter_i.range.extent) for iter_i in iters])
		else:
			loop = list()
			iters = init_state.stages[-1].iters
			for iter_i in iters:
				loop.append(int(iter_i.range.extent))
		loops.append(loop)
		print(task.workload_key)
	return loops







def get_transformed_loops_ansor(tasks, loops):
	'''
	Get the transformed loop extents of the given tasks using Ansor.
	(1) For group_conv2d, we divide "out_channel" into [#groups, #(filters in a group)]
	(2) For depthwise_conv2d_nchw, we divide "out_channel" into [in_channel, channel_multiplier]
	RMK: the input tasks must all need transformation on loops.
	INPUT:
		tasks:
			list of SearchTasks in Ansor. 
		loops:
			list of loops. The original loops corresponding to the tasks.
	OUTPUT:
		transformed_loops:
			list of loops. 
	'''
	import json
	def get_workload_dict(task):
		workload_dict = json.loads(s='{"wlk": ' + task.workload_key + '}')["wlk"]
		return workload_dict
	# 
	transformed_loops = list()
	for i in range(len(tasks)):
		task = tasks[i]
		loop = loops[i]
		workload = get_workload_dict(task)
		if workload[0] == 'group_conv2d_nchw':
			groups = workload[6]
			new_loop = [loop[0], groups, loop[1]//groups] + loop[2:]
			transformed_loops.append(new_loop)
		elif workload[0] == 'depthwise_conv2d_nchw':
			in_c, c_mul, _, _ = workload[2]
			assert in_c * c_mul == loop[1], "wrong in_c and c_mul for loop " +str(i) 
			new_loop = [loop[0], in_c, c_mul] + loop[2:]
			transformed_loops.append(new_loop)			
	return transformed_loops





def transform_back_loops_ansor(func, loops):
	'''
	Get the transformed loop extents of the given tasks using Ansor.
	(1) For group_conv2d, we divide "out_channel" into [#groups, #(filters in a group)]
	(2) For depthwise_conv2d_nchw, we divide "out_channel" into [in_channel, channel_multiplier]
	RMK: the input tasks must all need transformation on loops.
	INPUT:
		func:
			Function. Ansor task register function of the operator the loops correspond to. 
		loops:
			list of loops. The transformed loops corresponding to the tasks.
	OUTPUT:
		transformed_loops:
			list of loops.  The original loops.
	'''
	transformed_loops = list()
	for i in range(len(loops)):
		loop = loops[i]
		print(loop)
		if func == group_conv2d_nchw:
			print("group conv2d")
			new_loop = [loop[0], loop[1]*loop[2]] + loop[3:]
			transformed_loops.append(new_loop)
		elif func == depthwise_conv2d_nchw:
			new_loop = [loop[0], loop[1]*loop[2]] + loop[3:]
			transformed_loops.append(new_loop)			
	return transformed_loops




def loop2dummy_op(loop, op_type, task_names):
	'''
		Create a dummy op of type "op_type" from the given loop.
		INPUT:
			loop:		list of ints. The extents of each iterator of the op.
			op_type:	string. E.g., "conv2d".
			task_names:	list of strings. The op types that we want.
		OUTPUT:
			dummy_op:	Task and the corresponding SearchTask in Ansor.
	'''
	dummy_op = None
	dtype = "float32"
	target = tvm.target.cuda()
	if op_type == "conv2d":
		n, f, y, x, rc, ry, rx = loop
		# default strides = (1, 1)
		data_y = y - 1 + ry
		data_x = x - 1 + rx
		data = relay.var("x", shape=(n, rc, data_y, data_x), dtype=dtype)
		weight = relay.var("w", shape=(f, rc, ry, rx), dtype=dtype)
		net = relay.op.nn.nn.conv2d(
			data,
			weight,
			strides=(1, 1),
			padding=(0, 0),
			dilation=(1, 1),
			groups=1,
			channels=f,
			kernel_size=(ry, rx),
			data_layout="NCHW",
			kernel_layout="OIHW",
			out_layout="",
			out_dtype=dtype,
		)
		module = tvm.IRModule.from_expr(net)
		module = relay.transform.InferType()(module)
		tasks = autotvm.task.extract_from_program(module["main"], target=target, params={})
		for task in tasks:
			if task.name in task_names:
				loops, search_tasks = get_loops([task, ], conv2d_nchw)
				assert loops[0] == loop, "The dummy op has different loop from the given loop!"
				return task, search_tasks[0]
		assert False, "a loop does not have corresponding task!"













def loop2dummy_op_ansor(loop, func):
	'''
		Create a dummy op of using "func" from the given loop. This method uses Ansor's method.
		INPUT:
			loop:		list of ints. The extents of each iterator of the op.
			func:		Function. The registered func' name. E.g., conv2d_transpose_nchw.
		OUTPUT:
			SearchTask and SearchTask in Ansor.
	'''
	dtype = "float32"
	target = tvm.target.cuda()
	task = None
	if func == conv2d_nchw:
		# conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
		n, f, y, x, rc, ry, rx = loop
		data_shape = [n, rc, y - 1 + ry, x - 1 + rx]
		kernel_shape = [f, rc, ry, rx]
		task = auto_scheduler.SearchTask(
				func=func, args=(data_shape, kernel_shape, [1,1], [0, 0, 0, 0],\
					[1, 1], dtype, 'conv2d_nchw.cuda'), target=target)
	elif func == group_conv2d_nchw:
		# group_conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, groups, out_dtype, conv_type):
		# RMK: for this op, we transform the original loop of it before calling this method
		n, g, g_len_f, y, x, rc, ry, rx = loop
		data_shape = [n, g*rc, y - 1 + ry, x - 1 + rx]
		kernel_shape = [g*g_len_f, rc, ry, rx]
		task = auto_scheduler.SearchTask(
				func=func, args=(data_shape, kernel_shape, [1,1], [0, 0, 0, 0],\
					[1, 1], g, dtype, 'group_conv2d_nchw.cuda'), target=target)
	elif func == depthwise_conv2d_nchw:
		# depthwise_conv2d_nchw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
		# RMK: for this op, we transform the original loop of it before calling this method
		n, in_c, c_mul, y, x, ry, rx = loop
		data_shape = [n, in_c, y - 1 + ry, x - 1 + rx]
		kernel_shape = [in_c, c_mul, ry, rx]
		task = auto_scheduler.SearchTask(
				func=func, args=(data_shape, kernel_shape, [1,1], [0, 0, 0, 0],\
					[1, 1], dtype, 'depthwise_conv2d_nchw.cuda'), target=target)		
	elif func == conv2d_transpose_nchw:
		# conv2d_transpose_nchw(data_shape, kernel_shape, stride, padding, out_dtype, output_padding, conv_type)
		n, f, y, x, rc, ry, rx = loop
		data_shape = [n, rc, y+1-ry, x+1-rx]
		kernel_shape = [rc, f, ry, rx]
		task = auto_scheduler.SearchTask(
				func=func, args=(data_shape, kernel_shape, [1,1], [0, 0, 0, 0],\
					dtype, [0, 0], 'conv2d_transpose_nchw.cuda'), target=target)
	elif func == conv1d_ncw:
		# conv1d_ncw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
		n, f, x, rc, rx = loop
		data_shape = [n, rc, x - 1 + rx]
		kernel_shape = [f, rc, rx]
		task = auto_scheduler.SearchTask(
				func=func, args=(data_shape, kernel_shape, [1,], 0,\
					[1,], dtype, 'conv1d_ncw.cuda'), target=target)
	elif func == conv2d_capsule_nhwijc:
		# conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4):
		n, h, w, cap_i, cap_j, co, rh, rw, cap_k, rc = loop
		task = auto_scheduler.SearchTask(
				func=func, args=(n, h-1+rh, w-1+rw, rc, co, rh, 1, 0, cap_k),\
				target=target)
	elif func == conv3d_ncdhw:
		# conv3d_ncdhw(data_shape, kernel_shape, stride, padding, dilation_val, out_dtype, conv_type):
		n, f, z, y, x, rc, rz, ry, rx = loop
		data_shape = [n, rc, z-1+rz, y-1+ry, x-1+rx]
		kernel_shape = [f, rc, rz, ry, rx]
		task = auto_scheduler.SearchTask(
				func=func, args=(data_shape, kernel_shape, [1,1,1], [0, 0, 0, 0, 0, 0],\
					[1, 1, 1], dtype, 'conv3d_ncdhw.cuda'), target=target)
	elif func == batch_matmul:
		# batch_matmul(X_shape, Y_shape, oshape = None, op_name = None):
		b, i, j, k = loop
		# the default setting is X and Y's b is not 1 when batch > 1
		X_shape = [b, i, k]
		Y_shape = [b, j, k]
		task = auto_scheduler.SearchTask(
				func=func, args=(X_shape, Y_shape), target=target)
	elif func == frobenius_norm:
		b, i, j = loop
		# frobenius_norm(N, H, W)
		task = auto_scheduler.SearchTask(
				func=func, args=(b, i, j), target=target)		
	# assert whether the loop is correct or not
	if func in [group_conv2d_nchw, depthwise_conv2d_nchw]:
		assert (get_loops_ansor([task, ])[0] == [loop[0]] + [loop[1]*loop[2]] + loop[3:]),\
			"The dummy op has different loop from the given loop!"
	elif func == frobenius_norm:
		assert (get_op_para_ansor(task, None)['loop'] == loop),\
			"The dummy op has different loop from the given loop!"		
	else:
		assert get_loops_ansor([task, ])[0] == loop,\
			"The dummy op has different loop from the given loop!"
	return task, task















def test_given_knobs(knob_dict, reuse_config, to_tune_op, 
	no_tail,
	task_no, 
	tuner, measure_option, tmp_log_file,
	):
	'''
	Measure the cost on hardware of the configuration consisting of the given knobs
	'''
	tuned_configs = get_tuned_configs(knob_dict, reuse_config, to_tune_op, no_tail)
	# 
	prefix = "[Task %2d] " % (task_no)
	# create tuner
	if tuner == "xgb" or tuner == "xgb-rank":
		tuner_obj = XGBTuner(to_tune_op, loss_type="rank")
	elif tuner == "ga":
		tuner_obj = GATuner(to_tune_op, pop_size=100)
	elif tuner == "random":
		tuner_obj = RandomTuner(to_tune_op)
	elif tuner == "gridsearch":
		tuner_obj = GridSearchTuner(to_tune_op)
	else:
		raise ValueError("Invalid tuner: " + tuner)
	# 
	# if use_transfer_learning:
	#     if os.path.isfile(tmp_log_file):
	#         tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
	# 
	# do tuning
	tsk_trial = len(tuned_configs)
	measure_configs(
		tuner_obj, 
		tuned_configs, 
		measure_option, 
		callbacks=[
			autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
			autotvm.callback.log_to_file(tmp_log_file),
		],
	)
	# 
	# pick best records to a cache file
	# autotvm.record.pick_best(tmp_log_file, log_filename)
	# os.remove(tmp_log_file)
	# best_measure_pairs[to_tune_op] = tuner_obj.best_measure_pair
	return ( to_tune_op , tuner_obj.best_measure_pair)















# ########################################################################################################################################
# ########################################################################################################################################
def get_split_knobs_in_order_winograd(op_para, crossThrdReduction = False):
	if crossThrdReduction:
		keys = op_para['space_tile_knobs_set'][2][:-1] + ['tile_fusedR'] \
			+ op_para['space_tile_knobs_set'][0][:-1] \
			+ op_para['space_tile_knobs_set'][3] + [op_para['space_tile_knobs_set'][2][-1]] + [op_para['space_tile_knobs_set'][0][-1]]
	else:
		keys = op_para['space_tile_knobs_set'][2][:-1] + op_para['space_tile_knobs_set'][1] + op_para['reduc_tile_knobs_set'][1] \
			+ op_para['space_tile_knobs_set'][0][:-1] \
			+ op_para['space_tile_knobs_set'][3] + [op_para['space_tile_knobs_set'][2][-1]] + [op_para['space_tile_knobs_set'][0][-1]]
	return keys



def getTileInforFromTransformSteps(search_task, dag, loop, state, op_para, remain_same = False):
	'''
		Get the transform steps stored in state (which is the best optimized state by Ansor);
		and then try to extract the set of split steps for the compute op (e.g., conv2d);
		INPUT:
			search_task: SearchTask in Ansor;
			dag: ComputeDag in Ansor;
			state: StateObj in Ansor, we only need it contains the necessary transform steps information;
			op_para: dictionary. It stores the information of the op, which is done in hrc_tune_method.
			remain_same: bool. 
				True: then we would keep tile size the same; 
				False: means the tile size has more dimensions than we need in fine tuning, so we need to adjust them.
		OUTPUT:
			tile_knobs:	dictionary, including the tile information on spatial axes and reduction axes.
			Also the vectorization information in cooperative fetching,
			also the auto unroll information.
		RMK:
			We assume inner_to_outer == True.
	'''
	# 
	infor_idx = {"step_type":0, "stageId":1, "iterId":2, "extent":3, "inner_to_outer":4, "length_s":5}
	infor_idx_unroll = {"auto_unroll_max_step":3}
	auto_unroll_keys = ["auto_unroll_max_step"]
	if op_para['op_type'] == "conv2d_winograd":
		auto_unroll_keys = op_para['auto_unroll_keys']
	# 
	tile_knobs_list = list()
	tile_knobs = dict()
	ori_transform_steps = state.transform_steps
	target_stage_id = None
	multi_split_step_ids = list()
	vector_split_step_ids = list()
	vector_input_id = 0
	auto_unroll_step_ids = list()
	layout_rewrite_option = None
	sch, args = dag.apply_steps_from_state(
            state, layout_rewrite_option or search_task.layout_rewrite_option
        )
	# get all split steps information
	for i in range(len(state.transform_steps)):
		tmp = auto_scheduler._ffi_api.MyGetStepNodeInfor(dag, state, i)
		tmp = [int(j) for j in tmp]
		if len(tmp) == 0:
			continue
		if tmp[infor_idx["step_type"]] == 1:
			stage_id = tmp[infor_idx["stageId"]]
			stage_op_name = sch.stages[stage_id].op.name
			# We assume split step for fetching appears in the end when #stages would not change
			if ".shared" == stage_op_name[-len(".shared"):]:
				# this is the split step for fetching
				# tile_knobs["vector_" + stage_op_name[:-len(".shared")]] = tmp[-1]
				tile_knobs["vector_" + str(vector_input_id)] = tmp[-1]
				vector_input_id += 1
				vector_split_step_ids.append(i)
				continue
			# this is a split step (we assume it is for the compute op)
			if target_stage_id == None:
				target_stage_id = stage_id
				multi_split_step_ids.append(i)
			else:
				if (target_stage_id != stage_id) and (op_para['op_type'] != "conv2d_winograd"):
					assert False, "two multi level tiling in this subgraph!"
				else:
					multi_split_step_ids.append(i)
			tile_knobs_list.append(tmp[infor_idx["length_s"]:])
		elif tmp[infor_idx["step_type"]] == 2:
			# this is a unroll step, still need to get from c++ code
			auto_unroll_step_ids.append(i)
			tile_knobs[auto_unroll_keys[len(auto_unroll_step_ids)-1]] = tmp[infor_idx_unroll["auto_unroll_max_step"]]
			# tile_knobs["auto_unroll_max_step"] = tmp[infor_idx_unroll["auto_unroll_max_step"]]
	if (op_para['op_type'] != "conv2d_winograd"):
		assert len(auto_unroll_step_ids) == 1, "only one auto unroll pragma can be in this subgraph!"
	# 
	# print(tile_knobs_list)
	if op_para['op_type'] == "frobenius_norm":
		# since this op does not need multi-level tiling
		if len(tile_knobs_list) == op_para['max_loop_to_tile']:
			# tiled two loops independently
			tile_size = copy.deepcopy(tile_knobs_list[0])
			tile_knobs['tile_b1'] = tile_size
			tile_size = copy.deepcopy(tile_knobs_list[1])
			tile_knobs['tile_b2'] = tile_size
		else:
			# tiled by fused and crossThrdReduction
			assert len(tile_knobs_list) == 1, \
				"error in extracting tile infor for frobenius_norm"
			tile_size = copy.deepcopy(tile_knobs_list[0])
			# we do not deal with "remain_same" here
			tile_knobs['tile_fusedR'] = tile_size
		return tile_knobs, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids
	elif op_para['op_type'] == "conv2d_winograd":
		auto_unroll_keys = op_para['auto_unroll_keys']
		tile_lens = dict()
		# get keys related to split steps in order consistent with the tile_size_list infor
		keys = get_split_knobs_in_order_winograd(op_para, crossThrdReduction = False)
		# print(keys)
		for i in range(len(op_para['space_tile_knobs_set'])-1, -1, -1):
			keys_this_stage = op_para['space_tile_knobs_set'][i] + op_para['reduc_tile_knobs_set'][i]
			for key_i in range(len(keys_this_stage)):
				key = keys_this_stage[key_i]
				# print(key)
				if 'fuseS' in key:
					tile_lens[key] = get_product([op_para['loop_set'][i][j] for j in op_para['space_iters_set'][i]])
				else:
					iter_reorder = op_para['space_iters_set'][i] + op_para['reduc_iters_set'][i]
					tile_lens[key] = op_para['loop_set'][i][iter_reorder[key_i]]
		# print(tile_lens)
		# 
		if len(tile_knobs_list) > len(keys) - len(op_para['space_tile_knobs_set'][1]) + 1:
			# stage `bgemm` is multi-level tiled
			if len(tile_knobs_list) < len(keys):
				assert False, "Currently, we assume stage `conv2d_winograd` is large enough to be tiled (2 level)"
			assert len(tile_knobs_list) == len(keys), "Error: # split steps != # tile knobs we expect"
			for i in range(len(keys)):
				tile_size = copy.deepcopy(tile_knobs_list[i])
				tile_size[0] = -tile_lens[keys[i]] // get_product(tile_size)
				if remain_same:
					tile_knobs[keys[i]] = tile_size
				else:
					tile_knobs[keys[i]] = tile_size[:-2]
					tile_knobs[keys[i]].append(get_product(tile_size[-2:]))
		else:
			# stage `bgemm` is tiled with crossThreadReduction
			if len(tile_knobs_list) < len(keys) - len(op_para['space_tile_knobs_set'][1]) + 1:
				assert False, "Currently, we assume stage `conv2d_winograd` is large enough to be tiled (2 level)"
			# change keys order and tile_lens dict.
			keys = get_split_knobs_in_order_winograd(op_para, crossThrdReduction = True)
			tile_lens['tile_fusedR'] = get_product([op_para['loop_set'][1][iter_i] for iter_i in op_para['reduc_iters_set'][1]])
			# we do not deal with remain same here (since every split knobs has only 2 tile levels)
			for i in range(len(keys)):
				tile_size = copy.deepcopy(tile_knobs_list[i])
				tile_size[0] = -tile_lens[keys[i]] // get_product(tile_size)
				tile_knobs[keys[i]] = tile_size
		return tile_knobs, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids
	# 
	keys = op_para['space_tile_knobs'] + op_para['reduc_tile_knobs'] # the keys are listed in order in op_para
	if len(tile_knobs_list) >= len(keys):
		# multi_tile is done for this op, so we can extract the tile information on each axis in the op loop
		for i in range(len(keys)):
			tile_size = copy.deepcopy(tile_knobs_list[i])
			tile_size[0] = -loop[i] // get_product(tile_size)
			if remain_same:
				tile_knobs[keys[i]] = tile_size
			else:
				tile_knobs[keys[i]] = tile_size[:-2]
				tile_knobs[keys[i]].append(get_product(tile_size[-2:]))
	else:
		# the tile for this op must be cross_thread_reduction, i.e., tile fused reduction axes instead of space axes to bind to threads in gpu
		tile_size = copy.deepcopy(tile_knobs_list[0])
		tile_size[0] = -get_product([loop[i] for i in op_para['reduc_iters']]) // get_product(tile_size) #because this tile is on fused reduc axes
		# we do not deal with "remain_same" here
		tile_knobs['tile_fusedR'] = tile_size
	return tile_knobs, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids






'''
tile_knobs = getTileInforFromTransformSteps(task, sch, inp.state, "conv2d")


'''



def getLoopPerThrdBeforeUnroll(tile_knobs, op_para):
	'''
		RMK: the tile sizes in the tile_knobs should be complete, i.e., 5 levels for space loops, and 3 levels for reduc loops.
		Get the loop (workload) for a thread before unroll and is considered for unrolling.
		INPUT:
			tile_knobs: dictionary. It contains the tile sizes on each axis and the vectorization infor in shared_mem cache, 
				and the auto_unroll_max_step.
			op_para: dictionary. It stores the necessary op information.
		OUTPUT:
			loop_per_thread: list of int. The loop extents.
			axis_infor: list of ["S" (or "R", "V"), knob name], 
				i.e., this tiled axis is for space or reduc or vthrd, and for which exact original axis.
	'''	
	space_level = 3 # start from level 3 of S in "SSSRRSRS"
	reduc_level = 0 # start from level 0 of R in "SSSRRSRS"
	structure = "SSSRRSRS"
	tile_i = 3
	loop_per_thread = list()
	axis_infor = list() # stores the iter infor in loop_per_thread
	while tile_i < len(structure):
		if structure[tile_i] == "S":
			for knob in op_para['space_tile_knobs']:
				loop_per_thread.append(tile_knobs[knob][space_level])
				axis_infor.append(["S", knob])
			space_level += 1
		elif structure[tile_i] == "R":
			for knob in op_para['reduc_tile_knobs']:
				loop_per_thread.append(tile_knobs[knob][reduc_level])
				axis_infor.append(["R", knob])
			reduc_level += 1
		tile_i += 1
	# we need to add a loop for the fused virtual thread (we observed they are fused)
	vthrd_level = 1 # the first tile level corr. to virtual thread
	loop_per_thread.append(get_product([tile_knobs[knob][vthrd_level] for knob in op_para['space_tile_knobs']]))
	axis_infor.append(["V", "fused_vthrd"])	
	return loop_per_thread, axis_infor





def get_auto_unroll_max_step_knob_name(op_para):
	auto_unroll_max_step_knob = 'auto_unroll_max_step'
	if op_para['op_type'] == "conv2d_winograd":
		auto_unroll_max_step_knob = op_para['auto_unroll_keys'][1]
	return auto_unroll_max_step_knob



def calRegisterNumFromTileKnobs(tile_knobs, op_para, auto_unroll_max_step = None):
	'''
		RMK: the tile sizes in the tile_knobs should be complete, i.e., 5 levels for space loops, and 3 levels for reduc loops.
		Calculate the estimated number of registers that is needed in an unrolled loop.
		INPUT:
			tile_knobs: dictionary. It contains the tile sizes on each axis and the vectorization infor in shared_mem cache, 
				and the auto_unroll_max_step.
			op_para: dictionary. It stores the necessary op information.
			auto_unroll_max_step: int. If provided, use this value; otherwise, use the value in tile_knobs.
		OUTPUT:
			reg_num: int. The estimated number of registers needed in an unrolled loop.
			unroll_loop_num: int. The number of loops that are unrolled.
	'''
	auto_unroll_max_step_knob = get_auto_unroll_max_step_knob_name(op_para)
	# first get the shape of the unrolled loop before unrolling (for a thread).
	if auto_unroll_max_step == None:
		auto_unroll_max_step = tile_knobs[auto_unroll_max_step_knob]
	loop_per_thread, axis_infor = getLoopPerThrdBeforeUnroll(tile_knobs, op_para)
	# get the loops that would be unrolled
	prod = 1
	unroll_start_i = None
	for i in range(len(loop_per_thread) - 1, -1, -1):
		prod = prod * loop_per_thread[i]
		if prod >= auto_unroll_max_step:
			unroll_start_i = i + 1
			break
	if unroll_start_i == None:
		unroll_start_i = 0
	# compute the number of input the unrolled loop needs, first get necessary parameters
	blk_shape, thrd_shape, vthrd_shape, reduc_shape, load_shape = None, None, None, None, None
	blk_level, vthrd_level, thrd_level, load_level = 0, 1, 2, 0 
	if unroll_start_i == len(loop_per_thread):
		# this loop cannot be unrolled
		blk_shape = [get_product(tile_knobs[knob]) // tile_knobs[knob][blk_level] for knob in op_para['space_tile_knobs']]
		vthrd_shape = [1 for knob in op_para['space_tile_knobs']]
		thrd_shape = blk_shape
		reduc_shape = [1 for knob in op_para['reduc_tile_knobs']]
		load_shape = [get_product(tile_knobs[knob]) // tile_knobs[knob][load_level] for knob in op_para['reduc_tile_knobs']]
	else:
		blk_shape = [get_product(tile_knobs[knob]) // tile_knobs[knob][blk_level] for knob in op_para['space_tile_knobs']]
		vthrd_shape = [tile_knobs[knob][vthrd_level] for knob in op_para['space_tile_knobs']]
		thrd_shape = [blk_shape[i] // vthrd_shape[i] for i in range(len(blk_shape))]
		reduc_shape = [1 for knob in op_para['reduc_tile_knobs']]
		# adjust thrd shape and reduc shape according to the unrolled loop
		for unroll_i in range(unroll_start_i, len(loop_per_thread)):
			if axis_infor[unroll_i][0] == "S":
				for i in range(len(op_para['space_tile_knobs'])):
					if op_para['space_tile_knobs'][i] == axis_infor[unroll_i][1]:
						thrd_shape[i] = thrd_shape[i] // loop_per_thread[unroll_i]
						break
			elif axis_infor[unroll_i][0] == "R":
				for i in range(len(op_para['reduc_tile_knobs'])):
					if op_para['reduc_tile_knobs'][i] == axis_infor[unroll_i][1]:
						reduc_shape[i] = reduc_shape[i] * loop_per_thread[unroll_i]
						break
		# 
		load_shape = [get_product(tile_knobs[knob]) // tile_knobs[knob][load_level] for knob in op_para['reduc_tile_knobs']]
	# 
	input_needed = load_per_thread(blk_shape, thrd_shape, vthrd_shape, reduc_shape, op_para, load_shape)
	reg_num = sum(input_needed) + get_product([(get_product(tile_knobs[knob]) // tile_knobs[knob][blk_level]) // tile_knobs[knob][thrd_level] \
		for knob in op_para['space_tile_knobs']])
	return reg_num, len(loop_per_thread) - unroll_start_i





def get_min_factor(tot):
	if tot == 1:
		factor = 1
	else:
		factor = min(factorint(tot).keys())	
	return factor




def tileLoopPerThread(tile_knobs, op_para):
	'''
		RMK: this tile is done by heuristics, following the principle that the min factor is in the last tile level.
			 tiles sizes in tile_knobs must be imcomplete.
		Tile the workload (loops) per thread for better unrolling.
		INPUT:
			tile_knobs: dictionary. It contains the tile sizes on each axis and the vectorization infor in shared_mem cache, 
				and the auto_unroll_max_step.
			op_para: dictionary. It stores the necessary op information.
		OUTPUT:
			No output, we directly modify the tile_knobs, making the tile sizes COMPLETE.
	'''
	for knob in op_para['space_tile_knobs'] + op_para['reduc_tile_knobs']:
		to_tile = tile_knobs[knob][-1]
		tile_factor = get_min_factor(to_tile)
		tile_knobs[knob][-1] = to_tile // tile_factor
		tile_knobs[knob].append(tile_factor)






def getDefaultAutoUnroll(tile_knobs, op_para, reused_regNum, max_unroll_options = [0, 16, 64, 512, 1024]):
	'''
		RMK: the tile sizes in the tile_knobs should be complete, i.e., 5 levels for space loops, and 3 levels for reduc loops.
		RMK: the principle is that we must unroll at least one loop.
		Compute the default auto_unroll_max_step, such the registers needed is close (a little smaller or larger than) to the reused one.
		INPUT:
			tile_knobs: dictionary. It contains the tile sizes on each axis and the vectorization infor in shared_mem cache, 
				and the auto_unroll_max_step.
			op_para: dictionary. It stores the necessary op information.
			reused_regNum: int. The estimated number of registers of the reused config.
			max_unroll_options: list of ints. The possible values of auto_unroll_max_step.
		OUTPUT:
			auto_unroll_max_step: int. The parameter for the auto unroll pragma in Ansor.
	'''
	# first get the tiled loops (workload) per thread for us to analyse
	regNums = list()
	for i in range(len(max_unroll_options)):
		auto_unroll_max_step = max_unroll_options[i]
		tmp_regNum, unroll_loop_num = calRegisterNumFromTileKnobs(tile_knobs, op_para, auto_unroll_max_step)
		if unroll_loop_num == 0:
			# at least one loop must be unrolled
			continue
		if tmp_regNum > reused_regNum:
			if len(regNums) == 0:
				return auto_unroll_max_step
			else:
				# choose the closer one as the default value
				last_regNum = regNums[-1]
				if reused_regNum - last_regNum >= tmp_regNum - reused_regNum:
					return auto_unroll_max_step
				else:
					return max_unroll_options[i - 1]
		else:
			regNums.append(tmp_regNum)
	# If no value in regNums: This case is impossible, because a thread can have <= 255 registers, 
	# 	and the max unroll factor is 1024, the inner most loop is fused vthrd
	# 	any value in this case is ok
	# Else: return the max unroll factor, because the regNum is still <= reused_regNum
	return max(max_unroll_options)





def getDifferentAutoUnroll(tile_knobs, op_para, base_max_unroll, max_unroll_options = [0, 16, 64, 512, 1024]):
	'''
		RMK: the tile sizes in the tile_knobs should be complete, i.e., 5 levels for space loops, and 3 levels for reduc loops.
		Compute all the auto_unroll_max_step in given max_unroll_options that generate different unroll structure than base_max_unroll.
		INPUT:
			tile_knobs: dictionary. It contains the tile sizes on each axis and the vectorization infor in shared_mem cache, 
				and the auto_unroll_max_step.
			op_para: dictionary. It stores the necessary op information.
			base_max_unroll: int. The auto_unroll_max_step to be compared.
			max_unroll_options: list of ints. The possible values of auto_unroll_max_step.
		OUTPUT:
			other_auto_unroll_max_step: list of ints. auto_unroll_max_step's which have different unroll effect than base_max_unroll.		
	'''
	_, unroll_loop_num_base = calRegisterNumFromTileKnobs(tile_knobs, op_para, base_max_unroll)
	unroll_loop_num_list = list()
	ret = list()
	for auto_unroll_max_step in max_unroll_options:
		if auto_unroll_max_step == base_max_unroll:
			continue
		_, unroll_loop_num = calRegisterNumFromTileKnobs(tile_knobs, op_para, auto_unroll_max_step)
		if unroll_loop_num == unroll_loop_num_base:
			continue
		if unroll_loop_num not in unroll_loop_num_list:
			unroll_loop_num_list.append(unroll_loop_num)
			ret.append(auto_unroll_max_step)
	return ret



def setAutoUnrollInTileKnobs(op_para_t, tuned_knobs, op_para_r, tile_knobs_r, scaleRegNum = 1, NeedFurtherTile = True):
	'''
		Set the auto_unroll_max_step in tuned_knobs.
		INPUT:
			op_para_t, op_para_r: dictionary. Store the op infor.
			tuned_knobs: {knob: list of values}.
			tile_knobs_r: {knob: value}. The tile knobs of the reused op.
			scaleRegNum: float or list of floats. scaleRegNum * reused_regNum is the baseline we consider in this function.
			NeedFurtherTile: whether the loops need further tile or not.
		OUTPUT:
			NO output. We directly change the content of tuned_knobs.
	'''
	# set the correct auto unroll name (currently we only tune unroll for MultiLevelTiling)
	auto_unroll_max_step_knob = get_auto_unroll_max_step_knob_name(op_para_t)
	# 
	keys = op_para_t['space_tile_knobs'] + op_para_t['reduc_tile_knobs']
	num = len(tuned_knobs[keys[0]])
	if not isinstance(scaleRegNum, list):
		scaleRegNum = [scaleRegNum for i in range(num)]
	# 
	reused_regNum, _ = calRegisterNumFromTileKnobs(tile_knobs_r, op_para_r)
	tuned_knobs[auto_unroll_max_step_knob] = list()
	for i in range(num):	
		cur_tile_knobs = dict()
		for k in keys:
			cur_tile_knobs[k] = tuned_knobs[k][i]
		# first further tile the loop for better unrolling, we modify tile sizes in tuned_knobs at the same time
		if NeedFurtherTile:
			tileLoopPerThread(cur_tile_knobs, op_para_t)
		auto_unroll_max_step = getDefaultAutoUnroll(cur_tile_knobs, op_para_t, reused_regNum * scaleRegNum[i])
		tuned_knobs[auto_unroll_max_step_knob].append(auto_unroll_max_step)







def getPossibleVectorize(tot_data, thrd_num, vectorize_options = [1, 2, 4]):
	'''
		RMK:
			The vectorize_options is got from that CUDA supports at most 16-byte words. 
			Although larger values seems to be ok as well. But according to corr. CUDA code, when larger than 4, it uses float4 as well.
		Compute the default vectorization value in shared_mem caching, such the registers needed is close (or a little larger than) to the reused one.
		INPUT:
			tot_data: int. The total amount of data points the thread block needs to load.
			thrd_num: int. The total number of threads in a block.
			vectorize_options: list of ints. The options for vectorization in shared mem cache. 
		OUTPUT:
			possible_v: list of ints. The possible vectorization factors.
	'''
	loadPerThrd = math.ceil(tot_data / thrd_num)
	possible_v = [f for f in vectorize_options if (loadPerThrd % f == 0)]
	return possible_v







def tileKnobsToTransformSteps(op_para, tuned_knobs, remain_same = False):
	'''
		Convert the tuned tile knobs to transform steps in Ansor in order. E.g., for conv2d, the order is [n, f, y, x, rc, ry, rx].
		# Also generate the split steps for cooperative fetching. (do not do this part)
		We do not consider tuning "unrolling" currently.
		INPUT:
			op_para: dictionary. Stores op information, which is done in the hrc_tune_method.
			tuned_knobs: dictionary. {"tile_f" : [1, 1, 1, -1], ...}
			remain_same: bool. True if not assign default value to some dimension of the tile size; else False (currently, the default value is 1).
		OUTPUT: a list of tuned transform step parameters which can be used to build transform step. [para1, para2, ...], n sets of parameters are concatenated in one list.
	'''
	# ONE EXAMPLE: for "conv2d":
	# keys in tile_knobs ["tile_n", "tile_f", "tile_y", "tile_x", "tile_rc", "tile_ry", "tile_rx"]
	# keys in tile_knobs for ansor: ["tile_n", "tile_f", "tile_y", "tile_x", "tile_rc", "tile_ry", "tile_rx", "vector_kernel", "vector_pad_temp"]
	tile_sizes = list()
	auto_unroll_keys = ["auto_unroll_max_step"]
	# first get all tile knobs--------------------------
	if op_para['op_type'] == 'frobenius_norm':
		if 'tile_fusedR' in tuned_knobs.keys():
			keys = ['tile_fusedR']
		else:
			keys = ['tile_b1', 'tile_b2']
	elif op_para['op_type'] == 'conv2d_winograd':
		auto_unroll_keys = op_para['auto_unroll_keys']
		if op_para['space_tile_knobs_set'][1][0] not in tuned_knobs.keys():
			keys = get_split_knobs_in_order_winograd(op_para, crossThrdReduction = True)
		else:
			keys = get_split_knobs_in_order_winograd(op_para, crossThrdReduction = False)
	else:
		keys = op_para['space_tile_knobs'] + op_para['reduc_tile_knobs'] # the keys are listed in order in op_para
		if keys[0] not in tuned_knobs.keys():
			# this op is not multi_level tiled, we only deal with the tile in crossThreadReduction here
			keys = ['tile_fusedR', ]
	num = len(tuned_knobs[keys[0]])
	# 
	copyVectorize = False
	if ("vector_0" in tuned_knobs.keys()) and (len(tuned_knobs["vector_0"]) == num) \
		and ("vector_1" in tuned_knobs.keys()) and (len(tuned_knobs["vector_1"]) == num):
		copyVectorize = True
	# 
	# print("tuned_knobs in get transform steps: \n", tuned_knobs)
	# print("split keys we consider: ", keys)
	for i in range(num):
		tile_size_config = list()
		for k in keys:
			v = tuned_knobs[k][i]
			temp = list()
			# because tile size for split step with 5 parts only stores 4 length information
			for ele in v[1:]:
				temp.append(ele)
			if not remain_same:
				temp.append(1)
			tile_size_config.append(temp)
		# add tile size for vectorization (2 values here because we have 2 input data)
		if copyVectorize:
			tile_size_config.append([tuned_knobs["vector_0"][i]])
			tile_size_config.append([tuned_knobs["vector_1"][i]])
		else:
			tile_size_config.append([1])
			tile_size_config.append([1])
		# add auto_unroll_max_step
		for auto_unroll_key in auto_unroll_keys:
			tile_size_config.append([tuned_knobs[auto_unroll_key][i]])
		tile_sizes.append(tile_size_config)
	return tile_sizes




















def get_op_para_ansor(task, loop):
	'''
		Get op parameter dictionary. task is of type SearchTask in Ansor
	'''
	import json
	def get_workload_dict(task):
		workload_dict = json.loads(s='{"wlk": ' + task.workload_key + '}')["wlk"]
		return workload_dict
	# 
	op_para = None
	workload = get_workload_dict(task)
	if workload[0] == 'conv2d_nchw':
		stride = workload[3]
		dilation = workload[5]
		_, _, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh 
		op_para['kw'] = kw 
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['op_type'] = "conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5, 6]
		op_para['load_step_iter'] = 4 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
	elif workload[0] == 'group_conv2d_nchw':
		stride = workload[3]
		dilation = workload[5]
		groups = workload[6]
		_, in_channel, _, _ = workload[1]
		num_filter, _, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh
		op_para['kw'] = kw
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['groups'] = groups
		op_para['in_channel'] = in_channel
		op_para['num_filter'] = num_filter
		op_para['op_type'] = "group_conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5, 6]
		op_para['load_step_iter'] = 4 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
		# knob1 in space knobs (i.e., tile_f) must cover k groups in blk shape
		op_para['tile_by_group_knob_idx'] = 1 
	elif workload[0] == 'depthwise_conv2d_nchw':
		stride = workload[3]
		dilation = workload[5]
		_, in_channel, _, _ = workload[1]
		_, channel_multiplier, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh 
		op_para['kw'] = kw 
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['in_channel'] = in_channel
		op_para['channel_multiplier'] = channel_multiplier
		op_para['op_type'] = "depthwise_conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5]
		op_para['load_step_iter'] = None # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_ry', 'tile_rx']
		op_para['load_step_knob'] = None
	elif workload[0] == 'conv2d_transpose_nchw':
		_, _, kh, kw = workload[2]
		op_para = dict()
		op_para['kh'] = kh 
		op_para['kw'] = kw 
		op_para['op_type'] = "transpose_conv2d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3]
		op_para['reduc_iters'] = [4, 5, 6]
		op_para['load_step_iter'] = 4 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'		
	elif workload[0] == 'conv2d_capsule_nhwijc':
		# this op has not topi topi implementation, so we cannot extract autoTVM tasks
		# conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4)
		# workload_key example: '["conv2d_capsule_nhwijc", 1, 16, 16, 32, 32, 3, 2, 1, 4]'
		op_para = dict()
		op_para['kh'] = workload[2] 
		op_para['kw'] = workload[3] 
		op_para['stride'] = workload[7] # it is int, not list
		op_para['op_type'] = "conv2d_capsule"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3, 4, 5]
		op_para['reduc_iters'] = [6, 7, 8, 9]
		op_para['load_step_iter'] = 9 # the axis divided for loading input data
		op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x', 
										'tile_capi', 'tile_capj', 'tile_f']
		op_para['reduc_tile_knobs'] = ['tile_ry', 'tile_rx', 'tile_rcapk','tile_rc']
		op_para['load_step_knob'] = 'tile_rc'	
	elif workload[0] == 'conv1d_ncw':
		# we get op parameter of conv1d from ansor search task (not autotvm task)
		stride = workload[3]
		dilation = workload[5]
		_, _, kw = workload[2]
		op_para = dict()
		op_para['kw'] = kw
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['op_type'] = "conv1d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2]
		op_para['reduc_iters'] = [3, 4]
		op_para['load_step_iter'] = 3
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
	elif workload[0] == 'conv3d_ncdhw':
		stride = workload[3]
		dilation = workload[5]
		_, _, kd, kh, kw = workload[2]
		op_para = dict()
		op_para['kd'] = kd
		op_para['kh'] = kh
		op_para['kw'] = kw
		op_para['stride'] = stride
		op_para['dilation'] = dilation
		op_para['op_type'] = "conv3d"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2, 3, 4]
		op_para['reduc_iters'] = [5, 6, 7, 8]
		op_para['load_step_iter'] = 5
		op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_d', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_rd', 'tile_ry', 'tile_rx']
		op_para['load_step_knob'] = 'tile_rc'
	elif workload[0] == 'batch_matmul':
		# output shape is [batch, M, N], iteration space is b, y, x, k.
		op_para = dict()
		op_para['X_shape'] = workload[1]
		op_para['Y_shape'] = workload[2]
		op_para['op_type'] = "batch_matmul"
		op_para['loop'] = loop
		op_para['space_iters'] = [0, 1, 2]
		op_para['reduc_iters'] = [3]
		op_para['load_step_iter'] = 3
		op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x']
		op_para['reduc_tile_knobs'] = ['tile_k']
		op_para['load_step_knob'] = 'tile_k'
	elif workload[0] == 'frobenius_norm':
		op_para = dict()
		op_para['N'] = workload[1]
		op_para['H'] = workload[2]
		op_para['W'] = workload[3]
		op_para['op_type'] = "frobenius_norm"
		op_para['loop'] = [workload[1], workload[2], workload[3]]
		op_para['space_iters'] = [0]
		op_para['reduc_iters'] = [1, 2]
		op_para['max_loop_to_tile'] = 2 # there are at most 2 unfused loops to tile
	elif "conv2d_winograd" in task.compute_dag.init_state.stages[-1].__str__():
		# deal with winograd conv2d in nhwc layout
		op_para = dict()
		op_para['op_type'] = "conv2d_winograd"
		op_para['loop_set'] = loop
		op_para['loop'] = None # dynamically set it to the loop being optimized in 'loop_set'
		op_para['space_iters_set'] = [[2, 3], [0, 1, 2, 3], [2, 3], [0, 1, 2, 3]]
		op_para['reduc_iters_set'] = [[0, 1, 4, 5], [4,], [0, 1, 4, 5], []] # for datapack and inverse, [0,1] are unrolled space axes.
		op_para['space_iters'] = None
		op_para['reduc_iters'] = None
		# below is for space and reduc tile knobs of all stages
		# # the space axes for the ConstTensor stage: data_pack, are first tiled, reordered, fused, and then tiled
		op_para['space_tile_knobs_set'] = [['tile_p_datapack', 'tile_ci_datapack', 'tile_fuseS_datapack'], 
										['tile_eps', 'tile_nu', 'tile_p', 'tile_co'],
										['tile_p_inverse', 'tile_co_inverse', 'tile_fuseS_inverse'], 
										['tile_fuseS_winograd']]
		op_para['reduc_tile_knobs_set'] = [[], ['tile_ci'], [], []]
		op_para['space_tile_knobs'] = None
		op_para['reduc_tile_knobs'] = None		
		# below is for the stage: bgemm
		op_para['X_shape'] = copy.deepcopy(loop[0][:4]) # from data pack
		op_para['Y_shape'] = copy.deepcopy(loop[0][:2]) + [loop[1][3], loop[0][3]]
		op_para['load_step_iter'] = 4
		op_para['load_step_knob'] = 'tile_ci'
		op_para['auto_unroll_keys'] = ["auto_unroll_max_step_datapack", "auto_unroll_max_step_bgemm", "auto_unroll_max_step_inverse"]
	# 
	# print("OOOOOOOP_PARA: ", op_para)
	return op_para






def tune_frobenius_norm(rconfig_tile_knobs, op_para_t, op_para_r):
	inc_diff = get_product(op_para_t['loop']) / get_product(op_para_r['loop'])
	reduc_space_size = get_product([op_para_t['loop'][i] for i in op_para_t['reduc_iters'] ])
	reduc_space_size_r = get_product([op_para_r['loop'][i] for i in op_para_r['reduc_iters'] ])
	output_size = get_product([op_para_t['loop'][i] for i in op_para_t['space_iters'] ])
	output_size_r = get_product([op_para_r['loop'][i] for i in op_para_r['space_iters'] ]) 
	if 'tile_fusedR' in rconfig_tile_knobs.keys():
		# the reused op is tiled on reduction axes for parallelization
		rconfig_thrd_num = rconfig_tile_knobs['tile_fusedR'][1]
		thrd_sizes = all_dividers(0, reduc_space_size+1, reduc_space_size)
		# we only selected the thrd_sizes such that dt(blk_num*thrd_size) / dt(whole_space_size)
		sign_tmp = 1 if inc_diff > 1 else -1
		# seems here is a bug, for frobenius norm, there can be only one block [!there is no bug here, can have more than 1 block]
		# thrd_sizes = [thrd_size for thrd_size in thrd_sizes \
		# 	if ((1 * thrd_size * sign_tmp \
		# 		<= 1 * rconfig_thrd_num * inc_diff * sign_tmp) and \
		# 		(1 * thrd_size * sign_tmp\
		# 		>= 1 * rconfig_thrd_num * sign_tmp))]
		thrd_sizes = [thrd_size for thrd_size in thrd_sizes \
			if ((math.ceil(output_size / thrd_size) * thrd_size * sign_tmp \
				<= math.ceil(output_size_r / rconfig_thrd_num) * rconfig_thrd_num * inc_diff * sign_tmp) and \
				(math.ceil(output_size / thrd_size) * thrd_size * sign_tmp\
				>= math.ceil(output_size_r / rconfig_thrd_num) * rconfig_thrd_num * sign_tmp))]
		thrd_sizes = getValidThrdNumPerBlk(thrd_sizes)
		thrd_sizes.sort()
		enu_knobs = dict()
		if len(thrd_sizes) == 0: 
			# this case should not happen
			assert False
		# rconfig_tile_knobs stores a list of knobs, although the list len is 1
		for k in rconfig_tile_knobs:
			rconfig_tile_knobs[k] = [rconfig_tile_knobs[k]]
		# 
		enu_knobs['tile_fusedR'] = [[reduc_space_size//thrd_size, thrd_size] for thrd_size in thrd_sizes]
	else:
		# the two loops are tiled independently
		rconfig_thrd_num1 = rconfig_tile_knobs['tile_b1'][1]
		rconfig_thrd_num2 = rconfig_tile_knobs['tile_b2'][1]
		thrd_sizes1 = all_dividers(rconfig_thrd_num1, rconfig_thrd_num1 * output_size / output_size_r, output_size)
		if reduc_space_size == reduc_space_size_r:
			thrd_sizes2 = all_dividers(rconfig_thrd_num2, \
				rconfig_thrd_num2 * inc_diff, output_size)
			# if reduc space diff = 0, we force the method to find one extra thrd num beyond the reduc diff bound.
			if inc_diff > 1:
				extra_thrd_sizes2 = all_dividers(rconfig_thrd_num2 - 1, 1, output_size)
				if len(extra_thrd_sizes2) > 0:
					thrd_sizes2.append(max(extra_thrd_sizes2))
			else:
				extra_thrd_sizes2 = all_dividers(rconfig_thrd_num2 + 1, output_size, output_size)
				if len(extra_thrd_sizes2) > 0:
					thrd_sizes2.append(min(extra_thrd_sizes2))
		else:
			thrd_sizes2 = all_dividers(rconfig_thrd_num2 / reduc_space_size * reduc_space_size_r, \
				rconfig_thrd_num2 * inc_diff, output_size)
		thrd_sizes1 = getValidThrdNumPerBlk(thrd_sizes1)
		thrd_sizes1.sort()
		thrd_sizes2 = getValidThrdNumPerBlk(thrd_sizes2)
		thrd_sizes2.sort()
		enu_knobs = dict()
		if len(thrd_sizes1) == 0 or len(thrd_sizes2) == 0: 
			# this case should not happen
			assert False
		# rconfig_tile_knobs stores a list of knobs, although the list len is 1
		for k in rconfig_tile_knobs:
			rconfig_tile_knobs[k] = [rconfig_tile_knobs[k]]
		# 
		enu_knobs['tile_b1'] = [[output_size//thrd_size, thrd_size] for thrd_size in thrd_sizes1]
		enu_knobs['tile_b2'] = [[output_size//thrd_size, thrd_size] for thrd_size in thrd_sizes2]
	# print("BEFORE SUCCESS")
	tuned_knobs = change_configs(rconfig_tile_knobs, enu_knobs)
	# print("IN SUCCESS")
	# print(tuned_knobs)
	# 
	# print("AFTER SUCCESS")
	tile_sizes = tileKnobsToTransformSteps(op_para_t, tuned_knobs, True)
	# print(tile_sizes)
	return tile_sizes




def tune_ConstTensor_or_2LevelTile_sketch(rconfig_tile_knobs, op_para_t, op_para_r, knob_to_tune):
	''' 
		Like tune_frobenius_norm, this method generate possible knob values which need to be measured,
		but for the sketch tiled with SimplifyComputationWithConstTensor or 
			the sketch tiled with the simple 2-level tile structure (which fuses all outer space axes first and then tile it into 2 parts. 
		knob_to_tune: str. which knob is being tuned in this function.
		RMK:
			Before calling this function, make sure op_para_t['loop'] and op_para_r['loop'] is updated for the current stage being optimized
	'''
	# print("BEGIN TUNE CONSTTENSOR AND 2 LEVEL TILE!")
	inc_diff = get_product(op_para_t['loop']) / get_product(op_para_r['loop'])
	# print(inc_diff)
	reduc_space_size = get_product([op_para_t['loop'][i] for i in op_para_t['reduc_iters'] ])
	reduc_space_size_r = get_product([op_para_r['loop'][i] for i in op_para_r['reduc_iters'] ])
	dt_reduc_space = reduc_space_size / reduc_space_size_r
	# print(dt_reduc_space)
	rconfig_thrd_num = rconfig_tile_knobs[op_para_r['space_tile_knobs'][-1]][1]
	output_size = get_product([op_para_t['loop'][i] for i in op_para_t['space_iters'] ])
	thrd_sizes = all_dividers(rconfig_thrd_num / dt_reduc_space, rconfig_thrd_num * inc_diff, output_size)
	# 
	if reduc_space_size == reduc_space_size_r:
		if inc_diff > 1:
			extra_thrd_size = all_dividers(rconfig_thrd_num - 1, 1, output_size)
			if len(extra_thrd_size) > 0:
				thrd_sizes.append(max(extra_thrd_size))
		else:
			extra_thrd_size = all_dividers(rconfig_thrd_num + 1, output_size, output_size)
			if len(extra_thrd_size) > 0:
				thrd_sizes.append(min(extra_thrd_size))
	thrd_sizes = getValidThrdNumPerBlk(thrd_sizes)
	thrd_sizes.sort()
	enu_knobs = dict()
	if len(thrd_sizes) == 0: 
		# this case should not happen
		assert False
	# rconfig_tile_knobs stores a list of knobs, although the list len is 1
	# for k in rconfig_tile_knobs:
	# 	rconfig_tile_knobs[k] = [rconfig_tile_knobs[k]]
	# 
	enu_knobs[knob_to_tune] = [[output_size//thrd_size, thrd_size] for thrd_size in thrd_sizes]
	# print("BEFORE SUCCESS")
	# tuned_knobs = change_configs(rconfig_tile_knobs, enu_knobs)
	# print("IN SUCCESS")
	# print(tuned_knobs)
	# 
	# print("AFTER SUCCESS")
	# tile_sizes = tileKnobsToTransformSteps(op_para_t, tuned_knobs, True)
	# print(tile_sizes)
	return enu_knobs #tile_sizes






def set_stage_to_tune_for_MultiStages(op_para, loop_id):
	''' 
		Prepare infor in op_para for ops with multi stages (there are more than 1 loop nests after tuning). 
		INPUT:
			op_para: dict. Stores the information of an op.
			loop_id: int. Which loop nest (the order is as in the program after optimization) is currently being optimized.
	'''
	op_para['loop'] = op_para['loop_set'][loop_id]
	op_para['space_iters'] = op_para['space_iters_set'][loop_id]
	op_para['reduc_iters'] = op_para['reduc_iters_set'][loop_id]
	op_para['space_tile_knobs'] = op_para['space_tile_knobs_set'][loop_id]
	op_para['reduc_tile_knobs'] = op_para['reduc_tile_knobs_set'][loop_id]	




def get_default_knobs_for_a_sketch(sketch_type, op_para, loop_id = None, default_knobs = dict()):
	''' 
		NOTE: Since MultiLevelTile is too complex, we do not compute the default knobs for it. Instead, we directly tune it first. 
				Therefore, "CrossThread" will also not be dealt with in this function
		This function generates the default knobs for a specific type of sketch. 
		INPUT:
			sketch_type: str. possible values: "ConstTensor", "TwoLevelTile".
			op_para: dict. Stores the information of an op.
			rconfig_tile_knobs: dict. It is the same as used in the hrc tuning method.
			loop_id: int. 
				If there are more then one target loop nest in the initial program of an op, specify which loop nest this sketch is for.
				If `None`, then only one loop nest exists.
		OUTPUT:
			NO OUTPUT, we directly modify the content in the input default_knobs.
	'''
	# define some constant values
	warp_size = 32 # the number of threads in a warp
	if loop_id != None:
		# there are more than 1 loop nests in this op, e.g., winograd conv2d.
		set_stage_to_tune_for_MultiStages(op_para, loop_id)
	# 
	# default_knobs = None
	if sketch_type == "ConstTensor": 
		# currently, only conv2d_winograd has this sketch
		# default_knobs = dict()
		tot_iter_len = 1
		for knob_i in range(len(op_para['space_tile_knobs'])):
			knob = op_para['space_tile_knobs'][knob_i]
			if knob_i == len(op_para['space_tile_knobs']) - 1:
				default_knobs[knob] = [[math.ceil(tot_iter_len / warp_size), warp_size]]
			else:
				iter_len = op_para['loop'][op_para['space_iters'][knob_i]]
				tot_iter_len = tot_iter_len * iter_len
				default_knobs[knob] = [[iter_len , 1]]
	elif sketch_type == "TwoLevelTile":
		# this sketch type is generated if the loop is tiled in `InitThreadBind`.
		# default_knobs = dict()
		knob = op_para['space_tile_knobs'][0]
		tot_iter_len = get_product([op_para['loop'][i] for i in op_para['space_iters']])
		default_knobs[knob] = [[math.ceil(tot_iter_len / warp_size), warp_size]]
	# elif sketch_type == "CrossThread":
	# 	# when space axes is smaller than reduc axes, or when the loop nest does not have multi-level tiling.
	# 	default_knobs = dict()
	# 	tot_iter_len = get_product([op_para['loop'][i] for i in op_para['reduc_iters']])
	# 	default_knobs['tile_fusedR'] = [[math.ceil(tot_iter_len / warp_size), warp_size]]
	else:
		assert False, "We currently do not support default settings for other types of sketches! The error stage is: " + sketch_type





def get_default_knobs_conv2d_winograd(op_para):
	''' return a set of complete default knobs for a winograd conv2d operator. '''
	stages = ["ConstTensor", "MultiLevelTile", "ConstTensor", "TwoLevelTile"]
	default_knobs = dict()
	for i in range(len(stages)):
		if stages[i] != "MultiLevelTile":
			get_default_knobs_for_a_sketch(stages[i], op_para, loop_id = i, default_knobs = default_knobs)
	# print(default_knobs)
	return default_knobs







def tuneEnuKnobsONLY(to_tune_idx, op_para, dag_to_tune, state_reused_from,
	multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids,
	search_policy, tune_option,
	tot_config_num, wlk_to_tune, wlk_reuse_from, best_reuse_results, 
	best_res_this_blk_shape, 
	enu_knobs, curr_best_knobs, 
	):
	# curr_best_knobs stores a list of knobs, although the list len is 1
	for k in curr_best_knobs:
		curr_best_knobs[k] = [curr_best_knobs[k]]
	# 
	if len(enu_knobs[list(enu_knobs.keys())[0]]) == 0: 
		return list(), tot_config_num
	# print("BEFORE SUCCESS")
	tuned_knobs = change_configs(curr_best_knobs, enu_knobs)
	tile_sizes = tileKnobsToTransformSteps(op_para, tuned_knobs, True)
	# print("="*50)
	# print(prefix)
	# print("="*50)
	states_to_measure = auto_scheduler._ffi_api.MyGetStatesFromTunedKnobs(
		dag_to_tune, state_reused_from,
		tile_sizes, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids)
	states_to_measure = [dag_to_tune.infer_bound_from_state(state) for state in states_to_measure]
	stateObjs_to_measure = [state.state_object for state in states_to_measure]
	# time_ii = time.time()
	measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
			stateObjs_to_measure)
	# print(len(stateObjs_to_measure), time.time() - time_ii)
	tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
	best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
	if len(best_states) != 0:
		best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
		update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (best_states[0], float(best_result)), best_reuse_results)
		if best_res_this_blk_shape:
			# if best_res_this_blk_shape is None, then we do not need to update it.
			update_reuse_result_dict(to_tune_idx, (best_states[0], float(best_result)), best_res_this_blk_shape)
	return best_states, tot_config_num






def tuneStageWithPartialOptimization(op_para, op_para_r, 
	search_tasks, loops, to_tune_idx, reuse_from_idx,
	state_reused_from, rconfig_tile_knobs_completeTile, 
	search_policy, tune_option,
	tot_config_num, best_reuse_results, 
	get_enu_knob_func, stages_to_tune):
	'''
		This function is for tuning the stages `datapack`, `inverse`, and `conv2d_winograd` after we already optimize the stage `bgemm`.
		The difference between this function and function `tuneEnuKnobsONLY` is that 
			1. this function would compute enu_knobs.
			2. this function is used outside of the blk_size iteration.
		INPUT:
			get_enu_knob_func: func. The function use to compute the needed enu_knobs.
			stages_to_tune: list of int. A list of stage id (loop id) in order this function needs to tune.
		OUTPUT:
			tot_config_num: int. Update the total number of configs which are measured and returned.
	'''
	to_tune_task = search_tasks[to_tune_idx]
	reuse_from_task = search_tasks[reuse_from_idx]
	dag_to_tune = to_tune_task.compute_dag
	to_tune_loop = loops[to_tune_idx]
	wlk_to_tune = to_tune_task.workload_key
	wlk_reuse_from = reuse_from_task.workload_key
	# 
	if op_para["op_type"] == "conv2d_winograd":
		knob_to_tune_list = [op_para['space_tile_knobs_set'][i][-1] for i in stages_to_tune]
	else:
		assert False, "Only support multi stages for conv2d_winograd now!"
	# 
	for loop_id_i in range(len(stages_to_tune)):
		loop_id = stages_to_tune[loop_id_i]
		# print(stages_to_tune, loop_id)
		# first prepare op_para and op_para_r infor for this loop nest
		set_stage_to_tune_for_MultiStages(op_para, loop_id)
		set_stage_to_tune_for_MultiStages(op_para_r, loop_id)
		# print(op_para)
		knob_to_tune = knob_to_tune_list[loop_id_i]
		# print(knob_to_tune)
		# 
		curr_best_knobs, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids = getTileInforFromTransformSteps(
			to_tune_task, dag_to_tune, to_tune_loop, best_reuse_results[(wlk_to_tune, wlk_reuse_from)][0], op_para, True)
		enu_knobs = get_enu_knob_func(rconfig_tile_knobs_completeTile, op_para, op_para_r, knob_to_tune)
		# print(enu_knobs)
		best_states, tot_config_num = tuneEnuKnobsONLY(to_tune_idx, op_para, dag_to_tune, state_reused_from,
			multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids,
			search_policy, tune_option,
			tot_config_num, wlk_to_tune, wlk_reuse_from, best_reuse_results, 
			None, 
			enu_knobs, curr_best_knobs, 
			)
	return tot_config_num




def gen_and_tune_hierarchy_nonfactor_decompose_3TuneRound_nonblk_allops_largeRange_ANSOR_V2(
	search_policy, tune_option,
	search_tasks, tasks, loops, to_tune_idx, reuse_from_idx, 
	best_measure_pairs, best_reuse_results, config_num_dict):
	'''
	This is the 11th version of fine tune function. In this version, we allow various op types (not take "conv2d" by default).
	WE ALSO ENLARGE THE VALUE RANGE OF THE POSSIBLE FEATURES, LIKE blk_sizes, thrd_size, #load
	We try to tune knobs in a hierarchical way. 
	AT THE SAME TIME, WE MEASURE THE GENERATED CONFIGS to get intermediate feedback for better search efficiency.
	In this version, we do not limit the delta changes on features to specific factors. Instead, we enumerate all possible sizes or nums or #loads within specific ranges.
	RMK: in this version, we use the default value of "explicit_unroll":1, and then enumerate 0 and 1 first, and then enumerate #load [for given (blk_size)]
	RMK: config_num_dict: dictionary stores the number of configs for an op reuse pair, e.g., config_num_dict[(19, 0)] = 10.
	RMK: This version is not constrained by the block relationship condition, and smaller ops can reuse larger ops.
	RMK: for Ansor, a measure pair has 2 parts: input:{task, state}, results:{costs, ...}
	'''
	def avg_group_num_covered(group_len, tot_len, unit_len):
		'''
			Compute the average number of groups covered by a gpu block. This is for group_conv_2d.
			INPUT:
				group_len: int. the length of a group in the out channel.
				tot_len: int. the total length, i.e., #out_channel.
				unit_len: int. the # of out channels a gpu block is assigned.
			OUTPUT:
				avg_groups: int.
		'''
		group_num = tot_len // group_len
		unit_num = tot_len // unit_len
		if group_len > unit_len:
			# in this case, a gpu block can cover <= 2 groups
			cross_group_num = 0
			for i in range(1, group_num):
				if ((group_len * i) % unit_len) != 0:
					cross_group_num += 1
			return (unit_num + cross_group_num) / unit_num
		else:
			# unit len >= group len. we index from 0 here when computing.
			cover_group_num = 0
			for i in range(unit_num):
				cover_group_num = cover_group_num +\
				(((unit_num + 1) * unit_len - 1) // group_len) - ((unit_num * unit_len) // group_len) + 1
			return cover_group_num / unit_num
	# 
	def cal_blk_size(knobs, op_para):
		'''Calculate the block size given the tiling knobs'''
		blk_size = 1
		for k in op_para['space_tile_knobs']:
			blk_size = blk_size * get_product(knobs[k][1:])
		return blk_size
	# 
	def cal_thrd_num(knobs, op_para):
		'''Calculate the thread number given the tiling knobs'''
		thrd_num = 1
		for k in op_para['space_tile_knobs']:
			thrd_num = thrd_num * knobs[k][2]
		return thrd_num
	# 
	def cal_blk_shape(knobs, op_para):
		''' Compute the block shape from the knobs'''
		blk_shape = None
		# block shape is [n, f, y, x] for conv2d
		blk_shape = [get_product(knobs[k][1:]) for k in op_para['space_tile_knobs']]
		return blk_shape
	# 
	def cal_reduc_shape(knobs, op_para):
		''' Compute the reduc shape from the knobs '''
		reduc_shape = None
		# reduc shape is [in_channel, ry, rx]
		reduc_shape = [get_product(knobs[k][1:]) for k in op_para['reduc_tile_knobs']]
		return reduc_shape
	# 
	def get_reduc_shape_same_nload(rconfig_load_num, op_para):
		''' 
			Compute the reduc shape which keeps the number of loads the same as in reuse_tile_knobs 
			INPUT:	op_para is the parameter dictionary of to_tune_op.
		'''
		reduc_shape = list()
		for i in op_para['reduc_iters']:
			if i == op_para['load_step_iter']:
				reduc_shape.append(op_para['loop'][i] / rconfig_load_num)
			else:
				reduc_shape.append(op_para['loop'][i])
		return reduc_shape
	# 
	def enu_reduc_knobs(enu_knobs, reduc_shape, op_para):
		# store the tile knob values for reducion axes in enu_knobs: 
		# 	reduc_shape is the shape of elements on the reduc axis we load each time (for example).
		# RMK: We would modify enu_knobs directly in this function.
		for i in range(len(op_para['reduc_iters'])):
			r_iter = op_para['reduc_iters'][i]
			enu_knobs[op_para['reduc_tile_knobs'][i]] = [[op_para['loop'][r_iter] // reduc_shape[i], reduc_shape[i]]]
			# if r_iter == op_para['load_step_iter']:
			# 	enu_knobs[op_para['reduc_tile_knobs'][i]] = [[op_para['loop'][r_iter] // reduc, reduc]]
			# else:
			# 	enu_knobs[op_para['reduc_tile_knobs'][i]] = [[1, op_para['loop'][r_iter]]]
	# 
	def cal_load_diff(dt_blk_size, op_para, reuse_from_loop, reuse_knobs):
		'''
			Compute the load amount increase compared with reuse_from_knobs.
			INPUT:
				dt_blk_size: int. The block size ratio of the current setting of the to_tune_op over the block size in the reuse_config.
				op_para: the to_tune_op parameter dictionary.
				reuse_from_loop: the loop extents of the reuse_from_op.
				reuse_knobs: the tiling knobs of the reuse_from_op.
			OUTPUT:
				load diff: int. The load amount increase.
		'''
		load_diff = dt_blk_size
		for i in range(len(op_para['reduc_iters'])):
			r_iter = op_para['reduc_iters'][i]
			if r_iter == op_para['load_step_iter']:
				load_diff = load_diff * op_para['loop'][r_iter] / reuse_from_loop[r_iter]
			else:
				knob_name = op_para['reduc_tile_knobs'][i]
				load_diff = load_diff * op_para['loop'][r_iter] / reuse_knobs[knob_name][1]
		return load_diff
	# 
	def cal_data_load_1blk_1time(blk_shape, reduc_shape, op_para, getSum = True):
		'''
			Compute the data a block needs to load each time from global mem to shared mem.
			INPUT:
				blk_shape, reduc_shape: list of ints. The shape of output space a block is responsible for; and reduction space a block needs.
				op_para: the operator parameter dictionary.
				getSum: if True, get the total data a blk load a time, otherwise, get the individual value for each input.
			OUTPUT:
				load amount: int. The amount of data a block needs to load each loading time.
		'''
		load_amount = None
		# blk_shape = cal_blk_shape(knobs, op_para['op_type'])
		# reduc_shape = cal_reduc_shape(knobs, op_para['op_type'])
		if op_para['op_type'] == "conv2d":
			# the block_shape is [n, f, y, x]; the reduc_shape is [in_channel, ry, rx]
			kamount = blk_shape[1] * get_product(reduc_shape)
			damount = blk_shape[0] * reduc_shape[0] *\
						((blk_shape[2] - 1) * op_para['stride'][0] + (reduc_shape[1] - 1) * op_para['dilation'][0] + 1) *\
						((blk_shape[3] - 1) * op_para['stride'][1] + (reduc_shape[2] - 1) * op_para['dilation'][1] + 1)
			load_amount = [kamount, damount]
		elif op_para['op_type'] == "group_conv2d":
			# the block_shape is [n, f, y, x]; the reduc_shape is [in_channel, ry, rx]
			kamount = blk_shape[1] * get_product(reduc_shape)
			damount = blk_shape[0] * (reduc_shape[0] *\
						avg_group_num_covered(op_para['num_filter'] // op_para['groups'], op_para['num_filter'], blk_shape[1])) *\
						((blk_shape[2] - 1) * op_para['stride'][0] + (reduc_shape[1] - 1) * op_para['dilation'][0] + 1) *\
						((blk_shape[3] - 1) * op_para['stride'][1] + (reduc_shape[2] - 1) * op_para['dilation'][1] + 1)
			load_amount = [kamount, damount]
		elif op_para['op_type'] == "depthwise_conv2d":
			# the block_shape is [n, f, y, x]; the reduc_shape is [ry, rx]
			kamount = blk_shape[1] * get_product(reduc_shape)
			damount = blk_shape[0] *\
						avg_group_num_covered(op_para['channel_multiplier'], op_para['in_channel'] * op_para['channel_multiplier'], blk_shape[1]) *\
						((blk_shape[2] - 1) * op_para['stride'][0] + (reduc_shape[0] - 1) * op_para['dilation'][0] + 1) *\
						((blk_shape[3] - 1) * op_para['stride'][1] + (reduc_shape[1] - 1) * op_para['dilation'][1] + 1)
			load_amount = [kamount, damount]
		elif op_para['op_type'] == "transpose_conv2d":
			# the block_shape is [n, f, y, x]; the reduc_shape is [in_channel, ry, rx]
			kamount = blk_shape[1] * get_product(reduc_shape)
			damount = blk_shape[0] * reduc_shape[0] *\
						(blk_shape[2] - 1  + reduc_shape[1]) *\
						(blk_shape[3] - 1  + reduc_shape[2])
			load_amount = [kamount, damount]			
		elif op_para['op_type'] == "conv2d_capsule":
			# the block_shape is [n, y, x, cap_i, cap_j, f]; the reduc_shape is [ry, rx, cap_k, in_channel (rc)]
			# we assume rc is the iter which is divided for shared mem loading
			kamount = blk_shape[-2] * blk_shape[-1] * get_product(reduc_shape)
			damount = blk_shape[0] *\
						((blk_shape[1] - 1) * op_para['stride'] + reduc_shape[0]) *\
						((blk_shape[2] - 1) * op_para['stride'] + reduc_shape[1]) *\
						blk_shape[3] * reduc_shape[2] * reduc_shape[3]
			load_amount = [kamount, damount]
		elif op_para['op_type'] == "conv1d":
			# the block_shape is [n, f, x]; the reduc_shape is [rc (in_channel), rx]
			kamount = blk_shape[1] * get_product(reduc_shape)
			damount = blk_shape[0] *reduc_shape[0] *\
						((blk_shape[2] - 1) * op_para['stride'][0] + (reduc_shape[1] - 1) * op_para['dilation'][0] + 1)
			load_amount = [kamount, damount]
		elif op_para['op_type'] == "conv3d":
			# the block_shape is [n, f, d, y, x]; the reduc_shape is [rc (in_channel), rd, ry, rx]
			kamount = blk_shape[1] * get_product(reduc_shape)
			damount = blk_shape[0] *reduc_shape[0] *\
						((blk_shape[2] - 1) * op_para['stride'][0] + (reduc_shape[1] - 1) * op_para['dilation'][0] + 1) *\
						((blk_shape[3] - 1) * op_para['stride'][1] + (reduc_shape[2] - 1) * op_para['dilation'][1] + 1) *\
						((blk_shape[4] - 1) * op_para['stride'][2] + (reduc_shape[3] - 1) * op_para['dilation'][2] + 1)
			load_amount = [kamount, damount]
		elif op_para['op_type'] == "batch_matmul":
			# the block_shape is [b, y, x]; the reduc_shape is [rk]
			X_amount = (blk_shape[0] if op_para['X_shape'][0] != 1 else 1) * blk_shape[1] * reduc_shape[0]
			Y_amount = (blk_shape[0] if op_para['Y_shape'][0] != 1 else 1) * blk_shape[2] * reduc_shape[0]
			load_amount = [Y_amount, X_amount]
		elif op_para['op_type'] == "conv2d_winograd":
			# the block shape is [eps, nu, p, co]; the reduc_shape is [ci]
			X_amount = get_product(blk_shape[:3]) * reduc_shape[0]
			Y_amount = get_product(blk_shape[:2]) * blk_shape[3] * reduc_shape[0]
			load_amount = [Y_amount, X_amount]
		if load_amount and getSum:
			return sum(load_amount)
		return load_amount
	# 
	def get_op_para(op, loop):
		'''
			Obtain the op parameter dictionary for the given op.
			INPUT:
				op: Task.
			OUTPUT:
				op_para: dictionary of parameter values.
		'''
		if isinstance(op, tvm.auto_scheduler.search_task.SearchTask):
			op_para = get_op_para_ansor(op, loop)
			return op_para
		# 
		op_para = None
		if op.workload[0] == 'conv2d_nchw.cuda':
			stride = op.args[2]
			dilation = op.args[4]
			kh = op.args[1][1][-2]
			kw = op.args[1][1][-1]
			op_para = dict()
			op_para['kh'] = kh 
			op_para['kw'] = kw 
			op_para['stride'] = stride
			op_para['dilation'] = dilation
			op_para['op_type'] = "conv2d"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2, 3]
			op_para['reduc_iters'] = [4, 5, 6]
			op_para['load_step_iter'] = 4 # the axis divided for loading input data
			op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
			op_para['load_step_knob'] = 'tile_rc'
		elif op.workload[0] == 'group_conv2d_nchw.cuda':
			stride = op.args[2]
			dilation = op.args[4]
			groups = op.args[5]
			_, in_channel, _, _ = op.args[0][1]
			num_filter, _, kh, kw = op.args[1][1]
			op_para = dict()
			op_para['kh'] = kh
			op_para['kw'] = kw
			op_para['stride'] = stride
			op_para['dilation'] = dilation
			op_para['groups'] = groups
			op_para['in_channel'] = in_channel
			op_para['num_filter'] = num_filter
			op_para['op_type'] = "group_conv2d"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2, 3]
			op_para['reduc_iters'] = [4, 5, 6]
			op_para['load_step_iter'] = 4 # the axis divided for loading input data
			op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
			op_para['load_step_knob'] = 'tile_rc'
			# knob1 in space knobs (i.e., tile_f) must cover k groups in blk shape
			op_para['tile_by_group_knob_idx'] = 1 
		elif op.workload[0] == 'depthwise_conv2d_nchw.cuda':
			stride = op.args[2]
			dilation = op.args[4]
			_, in_channel, _, _ = op.args[0][1]
			_, channel_multiplier, kh, kw = op.args[1][1]
			op_para = dict()
			op_para['kh'] = kh 
			op_para['kw'] = kw 
			op_para['stride'] = stride
			op_para['dilation'] = dilation
			op_para['in_channel'] = in_channel
			op_para['channel_multiplier'] = channel_multiplier
			op_para['op_type'] = "depthwise_conv2d"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2, 3]
			op_para['reduc_iters'] = [4, 5]
			op_para['load_step_iter'] = None # the axis divided for loading input data
			op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_ry', 'tile_rx']
			op_para['load_step_knob'] = None
		elif op.workload[0] == 'conv2d_transpose_nchw.cuda':
			_, _, kh, kw = op.args[1][1]
			op_para = dict()
			op_para['kh'] = kh 
			op_para['kw'] = kw 
			op_para['op_type'] = "transpose_conv2d"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2, 3]
			op_para['reduc_iters'] = [4, 5, 6]
			op_para['load_step_iter'] = 4 # the axis divided for loading input data
			op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_y', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_ry', 'tile_rx']
			op_para['load_step_knob'] = 'tile_rc'			
		elif op.workload[0] == 'conv2d_capsule_nhwijc':
			# this op has not topi topi implementation, so we cannot extract autoTVM tasks
			op_para = dict()
			op_para['kh'] = None 
			op_para['kw'] = None 
			op_para['stride'] = None
			op_para['op_type'] = "conv2d_capsule"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2, 3, 4, 5]
			op_para['reduc_iters'] = [6, 7, 8, 9]
			op_para['load_step_iter'] = 9 # the axis divided for loading input data
			op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x', 
											'tile_capi', 'tile_capj', 'tile_f']
			op_para['reduc_tile_knobs'] = ['tile_ry', 'tile_rx', 'tile_rcapk','tile_rc']
			op_para['load_step_knob'] = 'tile_rc'
		elif op.workload[0] == 'conv1d_ncw.cuda':
			# we get op parameter of conv1d from ansor search task (not autotvm task)
			stride = op.args[2]
			dilation = op.args[4]
			_, _, kw = op.args[1][1]
			op_para = dict()
			op_para['kw'] = kw
			op_para['stride'] = stride
			op_para['dilation'] = dilation
			op_para['op_type'] = "conv1d"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2]
			op_para['reduc_iters'] = [3, 4]
			op_para['load_step_iter'] = 3
			op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_rx']
			op_para['load_step_knob'] = 'tile_rc'
		elif op.workload[0] == 'conv3d_ncdhw.cuda':
			stride = op.args[2]
			dilation = op.args[4]
			_, _, kd, kh, kw = op.args[1][1]
			op_para = dict()
			op_para['kd'] = kd
			op_para['kh'] = kh
			op_para['kw'] = kw
			op_para['stride'] = stride
			op_para['dilation'] = dilation
			op_para['op_type'] = "conv3d"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2, 3, 4]
			op_para['reduc_iters'] = [5, 6, 7, 8]
			op_para['load_step_iter'] = 5
			op_para['space_tile_knobs'] = ['tile_n', 'tile_f', 'tile_d', 'tile_y', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_rc', 'tile_rd', 'tile_ry', 'tile_rx']
			op_para['load_step_knob'] = 'tile_rc'
		elif op.workload[0] == 'batch_matmul.cuda':
			# output shape is [batch, M, N], iteration space is b, y, x, k.
			op_para = dict()
			op_para['X_shape'] = list(op.args[0][1]) # NOT SURE ABOUT THIS PART, NEED CHECK
			op_para['Y_shape'] = list(op.args[1][1])
			op_para['op_type'] = "batch_matmul"
			op_para['loop'] = loop
			op_para['space_iters'] = [0, 1, 2]
			op_para['reduc_iters'] = [3]
			op_para['load_step_iter'] = 3
			op_para['space_tile_knobs'] = ['tile_n', 'tile_y', 'tile_x']
			op_para['reduc_tile_knobs'] = ['tile_k']
			op_para['load_step_knob'] = 'tile_k'
		# print("OOOOOOOP_PARA: ", op_para)
		return op_para
	# 
	# def get_reuse_knobs(reuse_config):
	# 	reuse_knobs = dict()
	# 	for k,v in reuse_config._entity_map.items():
	# 		if isinstance(v, SplitEntity):
	# 			# get the init tile size for this knob
	# 			new_tile_size = copy.deepcopy( v.size )
	# 			reuse_knobs[k] = new_tile_size
	# 	return reuse_knobs
	# 
	def init_tuned_knobs(rconfig_tile_knobs):
		tuned_knobs = dict()
		for knob in rconfig_tile_knobs.keys():
			tuned_knobs[knob] = list()
		return tuned_knobs
	# 
	# print("THIS IS NEW FUCNTION!")
	EPSILON = 0.000001 
	fact_dict_refer = dict()
	# time_jj_list = list()
	tot_config_num = 0 # stores the total number of configurations measured for this op reuse pair
	to_tune_op = tasks[to_tune_idx] 
	reuse_from_op = tasks[reuse_from_idx] 
	to_tune_loop = loops[to_tune_idx] 
	reuse_from_loop = loops[reuse_from_idx]
	dag_to_tune = search_tasks[to_tune_idx].compute_dag
	dag_reuse_from = search_tasks[reuse_from_idx].compute_dag
	wlk_to_tune =  search_tasks[to_tune_idx].workload_key
	wlk_reuse_from = search_tasks[reuse_from_idx].workload_key
	state_reused_from = best_measure_pairs[wlk_reuse_from][0]
	# 
	# get some parameters needed in measuring
	to_tune_baseline = 0
	if wlk_to_tune in best_measure_pairs.keys():
		to_tune_baseline = best_measure_pairs[wlk_to_tune][1]
	reuse_from_baseline = best_measure_pairs[wlk_reuse_from][1]
	prefix = "[Task %2d Gflops: %f] reuse [Task %2d Gflops: %f]" % (to_tune_idx, to_tune_baseline, reuse_from_idx, reuse_from_baseline)
	# 
	blk_topk, thrd_topk = 1, 1
	# features = list(["nblock", "nthread"]) # the first kind of features we would iterate over whose whole possible values sets
	# 
	# prepare parameter dictionary
	op_para = get_op_para(to_tune_op, to_tune_loop) # the op_para for to_tune_op
	op_para_reuse = get_op_para(reuse_from_op, reuse_from_loop) # the op_para for reuse_from_op
	# below is to get all possible tuned knobs
	# output_size = get_product(to_tune_loop[0:4])
	# load_size = to_tune_loop[4]
	# rconfig_tile_knobs = get_reuse_knobs(reuse_config)
	# get the reused tile knobs from the reused state
	rconfig_tile_knobs, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids = \
				getTileInforFromTransformSteps(search_tasks[reuse_from_idx], dag_reuse_from, reuse_from_loop, state_reused_from, op_para_reuse)
	# get the complete tile infor for unroll
	rconfig_tile_knobs_completeTile, _, _, _ = \
				getTileInforFromTransformSteps(search_tasks[reuse_from_idx], dag_reuse_from, reuse_from_loop, state_reused_from, op_para_reuse, remain_same = True)
	# print(rconfig_tile_knobs_completeTile)
	# ******************************************************************************************************************************************
	default_knobs = None
	# we first deal with the tuning workflow of winograd conv2d
	if op_para['op_type'] == 'conv2d_winograd':
		default_knobs = get_default_knobs_conv2d_winograd(op_para) # get the default knobs first
		set_stage_to_tune_for_MultiStages(op_para, 1) # we tune stage 1 `bgemm` first, with other stages using default knobs.
		set_stage_to_tune_for_MultiStages(op_para_reuse, 1) # op_para_reuse also needs to be updated
		# update some variables
		to_tune_loop = op_para['loop']
		reuse_from_loop = op_para_reuse['loop']
	#
	inc_diff = get_product(to_tune_loop) / get_product(reuse_from_loop) 
	# here we deal with the special op which does not need multi-level tiling
	if op_para['op_type'] == 'frobenius_norm':
		tile_sizes = tune_frobenius_norm(rconfig_tile_knobs, op_para, op_para_reuse)
		# print("="*50)
		# print(prefix)
		# print("="*50)
		# print(tile_sizes, multi_split_step_ids, vector_split_step_ids)
		states_to_measure = auto_scheduler._ffi_api.MyGetStatesFromTunedKnobs(
			dag_to_tune, state_reused_from,
			tile_sizes, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids)
		states_to_measure = [dag_to_tune.infer_bound_from_state(state) for state in states_to_measure]
		stateObjs_to_measure = [state.state_object for state in states_to_measure]
		measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
				stateObjs_to_measure)
		tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
		best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
		if len(best_states) == 0:
			# there is no valid state in this measure round
			assert False # This case should not happen
		best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
		update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (best_states[0], float(best_result)), best_reuse_results)
		update_config_num_dict(config_num_dict, tot_config_num, (to_tune_idx, reuse_from_idx))
		return
	# 
	output_size = get_product([op_para['loop'][i] for i in op_para['space_iters']])
	load_size = op_para['loop'][op_para['load_step_iter']] if op_para['load_step_iter'] != None else None
	# WE NEED TO DEAL WITH MULTI_SKETCHES HERE
	if 'tile_fusedR' in rconfig_tile_knobs.keys():
		rconfig_thrd_num = rconfig_tile_knobs['tile_fusedR'][1]
		dt_output_space = output_size / get_product([reuse_from_loop[i] for i in op_para_reuse['space_iters']])
		dt_reduc_space = inc_diff / dt_output_space
		reduc_space_size = get_product([op_para['loop'][i] for i in op_para['reduc_iters']])
		# get all possible thread nums to be checked
		# to avoid compute result store error, we add 1 or minus 1 here
		# for other ops than frobenius_norm, the ruleCrossThrdReduction does not have fuse step
		if get_product(op_para['loop']) > get_product(op_para_reuse['loop']):
			thrd_sizes = all_dividers(rconfig_thrd_num / dt_output_space - 1, rconfig_thrd_num * dt_reduc_space + 1, reduc_space_size, fact_dict_refer)
			thrd_sizes.sort()
		else:
			thrd_sizes = all_dividers(rconfig_thrd_num / dt_output_space + 1, rconfig_thrd_num * dt_reduc_space - 1, reduc_space_size, fact_dict_refer)
			thrd_sizes.sort( reverse = True )
			# print("thrd_sizes: ", thrd_sizes)
		thrd_sizes = getValidThrdNumPerBlk(thrd_sizes)
		# 
		enu_knobs = dict()
		if len(thrd_sizes) == 0: 
			# this case should not happen
			assert False
		# rconfig_tile_knobs stores a list of knobs, although the list len is 1
		for k in rconfig_tile_knobs:
			rconfig_tile_knobs[k] = [rconfig_tile_knobs[k]]
		# 
		enu_knobs['tile_fusedR'] = [[reduc_space_size//thrd_size, thrd_size] for thrd_size in thrd_sizes]
		# print("BEFORE SUCCESS")
		# add support for multi stages ops like conv2d_winograd
		if default_knobs:
			tuned_knobs = change_configs(rconfig_tile_knobs, default_knobs)
			tuned_knobs = change_configs(tuned_knobs, enu_knobs)
		else:
			tuned_knobs = change_configs(rconfig_tile_knobs, enu_knobs)
		# print("IN SUCCESS")
		# print(tuned_knobs)
		# 
		# print("AFTER SUCCESS")
		tile_sizes = tileKnobsToTransformSteps(op_para, tuned_knobs, True)
		# print("="*50)
		# print(prefix)
		# print("="*50)
		states_to_measure = auto_scheduler._ffi_api.MyGetStatesFromTunedKnobs(
			dag_to_tune, state_reused_from,
			tile_sizes, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids)
		states_to_measure = [dag_to_tune.infer_bound_from_state(state) for state in states_to_measure]
		stateObjs_to_measure = [state.state_object for state in states_to_measure]
		measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
				stateObjs_to_measure)
		tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
		best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
		if len(best_states) == 0:
			# there is no valid state in this measure round
			assert False # This case should not happen
		best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
		update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (best_states[0], float(best_result)), best_reuse_results)
		if op_para["op_type"] == "conv2d_winograd":
			tot_config_num = tuneStageWithPartialOptimization(op_para, op_para_reuse, 
				search_tasks, loops, to_tune_idx, reuse_from_idx,
				state_reused_from, rconfig_tile_knobs_completeTile, 
				search_policy, tune_option,
				tot_config_num, best_reuse_results, 
				tune_ConstTensor_or_2LevelTile_sketch, [0, 2, 3])
		update_config_num_dict(config_num_dict, tot_config_num, (to_tune_idx, reuse_from_idx))
		return
	# in the case below, the reused op must be MULTI_LEVEL tiled=====================================================================================
	rconfig_blk_size = cal_blk_size(rconfig_tile_knobs, op_para_reuse)
	rconfig_thrd_num = cal_thrd_num(rconfig_tile_knobs, op_para_reuse)
	if op_para['load_step_iter'] != None:
		rconfig_load_num = reuse_from_loop[op_para_reuse['load_step_iter']] // rconfig_tile_knobs[op_para_reuse['load_step_knob']][1] 
	else:
		rconfig_load_num = None
	# compute the amount of data a block needs to load each loading time
	rconfig_blk_shape = cal_blk_shape(rconfig_tile_knobs, op_para_reuse)
	rconfig_reduc_shape = cal_reduc_shape(rconfig_tile_knobs, op_para_reuse)
	rconfig_data_load_amount = cal_data_load_1blk_1time(rconfig_blk_shape, rconfig_reduc_shape, op_para_reuse)
	# print(rconfig_data_load_amount, rconfig_blk_shape, rconfig_reduc_shape)
	# print(rconfig_blk_size, rconfig_thrd_num, rconfig_load_num)
	# 
	# print("output size ratio : ", output_size / get_product([reuse_from_loop[i] for i in op_para_reuse['space_iters']]))
	# get reduc shape of to_tune_op which keeps #load the same as reuse_config
	reduc_shape_same_nload = get_reduc_shape_same_nload(rconfig_load_num, op_para)
	# get all possible block sizes to be checked
	dt_output_space = output_size / get_product([reuse_from_loop[i] for i in op_para_reuse['space_iters']])
	dt_reduc_space = inc_diff / dt_output_space
	if get_product(to_tune_loop) > get_product(reuse_from_loop):
		if (dt_reduc_space >= 1 - EPSILON) and (dt_reduc_space <= 1 + EPSILON):
			blk_sizes = all_dividers(rconfig_blk_size / dt_reduc_space - 1, rconfig_blk_size * inc_diff + 1, output_size, fact_dict_refer)
			# we force the method to find one extra blk size beyond the lower bound if there is no reduc space diff
			extra_blk_sizes = all_dividers(rconfig_blk_size / dt_reduc_space - 2, 1, output_size, fact_dict_refer)
			if len(extra_blk_sizes) > 0:
				blk_sizes.append(max(extra_blk_sizes))
		else:
			blk_sizes = all_dividers(rconfig_blk_size / dt_reduc_space - 1, rconfig_blk_size * inc_diff + 1, output_size, fact_dict_refer)
		blk_sizes.sort( reverse = True )
	else:
		if (dt_reduc_space >= 1 - EPSILON) and (dt_reduc_space <= 1 + EPSILON):
			blk_sizes = all_dividers(rconfig_blk_size * inc_diff - 1, rconfig_blk_size / dt_reduc_space + 1, output_size, fact_dict_refer)
			# we force the method to find one extra blk size beyond the lower bound if there is no reduc space diff
			if len(blk_sizes) > 0:
				extra_blk_sizes = all_dividers(max(blk_sizes) + 1, output_size, output_size, fact_dict_refer)
			else:
				extra_blk_sizes = all_dividers(rconfig_blk_size / dt_reduc_space + 2, output_size, output_size, fact_dict_refer)
			if len(extra_blk_sizes) > 0:
				blk_sizes.append(min(extra_blk_sizes))
		else:
			blk_sizes = all_dividers(rconfig_blk_size * inc_diff - 1, rconfig_blk_size / dt_reduc_space + 1, output_size, fact_dict_refer)
		blk_sizes.sort()
	# 
	# print("rconfig_tile_knobs: ", rconfig_tile_knobs, "rconfig_blk_size: ", rconfig_blk_size, "inc_diff: ", inc_diff)
	# print("blk_sizes: ", blk_sizes)
	for blk_size in blk_sizes:
		# prepare container to store tuned knobs
		# tuned_knobs = init_tuned_knobs(reuse_config)
		dt_blk_size = blk_size / rconfig_blk_size
		# print("dt_blk_size: ", dt_blk_size)
		# get the best block shapes, each of which is list([ns, fs, ys, xs])
		blk_shapes = get_best_blk_shapes(blk_size, op_para, cal_32B_trans, blk_topk, simplify = True)
		# TODO: can add a cost model to choose the best vectorization (maybe no need, because the data threads load are consecutive)
		# Update: we deal with vectorization in the end
		# print("blk_shapes: ", blk_shapes)
		# get all possible thread nums to be checked
		# to avoid compute result store error, we add 1 here
		if inc_diff >= 1:
			thrd_sizes = all_dividers(rconfig_thrd_num / (dt_output_space / dt_blk_size) - 1, rconfig_thrd_num * inc_diff / (dt_output_space / dt_blk_size) + 1, blk_size, fact_dict_refer)
			thrd_sizes.sort()
		else:
			thrd_sizes = all_dividers(rconfig_thrd_num / (dt_output_space / dt_blk_size) + 1, rconfig_thrd_num * inc_diff / (dt_output_space / dt_blk_size) - 1, blk_size, fact_dict_refer)
			thrd_sizes.sort( reverse = True )
		# print(thrd_sizes, rconfig_thrd_num, (dt_output_space / dt_blk_size), inc_diff / (dt_output_space / dt_blk_size), blk_size)
		thrd_sizes = getValidThrdNumPerBlk(thrd_sizes)
		# print("thrd_sizes: ", thrd_sizes)
		for blk_shape in blk_shapes:
			# print("\n\n\n\n\n\nWE ARE HERE", blk_shape)
			best_res_this_blk_shape = dict()
			tuned_knobs = init_tuned_knobs(rconfig_tile_knobs)
			# get all possible load nums to be checked
			# load_diff = cal_load_diff(dt_blk_size, op_para, reuse_from_loop, rconfig_tile_knobs)
			load_diff = cal_data_load_1blk_1time(blk_shape, reduc_shape_same_nload, op_para) / rconfig_data_load_amount
			# print("load_diff: ", load_diff, cal_data_load_1blk_1time(blk_shape, reduc_shape_same_nload, op_para), rconfig_data_load_amount)
			# to avoid compute result store error, we add 1 here
			ld_n = None
			if op_para['load_step_iter'] != None:
				if load_diff < 1:
					load_nums = all_dividers(rconfig_load_num, \
						rconfig_load_num * load_diff - EPSILON, \
						load_size, fact_dict_refer)
					if len(load_nums) == 0:
						continue
					ld_n = min(load_nums)
					# ld_n = get_closest_value(load_nums, rconfig_load_num * load_diff - 1)
				else:
					load_nums = all_dividers(rconfig_load_num, \
						rconfig_load_num * load_diff + EPSILON, \
						load_size, fact_dict_refer)
					if len(load_nums) == 0:
						continue
					ld_n = max(load_nums)
					# ld_n = get_closest_value(load_nums, rconfig_load_num * load_diff + 1)
				# 
				# if inc_diff < 1:
				# 	load_nums = all_dividers_newRangeDef(rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) / inc_diff + 1,\
				# 		rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) - 1, \
				# 		load_size)
				# 	if len(load_nums) == 0:
				# 		continue
				# 	ld_n = get_closest_value(load_nums, rconfig_load_num * load_diff)
				# else:
				# 	load_nums = all_dividers_newRangeDef(rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) / inc_diff - 1,\
				# 		rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) + 1, \
				# 		load_size)
				# 	if len(load_nums) == 0:
				# 		continue
				# 	ld_n = get_closest_value(load_nums, rconfig_load_num * load_diff)
				# print("load_nums: ", load_nums, "default_ld_n: ", ld_n, "load_diff: ", load_diff, "rconfig_load_num: ", rconfig_load_num, "load_size: ", load_size)
			# 
			for thrd_size in thrd_sizes:
			# for blk_shape in blk_shapes:
				# first measure the default load num
				# print(blk_shape, blk_size, thrd_size, (load_size / ld_n) if ld_n != None else None)
				# reduc = load_size // ld_n
				# if reduc == 0: # this case means that ld_n is much larger than the original reduc
				# 	reduc = 1
				reduc_shape = [int(i) for i in get_reduc_shape_same_nload(ld_n, op_para)]
				tile_shapes = get_best_thrd_vthrd_shapes([blk_shape], thrd_size, reduc_shape, op_para, cal_bk_cflct, thrd_topk)
				enu_knobs = dict()
				enu_reduc_knobs(enu_knobs, reduc_shape, op_para)
				# print("tile_shapes:::")
				# print(tile_shapes)
				# enu_knobs['tile_rc'] = [[to_tune_loop[4] // reduc, reduc]]
				# enu_knobs['tile_ry'] = [[1, to_tune_loop[5]]]
				# enu_knobs['tile_rx'] = [[1, to_tune_loop[6]]]
				# TODO we currently do not tune "max_unroll_steps" here
				# enu_knobs['unroll_explicit'] = [reuse_config._entity_map['unroll_explicit'].val, ] # the default value of "unroll_explicit" is set to 1
				tile_shapes_2_knobs(tile_shapes, enu_knobs, op_para, tuned_knobs)
			# gen config immediately, and measure them
			if ld_n:
				setAutoUnrollInTileKnobs(op_para, tuned_knobs, op_para_reuse, rconfig_tile_knobs_completeTile, scaleRegNum = (rconfig_load_num / ld_n) * load_diff)
			else:
				setAutoUnrollInTileKnobs(op_para, tuned_knobs, op_para_reuse, rconfig_tile_knobs_completeTile, scaleRegNum = 1)
			# fill the knobs in other stages with default knob values
			if default_knobs:
				# print("\ntuned_knobs:\n", tuned_knobs)
				# rconfig_tile_knobs stores a list of knobs, although the list len is 1
				rconfig_tile_knobs_completeTile_tmp = dict()
				for k in rconfig_tile_knobs_completeTile:
					rconfig_tile_knobs_completeTile_tmp[k] = [rconfig_tile_knobs_completeTile[k]]
				tuned_knobs_base = change_configs(rconfig_tile_knobs_completeTile_tmp, default_knobs)
				# print("\ntuned_knobs_baset\n", tuned_knobs_base)
				delete_empty_knobs(tuned_knobs)
				if len(tuned_knobs) == 0:
					# no configuration to measure
					continue
				tuned_knobs = substitute_configs(tuned_knobs_base, tuned_knobs)
				# set default vectorization for the stage being tuned, which is no vectorization.
				tuned_knobs["vector_0"] = list()
				tuned_knobs["vector_1"] = list()
				# print("\ntuned_knobs\n", tuned_knobs)
			# 
			# convert tile knobs to Ansor transform steps
			tile_sizes = tileKnobsToTransformSteps(op_para, tuned_knobs, True)
			# for ii in tile_sizes:
			# 	print(ii)
			if len(tile_sizes) == 0:
				# no configuration to measure
				continue
			# best_result = list() # stores the best flops during this measure round
			# print("BEFORE WE MEASURE!")
			# 
			# print("="*50)
			# print(prefix)
			# print("="*50)
			# print(tile_sizes)
			# print(multi_split_step_ids) 
			# print(vector_split_step_ids)
			# print(auto_unroll_step_ids)
			# print(dag_to_tune)
			states_to_measure = auto_scheduler._ffi_api.MyGetStatesFromTunedKnobs(
				dag_to_tune, state_reused_from,
				tile_sizes, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids)
			# print("states_to_measure: ", states_to_measure)
			states_to_measure = [dag_to_tune.infer_bound_from_state(state) for state in states_to_measure]
			stateObjs_to_measure = [state.state_object for state in states_to_measure]
			# time_ii = time.time()
			measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
					stateObjs_to_measure)
			# time_jj_list.append(time.time() - time_ii)
			tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
			best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
			if len(best_states) == 0:
				# there is no valid state in this measure round
				# print("NO BEST RESULTS!", best_states)
				continue
			best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
			update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (best_states[0], float(best_result)), best_reuse_results)
			# print("\n\n\n\nworkload_key pair: ", wlk_to_tune, wlk_reuse_from)
			# assert (wlk_to_tune, wlk_reuse_from) in best_reuse_results.keys(), "VERY STRANGE BUG!"
			update_reuse_result_dict(to_tune_idx, (best_states[0], float(best_result)), best_res_this_blk_shape)
			# 
			# WE NEXT ENUMERATE tile_rc
			if op_para['load_step_iter'] != None:
				# NO NEED TO ENUMERATE TILE_RC if op_para['load_step_iter'] == None
				# continue
				# print("TUNING ld_n!!!!!")
				curr_best_knobs, _, _, _ = getTileInforFromTransformSteps(
					search_tasks[to_tune_idx], dag_to_tune, to_tune_loop, best_res_this_blk_shape[to_tune_idx][0], op_para, True)
				enu_knobs = dict()
				# curr_best_knobs stores a list of knobs, although the list len is 1
				for k in curr_best_knobs:
					curr_best_knobs[k] = [curr_best_knobs[k]]
				# 
				default_ld_n = curr_best_knobs[op_para['load_step_knob']][0][0] # get the ld_n in the curr_best_knobs
				# load_diff_larger = cal_load_diff(dt_blk_size, op_para, reuse_from_loop, rconfig_tile_knobs)
				if inc_diff >= 1:
					enu_knobs[op_para['load_step_knob']] = [[ld_n, (load_size // ld_n) // get_min_factor(load_size // ld_n), get_min_factor(load_size // ld_n)] \
					for ld_n in all_dividers(rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) / inc_diff - 1,\
							rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) + 1, \
							load_size, fact_dict_refer) if (ld_n != default_ld_n)]
				else:
					enu_knobs[op_para['load_step_knob']] = [[ld_n, (load_size // ld_n) // get_min_factor(load_size // ld_n), get_min_factor(load_size // ld_n)] \
					for ld_n in all_dividers(rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) / inc_diff + 1,\
							rconfig_load_num * load_diff * (dt_output_space / dt_blk_size) - 1, \
							load_size, fact_dict_refer) if (ld_n != default_ld_n)]				
					# all_dividers(rconfig_load_num, rconfig_load_num * load_diff_larger, load_size) if (ld_n != default_ld_n)]
				# enu_knobs['unroll_explicit'] = [best_unroll_explicit, ]
				if len(enu_knobs[op_para['load_step_knob']]) != 0: 
					# continue
					# print("BEFORE SUCCESS")
					tuned_knobs = change_configs(curr_best_knobs, enu_knobs)
					load_diffs = list()
					for ld_tile in enu_knobs[op_para['load_step_knob']]:
						load_diffs.append((rconfig_load_num / ld_tile[0]) * load_diff)
					setAutoUnrollInTileKnobs(op_para, tuned_knobs, op_para_reuse, rconfig_tile_knobs_completeTile, scaleRegNum = load_diffs, NeedFurtherTile = False)
					# print("IN SUCCESS")
					# print(tuned_knobs)
					# 
					# print("AFTER SUCCESS")
					tile_sizes = tileKnobsToTransformSteps(op_para, tuned_knobs, True)
					# print("="*50)
					# print(prefix)
					# print("="*50)
					# for kk in tile_sizes:
					# 	print(kk)
					states_to_measure = auto_scheduler._ffi_api.MyGetStatesFromTunedKnobs(
						dag_to_tune, state_reused_from,
						tile_sizes, multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids)
					states_to_measure = [dag_to_tune.infer_bound_from_state(state) for state in states_to_measure]
					stateObjs_to_measure = [state.state_object for state in states_to_measure]
					# time_ii = time.time()
					measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
							stateObjs_to_measure)
					# time_jj_list.append(time.time() - time_ii)
					tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
					best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
					# if len(best_states) == 0:
					# 	# there is no valid state in this measure round
					# 	continue
					if len(best_states) != 0:
						best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
						update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (best_states[0], float(best_result)), best_reuse_results)
						update_reuse_result_dict(to_tune_idx, (best_states[0], float(best_result)), best_res_this_blk_shape)
			# 
			# we need tune unroll next
			# print("TUNING unroll!!!!!")
			curr_best_knobs, _, _, _ = getTileInforFromTransformSteps(
				search_tasks[to_tune_idx], dag_to_tune, to_tune_loop, best_res_this_blk_shape[to_tune_idx][0], op_para, True)
			auto_unroll_max_step_knob = get_auto_unroll_max_step_knob_name(op_para)
			enu_knobs = {auto_unroll_max_step_knob : \
				getDifferentAutoUnroll(curr_best_knobs, op_para, curr_best_knobs[auto_unroll_max_step_knob], 
					max_unroll_options = [0, 16, 64, 512, 1024])}
				# [unroll_v for unroll_v in [0, 16, 64, 512, 1024] if unroll_v != curr_best_knobs["auto_unroll_max_step"]]}
			# print(enu_knobs)
			best_states, tot_config_num = tuneEnuKnobsONLY(to_tune_idx, op_para, dag_to_tune, state_reused_from,
				multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids,
				search_policy, tune_option,
				tot_config_num, wlk_to_tune, wlk_reuse_from, best_reuse_results, 
				best_res_this_blk_shape, 
				enu_knobs, curr_best_knobs, 
				)
			# if len(best_states) == 0:
			# 	continue
			# we next tune vectorization
			curr_best_knobs, _, _, _ = getTileInforFromTransformSteps(
				search_tasks[to_tune_idx], dag_to_tune, to_tune_loop, best_res_this_blk_shape[to_tune_idx][0], op_para, True)
			tot_data_list = cal_data_load_1blk_1time(blk_shape, cal_reduc_shape(curr_best_knobs, op_para), op_para, getSum = False)
			# 
			for tot_data_i in range(len(tot_data_list)):
				# print("TUNING vectorization %d !!!!!" % tot_data_i)
				enu_knobs = {"vector_" + str(tot_data_i) : \
					getPossibleVectorize(tot_data_list[tot_data_i], cal_thrd_num(curr_best_knobs, op_para), vectorize_options = [2, 4])}
				# 
				# print(enu_knobs)
				best_states, tot_config_num = tuneEnuKnobsONLY(to_tune_idx, op_para, dag_to_tune, state_reused_from,
					multi_split_step_ids, vector_split_step_ids, auto_unroll_step_ids,
					search_policy, tune_option,
					tot_config_num, wlk_to_tune, wlk_reuse_from, best_reuse_results, 
					best_res_this_blk_shape, 
					enu_knobs, curr_best_knobs, 
					)	
				if (tot_data_i == len(tot_data_list) - 1):
					break	
				curr_best_knobs, _, _, _ = getTileInforFromTransformSteps(
					search_tasks[to_tune_idx], dag_to_tune, to_tune_loop, best_res_this_blk_shape[to_tune_idx][0], op_para, True)
				tot_data_list = cal_data_load_1blk_1time(blk_shape, cal_reduc_shape(curr_best_knobs, op_para), op_para, getSum = False)		
			# print(wlk_to_tune, wlk_reuse_from)
			# assert (wlk_to_tune, wlk_reuse_from) in best_reuse_results.keys(), "VERY STRANGE BUG! BEFORE END OF BGEMM STAGE"			
	# 
	# print(wlk_to_tune, wlk_reuse_from)
	# assert (wlk_to_tune, wlk_reuse_from) in best_reuse_results.keys(), "VERY STRANGE BUG! AT END OF BGEMM STAGE"
	if op_para["op_type"] == "conv2d_winograd":
		tot_config_num = tuneStageWithPartialOptimization(op_para, op_para_reuse, 
			search_tasks, loops, to_tune_idx, reuse_from_idx,
			state_reused_from, rconfig_tile_knobs_completeTile, 
			search_policy, tune_option,
			tot_config_num, best_reuse_results, 
			tune_ConstTensor_or_2LevelTile_sketch, [0, 2, 3])
	update_config_num_dict(config_num_dict, tot_config_num, (to_tune_idx, reuse_from_idx))			
	# print(time_jj_list)


