
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



def get_corr_all_infor_ST(all_infor_ST_list, search_tasks_list):
	wlk_list = list()
	for search_tasks in search_tasks_list:
		for task in search_tasks:
			wlk_list.append(task.compute_dag.workload_key())
	for all_infor_ST in all_infor_ST_list:
		if set(all_infor_ST.keys()) == set(wlk_list):
			return all_infor_ST
	assert False 





# get the file containing full tuning logs for each task
def get_full_log_file_name_mapping(all_infor_ST_list, best_measure_pairs_all_unselected, all_search_tasks_list_dict):
	wlkAnsor2FullLogFile = dict()
	for model_name in all_search_tasks_list_dict.keys():
		log_file_ST_ori_test_num = 1
		if model_name == "resnet50_batch1":
			log_file_ST_ori_test_num = 3
		log_file_ST_ori = "allops_test_ST_"+model_name+"_Test"+str(log_file_ST_ori_test_num) + ".json"
		# get the corresponding all_infor_ST
		all_infor_ST = get_corr_all_infor_ST(all_infor_ST_list, all_search_tasks_list_dict[model_name])
		for k in all_infor_ST:
			if all_infor_ST[k]['wlk_r'] == None:
				if k not in wlkAnsor2FullLogFile:
					wlkAnsor2FullLogFile[k] = list()
				wlkAnsor2FullLogFile[k].append(log_file_ST_ori)
	for model_name in all_search_tasks_list_dict.keys():
		# get all tasks
		tasks = list()
		for search_tasks in all_search_tasks_list_dict[model_name]:
			tasks = tasks + search_tasks
		# add full_log_file_name if there is no such infor
		for task in tasks:
			wlk_ansor = task.compute_dag.workload_key()
			if (wlk_ansor not in wlkAnsor2FullLogFile) or (len(wlkAnsor2FullLogFile[wlk_ansor]) == 0):
				if wlk_ansor not in wlkAnsor2FullLogFile:
					wlkAnsor2FullLogFile[wlk_ansor] = list()
				wlkAnsor2FullLogFile[wlk_ansor].append("FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test1.json")
				assert task.workload_key in list(best_measure_pairs_all_unselected)
	return wlkAnsor2FullLogFile




def get_full_log_file_name_mapping_allInforSThasName(all_infor_ST_dict):
	# the key of all_infor_ST_dict is the model name in all_search_tasks_list_dict
	wlkAnsor2FullLogFile = dict()
	for model_name in all_infor_ST_dict.keys():
		log_file_ST_ori_test_num = 1
		if model_name == "resnet50_batch1":
			log_file_ST_ori_test_num = 3
		log_file_ST_ori = "allops_test_ST_"+model_name+"_Test"+str(log_file_ST_ori_test_num) + ".json"
		# get the corresponding all_infor_ST
		all_infor_ST = all_infor_ST_dict[model_name] 
		for k in all_infor_ST:
			if all_infor_ST[k]['wlk_r'] == None:
				if k not in wlkAnsor2FullLogFile:
					wlkAnsor2FullLogFile[k] = list()
				wlkAnsor2FullLogFile[k].append(log_file_ST_ori)
	for model_name in all_infor_ST_dict.keys():
		all_infor_ST = all_infor_ST_dict[model_name]
		for k in all_infor_ST:
			if (k not in wlkAnsor2FullLogFile): 
				wlkAnsor2FullLogFile[k] = ["FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test1.json",]
	return wlkAnsor2FullLogFile




# unselected_op_full_tune_log = "FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test2.json"
# unselected_op_full_tune_log_bertBase = "FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test3_DNNVariants.json"

def get_full_log_file_name_mapping_allInforSThasName_A_model(all_infor_ST_dict, 
	model_name, 
	unselected_op_full_tune_log,
	unselected_op_full_tune_log_bertBase):
	# 
	def get_log_file_ST_ori(model_name):
		log_file_ST_ori_test_num = 1
		if model_name == "resnet50_batch1":
			log_file_ST_ori_test_num = 3
		log_file_ST_ori = "allops_test_ST_"+model_name+"_Test"+str(log_file_ST_ori_test_num) + ".json"
		return log_file_ST_ori
	# 
	log_file_ST_ori = get_log_file_ST_ori(model_name)
	wlkAnsor2FullLogFile = dict()
	all_infor_ST = all_infor_ST_dict[model_name]
	for k in all_infor_ST:
		if all_infor_ST[k]['wlk_r'] == None:
			wlkAnsor2FullLogFile[k] = [log_file_ST_ori]
		else:
			if model_name == "bert_Base_batch1_seq_128":
				wlkAnsor2FullLogFile[k] = [unselected_op_full_tune_log_bertBase]
			else:
				for tmp_name, tmp_all_infor in all_infor_ST_dict.items():
					if (k in tmp_all_infor) and (tmp_all_infor[k]['wlk_r'] == None):
						wlkAnsor2FullLogFile[k] = [get_log_file_ST_ori(tmp_name)]
						break
			if k not in wlkAnsor2FullLogFile:
				# this op is unselected for all DNNs
				wlkAnsor2FullLogFile[k] = [unselected_op_full_tune_log]
	return wlkAnsor2FullLogFile


	

# for rerun end2end st variants
def assert_all_ST_infor_correspond_searchtasks(all_infor_ST, search_tasks_list):
	tasks = list()
	for search_tasks in search_tasks_list:
		tasks = tasks + search_tasks
	assert set([i.compute_dag.workload_key() for i in tasks]) ==\
		set(all_infor_ST.keys())









# below are methods to process the logs from the above end-to-end experiments on ST variants
# we only get entities {'config_num', 'search_time', 'latency', 'freqency'} in the new infor_dict structure
def get_tasks_weights_ansor(all_infor_ST_list, search_tasks_list):
	wlk_list = list()
	for search_tasks in search_tasks_list:
		for task in search_tasks:
			wlk_list.append(task.compute_dag.workload_key())
	for all_infor_ST in all_infor_ST_list:
		if set(all_infor_ST.keys()) == set(wlk_list):
			tasks_weights_ansor = dict()
			for k in all_infor_ST:
				tasks_weights_ansor[k] = all_infor_ST[k]['freqency']
			return tasks_weights_ansor
	assert False 




def get_best_measure_pairs_for_all_tasks(all_infor_ST_list, search_tasks_list, best_measure_pairs_single_ops_unselected):
	wlk_list = list()
	tasks = list()
	# get all search tasks and their workload_keys
	for search_tasks in search_tasks_list:
		for task in search_tasks:
			wlk_list.append(task.compute_dag.workload_key())
		tasks = tasks + search_tasks
	# 
	for all_infor_ST in all_infor_ST_list:
		if set(all_infor_ST.keys()) == set(wlk_list):
			best_measure_pairs_all_ops_Ansor = dict()
			for task in tasks:
				wlk = task.workload_key
				wlk_ansor = task.compute_dag.workload_key()
				if all_infor_ST[wlk_ansor]['wlk_r'] == None:
					best_measure_pairs_all_ops_Ansor[wlk] = task.compute_dag.flop_ct / all_infor_ST[wlk_ansor]['latency'] / 1e9
				else:
					if wlk not in best_measure_pairs_single_ops_unselected:
						# the op is tuned in another model
						for tmp_all_infor in all_infor_ST_list:
							if (wlk_ansor in tmp_all_infor) and (tmp_all_infor[wlk_ansor]['wlk_r'] == None):
								best_measure_pairs_all_ops_Ansor[wlk] = task.compute_dag.flop_ct / tmp_all_infor[wlk_ansor]['latency'] / 1e9
					else:
						best_measure_pairs_all_ops_Ansor[wlk] = best_measure_pairs_single_ops_unselected[wlk][1]
			return best_measure_pairs_all_ops_Ansor
	assert False 	





# this is then new version of getting best measure pairs for falling back
def get_best_measure_pairs_for_all_tasks_V2(model_name, all_infor_ST_dict, search_tasks_list, 
	best_measure_pairs_single_ops_unselected, 
	best_measure_pairs_single_ops_unselected_bertBase):
	tasks = list()
	# get all search tasks and their workload_keys
	for search_tasks in search_tasks_list:
		tasks = tasks + search_tasks
	# 
	all_infor_ST = all_infor_ST_dict[model_name]
	best_measure_pairs_all_ops_Ansor = dict()
	for task in tasks:
		wlk = task.workload_key
		wlk_ansor = task.compute_dag.workload_key()
		if all_infor_ST[wlk_ansor]['wlk_r'] == None:
			best_measure_pairs_all_ops_Ansor[wlk] = task.compute_dag.flop_ct / all_infor_ST[wlk_ansor]['latency'] / 1e9
		else:
			if model_name == "bert_Base_batch1_seq_128":
				best_measure_pairs_all_ops_Ansor[wlk] = best_measure_pairs_single_ops_unselected_bertBase[wlk][1]
			else:
				# this task is not selected to be tuned by Ansor
				if wlk not in best_measure_pairs_single_ops_unselected:
					# the op is tuned in another model
					for tmp_all_infor in all_infor_ST_dict.values():
						if (wlk_ansor in tmp_all_infor) and (tmp_all_infor[wlk_ansor]['wlk_r'] == None):
							best_measure_pairs_all_ops_Ansor[wlk] = task.compute_dag.flop_ct / tmp_all_infor[wlk_ansor]['latency'] / 1e9
							break
				else:
					best_measure_pairs_all_ops_Ansor[wlk] = best_measure_pairs_single_ops_unselected[wlk][1]
	return best_measure_pairs_all_ops_Ansor
	



def get_time_dict_fullTune_for_all_tasks_V2(model_name, all_infor_ST_dict, search_tasks_list, 
	time_dict_fullTune_single_ops_unselected, 
	time_dict_fullTune_single_ops_unselected_bertBase):
	tasks = list()
	# get all search tasks and their workload_keys
	for search_tasks in search_tasks_list:
		tasks = tasks + search_tasks
	# 
	all_infor_ST = all_infor_ST_dict[model_name]
	time_dict_fullTune_all_ops_Ansor = dict()
	for task in tasks:
		wlk = task.workload_key
		wlk_ansor = task.compute_dag.workload_key()
		if all_infor_ST[wlk_ansor]['wlk_r'] == None:
			time_dict_fullTune_all_ops_Ansor[wlk] = all_infor_ST[wlk_ansor]['search_time']
		else:
			if model_name == "bert_Base_batch1_seq_128":
				time_dict_fullTune_all_ops_Ansor[wlk] = time_dict_fullTune_single_ops_unselected_bertBase[(wlk, None)]
			else:
				# this task is not selected to be tuned by Ansor
				if (wlk,None) not in time_dict_fullTune_single_ops_unselected:
					# the op is tuned in another model
					for tmp_all_infor in all_infor_ST_dict.values():
						if (wlk_ansor in tmp_all_infor) and (tmp_all_infor[wlk_ansor]['wlk_r'] == None):
							time_dict_fullTune_all_ops_Ansor[wlk] = tmp_all_infor[wlk_ansor]['search_time']
							break
				else:
					time_dict_fullTune_all_ops_Ansor[wlk] = time_dict_fullTune_single_ops_unselected[(wlk,None)]
	return time_dict_fullTune_all_ops_Ansor


# check if a task is unselected in a model, then it cannot be selected in another model
def check_selection_conclusion_all_infor_ST(all_infor_ST_list):
	analysis = dict()
	for all_infor_ST in all_infor_ST_list:
		for k in all_infor_ST:
			if k not in analysis:
				analysis[k] = list()
			if all_infor_ST[k]['wlk_r'] == None:
				analysis[k].append(True)
			else:
				analysis[k].append(False)
	for k in analysis:
		if (True in analysis[k]) and (False in analysis[k]):
			print("error!")



# check_selection_conclusion_all_infor_ST(all_infor_ST_list) # the checking prints "error", failed.





#############################################################################



# compute total latency======================================
def get_latency_mytune(res_idx, best_measure_pairs_list, search_tasks, 
	tasks_ansor, task_weights_ansor):
	tot = 0
	best_measure_pairs = best_measure_pairs_list[res_idx]
	for i in range(ori_op_num_list[0]):
		wlk_t_ansor = search_tasks[i].compute_dag.workload_key()
		wlk_t = search_tasks[i].workload_key
		a = search_tasks[i].compute_dag.flop_ct /best_measure_pairs[wlk_t][1]/1e9
		b = None
		for j in range(len(tasks_ansor)):
			if wlk_t_ansor == tasks_ansor[j].compute_dag.workload_key():
				b = task_weights_ansor[j]
				break
		assert b!=None
		tot=tot+(a*b)
	return tot


# get_latency_mytune(1, best_measure_pairs_list, search_tasks, 
# 	tasks_ansor, task_weights_ansor)



# compute when did ansor get my latency========================
def get_time_ansor_need(my_score, all_infor_Ansor):
	print("When did Ansor get my latency: ", )
	for i in range(len(all_infor_Ansor.post_best_score_history)):
		tmp_score = all_infor_Ansor.post_best_score_history[i]
		if tmp_score and tmp_score < my_score:
			print("IN ROUND %d, used time: %f, trials: %d" % \
				(i, all_infor_Ansor.post_search_time_history[i], \
					all_infor_Ansor.post_config_num_history[i]))
			# print("ACCELETATE %f in time, %f in config num" % \
			# 	(all_infor_Ansor.post_search_time_history[i] / tot_time_num_mytune, \
			# 		all_infor_Ansor.post_config_num_history[i] / tot_config_num_mytune))
			break




# get_time_ansor_need(my_score=0.447774880845631, 
# 	all_infor_Ansor=all_infor_Ansor)


# some analysis functions--------------------------------------------

def get_SINGLE_pair_res_dict_for_figure( 
	best_measure_pairs_list, best_reuse_results_list, best_reuse_results_ST_list, 
	config_num_dict_list, time_dict_mytune_list, config_num_dict_ST_list, time_dict_ST_list, 
	log_file_ansor):
	''' print the data we need in the experiments '''
	import scipy
	def get_average(tot_list):
		assert len(tot_list) == 40
		return sum(tot_list) / len(tot_list)
	# 
	def get_normalized(ori_list, ref_list):
		assert (len(ori_list) == len(ref_list)) and (len(ori_list) == 40)
		return [ori_list[i]/ref_list[i] for i in range(len(ori_list))]
	# 
	for round_i in range(len(best_measure_pairs_list)):
		best_measure_pairs = best_measure_pairs_list[round_i] 
		best_reuse_results = best_reuse_results_list[round_i] 
		best_reuse_results_ST = best_reuse_results_ST_list[round_i] 
		config_num_dict = config_num_dict_list[round_i] 
		time_dict_mytune = time_dict_mytune_list[round_i] 
		config_num_dict_ST = config_num_dict_ST_list[round_i] 
		time_dict_ST = time_dict_ST_list[round_i]
		# 
		score_mytune = list()
		score_ansor = list()
		score_sametime_ansor = list()
		score_ST = list()
		# config_num_mytune = list()
		# config_num_Ansor = list()
		# config_num_Ansor_best = list()
		# config_num_ST = list()
		search_time_mytune = list()
		search_time_Ansor = list()
		search_time_Ansor_best = list()
		search_time_ST = list()
		for wlk_to_tune, wlk_reuse_from in best_reuse_results.keys():
			score_mytune.append(best_reuse_results[(wlk_to_tune, wlk_reuse_from)][1])
			score_ansor.append(best_measure_pairs[wlk_to_tune][1])
			score_ST.append(best_reuse_results_ST[(wlk_to_tune, wlk_reuse_from)][1])
			Ansor_best_ct, Ansor_best_time = get_best_ct_time_given_wlk(log_file_ansor, wlk_to_tune, \
				best_measure_pairs[wlk_to_tune][1])
			search_time_mytune.append(time_dict_mytune[(wlk_to_tune, wlk_reuse_from)])
			search_time_Ansor.append(time_dict_mytune[(wlk_to_tune, None)])
			print(time_dict_mytune[(wlk_to_tune, None)])
			search_time_Ansor_best.append(Ansor_best_time)
			search_time_ST.append(time_dict_ST[(wlk_to_tune, wlk_reuse_from)])
			Ansor_sametime_score, Ansor_sametime_config_num = get_perff_sametime_given_wlk(log_file_ansor, \
				wlk_to_tune, time_dict_mytune[(wlk_to_tune, wlk_reuse_from)])
			score_sametime_ansor.append(Ansor_sametime_score)
		#
		single_pair_res = dict()
		task_type = log_file_ansor[len('SINGLEpair_tune_Ansor_'):-len('_Test1')]
		single_pair_res[task_type] = {'my_score':score_mytune,                       
                           'ansor_score':score_ansor, 
                           'ansor_sametime_score':score_sametime_ansor,
                           'st_score': score_ST,
                           'my_time':search_time_mytune,
                           'ansor_time':search_time_Ansor,
                           'ansor_best_time':search_time_Ansor_best,
                           'st_time':search_time_ST
                           }
		for v in single_pair_res[task_type].values():
			assert len(v) == 40
		print('single_pair_res['+task_type+'] = ', single_pair_res[task_type])




get_SINGLE_pair_res_dict_for_figure( 
	best_measure_pairs_list, best_reuse_results_list, best_reuse_results_ST_list, 
	config_num_dict_list, time_dict_mytune_list, config_num_dict_ST_list, time_dict_ST_list, 
	log_file_ansor)



# get analysis results for single pair experiments
def print_SINGLE_pair_analysis(search_tasks_list, 
	best_measure_pairs_list, best_reuse_results_list, best_reuse_results_ST_list, 
	config_num_dict_list, time_dict_mytune_list, config_num_dict_ST_list, time_dict_ST_list, 
	log_file_ansor):
	''' print the data we need in the experiments '''
	import scipy
	def get_average(tot_list):
		assert len(tot_list) == 40
		return sum(tot_list) / len(tot_list)
	# 
	def get_normalized(ori_list, ref_list):
		assert (len(ori_list) == len(ref_list)) and (len(ori_list) == 40)
		return [ori_list[i]/ref_list[i] for i in range(len(ori_list))]
	# 
	for round_i in range(len(search_tasks_list)):
		search_tasks = search_tasks_list[round_i]
		best_measure_pairs = best_measure_pairs_list[round_i] 
		best_reuse_results = best_reuse_results_list[round_i] 
		best_reuse_results_ST = best_reuse_results_ST_list[round_i] 
		config_num_dict = config_num_dict_list[round_i] 
		time_dict_mytune = time_dict_mytune_list[round_i] 
		config_num_dict_ST = config_num_dict_ST_list[round_i] 
		time_dict_ST = time_dict_ST_list[round_i]
		# 
		normalized_perf_ratio_mytune = list()
		normalized_perf_ratio_ST = list()
		config_num_mytune = list()
		config_num_Ansor = list()
		config_num_Ansor_best = list()
		config_num_ST = list()
		search_time_mytune = list()
		search_time_Ansor = list()
		search_time_Ansor_best = list()
		search_time_ST = list()
		for to_tune_idx, reuse_from_idx in config_num_dict.keys():
			if reuse_from_idx == -1:
				continue
			print(to_tune_idx, reuse_from_idx)
			wlk_to_tune = search_tasks[to_tune_idx].workload_key
			wlk_reuse_from = search_tasks[reuse_from_idx].workload_key
			normalized_perf_ratio_mytune.append(best_reuse_results[(wlk_to_tune, wlk_reuse_from)][1] / \
				best_measure_pairs[wlk_to_tune][1])
			normalized_perf_ratio_ST.append(best_reuse_results_ST[(wlk_to_tune, wlk_reuse_from)][1] / \
				best_measure_pairs[wlk_to_tune][1])
			config_num_mytune.append(config_num_dict[(to_tune_idx, reuse_from_idx)])
			config_num_Ansor.append(config_num_dict[(to_tune_idx, -1)])
			Ansor_best_ct, Ansor_best_time = get_best_ct_time_given_wlk(log_file_ansor, wlk_to_tune, \
				search_tasks[to_tune_idx].compute_dag.flop_ct / best_measure_pairs[wlk_to_tune][1] / 1e9)
			config_num_Ansor_best.append(Ansor_best_ct)
			config_num_ST.append(config_num_dict_ST[(to_tune_idx, reuse_from_idx)])
			search_time_mytune.append(time_dict_mytune[(wlk_to_tune, wlk_reuse_from)])
			search_time_Ansor.append(time_dict_mytune[(wlk_to_tune, None)])
			print(time_dict_mytune[(wlk_to_tune, None)])
			search_time_Ansor_best.append(Ansor_best_time)
			search_time_ST.append(time_dict_ST[(wlk_to_tune, wlk_reuse_from)])
		# 
		# 1: performance ratio
		print("arithmatic: mytune, ST. normalized to Ansor ", get_average(normalized_perf_ratio_mytune), 
			get_average(normalized_perf_ratio_ST))
		print("geometric: mytune, ST. normalized to Ansor ", scipy.stats.gmean(normalized_perf_ratio_mytune), 
			scipy.stats.gmean(normalized_perf_ratio_ST))
		print("harmonic: mytune, ST. normalized to Ansor ", scipy.stats.hmean(normalized_perf_ratio_mytune), 
			scipy.stats.hmean(normalized_perf_ratio_ST))
		# 2: efficiency value (#config)
		print("avg #config: mytune, ST, Ansor best, Ansor total ", get_average(config_num_mytune), 
			get_average(config_num_ST), get_average(config_num_Ansor_best), get_average(config_num_Ansor))
		# 3: efficiency value (search time)
		print("#search time(s): mytune, ST, Ansor total ", get_average(search_time_mytune), 
			get_average(search_time_ST), get_average(search_time_Ansor))
		# 4: efficiency ratio (#config)
		print("avg #config ratio : mytune, ST. normalized to Ansor best", get_average(get_normalized(config_num_mytune, config_num_Ansor_best)), 
			get_average(get_normalized(config_num_ST, config_num_Ansor_best)))
		print("avg #config ratio : mytune, ST. normalized to Ansor total", get_average(get_normalized(config_num_mytune, config_num_Ansor)), 
			get_average(get_normalized(config_num_ST, config_num_Ansor)))
		print("geometric #config ratio : mytune, ST. normalized to Ansor best", scipy.stats.gmean(get_normalized(config_num_mytune, config_num_Ansor_best)), 
			scipy.stats.gmean(get_normalized(config_num_ST, config_num_Ansor_best)))
		print("geometric #config ratio : mytune, ST. normalized to Ansor total", scipy.stats.gmean(get_normalized(config_num_mytune, config_num_Ansor)), 
			scipy.stats.gmean(get_normalized(config_num_ST, config_num_Ansor)))
		print("harmonic #config ratio : mytune, ST. normalized to Ansor best", scipy.stats.hmean(get_normalized(config_num_mytune, config_num_Ansor_best)), 
			scipy.stats.hmean(get_normalized(config_num_ST, config_num_Ansor_best)))
		print("harmonic #config ratio : mytune, ST. normalized to Ansor total", scipy.stats.hmean(get_normalized(config_num_mytune, config_num_Ansor)), 
			scipy.stats.hmean(get_normalized(config_num_ST, config_num_Ansor)))
		# 5: efficiency ratio (#second(s))
		print("avg #search time (s) ratio : mytune, ST. normalized to Ansor best", get_average(get_normalized(search_time_mytune, search_time_Ansor_best)), 
			get_average(get_normalized(search_time_ST, search_time_Ansor_best)))
		print("avg #search time (s) ratio : mytune, ST. normalized to Ansor total", get_average(get_normalized(search_time_mytune, search_time_Ansor)), 
			get_average(get_normalized(search_time_ST, search_time_Ansor)))
		print("geometric #search time (s) ratio : mytune, ST. normalized to Ansor best", scipy.stats.gmean(get_normalized(search_time_mytune, search_time_Ansor_best)), 
			scipy.stats.gmean(get_normalized(search_time_ST, search_time_Ansor_best)))
		print("geometric #search time (s) ratio : mytune, ST. normalized to Ansor total", scipy.stats.gmean(get_normalized(search_time_mytune, search_time_Ansor)), 
			scipy.stats.gmean(get_normalized(search_time_ST, search_time_Ansor)))
		print("harmonic #search time (s) ratio : mytune, ST. normalized to Ansor best", scipy.stats.hmean(get_normalized(search_time_mytune, search_time_Ansor_best)), 
			scipy.stats.hmean(get_normalized(search_time_ST, search_time_Ansor_best)))
		print("harmonic #search time (s) ratio : mytune, ST. normalized to Ansor total", scipy.stats.hmean(get_normalized(search_time_mytune, search_time_Ansor)), 
			scipy.stats.hmean(get_normalized(search_time_ST, search_time_Ansor)))
		# 6: upper and lower bound
		print("[MIN, MAX]: mytune, ST. normalized to Ansor ", min(normalized_perf_ratio_mytune), max(normalized_perf_ratio_mytune),
			min(normalized_perf_ratio_ST), max(normalized_perf_ratio_ST))
		print("[MIN MAX] #config ratio : mytune, ST. normalized to Ansor best", min(get_normalized(config_num_mytune, config_num_Ansor_best)), 
			max(get_normalized(config_num_mytune, config_num_Ansor_best)), 
			min(get_normalized(config_num_ST, config_num_Ansor_best)), 
			max(get_normalized(config_num_ST, config_num_Ansor_best)))
		print("[MIN, MAX] #config ratio : mytune, ST. normalized to Ansor total", min(get_normalized(config_num_mytune, config_num_Ansor)), 
			max(get_normalized(config_num_mytune, config_num_Ansor)), 
			min(get_normalized(config_num_ST, config_num_Ansor)),
			max(get_normalized(config_num_ST, config_num_Ansor)))
		print("[MIN, MAX] #search time (s) ratio : mytune, ST. normalized to Ansor best", min(get_normalized(search_time_mytune, search_time_Ansor_best)), 
			max(get_normalized(search_time_mytune, search_time_Ansor_best)),
			min(get_normalized(search_time_ST, search_time_Ansor_best)), 
			max(get_normalized(search_time_ST, search_time_Ansor_best)))
		print("[MIN, MAX] #search time (s) ratio : mytune, ST. normalized to Ansor total", min(get_normalized(search_time_mytune, search_time_Ansor)), 
			max(get_normalized(search_time_mytune, search_time_Ansor)),
			min(get_normalized(search_time_ST, search_time_Ansor)), 
			max(get_normalized(search_time_ST, search_time_Ansor)))








# analysis functions for st
def get_tuning_log_infor_dict(tuning_log_infor_dict_DB, filename, workload_key, topks):
		if (filename, workload_key, tuple(topks)) not in list(tuning_log_infor_dict_DB.keys()):
			tuning_log_infor_dict_DB[(filename, workload_key, tuple(topks))] = get_best_cost_total_ct_time_given_wlk_TopK(filename, workload_key, topks)
		return copy.deepcopy(tuning_log_infor_dict_DB[(filename, workload_key, tuple(topks))])


def update_full_optimization_metric(log_file_ansor1, wlkAnsor2FullLogFile, search_task, all_infor_dict, tuning_log_infor_dict_DB):
	''' Update the all_infor_dict with full optimization results. '''
	wlk_t = search_task.workload_key
	wlk_t_ansor = search_task.compute_dag.workload_key()
	if log_file_ansor1 in wlkAnsor2FullLogFile[wlk_t_ansor]:
		infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, log_file_ansor1, wlk_t, [None])
	else:
		infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, wlkAnsor2FullLogFile[wlk_t_ansor][0], wlk_t, [None])
	# 
	assert infor_dict[None]['config_num'] == 1000, "this file does not have 1000 records" # check that the metrics for selected ops are right
	all_infor_dict[wlk_t_ansor]['latency'] = infor_dict[None]['latency']
	for k in ['search_time', 'config_num']:
		all_infor_dict[wlk_t_ansor][k] = all_infor_dict[wlk_t_ansor][k] + infor_dict[None][k]
	



def get_all_infor_ST_END2END_ST_variants(log_file, log_file_ansor1, all_infor_ST_variant_file,
	# all_infor_ST_ori, 
	wlkAnsor2FullLogFile,
	# best_measure_pairs_single_op_Ansor, # for selecting the right tuning log file.
	best_measure_pairs_baseline_fallback, # for deciding whether to fall back or not.
	# config_num_dict_single_op_Ansor,
	# time_dict_single_op_Ansor, 
	variant, variant_param, 
	search_tasks_list, 
	loops_list, 
	tasks_weights_ansor, 
	all_results_dict, # to store the metrics in
	tuning_log_infor_dict_DB, # stores the result of calling 'get_best_cost_total_ct_time_given_wlk_TopK'
	falling_back_ratios = [0.5, 0.7, 0.9]): # the ratio when ST's performance is not above it compared with Ansor, then falling back to full optimization
	'''
	! This function only do analysis work, it also stores the all_infor_ST variant to file.
	! This function would also generate the results analysis for each variant with falling back to full optimization. [We would generate two version with falling back: with or without feedback]
	! But for without feedback, some logs need to be re-generated [the variant: varying num_cluster_center_ops <- should be fast].
	!RMK: we would change the content in all_results_dict!!!!!!!
	log_file: str. The file to store tuning logs for this run.
	log_file_ansor1: str. The full tuning logs for this run to reuse. 
						There is log_file_ansor2, which stores the full tuning logs for search tasks in all DNNs that is not selected as center ops by ori_ST.
	# all_infor_ST_ori: dict. The all_infor_ST structure as stored in the corr. file for the DNN being optimized in this run.
	all_infor_ST_variant_file: str. The file to store the all_infor_ST_variant structure in.
	wlkAnsor2FullLogFile: mapping each workload_key to the file where there is the full optimization logs by Ansor.
	best_measure_pairs_single_op_Ansor: dict: [wlk: (state, GFLOPS)]. 
											The best_measure_pair structure as stored in the tuning process. 
											It stores the best measure pair for search tasks in all DNNs that is not selected as center ops by ori_ST.
	# config_num_dict_single_op_Ansor: dict: [wlk: config_num]. 
	# 										The same as best_measure_pairs_single_op_Ansor, but stores config num. 
	# 										!Need to do some transformation to get the structure before calling this func.
	# time_dict_single_op_Ansor
	variant: str. 
				"vary_K", we would change the K in TopK to 1000.
				"vary_reuse_clusterNum", we would make TopK reuse all the center cluster ops' configs in the order from SM high to low, with K = 3 and SM threshold being 0.01.
				"vary_SM_threshold", we would change the SM threshold in building the graph for finding cliques with K=3 and only reuse 1 cluster center op. 
					[!Only this variant would change the number of clusters.]
	'''
	# 
	def print_psm_range(psm):
		tmp_array = list()
		for row in psm:
			tmp_array = tmp_array + [i for i in row if i < 1]
		if len(tmp_array) > 0:
			print("max, min, avg in psm: ", max(tmp_array), min(tmp_array), sum(tmp_array) / len(tmp_array))
		# 
	# 
	time_begin_tot_ST_vary_K = time.time()
	best_reuse_results_ST_vary_K_list = list()
	config_num_dict_ST_vary_K_list = list()
	time_dict_ST_vary_K_list = list()
	# 
	# init all_infor_ST_variant and all_infor_ST_variant_fallback
	all_infor_ST_variant = dict()
	for paramV in variant_param:
		all_infor_ST_variant[paramV] = dict()
	# the first level of all_infor_ST_variant_fallback is the params, the second level is the fall back ratio.
	all_infor_ST_variant_fallback = dict() 
	for paramV in variant_param:
		all_infor_ST_variant_fallback[paramV] = dict()
		for fb_ratio in falling_back_ratios:
			all_infor_ST_variant_fallback[paramV][fb_ratio] = dict()
	# 
	for round_i in range(len(search_tasks_list)):
		search_tasks = search_tasks_list[round_i]
		loops = loops_list[round_i]
		best_reuse_results_ST_vary_K = dict()
		config_num_dict_ST_vary_K = dict()
		time_dict_ST_vary_K = dict()
		psm_for_variant = None
		# 
		op_para_list = [get_op_para_ansor(search_tasks[i], loops[i]) for i in range(len(search_tasks))]
		if variant == "vary_SM_threshold":
			if get_sketch_num(search_tasks[0]) > 1:
				mark_depend(op_para_list, True, variant_param[0])
			else:
				mark_depend(op_para_list, False, variant_param[0])
		else:
			if get_sketch_num(search_tasks[0]) > 1:
				mark_depend(op_para_list, True)
				psm_for_variant = compute_psm(op_para_list, True)
			else:
				mark_depend(op_para_list, False)
				psm_for_variant = compute_psm(op_para_list, False)
			print_psm_range(psm_for_variant)
		# 
		selected_ops = [i for i in range(len(search_tasks)) \
			if op_para_list[i]['depend'] == i]
		print("selected_ops: ", selected_ops, "total num: ", len(selected_ops))
		other_ops = [i for i in range(len(search_tasks)) \
			if op_para_list[i]['depend'] != i]
		print("other_ops: ", other_ops)
		# build selected op pairs
		if variant == "vary_reuse_clusterNum":
			default_SM_threshold = 0.01
			selected_op_pairs = list()
			for i in other_ops:
				possible_reuse_pairs_n_SM = [(i, j, psm_for_variant[i][j]) for j in selected_ops if psm_for_variant[i][j] >= default_SM_threshold]
				possible_reuse_pairs_n_SM = sorted(possible_reuse_pairs_n_SM, key = lambda pair_SM: pair_SM[-1], reverse = True)
				selected_op_pairs = selected_op_pairs + [j[:2] for j in possible_reuse_pairs_n_SM] 
			print(selected_op_pairs, '\ntotal pairs: ', len(selected_op_pairs))
		else:
			selected_op_pairs = [(i, op_para_list[i]['depend']) for i in other_ops]
			print(selected_op_pairs, '\ntotal pairs: ', len(selected_op_pairs))
		selected_op_pairs = selected_op_pairs + [(i, -1) for i in selected_ops]
		# 
		if variant == "vary_K":
			# topk = variant_param
			topks = variant_param # should be a list of ints [the topK values]
		else:
			# topk = 3
			topks = [None]
		target = tvm.target.Target("cuda")
		# 
		for to_tune_idx, reuse_from_idx in selected_op_pairs:
			wlk_t = search_tasks[to_tune_idx].workload_key
			wlk_t_ansor = search_tasks[to_tune_idx].compute_dag.workload_key()
			# wlk_r_ansor = search_tasks[reuse_from_idx].compute_dag.workload_key()
			if reuse_from_idx == -1:
				if log_file_ansor1 in wlkAnsor2FullLogFile[wlk_t_ansor]:
					infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, log_file_ansor1, wlk_t, [None])
				else:
					infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, wlkAnsor2FullLogFile[wlk_t_ansor][0], wlk_t, [None])
				# 
				# if wlk_t not in list(best_measure_pairs_single_op_Ansor.keys()):
				# 	infor_dict = get_best_cost_total_ct_time_given_wlk_TopK(log_file_ansor1, wlk_t, [None])
				# else:
				# 	infor_dict = get_best_cost_total_ct_time_given_wlk_TopK(log_file_ansor2, wlk_t, [None])
				assert infor_dict[None]['config_num'] == 1000, "this file does not have 1000 records" # check that the metrics for selected ops are right
				# if len(infor_dict) == 0:
				# 	# find a wrong file
				# 	infor_dict = get_best_cost_total_ct_time_given_wlk_TopK(log_file_ansor2, wlk_t, [None])
				# 	assert len(infor_dict) != 0, "no valid record can be found for this task tuned by Ansor! " + search_tasks[to_tune_idx].workload_key
				for paramV in variant_param:
					all_infor_ST_variant[paramV][wlk_t_ansor] = copy.deepcopy(infor_dict[None])
					all_infor_ST_variant[paramV][wlk_t_ansor]['freqency'] = tasks_weights_ansor[wlk_t_ansor]
					for fb_ratio in falling_back_ratios:
						all_infor_ST_variant_fallback[paramV][fb_ratio][wlk_t_ansor] = copy.deepcopy(infor_dict[None])
						all_infor_ST_variant_fallback[paramV][fb_ratio][wlk_t_ansor]['freqency'] = tasks_weights_ansor[wlk_t_ansor]
				# do some checking
				infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, log_file, wlk_t, [None])
				assert len(infor_dict) == 0, "Different representative tasks from the tuning logs!"
				if variant != "vary_SM_threshold":
					assert log_file_ansor1 in wlkAnsor2FullLogFile[wlk_t_ansor], "previous reused log file is incorrect for " +variant
			else:
				if wlk_t_ansor not in all_infor_ST_variant[variant_param[0]].keys():
					# infor_dict = get_tuning_log_infor_dict_givenRange(tuning_log_infor_dict_DB, log_file, wlk_t, wlk_r, [3, 10, 100, 1000], 
					# 	log_range_dict_ST[(wlk_t, wlk_r)], time_begin_dict_ST[(wlk_t, wlk_r)])
					# 
					infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, log_file, wlk_t, topks)
					assert len(infor_dict) != 0, "Different unselected tasks from the tuning logs!"
					if variant == "vary_K":
						for paramV in variant_param:
							all_infor_ST_variant[paramV][wlk_t_ansor] = copy.deepcopy(infor_dict[paramV])
							all_infor_ST_variant[paramV][wlk_t_ansor]['freqency'] = tasks_weights_ansor[wlk_t_ansor]
							for fb_ratio in falling_back_ratios:
								all_infor_ST_variant_fallback[paramV][fb_ratio][wlk_t_ansor] = copy.deepcopy(infor_dict[paramV])
								all_infor_ST_variant_fallback[paramV][fb_ratio][wlk_t_ansor]['freqency'] = tasks_weights_ansor[wlk_t_ansor]
								if search_tasks[to_tune_idx].compute_dag.flop_ct / infor_dict[paramV]['latency'] / 1e9 \
									/ best_measure_pairs_baseline_fallback[wlk_t] < fb_ratio:
									# need falling back to full optimization
									update_full_optimization_metric(log_file_ansor1, wlkAnsor2FullLogFile, \
										search_tasks[to_tune_idx], all_infor_ST_variant_fallback[paramV][fb_ratio], tuning_log_infor_dict_DB)
					else:
						for paramV in variant_param:
							all_infor_ST_variant[paramV][wlk_t_ansor] = copy.deepcopy(infor_dict[None])
							all_infor_ST_variant[paramV][wlk_t_ansor]['freqency'] = tasks_weights_ansor[wlk_t_ansor]
							for fb_ratio in falling_back_ratios:
								all_infor_ST_variant_fallback[paramV][fb_ratio][wlk_t_ansor] = copy.deepcopy(infor_dict[None])
								all_infor_ST_variant_fallback[paramV][fb_ratio][wlk_t_ansor]['freqency'] = tasks_weights_ansor[wlk_t_ansor]
								if search_tasks[to_tune_idx].compute_dag.flop_ct / infor_dict[None]['latency'] / 1e9 \
									/ best_measure_pairs_baseline_fallback[wlk_t] < fb_ratio:
									# need falling back to full optimization
									update_full_optimization_metric(log_file_ansor1, wlkAnsor2FullLogFile, \
										search_tasks[to_tune_idx], all_infor_ST_variant_fallback[paramV][fb_ratio], tuning_log_infor_dict_DB)
				else:
					assert variant == "vary_reuse_clusterNum", "more than 1 reuse pairs are for the same op!"
			# print(to_tune_idx, reuse_from_idx, all_infor_ST_variant[variant_param[0]][wlk_t_ansor])
	# 
	# print infor analysis
	# if variant not in all_results_dict:
	# 	all_results_dict[variant] = dict()
	# all_results_dict[variant]["no_full_opt"] = dict()
	# all_results_dict[variant]["full_opt"] = dict()
	for paramV in variant_param:
		ST_score, ST_tot_config_num, ST_tot_search_time = print_mytune_infor(all_infor_ST_variant[paramV])
		all_results_dict[variant]["no_full_opt"][paramV] = {"score": ST_score, "tot_config_num": ST_tot_config_num, "tot_search_time": ST_tot_search_time}
		all_results_dict[variant]["full_opt"][paramV] = dict()
		for fb_ratio in falling_back_ratios:
			ST_score, ST_tot_config_num, ST_tot_search_time = print_mytune_infor(all_infor_ST_variant_fallback[paramV][fb_ratio])
			all_results_dict[variant]["full_opt"][paramV][fb_ratio] = {"score": ST_score, "tot_config_num": ST_tot_config_num, "tot_search_time": ST_tot_search_time}
	# store the result in file
	if variant == "vary_K":
		print("minimum reused config: ", min([all_infor_ST_variant[1000][k]['config_num'] for k in all_infor_ST_variant[1000]]))
	with open(all_infor_ST_variant_file, 'w') as f:
		f.write('all_infor_ST_variant = ' + all_infor_ST_variant.__str__())
		f.write('all_infor_ST_variant_fallback = ' + all_infor_ST_variant_fallback.__str__())
	print("all_results_dict: ", all_results_dict)




def get_tuning_log_infor_dict_givenRange(tuning_log_infor_dict_DB, filename, wlk_t, wlk_r, topks, log_range, timestamp_begin):
		if (filename, (wlk_t, wlk_r), tuple(topks)) not in list(tuning_log_infor_dict_DB.keys()):
			tuning_log_infor_dict_DB[(filename, (wlk_t, wlk_r), tuple(topks))] = \
				get_best_cost_total_ct_time_given_wlk_TopK_givenRange(filename, wlk_t, topks, log_range, timestamp_begin)
		return copy.deepcopy(tuning_log_infor_dict_DB[(filename, (wlk_t, wlk_r), tuple(topks))])




def get_SINGLEpair_all_infor_ST_variants(
	search_tasks_list,
	# loops_list,
	selected_op_pairs_list,
	log_file_ST,
	all_infor_ST_file, 
	log_file_ansor,
	# all_results_dict, # to store the metrics in
	tuning_log_infor_dict_DB, # stores the result of calling 'get_best_cost_total_ct_time_given_wlk_TopK'
	falling_back_ratios = [0, 0.5, 0.7, 0.9]
	):
	# 
	all_infor_ST_variant_fallback = dict() 
	for paramV in [3, 10, 100, 1000]:
		all_infor_ST_variant_fallback[paramV] = dict()
		for fb_ratio in falling_back_ratios:
			all_infor_ST_variant_fallback[paramV][fb_ratio] = dict()
	# 
	with open('tmptmp.py', 'w') as fw:
		# transform the result in all_infor_ST_file to what we need
		fw.write('def get_dict_lists():\n')
		with open(all_infor_ST_file, 'r') as fr:
			for line in fr.readlines():
				fw.write('\t'+line)
		ret_vars = '(best_reuse_results_ST_variant_vary_K_list_no_state, '
		ret_vars += 'config_num_wlk_as_key_dict_ST_variant_vary_K_list, '
		ret_vars += 'time_dict_ST_variant_vary_K_list, '
		ret_vars += 'log_range_dict_ST_variant_vary_K_list, '
		ret_vars += 'time_begin_dict_ST_variant_vary_K_list)'
		fw.write('\treturn ' + ret_vars +'\n')
		fw.write('\n\n\n')
	# 
	import imp
	import tmptmp #avoid tmptmp has not been loaded, so reload would fail
	imp.reload(tmptmp) #update tmptmp
	_, config_num_wlk_as_key_dict_ST_list, _, log_range_dict_ST_list, time_begin_dict_ST_list = \
		tmptmp.get_dict_lists()
	# 
	for round_i in range(len(search_tasks_list)):
		search_tasks = search_tasks_list[round_i]
		# loops = loops_list[round_i]
		selected_op_pairs = selected_op_pairs_list[round_i]
		log_range_dict_ST = log_range_dict_ST_list[round_i]
		time_begin_dict_ST = time_begin_dict_ST_list[round_i]
		config_num_wlk_as_key_dict_ST = config_num_wlk_as_key_dict_ST_list[round_i]
		# target = tvm.target.Target("cuda")
		for to_tune_idx, reuse_from_idx in selected_op_pairs:
			assert reuse_from_idx != -1, "no op to be tuned by Ansor in this run!"
			wlk_t = search_tasks[to_tune_idx].workload_key
			wlk_r = search_tasks[reuse_from_idx].workload_key
			wlk_t_ansor = search_tasks[to_tune_idx].compute_dag.workload_key()
			wlk_r_ansor = search_tasks[reuse_from_idx].compute_dag.workload_key()
			# print(wlk_t, wlk_r)
			# get the performance with falling back ratios
			for paramV in [3, 10, 100, 1000]:
				for fb_ratio in falling_back_ratios:
					# print(paramV, fb_ratio)
					infor_dict = get_tuning_log_infor_dict_givenRange(tuning_log_infor_dict_DB, log_file_ST, wlk_t, wlk_r, [3, 10, 100, 1000], 
						log_range_dict_ST[(wlk_t, wlk_r)], time_begin_dict_ST[(wlk_t, wlk_r)])
					# print(infor_dict)
					assert len(infor_dict) != 0, "Different unselected tasks from the tuning logs!"
					all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)] = copy.deepcopy(infor_dict[paramV])
					if (paramV == 1000):
						# print(infor_dict)
						# print(infor_dict[paramV]['config_num'], config_num_wlk_as_key_dict_ST[(wlk_t, wlk_r)])
						assert infor_dict[paramV]['config_num'] == config_num_wlk_as_key_dict_ST[(wlk_t, wlk_r)]
					# get the latency by Ansor
					infor_dict_ansor = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, log_file_ansor, wlk_t, [None])
					if (search_tasks[to_tune_idx].compute_dag.flop_ct / infor_dict[paramV]['latency'] / 1e9) \
						/ (search_tasks[to_tune_idx].compute_dag.flop_ct / infor_dict_ansor[None]['latency'] / 1e9) < fb_ratio:
						assert fb_ratio!=0
						# need falling back to full optimization
						# infor_dict = get_tuning_log_infor_dict(tuning_log_infor_dict_DB, log_file_ansor, wlk_t, [None])
						assert infor_dict_ansor[None]['config_num'] == 1000, "this file does not have 1000 records" # check that the metrics for selected ops are right
						all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['latency'] = infor_dict_ansor[None]['latency']
						for k in ['search_time', 'config_num']:
							all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)][k] = \
								all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)][k] + infor_dict_ansor[None][k]
					all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['score'] = search_tasks[to_tune_idx].compute_dag.flop_ct / all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['latency'] / 1e9
					all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['normalized_score'] = \
						all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['score'] / \
							(search_tasks[to_tune_idx].compute_dag.flop_ct / infor_dict_ansor[None]['latency'] / 1e9)
					all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['normalized_search_time'] = \
						all_infor_ST_variant_fallback[paramV][fb_ratio][(wlk_t_ansor, wlk_r_ansor)]['search_time'] / infor_dict_ansor[None]['search_time']
	# 
	return all_infor_ST_variant_fallback


