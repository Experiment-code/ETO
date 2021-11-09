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


from helper_functions1 import *
from helper_functions2 import *
from helper_functions3 import * 


def tune_single_ops_ansor(search_tasks, log_file_ansor, infor_file_ansor):
	''' 
		search tasks is a list of tasks to tune;
		log_file_ansor is the name of the file to write logs in.
	'''
	def delete_state(inp_res_dict):
		new_dict = dict()
		for k,v in inp_res_dict.items():
			new_dict[k] = (None, v[1])
		return new_dict
	# 
	time_begin_tot = time.time()
	# 
	best_measure_pairs = dict()
	config_num_dict = dict()
	time_dict_mytune = dict()
	# 
	# log_file = "invalid_log.json"
	# log_file_ansor = "FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test1.json"
	for i in range(len(search_tasks)):
		line_num_before = get_lineNum(log_file_ansor)
		time_begin = time.time()
		tune_seach_task(search_tasks, i, best_measure_pairs, dict(), log_file_ansor)
		time_dict_mytune[(search_tasks[i].workload_key, None)] = \
			time.time() - time_begin
		# get the actual number of configs
		config_num_dict[(search_tasks[i].workload_key, None)] = get_lineNum(log_file_ansor) - line_num_before
		assert config_num_dict[(search_tasks[i].workload_key, None)] == 1000
		print(best_measure_pairs[search_tasks[i].workload_key][1])
		continue			
	# 
	tot_time_used = time.time() - time_begin_tot
	print('tot_time_used: ', tot_time_used)
	# store the results to infor file
	with open(infor_file_ansor, 'a') as f:
		f.write('TMP_best_measure_pairs = ' + delete_state(best_measure_pairs).__str__() + '\n')
		f.write('best_measure_pairs.update(TMP_best_measure_pairs)\n')
		f.write('TMP_config_num_dict = ' + config_num_dict.__str__() + '\n')
		f.write('config_num_dict.update(TMP_config_num_dict)\n')
		f.write('TMP_time_dict_mytune = ' + time_dict_mytune.__str__() + '\n')
		f.write('time_dict_mytune.update(TMP_time_dict_mytune)\n')
		f.write('log_file_ansor = "' + log_file_ansor.__str__() + '"\n')








# TUNE A SET OF OPS USING ANSOR ==========================================			
def tune_END2END_ansor(tasks_ansor, task_weights_ansor, log_file_ansor, infor_file_ansor):
	print("Begin tuning Using ANSOR...")
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	all_infor_Ansor = MyCollectTuneInfor()
	tuner = auto_scheduler.TaskScheduler(tasks_ansor, task_weights_ansor, 
		callbacks = [all_infor_Ansor,])
	# 
	# log_file = "allops_test_Ansor_bertBase_batch1.json"
	# log_file_ansor = "allops_tune_Ansor_DCGAN_batch1_Test1.json"
	tune_option = auto_scheduler.TuningOptions(
	    num_measure_trials=1000*len(tasks_ansor),  # change this to 20000 to achieve the best performance
	    runner=measure_ctx.runner,
	    measure_callbacks=[auto_scheduler.RecordToFile(log_file_ansor)],
	    verbose=0, #silent no print
	)
	# 
	time_begin_tot_ansor = time.time()
	tuner.tune(tune_option)
	tot_time_used_ansor = time.time() - time_begin_tot_ansor
	# 
	with open(infor_file_ansor, 'w') as f:
		f.write('all_infor_Ansor.pre_search_time_history = ' + all_infor_Ansor.pre_search_time_history.__str__() + '\n')
		f.write('all_infor_Ansor.pre_best_score_history = ' + all_infor_Ansor.pre_best_score_history.__str__() + '\n')
		f.write('all_infor_Ansor.pre_config_num_history = ' + all_infor_Ansor.pre_config_num_history.__str__() + '\n')
		f.write('all_infor_Ansor.post_search_time_history = ' + all_infor_Ansor.post_search_time_history.__str__() + '\n')
		f.write('all_infor_Ansor.post_best_score_history = ' + all_infor_Ansor.post_best_score_history.__str__() + '\n')
		f.write('all_infor_Ansor.post_config_num_history = ' + all_infor_Ansor.post_config_num_history.__str__() + '\n')
	# with open('tuner_resnet50_batch1_Test1.py', 'w') as f:
		f.write('best_costs = ' + tuner.best_costs.__str__() + '\n')
		f.write('task_cts = ' + tuner.task_cts.__str__() + '\n')
		f.write('task_costs_history  = ' + tuner.task_costs_history.__str__() + '\n')



