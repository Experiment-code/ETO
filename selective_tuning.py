'''
This file contains the code for the baseline method: selective tuning.
The code is implemented according to description of selective tuning in TVM's github.
'''
import time
import numpy as np
import json

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


def get_best_record_from_all_others(
	filename, workload_key, target, include_compatible=True):
	'''
		Extract the best record from the given record file.
		RMK: 
			(1) we only consider the record for other tasks 
			than the task with the exact workload_key.
		INPUT:
			filename:
				str. File name to load log from.
			workload_key: 
				str. The workload key of the compute declaration.
        		CANNOT BE `None`.
			target : 
    			tvm.target.Target. The target device.
        		CANNOT BE `None`.
			include_compatible: 
				bool. When set to True, 
				all compatible records in the log file will be considered.
    Returns
    -------
    input : auto_scheduler.measure.MeasureInput
        The best State's MeasureInput from this log fine.
    result : auto_scheduler.measure.MeasureResult
        The best State's MeasureResult from this log fine.
	'''
	from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureCallback
	from tvm.auto_scheduler.utils import calc_workload_dis_factor, decode_workload_key
	# 
	log_reader = auto_scheduler.measure_record.RecordReader(filename)
	best_cost = 1e30
	best_inp = None
	best_res = None
	# 
	for inp, res in log_reader:
		if res.error_no != MeasureErrorNo.NO_ERROR:
			continue
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		# 
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		# 
		if workload_key is not None:
			if workload_key == inp.task.workload_key:
				# we only reuse records from other tasks
				continue
			dis_f = calc_workload_dis_factor(
				decode_workload_key(workload_key), decode_workload_key(inp.task.workload_key)
			)
			if dis_f == float("inf"):
				continue
			if not include_compatible and dis_f != 1:
				continue
			# 
			# Since different workloads have different FLOPS, we multiply the factor to
			# eliminate this difference, which is basically the concept of throughput.
			cost *= dis_f
		#
		if cost < best_cost:
			best_cost = cost
			best_inp = inp
			best_res = res
	# 
	return best_inp, best_res







def get_topk_record_exact_match(
	filename, workload_key, target, topk):
	'''
		Extract the top k record from the given record file.
		RMK: 
			(1) we only consider the record for other tasks 
			than the task with the exact workload_key.
		INPUT:
			filename:
				str. File name to load log from.
			workload_key: 
				str. The workload key of the compute declaration.
        		CANNOT BE `None`.
			target: 
    			tvm.target.Target. The target device.
        		CANNOT BE `None`.
        	topk: 
        		int. The topk best records we want. 
        		If None, return all records in a sorted list.
    Returns
    -------
    input : auto_scheduler.measure.MeasureInput
        The best State's MeasureInput from this log fine.
    result : auto_scheduler.measure.MeasureResult
        The best State's MeasureResult from this log fine.
	'''
	from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureCallback
	from tvm.auto_scheduler.utils import calc_workload_dis_factor, decode_workload_key
	# 
	log_reader = auto_scheduler.measure_record.RecordReader(filename)
	# best_cost = [1e30 for i in range(topk)]
	# best_inp = list()
	# best_res = list()
	best_pairs = list()
	# 
	for inp, res in log_reader:
		if res.error_no != MeasureErrorNo.NO_ERROR:
			continue
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		if workload_key != inp.task.workload_key:
			continue
		# 
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		best_pairs.append([inp, res, cost])
		assert len(inp.state.stages) == 0
	# 
	# return topk small cost pairs
	best_pairs.sort(key=lambda pair: pair[-1])
	return best_pairs[:topk]










def my_calc_workload_similar_factor(op_para_t, op_para_r, sketch_hasCrossThrdReduc = False):
	'''
		This function calculate the similarity between two tasks, 
		using the method in selective tuning.
		INPUT:
			op_para_t, op_para_r: the dictionary storing necessary infor for 
				the task to tune and the task reused respectively.
			sketch_hasCrossThrdReduc: bool. if True, we consider the sketch which is tiled 
				using the rule of cross_thread_reduction as well, 
				besides the sketch tiled using multi_level_tile_rule.
		OUTPUT:
			similarity: float.
	'''
	def get_overlap_union(v1, v2, tile_level, dict_refer):
		'''
			Compute the #diff and #union of all tiling solutions with given tile level.
			The overlap ratio can be computed by:
				#(common tile solution) / #(union tile solution)
			INPUT:
				v1, v2: 
					int. the length of an axis to be tiled.
				tile_level: 
					int. if 2, then the axis is tiled into 2 parts.
				dict_refer: 
					dictionary. {(length, tile_level): [solution1, ...]}, 
					stores all the possible tile solution to refer is needed.
			OUTPUT:
				tuple: (#overlap, #union)
		'''
		keys = list(range(1, tile_level + 1))
		# print(keys)
		# 
		if (v1, tile_level) in dict_refer.keys():
			all_tiles1 = dict_refer[(v1, tile_level)]
		else:
			all_tiles1 = get_combinations(v1, keys)
			# delete the first level
			del all_tiles1[1] 
			all_tiles1 = dict2list(all_tiles1, keys[1:])
			dict_refer[(v1, tile_level)] = all_tiles1
		# 
		if (v2, tile_level) in dict_refer.keys():
			all_tiles2 = dict_refer[(v2, tile_level)]
		else:
			all_tiles2 = get_combinations(v2, keys)
			# delete the first level
			del all_tiles2[1] 
			all_tiles2 = dict2list(all_tiles2, keys[1:])
			dict_refer[(v2, tile_level)] = all_tiles2
		# compute the ratio
		overlap = 0
		diff = 0
		for i in all_tiles2:
			if i in all_tiles1:
				overlap += 1
			else:
				diff += 1
		return overlap , (len(all_tiles1) + diff)
	# 
	dict_refer = dict()
	loop_t = op_para_t['loop']
	loop_r = op_para_r['loop']
	tmp =\
		[get_overlap_union(loop_t[i], loop_r[i], 5, dict_refer)\
			for i in op_para_t['space_iters']] +\
		[get_overlap_union(loop_t[i], loop_r[i], 3, dict_refer)\
		for i in op_para_t['reduc_iters']]
	overlap = get_product([pair[0] for pair in tmp])*5 # because we have 5 unroll pragma
	tot = get_product([pair[1] for pair in tmp])*5
	if sketch_hasCrossThrdReduc:
		# we need to consider the second sketch as well
		fusedR_t = get_product([loop_t[i] for i in op_para_t['reduc_iters']])
		fusedR_r = get_product([loop_r[i] for i in op_para_r['reduc_iters']])
		tmp2 = get_overlap_union(fusedR_t, fusedR_r, 2, dict_refer)
		# tmp.append(tmp2)
		overlap = overlap + tmp2[0]*5 #because we have 5 unroll pragma
		tot = tot + tmp2[1]*5
	# overlap = get_product([pair[0] for pair in tmp])
	# tot = get_product([pair[1] for pair in tmp])
	# return (overlap + 10) / (tot + 10) # seems this 10 is of no use
	# print((overlap) / (tot))
	return (overlap) / (tot) # seems this 10 is of no use




# the functions below are from TVM directly, with some modification
def compute_psm(tasks, sketch_hasCrossThrdReduc = False):
	"""Compute a pairwise similarity matrix (PSM) for given tasks.
	Parameters
	----------
	tasks: List[op_paras]
	    the op_paras of the tasks to be computed.
	Returns
	-------
	psm: List[List[Float]]
	    a NxN PSM. PSM(i, j) is the similarity of task i and task j.
	"""
	psm = [[1.0 for _ in range(len(tasks))] for _ in range(len(tasks))]
	for idx1, task1 in enumerate(tasks):
		for idx2 in range(idx1 + 1, len(tasks)):
			psm[idx1][idx2] = psm[idx2][idx1] = my_calc_workload_similar_factor(task1, tasks[idx2], sketch_hasCrossThrdReduc)
	# 
	# print('Pairwise Similarity Matrix:')
	# for row in psm:
		# print('%s -> %.2f' % (', '.join(
		# 	['{:.2f}'.format(r) if r >= 0.01 else '----'
		# 		for r in row]), sum(row)))
	return psm



def psm_clustering(tasks, sketch_hasCrossThrdReduc = False, SM_threshold = 0.01):
	"""Cluster given tasks to several groups and select one task per group using
	pairwise simularity matrix (PSM) and graph cliques. We first compute the PSM
	that includes simularity rate (SM) of any two tasks. The simularity rate is the
	ratio of overlapped tuning space between tasks. Accordingly, we create a graph
	with tasks as nodes and SM (>=0.01) as edge weight, and then find all cliques
	in the graph as clusters. For the task that has been included by more than one
	cliques, we assign it to the clique with higher weight sum between the task and
	other tasks in the clique. Finally, every non-empty clique is a cluster, and
	the task with the highest weight sum is the representative task of that cluster,
	meaning that all other tasks in the cluster will depend on it.
	Parameters
	----------
	# tasks: List[autotvm.task.Task]
	#     the tasks to be clustered.
	tasks: List[op_paras]
	    the op_paras of the tasks to be computed.    
	Returns
	-------
	(centroids, labels): Tuple[List[int], List[int]]
	    the index of selected tasks and the cluster each task belongs to.
	"""
	try:
		import networkx as nx
	except ImportError:
		raise ImportError(
			'Missing required package for task selection. Please run "pip3 install networkx" '
			'and try again')
	# 
	def weight_sum(psm, prim, targets):
		"""Sum of the simularity rates of prim task to target tasks"""
		return sum([psm[prim][t] for t in targets])
	# 
	# Precompute the pairwise similarity matrix (PSM)
	psm = compute_psm(tasks, sketch_hasCrossThrdReduc)
	# 
	# Create a graph with task index as nodes and PSM as edge weights
	graph = nx.Graph()
	graph.add_nodes_from(range(len(tasks)))
	graph.add_edges_from([(i, j) for i in range(len(tasks))
							for j in range(i + 1, len(tasks))
							if psm[i][j] >= SM_threshold])
	# 
	# Cluster assignment for each task (List[clique index], assigned cluster index)
	assigned = [([], None) for _ in range(len(tasks))]
	# 
	# Find cliques and initialize clusters
	clusters = []
	for cidx, clique in enumerate(nx.find_cliques(graph)):
		clusters.append(set())
		for idx in clique:
			assigned[idx][0].append(cidx)
	# 
	# Assign the tasks that only belong to one clique to the cluster
	for idx in range(len(tasks)):
		if len(assigned[idx]) == 1:
			clusters[assigned[idx][0][0]].add(idx)
			assigned[idx][1] = assigned[idx][0][0]
	# 
	changed = True
	while changed:
		# if True: #logger.isEnabledFor(logging.DEBUG):
			# print('Round')
			# for idx, clut in enumerate(clusters):
			# 	print('%d: %s' % (idx, ','.join([str(i) for i in clut])))
		changed = False
		for idx in range(len(tasks)):
			if len(assigned[idx]) == 1:
				continue
			new_cidx = max(assigned[idx][0], key=lambda c: weight_sum(psm, idx, clusters[c]))
			if new_cidx != assigned[idx][1]:
				changed = True
				clusters[new_cidx].add(idx)
				if assigned[idx][1] is not None:
					clusters[assigned[idx][1]].remove(idx)
				assigned[idx] = (assigned[idx][0], new_cidx)
	# 
	# Create labels
	labels = [label for _, label in assigned]
	# 
	# For each cluster, select the task that has the maximum weight sum to other tasks in cluster
	centroids = []
	for clut in clusters:
		if clut:
			centroids.append(max(clut, key=lambda p: weight_sum(psm, p, clut)))
		else: # Empty cluster
			centroids.append(-1)
	return centroids, labels



def mark_depend(tasks, sketch_hasCrossThrdReduc = False, SM_threshold = 0.01):
	"""Mark some tasks to depend on other tasks by assigning task.depend = other_task.
	It means the tuner will only use the top schedules of the dependent task when
	that task has been tuned already.
	Parameters
	----------
	# tasks: List[tvm.autotvm.task.Task]
	#     the tasks to be analyzed and marked.
	tasks: List[op_paras]
        the op_paras of the tasks to be computed.    
	"""
    # 
	# assert all([t.workload is not None
	#             for t in tasks]), "One or more tasks have undefined workload"
	# 
	# if not tasks or tasks[0].target.target_name != 'cuda':
	#     logging.warning('Make dependent tasks for CPU is unstable')
	# 
	centroids, labels = psm_clustering(tasks, sketch_hasCrossThrdReduc, SM_threshold)
    # 
    # if logger.isEnabledFor(logging.DEBUG):
	# print('Selected task index: %s' % ', '.join([str(c) for c in centroids]))
	# print('Dependent task index: %s' % ', '.join([str(p) for p in labels]))
	# 
	for idx, task in enumerate(tasks):
		if labels[idx] != -1:
			task['depend'] = centroids[labels[idx]] #tasks[centroids[labels[idx]]]
		else:  # Outliers depend on itself to guarantee the performance
			task['depend'] = idx
			# logger.debug('task %s does not have dependent', str(task))
	# 
    # logger.info('Select %d tasks over %d tasks ',
    #             sum([1 if t.depend == t else 0 for t in tasks]), len(tasks))
	# print('Select %d tasks over %d tasks ',
 #                sum([1 if tasks[i]['depend'] == i else 0 for i in range(len(tasks))]), len(tasks))






# =============================================================================
# THE FUNCTION FOR RUNNING SINGLE PAIR EXPERIMENTS
def ST_with_feedback(search_tasks, to_tune_idx, reuse_from_idx, 
	best_reuse_results_ST, config_num_dict_ST, time_dict_ST, 
	log_file, tune_option, topk, time_begin_dict_ST = dict()):
	''' 
		Get the configs of the reused task from log_file. 
		This mathod keep on finding valid configs for the task to be tuned, with a stride 3. 
	'''
	# print(reuse_from_idx)
	time_begin = time.time()
	target = tvm.target.Target("cuda")
	topk_begin = 0
	best_pairs_list = None
	# 
	task_to_tune = search_tasks[to_tune_idx]
	wlk_to_tune = task_to_tune.workload_key
	# j = op_para_list[i]['depend']
	task_reuse_from = search_tasks[reuse_from_idx]
	wlk_reuse_from = task_reuse_from.workload_key	
	best_pairs_list = get_topk_record_exact_match(
		log_file, wlk_reuse_from, target, None)
	# inputs_reuse_from = inputs_reuse_from + [pair[0] for pair in topk_pairs]
	search_policy = get_search_policy(task_to_tune)
	tot_config_num = 0
	while topk_begin < len(best_pairs_list):
		topk_pairs = best_pairs_list[topk_begin : topk_begin + topk]
		topk_begin += topk
		inputs_reuse_from = [pair[0] for pair in topk_pairs]
		state_objs_r = [inp.state if isinstance(inp.state, auto_scheduler.loop_state.StateObject) else inp.state.state_object for inp in inputs_reuse_from]
		states_to_measure = [task_to_tune.compute_dag.infer_bound_from_state(state_obj_r) for state_obj_r in state_objs_r]
		stateObjs_to_measure = [state.state_object for state in states_to_measure]
		measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
				stateObjs_to_measure)
		tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
		best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
		if len(best_states) == 0:
			# there is no valid state in this measure round
			update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (None, float(0)), best_reuse_results_ST)
			continue
		best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
		update_reuse_result_dict((wlk_to_tune, wlk_reuse_from), (best_states[0], float(best_result)), best_reuse_results_ST)
		break
	config_num_dict_ST[(to_tune_idx,reuse_from_idx)] = tot_config_num
	time_dict_ST[(wlk_to_tune, wlk_reuse_from)] = time.time() - time_begin
	time_begin_dict_ST[(wlk_to_tune, wlk_reuse_from)] = time_begin # store the begin time of this round of reuse, for the experiment of ST vary K
	# print((i,j), best_measure_pairs_ST[wlk_to_tune][1])






# ============================================================================
# THIS FUNCTION STORES THE PERFORMANCE LOG FOR EACH REUSED SCHEDULE
# there is no feedback: it tunes exactly topK configurations.
def ST_no_feedback_store_each_record_in_dict_given_states(task_to_tune, topk_pairs, 
	all_reuse_results_ST, config_num_dict_ST, time_dict_ST, 
	tune_option, time_begin_dict_ST = dict()):
	''' 
		Get the configs of the reused task from log_file. 
		This mathod keep on finding valid configs for the task to be tuned, with a stride 3. 
	'''
	time_begin = time.time()
	target = tvm.target.Target("cuda")
	# 
	# task_to_tune = search_tasks[to_tune_idx]
	wlk_to_tune = task_to_tune.workload_key
	# 
	search_policy = get_search_policy(task_to_tune)
	tot_config_num = 0
	all_reuse_results_ST[wlk_to_tune] = dict()
	# while topk_begin < len(best_pairs_list):
	# 	topk_pairs = best_pairs_list[topk_begin : topk_begin + topk]
	# 	topk_begin += topk
	for pair in topk_pairs:
		inputs_reuse_from = [pair[0]]
		state_objs_r = [inp.state if isinstance(inp.state, auto_scheduler.loop_state.StateObject) else inp.state.state_object for inp in inputs_reuse_from]
		states_to_measure = [task_to_tune.compute_dag.infer_bound_from_state(state_obj_r) for state_obj_r in state_objs_r]
		stateObjs_to_measure = [state.state_object for state in states_to_measure]
		measurer = auto_scheduler._ffi_api.MyMeasureStoreBestState(search_policy, tune_option,
				stateObjs_to_measure)
		tot_config_num = tot_config_num + auto_scheduler._ffi_api.MyGetProgramMeasurerCt(measurer)
		best_states = auto_scheduler._ffi_api.MyGetProgramMeasurerBestState(measurer, wlk_to_tune)
		if len(best_states) == 0:
			# there is no valid state in this measure round
			all_reuse_results_ST[wlk_to_tune][inputs_reuse_from[0]] = (None, float(0))
		else:
			best_result = auto_scheduler._ffi_api.MyGetProgramMeasurerBestFlops(measurer, wlk_to_tune) 
			all_reuse_results_ST[wlk_to_tune][inputs_reuse_from[0]] = (best_states[0], float(best_result))
	# 
	config_num_dict_ST[wlk_to_tune] = tot_config_num
	time_dict_ST[wlk_to_tune] = time.time() - time_begin
	time_begin_dict_ST[wlk_to_tune] = time_begin # store the begin time of this round of reuse, for the experiment of ST vary K
	# print((i,j), best_measure_pairs_ST[wlk_to_tune][1])






# =============================================================================
# below runs the SINGLE PAIRS WITH SELECTIVE TUNING VARAINTS (VARY-K)
def SINGLE_PAIR_ST_variants(
	search_tasks_list,
	# loops_list,
	selected_op_pairs_list,
	log_file_ST,
	all_infor_ST_file, 
	log_file_ansor_mapping,
	):
	def delete_state(inp_res_list):
		new_list = list()
		for old_dict in inp_res_list:
			new_dict = dict()
			for k,v in old_dict.items():
				new_dict[k] = (None, v[1])
			new_list.append(new_dict)
		return new_list
	# 
	time_begin_tot_ST = time.time()
	best_reuse_results_ST_list = list()
	config_num_dict_ST_list = list()
	config_num_wlk_as_key_dict_ST_list = list()
	time_dict_ST_list = list()
	log_range_dict_ST_list = list()
	time_begin_dict_ST_list = list()
	# 
	for round_i in range(len(search_tasks_list)):
		search_tasks = search_tasks_list[round_i]
		# loops = loops_list[round_i]
		selected_op_pairs = selected_op_pairs_list[round_i]
		best_reuse_results_ST = dict()
		config_num_dict_ST = dict()
		config_num_wlk_as_key_dict_ST = dict()
		time_dict_ST = dict()
		log_range_dict_ST = dict()
		time_begin_dict_ST = dict()
		# 
		# log_file_ST = "SINGLEpair_test_ST_conv2d_winograd_Test1.json"
		# log_file_ansor = "SINGLEpair_tune_Ansor_conv2d_winograd_Test1.json"
		measure_ctx_ST = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
		tune_option_ST = auto_scheduler.TuningOptions(
		    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
		    runner=measure_ctx_ST.runner,
		    measure_callbacks=[auto_scheduler.RecordToFile(log_file_ST)],
		    verbose=0,
		)
		# 
		topk = 3 #1000
		# target = tvm.target.Target("cuda")
		for to_tune_idx, reuse_from_idx in selected_op_pairs:
			assert reuse_from_idx != -1, "no op to be tuned by Ansor in this run!"
			if not os.path.isfile(log_file_ST):
				lineNum_before_run = 0
			else:
				lineNum_before_run = get_lineNum(log_file_ST)
			log_file_ansor = None
			if type(log_file_ansor_mapping) == str:
				log_file_ansor = log_file_ansor_mapping
			else:
				log_file_ansor = log_file_ansor_mapping[search_tasks[reuse_from_idx].compute_dag.workload_key()][0]
			# 
			ST_with_feedback(search_tasks, to_tune_idx, reuse_from_idx, 
				best_reuse_results_ST, config_num_dict_ST, time_dict_ST, 
				log_file_ansor, tune_option_ST, topk, time_begin_dict_ST)
			lineNum_after_run = get_lineNum(log_file_ST)
			# record the log range for this reuse pair: [include_lower_bound, exclude_upper_bound)
			log_range_dict_ST[(search_tasks[to_tune_idx].workload_key, search_tasks[reuse_from_idx].workload_key)] = (lineNum_before_run, lineNum_after_run)
			config_num_wlk_as_key_dict_ST[(search_tasks[to_tune_idx].workload_key, search_tasks[reuse_from_idx].workload_key)] = \
				config_num_dict_ST[(to_tune_idx,reuse_from_idx)]
			assert (lineNum_after_run - lineNum_before_run) == config_num_dict_ST[(to_tune_idx,reuse_from_idx)], "Different log line num from config_num dict record!"
			# print((to_tune_idx, reuse_from_idx), best_measure_pairs_ST[wlk_to_tune][1], )
			# print((to_tune_idx, reuse_from_idx), "  ST: ", \
			# 	best_reuse_results_ST[(search_tasks[to_tune_idx].workload_key, search_tasks[reuse_from_idx].workload_key)][1], \
			# 	"  baseline: ", best_measure_pairs[search_tasks[to_tune_idx].workload_key][1])
		# 
		best_reuse_results_ST_list.append(best_reuse_results_ST)
		config_num_dict_ST_list.append(config_num_dict_ST)
		config_num_wlk_as_key_dict_ST_list.append(config_num_wlk_as_key_dict_ST)
		time_dict_ST_list.append(time_dict_ST)
		log_range_dict_ST_list.append(log_range_dict_ST)
		time_begin_dict_ST_list.append(time_begin_dict_ST)
	tot_time_used_ST = time.time() - time_begin_tot_ST
	print("total search time: ", tot_time_used_ST)
	# store the meta_analysis_results to file
	with open(all_infor_ST_file, 'w') as f:
		f.write('best_reuse_results_ST_variant_vary_K_list_no_state = ' + delete_state(best_reuse_results_ST_list).__str__() + '\n')
		f.write('config_num_wlk_as_key_dict_ST_variant_vary_K_list = ' + config_num_wlk_as_key_dict_ST_list.__str__()+ '\n')
		f.write('time_dict_ST_variant_vary_K_list = ' + time_dict_ST_list.__str__()+ '\n')
		f.write('log_range_dict_ST_variant_vary_K_list = ' + log_range_dict_ST_list.__str__()+ '\n')
		f.write('time_begin_dict_ST_variant_vary_K_list = ' + time_begin_dict_ST_list.__str__() + '\n')








# =============================================================================
# =============================================================================
# BELOW IS THE CODE FOR RUNNING END-TO-END SELECTIVE TUNING (WITH FALL-BACK MECHANISM) with K = 1000 for top K
# suppose we already have list of search_tasks, and corresponding loops
# SUPPOSE the tuning result for the center op of each cluster has been stored in the log file `log_file_ansor`.
def END2END_ST_variants(log_file, all_infor_ST_file, #log_file_ansor1, # if log_file_ansor1 contains the full logs according to the wlkAnsor2FullLogFile, use it
	# wlkAnsor2FullLogFile, 
	# all_infor_ST_ori, 
	# best_measure_pairs_single_op_Ansor, # for print purpose
	# config_num_dict_single_op_Ansor,
	# time_dict_single_op_Ansor, 
	variant, variant_param, # example SM_threshold_choices = [0.001, 0.1, 0.5]
	search_tasks_list, 
	loops_list, 
	# ALL_MODEL_TIME_BEGIN_DICT_file, 
	# model_name
	):
	'''
	! This function does not do analysis work, it only print logs to file.
	log_file: str. The file to store tuning logs for this run.
	log_file_ansor1: str. The full tuning logs for this run to reuse. 
						There is log_file_ansor2, which stores the full tuning logs for search tasks in all DNNs that is not selected as center ops by ori_ST.
	# wlkAnsor2FullLogFile: mapping each workload_key to the file where there is the full optimization logs by Ansor.
	# best_measure_pairs_single_op_Ansor: dict: [wlk: (state, GFLOPS)]. 
	# 										The best_measure_pair structure as stored in the tuning process. 
	# 										It stores the best measure pair for search tasks in all DNNs that is not selected as center ops by ori_ST.
	# ALL_MODEL_TIME_BEGIN_DICT_file: the name of the file which stores the time begin infor for each reuse pair in each model by each variant
	variant: str. 
				"vary_K", we would change the K in TopK to 1000.
				"vary_reuse_clusterNum", we would make TopK reuse all the center cluster ops' configs in the order from SM high to low, with K = 3 and SM threshold being 0.01.
				"vary_SM_threshold", we would change the SM threshold in building the graph for finding cliques with K=3 and only reuse 1 cluster center op. 
					[!Only this variant would change the number of clusters.]
	'''
	def delete_state(inp_res_list):
		new_list = list()
		for old_dict in inp_res_list:
			new_dict = dict()
			for k,v in old_dict.items():
				new_dict[k] = (None, v[1])
			new_list.append(new_dict)
		return new_list
	# 
	def print_psm_range(psm):
		tmp_array = list()
		for row in psm:
			tmp_array = tmp_array + [i for i in row if i < 1]
		if len(tmp_array) > 0:
			print("max, min, avg in psm: ", max(tmp_array), min(tmp_array), sum(tmp_array) / len(tmp_array))
		# 
	time_begin_tot_ST_vary_K = time.time()
	best_measure_pairs_ST_vary_K_list = list()
	best_reuse_results_ST_vary_K_list = list()
	config_num_dict_ST_vary_K_list = list()
	config_num_wlk_as_key_dict_ST_list = list()
	time_dict_ST_vary_K_list = list()
	time_begin_dict_ST_list = list()
	# 
	for round_i in range(len(search_tasks_list)):
		search_tasks = search_tasks_list[round_i]
		loops = loops_list[round_i]
		best_measure_pairs_ST_vary_K = dict()
		best_reuse_results_ST_vary_K = dict()
		config_num_dict_ST_vary_K = dict()
		config_num_wlk_as_key_dict_ST = dict()
		time_dict_ST_vary_K = dict()
		time_begin_dict_ST = dict()
		psm_for_variant = None
		# 
		op_para_list = [get_op_para_ansor(search_tasks[i], loops[i]) for i in range(len(search_tasks))]
		if variant == "vary_SM_threshold":
			if get_sketch_num(search_tasks[0]) > 1:
				mark_depend(op_para_list, True, variant_param)
			else:
				mark_depend(op_para_list, False, variant_param)
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
		# return
		# first tune selected tasks
		for i in selected_ops:
			reuse_pair = (i, -1)
			# need to tune this op
			time_begin = time.time()
			tune_seach_task(search_tasks, i, best_measure_pairs_ST_vary_K, dict(), log_file)
			time_dict_ST_vary_K[(search_tasks[i].workload_key, None)] = \
				time.time() - time_begin
			time_begin_dict_ST[(search_tasks[i].workload_key, None)] = time_begin
			# get the actual number of configs
			config_num_dict_ST[reuse_pair] = get_lineNum(log_file) - sum(config_num_dict_ST.values()) - \
				sum([sum(config_num_dict_ST_tmp.values()) for config_num_dict_ST_tmp in config_num_dict_ST_list])
			config_num_wlk_as_key_dict_ST[(search_tasks[i].workload_key, None)] = \
				config_num_dict_ST[reuse_pair]
		# print((i, -1), best_measure_pairs_ST_vary_K[search_tasks[i].workload_key][1])
		# 
		# log_file_ST_vary_K = "allops_test_ST_vary_K_resnet50_batch1_Test1_reuse_ST_Test3.json"
		# log_file_ansor1 = "allops_test_ST_resnet50_batch1_Test3.json"
		# log_file_ansor2 = "FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test1.json"
		measure_ctx_ST_vary_K = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
		tune_option_ST_vary_K = auto_scheduler.TuningOptions(
		    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
		    runner=measure_ctx_ST_vary_K.runner,
		    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
		    verbose=0,
		)
		# 
		if variant == "vary_K":
			topk = variant_param
		else:
			topk = 3
		target = tvm.target.Target("cuda")
		for to_tune_idx, reuse_from_idx in selected_op_pairs:
			wlk_r_ansor = search_tasks[reuse_from_idx].compute_dag.workload_key()
			if reuse_from_idx == -1:
				continue
			# if variant != "vary_SM_threshold":
			# 	assert log_file_ansor1 in wlkAnsor2FullLogFile[wlk_r_ansor], variant + " selects different centers."
			# 
			# if search_tasks[reuse_from_idx].workload_key not in list(best_measure_pairs_single_op_Ansor.keys()):
			# 	assert log_file_ansor1 in wlkAnsor2FullLogFile[wlk_r_ansor], "reuse from wrong file for " + variant
			# if log_file_ansor1 in wlkAnsor2FullLogFile[wlk_r_ansor]:
			# 	ST_with_feedback(search_tasks, to_tune_idx, reuse_from_idx, 
			# 		best_reuse_results_ST_vary_K, config_num_dict_ST_vary_K, time_dict_ST_vary_K, 
			# 		log_file_ansor1, tune_option_ST_vary_K, topk, time_begin_dict_ST)
			# else:
			# 	ST_with_feedback(search_tasks, to_tune_idx, reuse_from_idx, 
			# 		best_reuse_results_ST_vary_K, config_num_dict_ST_vary_K, time_dict_ST_vary_K, 
			# 		wlkAnsor2FullLogFile[wlk_r_ansor][0], tune_option_ST_vary_K, topk, time_begin_dict_ST)
			ST_with_feedback(search_tasks, to_tune_idx, reuse_from_idx, 
					best_reuse_results_ST_vary_K, config_num_dict_ST_vary_K, time_dict_ST_vary_K, 
					log_file, tune_option_ST_vary_K, topk, time_begin_dict_ST)
			config_num_wlk_as_key_dict_ST[(search_tasks[to_tune_idx].workload_key, search_tasks[reuse_from_idx].workload_key)] = \
				config_num_dict_ST[(to_tune_idx,reuse_from_idx)]
		# 
		best_measure_pairs_ST_vary_K_list.append(best_measure_pairs_ST_vary_K)
		best_reuse_results_ST_vary_K_list.append(best_reuse_results_ST_vary_K)
		config_num_dict_ST_vary_K_list.append(config_num_dict_ST_vary_K)
		config_num_wlk_as_key_dict_ST_list.append(config_num_wlk_as_key_dict_ST)
		time_dict_ST_vary_K_list.append(time_dict_ST_vary_K)
		time_begin_dict_ST_list.append(time_begin_dict_ST)
	# 
	tot_time_used_ST_vary_K = time.time() - time_begin_tot_ST_vary_K
	print("total search time (reuse part): ", tot_time_used_ST_vary_K)
	# with open(ALL_MODEL_TIME_BEGIN_DICT_file, 'a') as f:
	# 	f.write('time_begin_dict_ST_variant[{}][{}][{}] = {}\n'.format(variant, variant_param, model_name, time_begin_dict_ST_list.__str__()))
	# store the meta_analysis_results to file
	with open(all_infor_ST_file, 'w') as f:
		f.write('best_measure_pairs_ST_variant_vary_K_list_no_state = ' + delete_state(best_measure_pairs_ST_vary_K_list).__str__() + '\n')
		f.write('best_reuse_results_ST_variant_vary_K_list_no_state = ' + delete_state(best_reuse_results_ST_vary_K_list).__str__() + '\n')
		f.write('config_num_wlk_as_key_dict_ST_variant_vary_K_list = ' + config_num_wlk_as_key_dict_ST_list.__str__()+ '\n')
		f.write('time_dict_ST_variant_vary_K_list = ' + time_dict_ST_vary_K_list.__str__()+ '\n')
		f.write('time_begin_dict_ST_variant_vary_K_list = ' + time_begin_dict_ST_list.__str__() + '\n')




