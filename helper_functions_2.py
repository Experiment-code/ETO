
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




def get_search_policy(search_task):
	cost_model = auto_scheduler.cost_model.XGBModel()
	search_policy = auto_scheduler.search_policy.SketchPolicy(\
		search_task, cost_model, verbose = 0)
	return search_policy



def can_reuse_directly(loop1, loop2):
	'''
		If loop1 can reuse loop2 directly, return (True, None); else return (False, intermediate dummy loop).
	'''
	smaller_n = 0
	larger_n = 0
	dummy_loop = list()
	assert len(loop1) == len(loop2), "These two ops cannot reuse each other:" + loop1 + ", " + loop2
	for i in range(len(loop1)):
		if loop1[i] < loop2[i]:
			smaller_n += 1
			dummy_loop.append(loop1[i])
		elif loop1[i] > loop2[i]:
			larger_n += 1
			dummy_loop.append(loop2[i])
		else:
			dummy_loop.append(loop1[i])
	if smaller_n > 0 and larger_n > 0:
		return (False, dummy_loop)
	else:
		return (True, None)




def get_sketch_num(task):
	search_policy = get_search_policy(task)
	sketches = search_policy.generate_sketches(print_for_debug=False)
	return len(sketches)



def group_tasks(tasks):
	groups = dict()
	for task in tasks:
		groups[task.name] = list()
	for task in tasks:
		groups[task.name].append(task)
	return groups






def group_tasks_ansor(tasks):
	# the input are SearchTasks of Ansor
	import json
	def get_workload_dict(task):
		workload_dict = json.loads(s='{"wlk": ' + task.workload_key + '}')["wlk"]
		return workload_dict
	# 
	groups = dict()
	for task in tasks:
		workload = get_workload_dict(task)
		groups[workload[0]+'.cuda'] = list()
	for task in tasks:
		workload = get_workload_dict(task)
		groups[workload[0]+'.cuda'].append(task)
	return groups







def print_code_for_reuse_graph(edge_cost_dict, ori_op_num, tot_op_num):
	func_no = 1 # the index of the function
	idx_offset = 2
	MAX_LEN_FUNC = 40000
	edge_list = list(edge_cost_dict.keys())
	with open('SelectiveReusePairs.java', 'w') as f:
		edge_no = 0 # start from edge 0
		while True:
			f.write('public static DirectedGraph get_dg%d() {' % func_no)
			if edge_no == 0:
				f.write("DirectedGraph dg = new DirectedGraph();")
				# add vertice
				f.write("for (int i = %d; i <= %d; i++)" % (1, 1+tot_op_num))
				f.write("\tdg.addVertice(i);")
			else:
				f.write("DirectedGraph dg = get_dg%d();" % (func_no-1))
			# add edges
			edge_str = ""
			for k_i in range(edge_no, len(edge_list)): # edge_cost_dict.keys():
				k = edge_list[k_i]
				edge_str = edge_str + ("dg.addDirectedEdges(%d, %d);" % \
					(k[1] + idx_offset, k[0] + idx_offset)) # change the order
				if len(edge_str) > MAX_LEN_FUNC:
					# we need a new func
					func_no += 1
					edge_no = k_i + 1
					break
			f.write(edge_str)
			f.write("return dg;}\n")
			if len(edge_str) <= MAX_LEN_FUNC:
				# all edges are enumerated
				break
		# 
		# print edge costs
		func_no += 1
		edge_no = 0 # start from edge 0
		while True:
			f.write('public static SteinerDirectedInstance get_dg%d() {' % func_no)
			if edge_no == 0:
				f.write("DirectedGraph dg = get_dg%d();" % (func_no-1))
				f.write("SteinerDirectedInstance sdi = new SteinerDirectedInstance(dg);")
				# set root
				f.write("sdi.setRoot(1);")
				# set terminals
				terminals = str(tuple(range(2, ori_op_num + idx_offset)))
				f.write("sdi.setRequiredNodes"+ terminals +";")
			else:
				f.write("SteinerDirectedInstance sdi = get_dg%d();" % (func_no-1))
			# set edge costs
			edge_str = ""
			for k_i in range(edge_no, len(edge_list)): # edge_cost_dict.keys():
				k = edge_list[k_i]
				v = edge_cost_dict[k]
				edge_str = edge_str + ("sdi.setCost(%d, %d, %.2f);" % \
					(k[1] + idx_offset, k[0] + idx_offset, v)) # change the order
				if len(edge_str) > MAX_LEN_FUNC:
					# we need a new func
					func_no += 1
					edge_no = k_i + 1
					break
			f.write(edge_str)
			f.write("return sdi;}\n")
			if len(edge_str) <= MAX_LEN_FUNC:
				# all edges are enumerated
				break
		# 
		f.write('public static void select_reuse_pairs() {')
		f.write("SteinerDirectedInstance sdi = get_dg%d();" % func_no)
		# solve the problem
		f.write("SteinerArborescenceApproximationAlgorithm alg = new GFLACAlgorithm();")
		f.write("alg.setInstance(sdi);")
		f.write("alg.compute();")
		# the remaining part are omitted, because we do not change it.
		f.write('System.out.println("Returned solution : " + alg.getArborescence());')
		f.write('System.out.println("Cost: " + alg.getCost());')
		f.write('System.out.println("Running Time: " + alg.getTime() + " ms");')
		f.write('String selected = "[";')
		f.write('for (Arc a : alg.getArborescence())')
		f.write('selected += "("+a.getInput()+", " + a.getOutput() + "),";')
		f.write('selected += "]";')
		f.write('System.out.println(selected);')
		f.write('for (Arc a : alg.getArborescence())')
		f.write('sdi.getGraph().setColor(a, Color.RED);')
		f.write('new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());')
		f.write('}')
		# 
	# with open('SelectiveReusePairs.java', 'w') as f:
	# 	f.write('public static void select_reuse_pairs() {')
	# 	f.write("DirectedGraph dg = new DirectedGraph();")
	# 	idx_offset = 2
	# 	# add vertice
	# 	f.write("for (int i = %d; i <= %d; i++)" % (1, 1+tot_op_num))
	# 	f.write("\tdg.addVertice(i);")
	# 	# add edges
	# 	for k in edge_cost_dict.keys():
	# 		f.write("dg.addDirectedEdges(%d, %d);" % \
	# 			(k[1] + idx_offset, k[0] + idx_offset)) # change the order
	# 	f.write("SteinerDirectedInstance sdi = new SteinerDirectedInstance(dg);")
	# 	# set root
	# 	f.write("sdi.setRoot(1);")
	# 	# set terminals
	# 	terminals = str(tuple(range(2, ori_op_num + idx_offset)))
	# 	f.write("sdi.setRequiredNodes"+ terminals +";")
	# 	# set edge costs
	# 	for k,v in edge_cost_dict.items():
	# 		f.write("sdi.setCost(%d, %d, %.2f);" % \
	# 			(k[1] + idx_offset, k[0] + idx_offset, v)) # change the order
	# 	# solve the problem
	# 	f.write("SteinerArborescenceApproximationAlgorithm alg = new GFLACAlgorithm();")
	# 	f.write("alg.setInstance(sdi);")
	# 	f.write("alg.compute();")
	# 	# the remaining part are omitted, because we do not change it.
	# 	f.write('System.out.println("Returned solution : " + alg.getArborescence());')
	# 	f.write('System.out.println("Cost: " + alg.getCost());')
	# 	f.write('System.out.println("Running Time: " + alg.getTime() + " ms");')
	# 	f.write('String selected = "[";')
	# 	f.write('for (Arc a : alg.getArborescence())')
	# 	f.write('selected += "("+a.getInput()+", " + a.getOutput() + "),";')
	# 	f.write('selected += "]";')
	# 	f.write('System.out.println(selected);')
	# 	f.write('for (Arc a : alg.getArborescence())')
	# 	f.write('sdi.getGraph().setColor(a, Color.RED);')
	# 	f.write('new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());')
	# 	f.write('}')






def solve_directed_spanning_tree(edge_cost_dict, ori_op_num, tot_op_num):
	idx_offset = 2
	G = networkx.DiGraph()
	# add vertice
	G.add_nodes_from(range(1, 2+tot_op_num))
	# add edges
	for k,v in edge_cost_dict.items():
		G.add_edge(k[1] + idx_offset, k[0] + idx_offset, weight=v) # change the order
	# 
	mst = networkx.algorithms.tree.branchings.minimum_spanning_arborescence(G)
	# mst = mysolver.find_optimum(kind="min",)
	selected_op_pairs = list()
	for u, v, weight in mst.edges(data="weight"):
		selected_op_pairs.append((u, v))
	print(networkx.algorithms.tree.branchings.branching_weight(mst))
	return selected_op_pairs



def process_selected_op_pairs(selected_op_pairs):
	# the selected op pairs need to be sorted in the order of tuning
	# they also need minus the idx_offset, and reversed
	new_selected_pairs = list()
	visited = list()
	root = 1 # the root in the graph in java
	idx_offset = 2
	edge_dict = dict()
	for edge in selected_op_pairs:
		if edge[0] not in edge_dict.keys():
			edge_dict[edge[0]] = [edge[1]]
		else:
			edge_dict[edge[0]].append(edge[1])
	# 
	new_visited = [root]
	while(len(new_visited) > 0):
		tmp = list()
		for v1 in new_visited:
			if v1 not in edge_dict.keys():
				continue
			for v2 in edge_dict[v1]: 
				new_selected_pairs.append((v2-idx_offset, v1-idx_offset))
				tmp.append(v2)
		new_visited = tmp
	# 
	assert len(new_selected_pairs) == len(selected_op_pairs)
	for v1, v2 in selected_op_pairs:
		assert (v2-idx_offset, v1-idx_offset) in new_selected_pairs
	# 
	return new_selected_pairs



def get_supported_Tasks_n_Weights_for_Ansor(tasks_ansor, task_weights_ansor, 
	search_tasks_list, ori_op_num_list):
	supported_idx = list()
	for i in range(len(tasks_ansor)):
		flag = False
		for j in range(len(search_tasks_list)):
			for k in range(ori_op_num_list[j]):
				if search_tasks_list[j][k].compute_dag.workload_key() == \
					tasks_ansor[i].compute_dag.workload_key():
					supported_idx.append(i)
					flag = True
					break
			if flag:
				break
	assert len(supported_idx) == sum(ori_op_num_list)
	new_tasks_ansor = [tasks_ansor[i] for i in supported_idx]
	new_task_weights_ansor = [task_weights_ansor[i] for i in supported_idx]
	return new_tasks_ansor, new_task_weights_ansor





def tune_seach_task(search_tasks, to_tune_idx, best_measure_pairs, time_dict,\
	log_file, num_measure_trials=1000):
	task = search_tasks[to_tune_idx]
	# log_file = "allops_test_new_hrc_bertBase_batch1.json"
	measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
	tune_option = auto_scheduler.TuningOptions(
		num_measure_trials=num_measure_trials,  # change this to 1000 to achieve the best performance
		runner=measure_ctx.runner,
		measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
		verbose=0, #silent for no print
	)
	# Run auto-tuning (search)
	time_begin = time.time()
	task.tune(tune_option, search_policy = get_search_policy(task))
	time_dict[task.workload_key] = time.time() - time_begin
	# Apply the best schedule
	inp, res = auto_scheduler.measure_record.load_best_record(log_file, task.workload_key)
	if inp == None:
		raise RuntimeError(
			"Cannot find any valid schedule for %s in file %s" % (task.workload_key, log_file)
		)
	# 
	inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
	best_measure_pairs[task.workload_key] = (inp.state, task.compute_dag.flop_ct / res.costs[0].value / 1e9)
	# Kill the measurement process
	del measure_ctx





class MyHardwareAnalyser():
	"""
	Store the hardware information, and do some simple analysis.
	"""
	def __init__(self):
		# all mem using bit as the unit
		self.KB = 1024*8 # bit
		self.Word = 32 # bit
		self.K = 1024
		self.max_shared_mem_perblock = 48 * self.KB
		self.max_shared_mem_perSM = 64 * self.KB
		self.max_blk_perSM = 32
		self.max_register_perSM = 65536 * self.Word
		self.max_register_perBlk = 65536 * self.Word
		self.max_register_perThrd = 255 * self.Word
		self.max_thread_perSM = 2048
		self.max_thread_perBlk = 1024
		self.blkNum_perSM = None
		self.thread_required_perBlk = None
		self.shared_mem_required_perBlk = None # for a block
		self.register_perThrd = None
		self.min_registers_required_perThrd = None # for a thread
	# 
	def reset():
		self.blkNum_perSM = None
		self.thread_required_perBlk = None
		self.shared_mem_required_perBlk = None # for a block
		self.register_perThrd = None
		self.min_registers_required_perThrd = None # for a thread
	# 
	def set_ThrdNum(thrd_size):
		''' Check whether the required thrd num is under limit and set. '''
		self.thread_required_perBlk = thrd_size
		if thrd_size > self.max_thread_perBlk:
			return False
		else:
			return True
	# 
	def set_sharedmemRequired(shared_mem_amount):
		''' Check whether the required shared mem amout is under limit and set. '''
		self.shared_mem_required_perBlk = shared_mem_amount
		if self.shared_mem_required_perBlk > self.max_shared_mem_perblock:
			return False
		else:
			return True
	# 
	def estimate_blk_num_perSM():
		'''
			Estimate the number of blocks on a SM at the same time
		'''
		self.blkNum_perSM = min(self.max_shared_mem_perSM // \
			self.shared_mem_required, self.max_blk_perSM)
		if self.blkNum_perSM * self.thread_required_perBlk>self.max_thread_perSM:
			self.blkNum_perSM = 0
		return self.blkNum_perSM
	# 
	def get_registerNum_perThrd():
		''' Compute the number of registers assigned to a thread. '''
		self.register_perThrd = min(self.max_register_perThrd, 
			self.max_register_perSM // self.blkNum_perSM // self.thread_required_perBlk)
		return self.register_perThrd
	# 





def get_lineNum(filename):
	if os.path.isfile(filename) == False:
		# this file does not exist
		return 0
	count=-1
	for count, line in enumerate(open(filename,'rU')):
		pass
	count+=1
	return count





def get_best_ct_time_given_wlk(filename, workload_key, cost_compare):
	from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureCallback
	from tvm.auto_scheduler.utils import calc_workload_dis_factor, decode_workload_key
	# 
	log_reader = auto_scheduler.measure_record.RecordReader(filename)
	target = tvm.target.Target("cuda")
	# best_cost = [1e30 for i in range(topk)]
	# best_inp = list()
	# best_res = list()
	best_pairs = list()
	timestamp_begin = None
	# 
	for inp, res in log_reader:
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		if workload_key != inp.task.workload_key:
			continue
		# 
		if timestamp_begin == None:
			timestamp_begin=res.timestamp
		if res.error_no != MeasureErrorNo.NO_ERROR:
			best_pairs.append([inp, res, float('inf')])
			continue
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		best_pairs.append([inp, res, cost])
		assert len(inp.state.stages) == 0
	# 
	# return the index of the smallest res
	cost_list = [i[2] for i in best_pairs]
	print(workload_key, len(best_pairs))
	assert min(cost_list) == best_pairs[np.argmin(np.array(cost_list))][2]
	# if cost_compare > 1:
	# 	# the cost is GFLOPS
	# 	cost_compare = best_pairs[np.argmin(np.array(cost_list))][0].task.compute_dag.flop_ct/cost_compare/1e9
	# 
	print(np.argmin(np.array(cost_list)), " out of ", len(best_pairs), "best by ansor: ", best_pairs[np.argmin(np.array(cost_list))][2], " ", cost_compare, 
		(cost_compare >= best_pairs[np.argmin(np.array(cost_list))][2] - 0.000001) \
			and (cost_compare <= best_pairs[np.argmin(np.array(cost_list))][2] + 0.000001), 
			best_pairs[np.argmin(np.array(cost_list))][1].timestamp - timestamp_begin)
	return np.argmin(np.array(cost_list)), \
		best_pairs[np.argmin(np.array(cost_list))][1].timestamp - timestamp_begin







def get_perff_sametime_given_wlk(filename, workload_key, given_time):
	from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureCallback
	from tvm.auto_scheduler.utils import calc_workload_dis_factor, decode_workload_key
	# 
	log_reader = auto_scheduler.measure_record.RecordReader(filename)
	target = tvm.target.Target("cuda")
	# best_cost = [1e30 for i in range(topk)]
	# best_inp = list()
	# best_res = list()
	best_pairs = list()
	timestamp_begin = None
	# 
	for inp, res in log_reader:
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		if workload_key != inp.task.workload_key:
			continue
		# 
		if timestamp_begin == None:
			timestamp_begin=res.timestamp
		if res.error_no != MeasureErrorNo.NO_ERROR:
			best_pairs.append([inp, res, float('inf')])
			# check the timestamp
			if res.timestamp - timestamp_begin > given_time:
				break
			continue
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		best_pairs.append([inp, res, cost])
		assert len(inp.state.stages) == 0
		# check the timestamp
		if res.timestamp - timestamp_begin > given_time:
			break
	# 
	# return the index of the smallest res
	cost_list = [i[2] for i in best_pairs]
	assert min(cost_list) == best_pairs[np.argmin(np.array(cost_list))][2]
	print("ansor time: ", res.timestamp - timestamp_begin, " given_time: ", given_time, abs(res.timestamp - timestamp_begin - given_time) < 10)
	best_inp = best_pairs[np.argmin(np.array(cost_list))][0]
	best_inp = auto_scheduler.measure.recover_measure_input(best_inp, rebuild_state=True)
	print("GFLOPS same time: ", best_inp.task.compute_dag.flop_ct/min(cost_list)/1e9)
	return best_inp.task.compute_dag.flop_ct/min(cost_list)/1e9, len(cost_list)






def get_best_cost_total_ct_time_given_wlk_TopK(filename, workload_key, topks):
	'''
		Get the ST result when K in TopK is set to topk. The ST method is with feedback.
		Return the minimum latency in the first K records, the total time and the total config_num to when running the topK method.
		RMK: 
			1. topks can be a list of ints. 
			2. the returned results can be dictionary of (best_cost, total_search_time, tot_config_num).
			3. if topK is None, then just read all the records related to workload_key.
	'''
	from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureCallback
	from tvm.auto_scheduler.utils import calc_workload_dis_factor, decode_workload_key
	# 
	log_reader = auto_scheduler.measure_record.RecordReader(filename)
	target = tvm.target.Target("cuda")
	# best_cost = [1e30 for i in range(topk)]
	# best_inp = list()
	# best_res = list()
	best_pairs = list()
	timestamp_begin = None
	# 
	print(filename, workload_key, topks)
	for inp, res in log_reader:
		if target and inp.task.target.kind.name != target.kind.name:
			continue
		if workload_key != inp.task.workload_key:
			continue
		# 
		if timestamp_begin == None:
			timestamp_begin=res.timestamp
		if res.error_no != MeasureErrorNo.NO_ERROR:
			best_pairs.append([inp, res, float('inf')])
			continue
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		best_pairs.append([inp, res, cost])
		assert len(inp.state.stages) == 0
	# 
	if len(best_pairs) == 0: # this case would happen when we search result for tasks tuned by Ansor in a wrong file
		return dict()
	# 
	print('len(best_pairs), ', len(best_pairs))
	print("time begin of searching without range:", timestamp_begin)
	infor_dict = dict()
	for topk in topks:
		record_begin = 0
		ori_topk = topk
		if topk == None:
			topk = len(best_pairs)
		# 
		best_cost = float('inf')
		tot_config_num = None
		total_search_time = None
		while record_begin < len(best_pairs):
			best_cost = min([i[2] for i in best_pairs[record_begin:topk + record_begin]])
			if best_cost == float('inf'):
				record_begin = topk + record_begin
			else:
				# find a valid record, stop
				if len(best_pairs) >= topk + record_begin:
					tot_config_num = topk + record_begin
				else:
					tot_config_num = len(best_pairs)
				total_search_time = best_pairs[tot_config_num-1][1].timestamp - timestamp_begin
				break
		if best_cost == float('inf'):
			# fail to find valid record
			tot_config_num = len(best_pairs)
			total_search_time = best_pairs[tot_config_num-1][1].timestamp - timestamp_begin
		# 
		infor_dict[ori_topk] = {'latency': best_cost, 'config_num': tot_config_num, 'search_time': total_search_time}
	return infor_dict









def get_best_cost_total_ct_time_given_wlk_TopK_givenRange(filename, workload_key, topks, 
	log_range, timestamp_begin):
	'''
		Get the ST result when K in TopK is set to topk. The ST method is with feedback.
		Return the minimum latency in the first K records, the total time and the total config_num to when running the topK method.
		RMK: 
			1. topks can be a list of ints. 
			2. the returned results can be dictionary of (best_cost, total_search_time, tot_config_num).
			3. if topK is None, then just read all the records related to workload_key.
		RMK: WE CANNOT USE THIS TIMESTAMP_BEGIN, because there is about 10 seconds for building all 1000 schedules, and reading logs. 
			We decide to ignore the time for reading logs and doing bound_infer.
			The search time of each schedule is the build time+ running time
	'''
	from tvm.auto_scheduler.measure import MeasureErrorNo, MeasureCallback
	from tvm.auto_scheduler.utils import calc_workload_dis_factor, decode_workload_key
	# 
	log_reader = auto_scheduler.measure_record.RecordReader(filename)
	target = tvm.target.Target("cuda")
	best_pairs = list()
	# 
	print(filename, workload_key, topks, log_range, timestamp_begin)
	for line_counter, (inp, res) in enumerate(log_reader):
		# print(line_counter)
		if line_counter >= log_range[1]:
			break
		elif line_counter < log_range[0]:
			continue
		assert target and inp.task.target.kind.name == target.kind.name
		assert workload_key == inp.task.workload_key
		# if target and inp.task.target.kind.name != target.kind.name:
		# 	continue
		# if workload_key != inp.task.workload_key:
		# 	continue
		# 
		# if timestamp_begin == None:
		# 	timestamp_begin=res.timestamp
		if res.error_no != MeasureErrorNo.NO_ERROR:
			best_pairs.append([inp, res, float('inf')])
			continue
		costs = [v.value for v in res.costs]
		cost = np.mean(costs)
		best_pairs.append([inp, res, cost])
		assert len(inp.state.stages) == 0
	# 
	if len(best_pairs) == 0: # this case would happen when we search result for tasks tuned by Ansor in a wrong file
		return dict()
	print('len(best_pairs), ', len(best_pairs))
	assert len(best_pairs) == len(range(log_range[0], log_range[1]))
	# 
	print("time begin of search with range:", timestamp_begin)
	infor_dict = dict()
	for topk in topks:
		record_begin = 0
		ori_topk = topk
		if topk == None:
			topk = len(best_pairs)
		# 
		best_cost = float('inf')
		tot_config_num = None
		total_search_time = None
		while record_begin < len(best_pairs):
			best_cost = min([i[2] for i in best_pairs[record_begin:topk + record_begin]])
			if best_cost == float('inf'):
				record_begin = topk + record_begin
			else:
				# find a valid record, stop
				if len(best_pairs) >= topk + record_begin:
					tot_config_num = topk + record_begin
				else:
					tot_config_num = len(best_pairs)
				# total_search_time = best_pairs[tot_config_num-1][1].timestamp - timestamp_begin
				break
		if best_cost == float('inf'):
			# fail to find valid record
			tot_config_num = len(best_pairs)
			# total_search_time = best_pairs[tot_config_num-1][1].timestamp - timestamp_begin
		# 
		total_search_time = sum([best_pairs[i][1].all_cost for i in range(tot_config_num)])
		infor_dict[ori_topk] = {'latency': best_cost, 'config_num': tot_config_num, 'search_time': total_search_time}
	return infor_dict



# get_best_cost_total_ct_time_given_wlk_TopK_givenRange('SINGLEpair_test_ST_vary_K_bmm_Test2_reuse_Ansor_Test1.json','["batch_matmul", [1, 64, 128], [1, 64, 128]]', [3, 10, 100, 1000], (995, 1992), 1626519692.2621899)




def trans_log_into_dict(filename):
	newfile = 'noState_' + filename
	wf = open(newfile,'w')
	with open(filename,'r') as f:
		for count, line in enumerate(f):
			if 'Placeholder' in line:
				if 'capsule' in filename:
					new_line = line[:-len('Placeholder: inputs, weight\n')] + 'None'
				elif '2norm' in filename:
					new_line = line[:-len('Placeholder: inputs\n')] + 'None'
				else:
					new_line = line[:-len('Placeholder: placeholder, placeholder\n')] + 'None'
				wf.write(new_line)
			elif line[0] == ',':
				wf.write(line)
			elif ('config_num_dict_list = ' in line) or\
				('time_dict_mytune_list = ' in line) or\
				('config_num_dict_ST_list = ' in line) or\
				('time_dict_ST_list = ' in line) or\
				('log_file_ansor = "' in line):
				wf.write(line)
	wf.close()



'''
'SINGLEpair_all_infor_conv1d_TestA_M_S_111.py'
'SINGLEpair_all_infor_bmm_TestA_M_S_111.py'
'SINGLEpair_all_infor_transpose_conv2d_TestA_M_S_111.py'
'SINGLEpair_all_infor_capsule_conv2d_TestA_M_S_111.py'
'SINGLEpair_all_infor_matrix_2norm_TestA_M_S_222.py'
'SINGLEpair_all_infor_group_conv2d_TestA_M_S_111.py'
'SINGLEpair_all_infor_conv2d_TestA_M_S_111.py'
'SINGLEpair_all_infor_depthwise_conv2d_TestA_M_S_111.py'
'SINGLEpair_all_infor_conv3d_TestA_M_S_227.py'


'''




def merge_search_tasks(all_search_tasks, all_workload_keys, to_add_list):
	'''
		all_search_tasks is a list of search tasks;
		all_workload_keys is a list of workload_keys (obtained from task.workload_key);
		to_add is a list of search task list;
		This function directly modifies the content of all_search_tasks.
	'''
	for tasks in to_add_list:
		for task in tasks:
			if task.workload_key not in all_workload_keys:
				all_search_tasks.append(task)
				all_workload_keys.append(task.workload_key)




def get_search_tasks_tuned_by_ansor(all_infor_ST):
	'''
		all_infor_ST is a dictionary: [ansor_wlk]:{'wlk_r':..., }
		If 'wlk_r' in value is None, 
		then the corresponding task is full optimized by by Ansor.
	'''
	full_tuned_wlk = list()
	for k, v in all_infor_ST.items():
		if v['wlk_r'] == None:
			full_tuned_wlk.append(k)
			assert v['config_num'] == 1000
	return full_tuned_wlk








# my method to extract tasks===============================================================
def call_all_topi_funcs(mod, params, target):
    """Call all TOPI compute to extract auto_scheduler tasks in a Relay program"""
    # pylint: disable=import-outside-toplevel
    from tvm import autotvm, transform
    from tvm.ir.transform import PassContext
    from tvm import relay
    from tvm.relay.backend import graph_runtime_codegen
    # 
    # Turn off AutoTVM config not found warnings
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True
    # 
    with transform.PassContext(
        opt_level=0,
        config={
            "relay.backend.use_auto_scheduler": True,
            "relay.backend.disable_compile_engine_cache": True,
        },
        disabled_pass={"AutoSchedulerLayoutRewrite"},
    ):
        try:
            opt_mod, _ = relay.optimize(mod, target, params)
            grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
            grc.codegen(opt_mod["main"])
        except tvm.TVMError:
            print(
                "Get errors with GraphRuntimeCodegen for task extraction. "
                "Fallback to VMCompiler."
            )
            compiler = relay.vm.VMCompiler()
            if params:
                compiler.set_params(params)
            mod = tvm.IRModule.from_expr(mod) if isinstance(mod, relay.Function) else mod
            compiler.lower(mod, target)
    # 
    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent




def extract_tasks(
    mod, params, target, target_host=None, hardware_params=None, include_simple_tasks=False
):
    """Extract tuning tasks from a relay program.
    Parameters
    ----------
    mod: tvm.IRModule or relay.function.Function
        The module or function to tune
    params: dict of str to numpy array
        The associated parameters of the program
    target: Union[tvm.target.Target, str]
        The compilation target
    target_host: Optional[Union[tvm.target.Target, str]]
        The host compilation target
    hardware_params : Optional[HardwareParams]
        Hardware parameters used for the search tasks
    include_simple_tasks: bool
        Whether to extract simple tasks that do not include complicated ops.
    Returns
    -------
    tasks: List[SearchTask]
        The tasks in this network
    weights: List[int]
        The weight (i.e. the number of appearance) of extracted tasks
    """
    # pylint: disable=import-outside-toplevel
    import threading
    from tvm.auto_scheduler.compute_dag import LayoutRewriteOption
    from tvm.auto_scheduler.search_task import SearchTask
    from tvm.auto_scheduler.relay_integration import TracingMode, TracingEnvironment
    # 
    if isinstance(target, str):
        target = tvm.target.Target(target)
    if isinstance(target_host, str):
        target_host = tvm.target.Target(target_host)
    # 
    # Run the compiler to collect all TOPI calls during compilation.
    env = TracingEnvironment(
        TracingMode.EXTRACT_TASK if include_simple_tasks else TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )
    with env:
        # Wrap build call in a new thread to avoid the conflict
        # between python's multiprocessing and tvm's thread pool
        build_thread = threading.Thread(target=call_all_topi_funcs, args=(mod, params, target))
        build_thread.start()
        build_thread.join()
    # 
    # create search tasks
    tasks = []
    weights = []
    for wkl_key, weight in env.wkl_key_to_weight.items():
        tasks.append(
            SearchTask(
                workload_key=wkl_key,
                target=target,
                target_host=target_host,
                hardware_params=hardware_params,
                # When auto scheduler is used in end to end network, try to apply layout rewrite
                # to improve the overall performance
                layout_rewrite_option=LayoutRewriteOption.get_target_default(target, True),
            )
        )
        weights.append(weight)
    # 
    return tasks, weights







# the structure to ansor tuning infor====================================================
from tvm.auto_scheduler.task_scheduler import TaskSchedulerCallback
class MyCollectTuneInfor(TaskSchedulerCallback):
	"""
	The callback that collects the tuning information of the whole progress.
	Parameters
	----------
	search_time_history: search time from beginning to current round for each round.
	best_score_history: best score (weighted sum of best costs) for each round.

	RMK: NEED TO CALL pre_tune AFTER THE TUNE IS FINISHED TO GET THE LAST INFOR.
	"""
	def __init__(self):
		self.pre_search_time_history = list()
		self.pre_best_score_history = list()
		self.pre_config_num_history = list()
		self.post_search_time_history = list()
		self.post_best_score_history = list()
		self.post_config_num_history = list()
	# 
	def pre_tune(self, task_scheduler, task_id):
		self.pre_search_time_history.append(time.time() - task_scheduler.tic)
		self.pre_config_num_history.append(task_scheduler.ct)
		# overall info
		if all(cost < 1e9 for cost in task_scheduler.best_costs):
			self.pre_best_score_history.append(task_scheduler.cur_score)
		else:
			self.pre_best_score_history.append(None)
	# 
	def post_tune(self, task_scheduler, task_id):
		self.post_search_time_history.append(time.time() - task_scheduler.tic)
		self.post_config_num_history.append(task_scheduler.ct)
		# overall info
		if all(cost < 1e9 for cost in task_scheduler.best_costs):
			self.post_best_score_history.append(task_scheduler.cur_score)
		else:
			self.post_best_score_history.append(None)

			
