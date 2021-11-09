from helper_functions1 import *
from helper_functions2 import *
from helper_functions3 import * 



def get_lineNum(filename):
	count=-1
	for count, line in enumerate(open(filename,'rU')):
		pass
	count+=1
	return count




def tune_SINGLE_PAIRS_hrc(tasks,
	search_tasks,
	loops,
	selected_op_pairs,
	log_file,  # to store the result for hrc
	log_file_ansor, # to store the result for full tuning
	infor_log_file, # to store the infor dict of hrc
	best_measure_pairs = None # store the best measure pairs that may be reused
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
	tasks_list = [tasks,]
	search_tasks_list = [search_tasks,]
	loops_list = [loops,]
	selected_op_pairs_list = [selected_op_pairs,]
	time_begin_tot = time.time()
	best_measure_pairs_list = list()
	best_reuse_results_list = list()
	config_num_dict_list = list()
	time_dict_mytune_list = list()
	error_pairs = list()
	# 
	stop_flag = False
	# all round share the best measure pairs
	if best_measure_pairs == None:
		best_measure_pairs = dict()
	for round_i in range(len(tasks_list)):
		tasks = tasks_list[round_i]
		search_tasks = search_tasks_list[round_i]
		loops = loops_list[round_i]
		# transformed_loops = transformed_loops_list[round_i]
		selected_op_pairs = selected_op_pairs_list[round_i]
		best_reuse_results = dict()
		config_num_dict = dict()
		time_dict_mytune = dict()
		# 
		# ori_op_num = len(tasks)
		# for task_id in range(ori_op_num):
		# 	tune_seach_task(search_tasks, task_id, best_measure_pairs)
		# 
		# log_file = "SINGLEpair_test_new_hrc_conv2d_winograd_Test1.json"
		# log_file_ansor = "SINGLEpair_tune_Ansor_conv2d_winograd_Test1.json"
		# log_file = "invalid_log.json"
		# log_file_ansor = "FULL_OPTIMIZE_DATABASE_tune_Ansor_all_end2end_ops_Test1.json"
		# log_file = "SINGLEpair_test_new_hrc_matrix_2norm_Test4_newdata.json"
		# log_file_ansor = "SINGLEpair_tune_Ansor_matrix_2norm_Test4_newdata.json"
		measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
		tune_option = auto_scheduler.TuningOptions(
		    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
		    runner=measure_ctx.runner,
		    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
		    verbose=0, #silent no print
		)
		# 
		if stop_flag:
			break
		for reuse_pair in selected_op_pairs:
			print(reuse_pair)
			i, j = reuse_pair
			if j == -1:
				print(loops[i])
				# continue
				time_begin = time.time()
				tune_seach_task(search_tasks, i, best_measure_pairs, dict(), log_file_ansor)
				time_dict_mytune[(search_tasks[i].workload_key, None)] = \
					time.time() - time_begin
				# get the actual number of configs
				config_num_dict[reuse_pair] = get_lineNum(log_file_ansor) \
					- sum(config_num_dict.values()) - \
					sum([sum(config_num_dict_tmp.values()) for config_num_dict_tmp in config_num_dict_list])
				print(best_measure_pairs[search_tasks[i].workload_key][1])
				continue			
			elif search_tasks[j].workload_key not in best_measure_pairs.keys():
				assert False
			# start hrc tuning
			to_tune_idx = reuse_pair[0]
			reuse_from_idx = reuse_pair[1]
			wlk_to_tune = search_tasks[to_tune_idx].workload_key
			wlk_reuse_from = search_tasks[reuse_from_idx].workload_key
			reuse_key = (wlk_to_tune, wlk_reuse_from)
			# 
			search_policy = get_search_policy(search_tasks[to_tune_idx])
			# 
			try:
				time_begin = time.time()
				gen_and_tune_hierarchy_nonfactor_decompose_3TuneRound_nonblk_allops_largeRange_ANSOR_V2(
					search_policy, tune_option,
					search_tasks, tasks, loops, to_tune_idx, reuse_from_idx, 
					best_measure_pairs, best_reuse_results, config_num_dict
				)
				time_dict_mytune[reuse_key] = time.time() - time_begin
				# print("WE CAN REACH HERE")
				# best_measure_pairs[wlk_to_tune] = best_reuse_results[reuse_key]
				# print(best_measure_pairs[wlk_to_tune][1])
				print("mine: ", best_reuse_results[reuse_key][1], \
					"  baseline: ", best_measure_pairs[wlk_to_tune][1])
				#
			except KeyboardInterrupt:
				stop_flag = True
				break
			except:
				error_pairs.append(reuse_pair)
		# 
		# best_measure_pairs_list.append(best_measure_pairs)
		best_reuse_results_list.append(best_reuse_results)
		config_num_dict_list.append(config_num_dict)
		time_dict_mytune_list.append(time_dict_mytune)
	# 
	best_measure_pairs_list.append(best_measure_pairs)
	tot_time_used = time.time() - time_begin_tot
	# store the results in file
	with open(infor_log_file, 'w') as f:
		f.write('best_measure_pairs_list_no_state = ' + delete_state(best_measure_pairs_list).__str__() + '\n')
		f.write('best_reuse_results_list_no_state = ' + delete_state(best_reuse_results_list).__str__() + '\n')
		f.write('config_num_dict_list = ' + config_num_dict_list.__str__()+ '\n')
		f.write('time_dict_mytune_list = ' + time_dict_mytune_list.__str__()+ '\n')






# TUNE TARGET ROUND=========================================================
# time_begin_tot = time.time()

def tune_END2END_hrc(tasks_list,
	search_tasks_list,
	loops_list,
	selected_op_pairs_list,
	log_file,  # to store the result for hrc
	infor_log_file, # to store the infor dict of hrc
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
	time_begin_tot = time.time()
	best_measure_pairs_list = list()
	best_reuse_results_list = list()
	config_num_dict_list = list()
	time_dict_mytune_list = list()
	error_pairs = list()
	# 
	stop_flag = False
	for round_i in range(len(tasks_list)):
		tasks = tasks_list[round_i]
		search_tasks = search_tasks_list[round_i]
		loops = loops_list[round_i]
		# transformed_loops = transformed_loops_list[round_i]
		selected_op_pairs = selected_op_pairs_list[round_i]
		best_measure_pairs = dict()
		best_reuse_results = dict()
		config_num_dict = dict()
		time_dict_mytune = dict()
		# 
		measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
		tune_option = auto_scheduler.TuningOptions(
		    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
		    runner=measure_ctx.runner,
		    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
		    verbose=0, #silent no print
		)
		# 
		if stop_flag:
			break
		for reuse_pair in selected_op_pairs:
			print(reuse_pair)
			i, j = reuse_pair
			if j == -1:
				# continue
				time_begin = time.time()
				tune_seach_task(search_tasks, i, best_measure_pairs, dict(), log_file)
				time_dict_mytune[(search_tasks[i].workload_key, None)] = \
					time.time() - time_begin
				# get the actual number of configs
				config_num_dict[reuse_pair] = get_lineNum(log_file) - sum(config_num_dict.values()) - \
					sum([sum(config_num_dict_tmp.values()) for config_num_dict_tmp in config_num_dict_list])
				print(best_measure_pairs[search_tasks[i].workload_key][1])
				continue			
			elif search_tasks[j].workload_key not in best_measure_pairs.keys():
				assert False
			# start hrc tuning
			to_tune_idx = reuse_pair[0]
			reuse_from_idx = reuse_pair[1]
			wlk_to_tune = search_tasks[to_tune_idx].workload_key
			wlk_reuse_from = search_tasks[reuse_from_idx].workload_key
			reuse_key = (wlk_to_tune, wlk_reuse_from)
			# 
			search_policy = get_search_policy(search_tasks[to_tune_idx])
			# 
			try:
				time_begin = time.time()
				gen_and_tune_hierarchy_nonfactor_decompose_3TuneRound_nonblk_allops_largeRange_ANSOR_V2(
					search_policy, tune_option,
					search_tasks, tasks, loops, to_tune_idx, reuse_from_idx, 
					best_measure_pairs, best_reuse_results, config_num_dict
				)
				time_dict_mytune[reuse_key] = time.time() - time_begin
				best_measure_pairs[wlk_to_tune] = best_reuse_results[reuse_key]
				print(best_measure_pairs[wlk_to_tune][1])
				# print("mine: ", best_reuse_results[reuse_key][1], \
				# 	"  baseline: ", best_measure_pairs[wlk_to_tune][1])
				#
			except KeyboardInterrupt:
				stop_flag = True
				break
			except:
				error_pairs.append(reuse_pair)
		# 
		best_measure_pairs_list.append(best_measure_pairs)
		best_reuse_results_list.append(best_reuse_results)
		config_num_dict_list.append(config_num_dict)
		time_dict_mytune_list.append(time_dict_mytune)
	# 
	tot_time_used = time.time() - time_begin_tot
	# store the results in file
	with open(infor_log_file, 'w') as f:
		f.write('best_measure_pairs_list_no_state = ' + delete_state(best_measure_pairs_list).__str__() + '\n')
		f.write('best_reuse_results_list_no_state = ' + delete_state(best_reuse_results_list).__str__() + '\n')
		f.write('config_num_dict_list = ' + config_num_dict_list.__str__()+ '\n')
		f.write('time_dict_mytune_list = ' + time_dict_mytune_list.__str__()+ '\n')






