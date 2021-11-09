
# ===========================================================================================================

# batch = 1
# tasks = get_tasks_from_3rdparty_dnn_lib("resnet50_v1", "mxnet", 
# 		task_names=[], batch_size = batch, input_shape = [3, 224, 224])



tasks = list()
wlk_set = list()
for batch in [1]:#, 16]:
	for model_name in ["resnet50_v1"]:
		ori_tasks = get_tasks_from_3rdparty_dnn_lib(model_name, "mxnet", 
			task_names=[], batch_size = batch, input_shape = [3, 224, 224])
		for task in ori_tasks:
			if task.workload not in wlk_set:
				wlk_set.append(task.workload)
				tasks.append(task)





grouped_tasks = group_tasks(tasks)
taskName2Func = {'conv2d_nchw.cuda':conv2d_nchw, 'group_conv2d_nchw.cuda':group_conv2d_nchw, 
	'depthwise_conv2d_nchw.cuda':depthwise_conv2d_nchw, 'conv2d_transpose_nchw.cuda':conv2d_transpose_nchw, 
	'conv1d_ncw.cuda':conv1d_ncw, 'conv3d_ncdhw.cuda':conv3d_ncdhw, 
	"batch_matmul.cuda":batch_matmul}

# the original tasks that our system support
groups_to_tune = dict()
for k,v in grouped_tasks.items():
	if k in taskName2Func.keys():
		groups_to_tune[k] = v





# group the ops according to their sketch sets
tasks_list = list()
search_tasks_list = list()
loops_list = list()
transformed_loops_list = list()
task_type_list = list()
ori_op_num_list = list()
edge_cost_dict_list = list()
for name, tasks in groups_to_tune.items():
	func = taskName2Func[name]
	loops, search_tasks = get_loops(tasks, func)
	tmp = dict()
	for i in range(len(search_tasks)):
		tmp[get_sketch_num(search_tasks[i])] = list()
	for i in range(len(search_tasks)):
		tmp[get_sketch_num(search_tasks[i])].append(i)
	for tmp_v in tmp.values():
		if len(tmp_v) > 0:
			tasks_list.append([tasks[i] for i in tmp_v])
			search_tasks_list.append([search_tasks[i] for i in tmp_v])
			loops_list.append([loops[i] for i in tmp_v])
			task_type_list.append(name)
			ori_op_num_list.append(len(tmp_v))



# prepare enlarged dataset and reuse cost dicts
fact_dict_refer = dict() # for speed up computation
for i in range(len(tasks_list)):
	func = taskName2Func[task_type_list[i]]
	tasks = tasks_list[i]
	loops = loops_list[i]
	search_tasks = search_tasks_list[i]
	# we need to create 1 hop dummy ops
	ori_op_num = ori_op_num_list[i]
	for i in range(ori_op_num):
		for j in range(i+1, ori_op_num):
			can_reuse, dummy_loop = can_reuse_directly(loops[i], loops[j])
			if (not can_reuse) and (dummy_loop not in loops):
				dummyop1, dummyop2 = loop2dummy_op_ansor(dummy_loop, func)
				if get_sketch_num(dummyop1) != get_sketch_num(search_tasks[i]):
					continue
				loops.append(dummy_loop)
				tasks.append(dummyop1)
				search_tasks.append(dummyop2)
				assert get_loops_ansor([dummyop1])[0] == dummy_loop	
	transformed_loops_list.append(loops)
	# we need to compute the reuse costs
	edge_cost_dict = dict()
	for i in range(len(tasks)):
		edge_cost_dict[(i,-1)] = 1000
		for j in range(len(tasks)):
			if i!=j:
				can_reuse, dummy_loop = can_reuse_directly(loops[i], loops[j])
				if can_reuse:
					# compute the reuse cost for this ordered pair
					cost = bf_avg_config_esitimator_decompose_allops_largeRange_V3(
						search_tasks, loops, 
						i, j,
						blk_topk=1, thrd_topk=1, sample_n = 10, fact_dict_refer = fact_dict_refer)
					edge_cost_dict[(i,j)] = cost
	edge_cost_dict_list.append(edge_cost_dict)



# then we need to use java to compute the directed steiner tree for each "tasks" in tasks_list
# print the code we need in the java program
for i in range(len(tasks_list)):
	print_code_for_reuse_graph(edge_cost_dict_list[i], 
		ori_op_num_list[i], len(tasks_list[i]))



# selected_op_pairs_list = list()
# for i in range(len(tasks_list)):
# 	selected_op_pairs = solve_directed_spanning_tree(edge_cost_dict_list[i], \
# 		ori_op_num_list[i], len(tasks_list[i]))
# 	selected_op_pairs = process_selected_op_pairs(selected_op_pairs)
# 	selected_op_pairs_list.append(selected_op_pairs)



# after run the java project, we know the selected_op_pairs
selected_op_pairs = None
selected_op_pairs = process_selected_op_pairs(selected_op_pairs)
# then we can tune the whole op set






tasks_ansor, task_weights_ansor = list(), list()
wlk_weight_dict = dict()
for batch in [1, 16]:
	for model_name in ["resnet50_v1"]:
		mod, params = get_network_from_3rdparty_dnn_lib(model_name, "mxnet", 
			task_names=[], batch_size = batch, input_shape = [3, 224, 224])
		target = tvm.target.Target("cuda")
		new_tasks_ansor, new_task_weights_ansor = extract_tasks(mod["main"], params, target)
		# tasks_ansor = tasks_ansor + new_tasks_ansor
		# task_weights_ansor = task_weights_ansor + new_task_weights_ansor
		for i in range(len(new_tasks_ansor)):
			task = new_tasks_ansor[i]
			if task.compute_dag.workload_key() not in wlk_weight_dict.keys():
				wlk_weight_dict[task.compute_dag.workload_key()] = \
					new_task_weights_ansor[i]
				tasks_ansor.append(task)
			else:
				wlk_weight_dict[task.compute_dag.workload_key()] = \
					wlk_weight_dict[task.compute_dag.workload_key()] + new_task_weights_ansor[i]



task_weights_ansor = [wlk_weight_dict[i.compute_dag.workload_key()] for i in tasks_ansor]
# target = tvm.target.Target("cuda")
# tasks_ansor, task_weights_ansor = extract_tasks(mod["main"], params, target)

tasks_ansor, task_weights_ansor = \
	get_supported_Tasks_n_Weights_for_Ansor(tasks_ansor, task_weights_ansor, 
		search_tasks_list, ori_op_num_list)





