# THIS FILE PREPARE CONV2D OPS

func = conv2d_nchw 
tasks_list = list()
loops_list = list()
search_tasks_list = list()

for batch in [1, 16]:
	tasks = get_tasks_from_3rdparty_dnn_lib("resnet50_v1", "mxnet", 
			task_names=['conv2d_nchw.cuda', ], batch_size = batch, input_shape = [3, 224, 224])
	loops, search_tasks = get_loops(tasks, func)
	# 
	ori_op_num = len(tasks)
	for i in range(ori_op_num):
		for j in range(i+1, ori_op_num):
			can_reuse, dummy_loop = can_reuse_directly(loops[i], loops[j])
			if (not can_reuse) and (dummy_loop not in loops):
				loops.append(dummy_loop)
				dummyop1, dummyop2 = loop2dummy_op_ansor(dummy_loop, func)
				tasks.append(dummyop1)
				search_tasks.append(dummyop2)
				assert get_loops_ansor([dummyop1])[0] == dummy_loop	
	search_tasks_list.append(search_tasks)
	tasks_list.append(tasks)
	loops_list.append(loops)	



# because every two op can already reuse each other
selected = np.random.choice(a=len(tasks_list[0]), \
	size=5, replace=False, p=None)


tasks = [tasks_list[0][i] for i in selected] +\
		[tasks_list[1][i] for i in selected]


loops = [loops_list[0][i] for i in selected] +\
		[loops_list[1][i] for i in selected]


search_tasks = [search_tasks_list[0][i] for i in selected] +\
		[search_tasks_list[1][i] for i in selected]


# transformed_loops = [transformed_loops_list[0][i] for i in selected] +\
# 		[transformed_loops_list[1][i] for i in selected]

transformed_loops = loops

count = 0
op_pairs = list()
for i in range(len(search_tasks)):
	for j in range(len(search_tasks)):
		if i==j:
			continue
		res, _ = can_reuse_directly(transformed_loops[i], transformed_loops[j])
		if res:
			count +=1
			op_pairs.append((i, j))
			# op_pairs.append((j, i))


count
for loop in loops:
	print(loop)



fact_dict_refer = dict() # for speed up computation
selected_op_pairs = list()
for i, j in op_pairs:
	cost = bf_avg_config_esitimator_decompose_allops_largeRange_V3(
			search_tasks, loops, 
			i, j,
			blk_topk=1, thrd_topk=1, sample_n = 10, fact_dict_refer = fact_dict_refer)
	print(i, j, \
		max(get_product(loops[i])/get_product(loops[j]), \
		get_product(loops[j])/get_product(loops[i])), 
		cost)
	if cost < 1000:
		selected_op_pairs.append((i, j))


len(selected_op_pairs)
# sample a subset of 40 pairs
selected_op_pairs_idx = np.random.choice(len(selected_op_pairs), size=40, replace=False, p=None)
selected_op_pairs = [selected_op_pairs[i] for i in selected_op_pairs_idx]
invloved_ops=set()
for i,j in selected_op_pairs:
	invloved_ops.add(i)
	invloved_ops.add(j)


invloved_ops = list(invloved_ops)
invloved_ops.sort()
selected_op_pairs = [(i, -1) for i in invloved_ops] + selected_op_pairs


# print estimated total config num
print(sum([bf_avg_config_esitimator_decompose_allops_largeRange_V3(
			search_tasks, loops, 
			i, j,
			blk_topk=1, thrd_topk=1, sample_n = 10, fact_dict_refer = fact_dict_refer) for \
		i,j in selected_op_pairs if j!=-1])\
	+ 1000*len(invloved_ops))



