#### 1.code structure

```
--> python_simulator.py
	--> arguments.py
	--> PolyRect.py
	--> iLQR.py
		--> vehicle_model.py
		--> local_planner.py
		--> constraints.py
			--> Obstacle.py
	
```



#### 2. python_simulator

simulate_npc：调用run_model_simulation，让人工驾驶车辆从初始状态循环到control seq结束

create_global_plan：设置总的规划目标（红线）

get_ego_states：返回自动驾驶车辆的状态

get_npc_bounding_box：返回人工驾驶车辆的长方形位置

get_npc_states：获得人工驾驶车辆的状态

create_ilqr_agent：首先调用create_global_plan，navigation_agent中设置npc的参数、bounding box的位置以及自动驾驶车辆的global plan

run_step_ilqr：仿真一步，得到desired plan，local plan和控制输入

animate：作为回调函数，得到可视化的参数

run_simulation：主程序，调用上面的程序进行仿真

run_model_simulation：仿真一步后dynamics的变化



#### 3. iLQR

set_global_plan：设置global plan

get_nominal_trajectory：获得在horizon内的nominal trajectory

forward pass与backward pass：具体iLQR的更新算法

run_step：进行一步仿真

get_optimal_control_seq：寻找最优的代价J，返回最优的状态序列X和控制输入序列U

filter_control：暂时时不知道有啥用

plot：可视化trajectory和control













