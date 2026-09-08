import ray
import gym
import numpy as np
import os
import shutil
import tempfile
from typing import List, Dict, Any
from agent_system.environments.env_package.bfcl.bfcl.env import BFCLEnv
import random
from typing import Tuple

import numpy as np
from typing import List

def sample_balanced_eval_ids(
    valid_pool_ids: List[str], 
    env_num: int, 
    categories: List[str] = None
) -> List[str]:
    """
    从验证集 ID 池中，分层均匀采样出 env_num 个 ID。
    
    机制：
    1. 分类：将 ID 按 Category 分组。
    2. 排序：确保每次运行顺序一致 (消除 Set/Dict 的随机性)。
    3. 配额：计算每个 Category 应分得多少个名额。
    4. 采样：使用 np.linspace 进行均匀间隔采样（而非随机采样），保证覆盖面。
    
    Args:
        valid_pool_ids: 候选的 ID 列表。
        env_num: 需要采样的总数 (对应 distinct tasks 的数量)。
        categories: 类别列表，若为 None 则使用 BFCL 默认定义的4类。
    
    Returns:
        sampled_ids: 长度为 env_num 的 ID 列表。
    """
    
    if categories is None:
        categories = [
            "multi_turn_base",
            "multi_turn_long_context",
            "multi_turn_miss_func",
            "multi_turn_miss_param"
        ]
    
    ids_by_category = {cat: [] for cat in categories}
    
    for tid in valid_pool_ids:
        for cat in categories:
            if tid.startswith(cat):
                ids_by_category[cat].append(tid)
                break
    
    for cat in categories:
        ids_by_category[cat].sort()

    total_cats = len(categories)
    base_count = env_num // total_cats
    remainder = env_num % total_cats
    
    sampled_ids = []
    
    for i, cat in enumerate(categories):
        count_needed = base_count + (1 if i < remainder else 0)
        
        available_ids = ids_by_category[cat]
        n_available = len(available_ids)
        
        if n_available == 0:
            continue
            
        if count_needed >= n_available:
            indices = [x % n_available for x in range(count_needed)]
        else:
            indices = np.linspace(0, n_available - 1, count_needed, dtype=int)
            
        cat_sampled = [available_ids[idx] for idx in indices]
        sampled_ids.extend(cat_sampled)
    
    if len(sampled_ids) < env_num:
        needed = env_num - len(sampled_ids)
        existing_set = set(sampled_ids)
        remaining = sorted([x for x in valid_pool_ids if x not in existing_set])
        
        if remaining:
            extra_indices = np.linspace(0, len(remaining) - 1, needed, dtype=int)
            sampled_ids.extend([remaining[idx] for idx in extra_indices])
            
    return sampled_ids

def get_bfcl_fixed_split() -> Tuple[List[str], List[str]]:
    """
    生成 BFCL 多轮对话数据的固定 Train/Val ID 划分。
    
    修改说明：
    由于四个 category (base, long_context, miss_func, miss_param) 的同一 idx 对应相同的基础样本，
    为了绝对避免数据泄漏，不再使用随机打乱。
    
    划分规则 (Hard Split):
    - Train: 所有分类的索引 0 - 99 (共 400 条)
    - Val:   所有分类的索引 100 - 199 (共 400 条)
    
    Returns:
        train_ids (List[str]): 训练集 ID 列表
        val_ids (List[str]):   验证集 ID 列表
    """
    categories = [
        "multi_turn_base",
        "multi_turn_long_context",
        "multi_turn_miss_func",
        "multi_turn_miss_param"
    ]
    
    SPLIT_INDEX = 100
    TOTAL_ITEMS = 200
    
    train_ids = []
    val_ids = []
    
    for category in categories:
        for idx in range(SPLIT_INDEX):
            train_ids.append(f"{category}_{idx}")
            
        for idx in range(SPLIT_INDEX, TOTAL_ITEMS):
            val_ids.append(f"{category}_{idx}")
        
    return train_ids, val_ids



@ray.remote
class BFCLDataServer:
    def __init__(self, data_map, gt_map):
        print("Initializing Data Server... loading data into memory...")
        self.data_map = data_map 
        self.gt_map = gt_map
        print(f"Data Server Ready. Loaded {len(self.data_map)} data entries.")

    def get_task_data(self, task_id: str):
        return self.data_map.get(task_id), self.gt_map.get(task_id)

class BFCLWorker:
    """
    Ray remote actor.
    Holds reference to the FULL dataset map.
    Resets dynamically based on is_train flag.
    """
    def __init__(self, 
                 env_class, 
                 data_server_handle,  # <--- 变更为 Handle
                 target_ids: List[str], # 这是一个 ID 池
                 is_train: bool,
                 is_diverse: bool = False,
                 fixed_id: str = None): # 仅在 is_train=False 时使用
        
        self.env_class = env_class
        
        self.data_server = data_server_handle # 保存句柄 (引用计数，很小)
        
        self.target_ids = target_ids
        self.is_train = is_train
        self.fixed_id = fixed_id
        self.is_diverse = is_diverse
        
        self.env = None
        self.work_dir = tempfile.mkdtemp(prefix="bfcl_worker_")

        self.global_step_count = 0

    def set_training_step(self, step: int):
        self.training_step = step

    def reset(self, seed: int = None):
        self.global_step_count = getattr(self, "training_step", self.global_step_count + 1)
        print("global_step_count:", self.global_step_count)

        os.chdir(self.work_dir)
        
        current_data = None
        current_gt = None

        if self.is_train:
            rng = random.Random(seed)
            selected_id = rng.choice(self.target_ids)
        else:
            if self.fixed_id is None:
                raise ValueError("Eval mode requires fixed_id")
            selected_id = self.fixed_id
        
        current_data, current_gt = ray.get(self.data_server.get_task_data.remote(selected_id))
        
        if current_data is None:
             raise ValueError(f"ID {selected_id} not found in Data Server!")
        if self.is_train:
            print(f"Training mode reset: choose_data {selected_id}")
        else:
            print(f"Validation mode reset: choose data {selected_id}")
        

        is_diverse = False
        if self.is_train and self.is_diverse:
            rng = random.Random(seed)
            if rng.random() > 0.5:
                is_diverse = True
        is_diverse = is_diverse if self.is_train else False #  验证的时候强制关闭提示


        action_guidance = False
        observation_enrichment = False
        if is_diverse:
            if self.global_step_count <= 100: # 可以看情况进行一定的更改
                action_guidance = True
                observation_enrichment = False
            else:
                action_guidance = False
                observation_enrichment = True
            
        self.env = self.env_class(current_data, current_gt, action_guidance=action_guidance,observation_enrichment=observation_enrichment)
        
        obs, info = self.env.reset(seed=seed)
        
        return obs, info

    def step(self, action: str):
        os.chdir(self.work_dir)
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.step(action)

    def close(self):
        if self.env:
            self.env.close()
        try:
            shutil.rmtree(self.work_dir)
        except Exception:
            pass
 


class BFCLMultiProcessEnv(gym.Env):
    def __init__(self,
                 data_map: Dict[str, Any], 
                 gt_map: Dict[str, Any],
                 target_ids_pool: List[str], # 所有的可用 ID (train set or val set)
                 fixed_ids_list: List[str] = None, # 仅 eval 模式需要，长度需等于 num_processes
                 seed=0, 
                 env_num=1, 
                 group_n=1, 
                 mode='text',
                 resources_per_worker={"num_cpus": 1},
                 is_train=True,
                 is_diverse=False,
                 env_kwargs=None):
        
        super().__init__()
        if not ray.is_initialized():
            ray.init()
            
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n

        data_ref = ray.put(data_map)
        gt_ref = ray.put(gt_map)
        
        self.data_server = BFCLDataServer.remote(data_ref, gt_ref)
        
        ray.get(self.data_server.get_task_data.remote(target_ids_pool[0]))
        print("Data Server initialized successfully.")
        
        np.random.seed(seed)
        env_worker_cls = ray.remote(**resources_per_worker)(BFCLWorker)

        
        self.workers = []
        print(f"Starting {self.num_processes} workers...")
        for i in range(self.num_processes):
            worker_fixed_id = fixed_ids_list[i] if (fixed_ids_list is not None) else None
            
            worker = env_worker_cls.remote(
                env_class=BFCLEnv,
                data_server_handle=self.data_server, # 传递 Actor Handle
                target_ids=target_ids_pool, # 训练时 worker 从这里随机取
                is_train=is_train,
                fixed_id=worker_fixed_id,    # 验证时 worker 锁死这个 ID
                is_diverse=is_diverse,
            )
            self.workers.append(worker)

    def step(self, actions: List[str]):
        assert len(actions) == self.num_processes
        futures = [worker.step.remote(act) for worker, act in zip(self.workers, actions)]
        results = ray.get(futures)
        
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, truncated, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done or truncated)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def set_training_step(self, step: int):
        ray.get([worker.set_training_step.remote(step) for worker in self.workers])

    def reset(self):
        if self.is_train:
            base_seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            base_seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        seeds = np.repeat(base_seeds, self.group_n).tolist()

        futures = [worker.reset.remote(s) for worker, s in zip(self.workers, seeds)]
        results = ray.get(futures)
        
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
            
        return obs_list, info_list
        
    def close(self):
        for worker in self.workers:
            ray.kill(worker)

 
def build_bfcl_envs(
        seed=0, 
        env_num=1, 
        group_n=1, 
        mode='text',
        resources_per_worker={"num_cpus": 1},
        is_train=True,
        env_kwargs=None,
        is_diverse=False,
        **kwargs):
    
    from bfcl_eval.utils import load_dataset_entry
    from bfcl_eval.utils import load_ground_truth_entry

    categories = ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"]
    all_test_entries = [entry for category in categories for entry in load_dataset_entry(category)]
    
    all_gt = []
    for key in ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"]:
        all_gt.extend(load_ground_truth_entry(key))
        
    gt_map = {item['id']: item for item in all_gt}
    data_map = {item['id']: item for item in all_test_entries}

    fixed_train_ids, fixed_val_ids = get_bfcl_fixed_split()
    pool_ids = fixed_train_ids if is_train else fixed_val_ids
    valid_pool_ids = [tid for tid in pool_ids if tid in data_map and tid in gt_map]
    if len(valid_pool_ids) != len(pool_ids):
        raise ValueError("BFCL data or ground truth is incomplete for the requested split")
    
    fixed_ids_list = None
    if not is_train:
        distinct_ids = sample_balanced_eval_ids(valid_pool_ids, env_num)
        
        fixed_ids_list = np.repeat(distinct_ids, group_n).tolist()

    envs = BFCLMultiProcessEnv(
        data_map=data_map,
        gt_map=gt_map,
        target_ids_pool=valid_pool_ids, # 训练时 worker 在这里面随机游走
        fixed_ids_list=fixed_ids_list,  # 验证时 worker 被钉在这个列表对应的位置
        seed=seed,
        env_num=env_num,
        group_n=group_n, 
        mode=mode,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        is_diverse=is_diverse,
        env_kwargs=env_kwargs
    )
    
    return envs



