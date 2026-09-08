import gym
import json
import re
import copy
import importlib
import inspect
import threading
from typing import List, Dict, Any, Tuple, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_system.environments.env_package.bfcl.bfcl.observation_enricher import ENRICHMENT_HANDLERS
from functools import wraps
import traceback

from bfcl_eval.constants.executable_backend_config import STATELESS_CLASSES, CLASS_FILE_PATH_MAPPING
from bfcl_eval.model_handler.utils import extract_prompt_format_from_id, formulate_system_prompt,default_decode_execute_prompting
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import state_checker, response_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response

FORBIDDEN_FUNCTIONS = ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]

print_lock = threading.Lock()

def clean_model_output(raw_text: str) -> str:
    """
    只保留 </think> 标签之后的内容。
    如果不存在 </think>，则返回原始文本。
    """
    if not raw_text:
        return ""
    
    tag = "</think>"
    if tag in raw_text:
        _, _, post_think_content = raw_text.rpartition(tag)
        return post_think_content.strip()
    
    return raw_text.strip()

def system_prompt_pre_processing_chat_model(
    function_docs: list[dict], test_entry_id: str
) -> list[dict]:
    """
    Add a system prompt to the chat model to instruct the model on the available functions and the expected response format.
    If the prompts list already contains a system prompt, append the additional system prompt content to the existing system prompt.
    """
    prompt_format = extract_prompt_format_from_id(test_entry_id)

    system_prompt = formulate_system_prompt(
        format_sensitivity_config=prompt_format, functions=function_docs
    )

    return system_prompt

class BFCLEnv(gym.Env):
    def __init__(self, data: Dict[str, Any],ground_truth: Dict[str, Any],action_guidance=False,observation_enrichment=False):
        """
        初始化环境。
        Args:
            data: BFCL 数据集中的单个样本 dict (如你提供的 results[0])
        """
        assert data["id"] == ground_truth["id"]
        super().__init__()
        self.data = copy.deepcopy(data)
        self.ground_truth = copy.deepcopy(ground_truth["ground_truth"])
        self.test_entry_id = data.get("id", "unknown")
        
        self.all_turns = data.get("question", []) # List[List[Dict]] (多轮对话列表)
        self.initial_config = data.get("initial_config", {})
        self.involved_classes = data.get("involved_classes", [])
        self.tool_definitions = data.get("function", [])
        self.holdout_function = data.get("missed_function", {}) # Holdout逻辑



        if action_guidance:
            hint_mapping = {
                "miss_func": (
                    "A required function to fulfill the user's request is currently missing. "
                    "Explicitly state that there is no available function to handle this request."
                ),
                "miss_param": (
                    "Essential information is missing from the user's request. "
                    "Do not make any assumptions; explicitly ask the user to clarify the missing parameters."
                ),
            }
            if "miss_func" in self.test_entry_id:
                category = "miss_func"
            elif "miss_param" in self.test_entry_id:
                category = "miss_param"
            elif "long_context" in self.test_entry_id:
                category = "long_context"
            elif "base" in self.test_entry_id:
                category = "base"
            else:
                assert False, f"Unknown test entry id: {self.test_entry_id}"
            for i in range(3):
                if i < len(self.all_turns) and len(self.all_turns[i]) > 0 and i < len(self.ground_truth):
                    if len(self.ground_truth[i]) > 0:
                        hint = f"\nHint: considering start with the following operation: \n{self.ground_truth[i][0]}" 
                        self.all_turns[i][0]["content"] = self.all_turns[i][0]["content"] + hint
                    else: # len(self.ground_truth[i]) == 0
                        hint = hint_mapping.get(category, "")
                        if hint:
                            self.all_turns[i][0]["content"] = self.all_turns[i][0]["content"] + f"\n Hint: {hint}"

        self.observation_enrichment = observation_enrichment
        self.action_space = gym.spaces.Text(max_length=50000)
        self.observation_space = gym.spaces.Dict({
            "messages": gym.spaces.Text(max_length=200000)
        })

        self.active_instances = {}
        self.class_method_name_mapping = {}
        self.messages = []
        self.turn_idx = 0         # 当前在第几个 User Turn
        self.step_in_turn = 0     # 当前 Turn 内的第几次交互
        self.max_steps_per_turn = 20
        self.is_done = False

        self.is_success = False
        self.current_turn_model_exec_results = []
        self.all_turn_model_exec_results = []
        self.model_snapshots = []   

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[List[Dict], Dict]:
        super().reset(seed=seed)
        
        self._init_instances(is_gt=False)
        self.is_done = False # 重置 done 状态
        self.gt_snapshots = self._precompute_gt_snapshots()
        
        self.messages = []
        self.turn_idx = 0
        self.step_in_turn = 0

        self.is_success = False
        self.current_turn_model_exec_results = []
        self.all_turn_model_exec_results = []
        self.model_snapshots = []

        self.step_counts_per_turn = []

        self.current_turn_decoded_responses = [] 
        self.all_turn_decoded_responses = []
        
        system_prompt = system_prompt_pre_processing_chat_model(
            self.tool_definitions, self.test_entry_id
        )
        self.messages.append({"role": "system", "content": system_prompt})
            
        self.turn_idx = -1 
        self._load_next_turn() 
        
        return self.messages, {"id": self.test_entry_id}

    def step(self, action: str):
        """
        Action: 模型的原始回复字符串。
        """
        if self.is_done:
            return self.messages, 0, True, True, {"id": self.test_entry_id,"success": self.is_success}
        
        self.step_in_turn += 1
        reward = 0
        done = False
        truncated = False

        info = {"id": self.test_entry_id,"success": False}
        

        cleaned_action = clean_model_output(action)

        self.messages.append({"role": "assistant", "content": cleaned_action})
        
        try:
            decoded_model_responses = default_decode_execute_prompting(
                    cleaned_action,
                    has_tool_call_tag=False,
            )
        except Exception as e:
            decoded_model_responses = None

        if decoded_model_responses:
            
            execution_results = self._execute_func_calls_logic(decoded_model_responses,self.active_instances)
            self.current_turn_model_exec_results.extend(execution_results)
            self.current_turn_decoded_responses.append(decoded_model_responses)
            
            for res, decoded_model_response in zip(execution_results, decoded_model_responses):
                self.messages.append(
                    {"role": "tool", "content": res, "name": decoded_model_response}
                )
            
            if self.step_in_turn >= self.max_steps_per_turn:
                truncated = True
                
        else:
            self.step_counts_per_turn.append(self.step_in_turn)
            
            
            self.all_turn_model_exec_results.extend(self.current_turn_model_exec_results)
            self.all_turn_decoded_responses.append(self.current_turn_decoded_responses)
            
            self._save_model_snapshot()
            self.current_turn_decoded_responses = []
            self.current_turn_model_exec_results = []
            
            has_next_turn = self._load_next_turn()
            
            if has_next_turn:
                self.step_in_turn = 0
            else:
                done = True
                self.is_done = True # 下一次就不再进行step操作了
                is_success = self._evaluate_full_episode()
                
                info["success"] = is_success
                self.is_success = is_success
                reward = 1 if is_success else 0
        
        return self.messages, reward, done, truncated, info


    def _evaluate_full_episode(self) -> bool:
        """
        在 Episode 结束时，对比保存的 model_snapshots 和 gt_snapshots。
        只有每一轮都正确，才返回 True。
        """
        ground_truth_list = self.ground_truth
        model_response_list = self.all_turn_decoded_responses
        if len(ground_truth_list) != len(model_response_list):
            return False
        for single_turn_ground_truth_list,single_turn_model_response_list in zip(ground_truth_list,model_response_list):
            if len(single_turn_ground_truth_list) > 0:
                if not single_turn_model_response_list or is_empty_execute_response(
                    single_turn_model_response_list
                ):
                    return False
        
        assert len(self.gt_snapshots) == len(ground_truth_list)
        for idx, (model_snap, gt_snap) in enumerate(zip(self.model_snapshots, self.gt_snapshots)):
            gt_responses = gt_snap['gt_responses']

    
            if not gt_responses:
                continue 

            state_res = state_checker(model_snap['instances'], gt_snap['instances'])
            if not state_res["valid"]:
                return False
                
            resp_res = response_checker(
                model_snap['cumulative_results'],
                gt_snap['exec_results'],
                turn_index=idx
            )
            if not resp_res["valid"]:
                return False
                
        return True

    def _load_next_turn(self) -> bool:
        """
        加载下一轮 User Message。
        对应原代码中 Outer Loop 的迭代逻辑以及 Holdout 处理。
        """
        next_idx = self.turn_idx + 1
        
        if next_idx >= len(self.all_turns):
            return False
        
        self.turn_idx = next_idx
        
        current_turn_msgs = self.all_turns[self.turn_idx]
        
        if str(self.turn_idx) in self.holdout_function:
            missed_funcs = self.holdout_function[str(self.turn_idx)]
            prompt_content = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                functions=missed_funcs
            )
            current_turn_msgs = [{
                "role": "user",
                "content": prompt_content
            }]
            
        self.messages.extend(current_turn_msgs)
        return True


    def _execute_func_calls_logic(self, func_call_list: List[str], context: Dict) -> List[str]:
        results = []
        exec_globals = context.copy()
        for func_call in func_call_list:
            func_call = self._process_method_calls(func_call)
            try:
                self._check_safety(func_call)
                res = eval(func_call, exec_globals)

                if self.observation_enrichment:
                    try:
                        print(f"enriching for func_call: {func_call}")
                        call_header = func_call.split('(')[0]  # "TradingBot.place_order"
                        if '.' in call_header:
                            instance_name, method_name = call_header.split('.', 1)
                            
                            handler = ENRICHMENT_HANDLERS.get(instance_name)
                            instance_obj = exec_globals.get(instance_name)
                            
                            if handler and instance_obj:
                                res = handler(method_name, res, instance_obj)
                    except Exception as e:
                        print(f"Enrichment failed for func_call: {func_call}")
                        print(f"Error message: {str(e)}")
                        print(traceback.format_exc())
                        print("You can safely ignore this error.")

                results.append(self._format_result(res))
            except Exception as e:
                results.append(f"Error: {str(e)}")
        return results
    
    def _precompute_gt_snapshots(self) -> List[Dict]:
        """在 Reset 时预计算所有 GT 状态"""
        snapshots = []
        gt_instances = {}
        self._init_instances(is_gt=True, target_dict=gt_instances)
        
        for turn_i, gt_calls in enumerate(self.ground_truth):
            gt_results = self._execute_func_calls_logic(gt_calls, gt_instances)
            
            snap = {
                "instances": {k: copy.deepcopy(v) for k, v in gt_instances.items()},
                "exec_results": copy.deepcopy(gt_results),
                "gt_responses": copy.deepcopy(gt_calls),
            }
            snapshots.append(snap)
            
        return snapshots


    def _save_model_snapshot(self):
        """保存当前时刻 Agent 的状态和结果快照"""
        snapshot = {
            "instances": {k: copy.deepcopy(v) for k, v in self.active_instances.items()},
            "cumulative_results": copy.deepcopy(self.all_turn_model_exec_results),
            "current_turn_decoded_responses": self.current_turn_decoded_responses,
            "all_turn_decoded_responses": copy.deepcopy(self.all_turn_decoded_responses),
        }
        self.model_snapshots.append(snapshot)

    def _init_instances(self, is_gt: bool, target_dict: Dict = None):
        """初始化实例。如果 target_dict 未传，默认使用 self.active_instances"""
        if target_dict is None:
            self.active_instances = {}
            target_dict = self.active_instances
            
        if not is_gt:
            self.class_method_name_mapping = {}

        test_cat = self.test_entry_id.rsplit("_", 1)[0]
        long_ctx = "long_context" in test_cat or "composite" in test_cat
        
        for cls_name in self.involved_classes:
            if cls_name not in CLASS_FILE_PATH_MAPPING: continue
            try:
                mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls_name])
                cls = getattr(mod, cls_name)
                instance = cls()
                if cls_name not in STATELESS_CLASSES:
                    cfg = copy.deepcopy(self.initial_config.get(cls_name, {}))
                    instance._load_scenario(cfg, long_context=long_ctx)
                target_dict[cls_name] = instance
                
                if not is_gt:
                    for name, _ in inspect.getmembers(instance, predicate=inspect.ismethod):
                        if not name.startswith("_"):
                            self.class_method_name_mapping[name] = cls_name
            except Exception as e:
                print(f"Init Error ({cls_name}): {e}")

        
    
    def _process_method_calls(self, func_call: str) -> str:

        def replace_function(match):
            func_name = match.group(1)
            
            if func_name in self.class_method_name_mapping:
                return f"{self.class_method_name_mapping[func_name]}.{func_name}"
            return func_name

        pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"

        processed_string = re.sub(pattern, replace_function, func_call)

        return processed_string


    def _check_safety(self, func_call: str):
        name = func_call.split("(")[0]
        if "." in name: name = name.split(".")[1]
        if name in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
            raise ValueError(f"Forbidden: {name}")

    def _format_result(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            try: return json.dumps(result)
            except: pass
        return str(result)

