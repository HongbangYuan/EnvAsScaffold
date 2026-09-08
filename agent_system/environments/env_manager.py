# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
import random
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from omegaconf import OmegaConf
import re
import copy
from pathlib import Path
class BFCLEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs=None):
        obs_list, infos = self.envs.reset()
        
        text_obs = self.build_text_obs(obs_list)
        
        observations = {
            'text': text_obs,
            'image': None,
            'anchor': None,
            "messages":obs_list,
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs_list, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = valids[i]
            if 'success' in info:
                info['won'] = info['success']
            else:
                info['won'] = False

        next_text_obs = self.build_text_obs(next_obs_list)

        next_observations = {
            'text': next_text_obs,
            'image': None,
            'anchor': None,
            "messages":next_obs_list,
        }
        
        rewards = np.array(rewards)
        dones = np.array(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, obs_list: List[List[Dict]]) -> List[str]:
        import json
        processed_obs = []
        for messages in obs_list:
            messages_for_model = []
            for msg in messages:
                messages_for_model.append(msg)
            processed_obs.append(json.dumps(messages_for_model))
        return processed_obs


class SciWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        print("type(self.config)",type(self.config))
        self.plannings = []
        self.meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        super().__init__(envs, projection_f, config)

    def reset(self,kwargs=None):
        text_obs, infos = self.envs.reset()

        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos], init=True)
        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos])

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")

    def build_text_obs(self, text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_MC
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_MC
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""

                for j in range(recent_start_index):
                    action = all_actions[j]
                    step_number = j + 1
                    action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

                for j, record in enumerate(recent_history):
                    step_number = recent_start_index + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.config is not None and hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') and self.config.env.sciworld.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"

                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        planning=self.plannings[i],
                        available_actions=available_actions[i]
                    )
                else:
                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=available_actions[i]
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type


class SciWorldHintV7EnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        print("type(self.config)",type(self.config))
        self.plannings = []
        self.meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        super().__init__(envs, projection_f, config)
        
        if hasattr(envs, 'num_processes'):
            self.num_envs = envs.num_processes
        elif hasattr(envs, 'batch_size'):
             self.num_envs = envs.batch_size
        else:
            try:
                self.num_envs = len(envs)
            except:
                self.num_envs = 1

        self.gold_paths = [[] for _ in range(self.num_envs)]

        self.global_step_count = 0
        
    def _filter_gold_actions(self, actions: List[str]) -> List[str]:
        """
        预处理 Ground Truth：合并 open -> go
        """
        if not actions: return []
        filtered = []
        i = 0
        while i < len(actions):
            curr_act = actions[i]
            if i < len(actions) - 1:
                curr_str = curr_act.lower().strip()
                next_str = actions[i+1].lower().strip()
                match = re.match(r"^open door to\s+(.*)$", curr_str)
                if match:
                    location = match.group(1)
                    if next_str == f"go to {location}":
                        i += 1
                        continue
            filtered.append(curr_act)
            i += 1
        return filtered

    def reset(self, kwargs=None):
        self.global_step_count = getattr(self, "training_step", self.global_step_count + 1)
        print("global_step_count:", self.global_step_count)
        text_obs, infos = self.envs.reset()

        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        for i in range(len(infos)):
            raw_gold = infos[i].get('gold_actions', [])
            if i < len(self.gold_paths):
                self.gold_paths[i] = self._filter_gold_actions(raw_gold) if raw_gold else []

        full_text_obs = self.build_text_obs(
            infos,
            text_obs, 
            [info['available_actions'] for info in infos], 
            init=True
        )
        
        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def build_text_obs(self,infos: List[dict], text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        postprocess_text_obs = []
        
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_MC
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_MC
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            obs = ""
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""
                
                for j in range(max(0, recent_start_index)):
                     action = all_actions[j]
                     step_number = j + 1
                     action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"
                for j, record in enumerate(recent_history):
                    step_number = max(0, recent_start_index) + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                obs = _SCIWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )

            if i < len(self.gold_paths):
                current_gold_path = self.gold_paths[i]

                if i < len(text_obs) // 2:
                    steps_taken = len(self.buffers[i])
                    if 0 <= steps_taken <= 3 and self.global_step_count <= 100:
                        hint_msg = f"[Caution!] Start with the following action to start your exploration!\n{current_gold_path[:3]}"
                        obs += hint_msg
                    elif 6 <= steps_taken <= 10 and  self.global_step_count > 100:
                        info = infos[i]
                        if "goal_progress" in info:
                            hint_msg = f"[Hint]Considering The current goal progress is {info['goal_progress']}.\n"
                            obs += hint_msg
                    else:
                        pass

            postprocess_text_obs.append(obs)

        return postprocess_text_obs
    
    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")
    
    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type
    

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(infos, text_obs, [info['available_actions'] for info in infos])

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos


class SciWorldHintV8EnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        print("type(self.config)",type(self.config))
        self.plannings = []
        self.meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        super().__init__(envs, projection_f, config)
        
        if hasattr(envs, 'num_processes'):
            self.num_envs = envs.num_processes
        elif hasattr(envs, 'batch_size'):
             self.num_envs = envs.batch_size
        else:
            try:
                self.num_envs = len(envs)
            except:
                self.num_envs = 1

        self.gold_paths = [[] for _ in range(self.num_envs)]

        self.global_step_count = 0
        
    def _filter_gold_actions(self, actions: List[str]) -> List[str]:
        """
        预处理 Ground Truth：合并 open -> go
        """
        if not actions: return []
        filtered = []
        i = 0
        while i < len(actions):
            curr_act = actions[i]
            if i < len(actions) - 1:
                curr_str = curr_act.lower().strip()
                next_str = actions[i+1].lower().strip()
                match = re.match(r"^open door to\s+(.*)$", curr_str)
                if match:
                    location = match.group(1)
                    if next_str == f"go to {location}":
                        i += 1
                        continue
            filtered.append(curr_act)
            i += 1
        return filtered

    def reset(self, kwargs=None):
        self.global_step_count = getattr(self, "training_step", self.global_step_count + 1)
        print("global_step_count:", self.global_step_count)
        text_obs, infos = self.envs.reset()

        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        for i in range(len(infos)):
            raw_gold = infos[i].get('gold_actions', [])
            if i < len(self.gold_paths):
                self.gold_paths[i] = self._filter_gold_actions(raw_gold) if raw_gold else []

        full_text_obs = self.build_text_obs(
            infos,
            text_obs, 
            [info['available_actions'] for info in infos], 
            init=True
        )
        
        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def build_text_obs(self,infos: List[dict], text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        postprocess_text_obs = []
        
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_MC
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_MC
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            obs = ""
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""
                
                for j in range(max(0, recent_start_index)):
                     action = all_actions[j]
                     step_number = j + 1
                     action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"
                for j, record in enumerate(recent_history):
                    step_number = max(0, recent_start_index) + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                obs = _SCIWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )

            if i < len(self.gold_paths):
                current_gold_path = self.gold_paths[i]

                if i % 2 == 0: # 针对偶数的环境，返回hint，这样同一个group内的返回，也是不一样的了
                    steps_taken = len(self.buffers[i])
                    if 0 <= steps_taken <= 3 and self.global_step_count <= 100:
                        hint_msg = f"[Caution!] Start with the following action to start your exploration!\n{current_gold_path[:3]}"
                        obs += hint_msg
                    elif 6 <= steps_taken <= 10 and  self.global_step_count > 100:
                        info = infos[i]
                        if "goal_progress" in info:
                            hint_msg = f"[Hint]Considering The current goal progress is {info['goal_progress']}.\n"
                            obs += hint_msg
                    else:
                        pass

            postprocess_text_obs.append(obs)

        return postprocess_text_obs
    
    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")
    
    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type
    

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(infos, text_obs, [info['available_actions'] for info in infos])

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos



def make_envs(config):
    """Build only SciWorld/BFCL; validation always uses standard feedback."""
    import json
    group_n = config.env.rollout.n
    if not isinstance(group_n, int) or group_n < 1:
        raise ValueError("env.rollout.n must be a positive integer")
    resources = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)
    name = config.env.env_name.lower()
    if name in ("bfcl", "bfcl_diversev3"):
        from agent_system.environments.env_package.bfcl.envs import build_bfcl_envs
        train = build_bfcl_envs(config.env.seed, config.data.train_batch_size, group_n,
                               is_train=True, resources_per_worker=resources,
                               is_diverse=name == "bfcl_diversev3")
        val = build_bfcl_envs(config.env.seed + 1000, config.data.val_batch_size, 1,
                             is_train=False, resources_per_worker=resources)
        projection = lambda actions: (actions, [True] * len(actions))
        return (BFCLEnvironmentManager(train, projection, config),
                BFCLEnvironmentManager(val, projection, config))
    managers = {"sciworld": SciWorldEnvironmentManager,
                "sciworld_diversev7": SciWorldHintV7EnvironmentManager,
                "sciworld_diversev8": SciWorldHintV8EnvironmentManager}
    if name not in managers:
        raise ValueError(f"Unsupported environment: {name}. Choose {list(managers)} or bfcl/bfcl_diversev3")
    from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
    level = config.env.sciworld.generalization_level
    if level not in (0, 1, 2):
        raise ValueError("generalization_level must be 0, 1 or 2")
    variation_path = Path(__file__).parent / "env_package" / "sciworld" / "variations_idx" / f"L{level}_idx.json"
    with variation_path.open() as stream:
        variations = json.load(stream)
    options = dict(simplifications_preset=config.env.sciworld.get("simplifications_preset", "easy"),
                   env_step_limit=config.env.sciworld.get("env_step_limit", 100),
                   jar_path=config.env.sciworld.get("jar_path", None))
    train = build_sciworld_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                              group_n=group_n, variations_idx=variations["train"], **options)
    val = build_sciworld_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                            group_n=1, variations_idx=variations["test"], **options)
    projection = partial(sciworld_projection)
    return managers[name](train, projection, name, config), SciWorldEnvironmentManager(val, projection, name, config)
