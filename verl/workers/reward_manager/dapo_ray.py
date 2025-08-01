"""DAPO reward manager that runs scoring in Ray tasks."""
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import ray
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager

# Re-use the global Ray cluster if already initialized
ray.init(address="auto", namespace="reward", ignore_reinit_error=True)


@ray.remote(num_cpus=1, num_gpus=0)
def _remote_score(compute_fn, args):
    """Lightweight remote wrapper for reward scoring."""
    return compute_fn(**args)


@register("dapo_ray")
class DAPORayRewardManager(DAPORewardManager):
    """DAPO reward manager that evaluates rewards in parallel using Ray."""

    def _build_remote_args(self, data: DataProto):
        arg_list = []
        ctx_list = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            arg_list.append(
                {
                    "data_source": data_source,
                    "solution_str": response_str,
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                }
            )
            ctx_list.append(
                {
                    "valid_response_length": valid_response_length,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                }
            )
        return arg_list, ctx_list

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        arg_list, ctx_list = self._build_remote_args(data)
        futures = [_remote_score.remote(self.compute_score, args) for args in arg_list]
        scores = ray.get(futures)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print = {}

        for i, result in enumerate(scores):
            ctx = ctx_list[i]
            valid_response_length = ctx["valid_response_length"]
            data_source = ctx["data_source"]
            prompt_str = ctx["prompt_str"]
            response_str = ctx["response_str"]
            ground_truth = ctx["ground_truth"]

            reward = result["score"] if isinstance(result, dict) else result
            if isinstance(result, dict):
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                reward_extra_info["acc"].append(result)

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if already_print.get(data_source, 0) < self.num_examine:
                already_print[data_source] = already_print.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", result)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
