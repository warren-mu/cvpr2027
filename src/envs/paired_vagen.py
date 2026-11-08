"""CPU-only VAGEN interface scaffold, not a registered/training-ready environment.

Implements the pinned legacy BaseEnv method/return shapes by duck typing. The
caller must supply episode selection, the actual tokenizer, a response parser,
and RGB-to-VAGEN image conversion. No VAGEN/PIL/torch import or private model
inputs are added. Service registration and real L/P integration remain pending.
"""
import uuid


class DraftVagenAdapter:
    def __init__(self, environment, episode_selector, image_adapter, response_parser, token_counter):
        if not all(callable(callback) for callback in (episode_selector, image_adapter, response_parser, token_counter)):
            raise ValueError("Explicit selectors, image conversion, parser and actual tokenizer are required")
        self.environment = environment
        self.episode_selector = episode_selector
        self.image_adapter = image_adapter
        self.response_parser = response_parser
        self.token_counter = token_counter

    def _obs(self, observation):
        # image_adapter receives only the public marked RGB, never BackendFrame/info.
        return {"obs_str": "<image>\n" + observation.text,
                "multi_modal_data": {"<image>": [self.image_adapter(observation.image)]}}

    @staticmethod
    def _info(info, raw=None):
        return {**info, "llm_raw_response": raw, "llm_response": raw,
                "metrics": {"turn_metrics": {"action_is_valid": info.get("action_valid", False),
                                             "action_is_effective": info.get("action_success", False)},
                            "traj_metrics": {"success": info["success"]}}}

    def reset(self, seed=None):
        pair_id, world = self.episode_selector(seed)
        observation = self.environment.reset(pair_id, world)
        return self._obs(observation), self._info(self.environment.info)

    def step(self, llm_raw_response):
        if not isinstance(llm_raw_response, str):
            raise ValueError("VAGEN response must be text")
        token_count = self.token_counter(llm_raw_response)
        if type(token_count) is not int or token_count < 0:
            raise ValueError("Actual response-token count is required")
        self.environment.record_model_call(uuid.uuid4().hex)
        exceeded = token_count > self.environment.budget.max_response_tokens
        # A malformed/multiple action never partially executes an observation.
        try:
            action = "" if exceeded else self.response_parser(llm_raw_response)
        except ValueError:
            action = ""
        observation, reward, done, info = self.environment.step(action)
        info.update(response_tokens=token_count, response_token_budget_exceeded=exceeded)
        return self._obs(observation), reward, done, self._info(info, llm_raw_response)

    def system_prompt(self):
        return ("Choose exactly one available action. A move, 45-degree turn, or open action ends this decision "
                "and returns a new image. Use select with the current numbered candidate to finish.")

    def compute_reward(self):
        return 0.0  # Success and observation cost already appear in step rewards.

    def close(self):
        self.environment.close()
