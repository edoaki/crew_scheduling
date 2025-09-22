from tensordict.tensordict import TensorDict
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from models.decoding_strategy import get_decoding_strategy

class Policy(nn.Module):
    """Policy for PARCO model"""

    def __init__(
        self,
        encoder=None,
        decoder=None,
        device: Optional[torch.device] = None,
  
       ):
        super(Policy, self).__init__()

        self.encoder = encoder   # PARCOEncoder
        self.decoder = decoder # PARCODecoder
        self.device = device

    def forward(
        self,
        env_out: TensorDict,
        vec_env,
        phase: str = "train",
    ) -> dict:
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, static_task_mask = self.encoder(env_out)
        done = vec_env.done
        decoding_strategy= get_decoding_strategy(
                phase=phase,
            )
        # まず1stepだけで確認
        cache = self.decoder.pre_decoder_hook(hidden,static_task_mask)

        while not vec_env.done.all().item():
            logit,action_mask= self.decoder(env_out,cache)
            selected,logprobs = decoding_strategy.step(logit,action_mask,env_out,done)
            env_out,rewards,dones,info = vec_env.step(selected,env_out)

        logprobs = decoding_strategy.post_decoder_hook()

        #報酬の計算
        reward = 0
        logprobs = 0

        outdict = {
            "reward": reward,
            "log_likelihood": logprobs,
        }

        return outdict