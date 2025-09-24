from tensordict.tensordict import TensorDict
from typing import Optional
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
        phase: str = "train", # "train", "val"
    ) -> dict:
        
        hidden, static_task_mask = self.encoder(env_out)
        
        done = vec_env.done
        decoding_strategy= get_decoding_strategy(
                phase=phase,
                temperature=1.0,
                tanh_clipping=10.0,
            )
        # まず1stepだけで確認
        cache = self.decoder.pre_decoder_hook(hidden,static_task_mask)
        
        step = 0
        while not vec_env.done.all().item():
            # print(f"step {step} / done {vec_env.done}")
            logit,action_mask= self.decoder(env_out,cache)
         
            selected = decoding_strategy.step(logit,action_mask,done)
           
            env_out,rewards,dones,info = vec_env.step(selected,env_out)
            step += 1

        logprobs = decoding_strategy.post_decoder_hook()
        sol = vec_env.return_solution()
        
        sol = [s.to(self.device) for s in sol]
      
        reward = 0 # dummy

        outdict = {
            "solution": sol,
            "reward": reward,
            "log_likelihood": logprobs,
        }

        return outdict