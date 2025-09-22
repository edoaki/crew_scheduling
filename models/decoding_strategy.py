import abc

from typing import Tuple

import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict

def get_decoding_strategy(phase):
    if phase == "train":
        decode_strategy = Sampling()
    elif phase in ["val", "test"]:
        decode_strategy = Greedy()
    else:
        raise ValueError(f"Unknown phase: {phase}")
    return decode_strategy

class DecodingStrategy(metaclass=abc.ABCMeta):
    name = "base"

    def __init__(
        self,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
    ):
       
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
       
        # initialize buffers
        self.actions = []
        self.logprobs = []

    @abc.abstractmethod
    def _step(
        self,
        logit,
        action_mask,
        done,
    ):
        raise NotImplementedError("Must be implemented by subclass")

    def post_decoder_hook(self):
       return self.logprobs
       

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        env_out: TensorDict,
        done: torch.Tensor, # shape [B], bool
    ):
        selected, logprobs = self._step(logits, mask,done)
        self.actions.append(selected)
        self.logprobs.append(logprobs)
        return selected,logprobs

class Greedy(DecodingStrategy):
    name = "greedy"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select the action with the highest log probability"""
        selected = self.greedy(logprobs, mask)
        return logprobs, selected


class Sampling(DecodingStrategy):
    name = "sampling"

    def _step(self, logit: torch.Tensor, mask: torch.Tensor, done) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling step.
        Args:
        logit: [B, C, T] 各タスク列 t ごとのクルーロジット
        mask : [B, C, T] True=選択可能
        done : [B] もしくは [B, T] のbool。Trueの箇所は計算しない
        Returns:
        selected: [B, T] 各タスク列に割り当てたクルーID（割当なしは -1）
        logprobs: [B, T] 選択の対数確率（割当なしは 0）
        """
        B, C, T = logit.shape
        if isinstance(done, torch.Tensor):
            if done.dim() == 1:
                done_bt = done.view(B, 1).expand(B, T)
            else:
                done_bt = done
        else:
            done_bt = torch.zeros((B, T), dtype=torch.bool, device=logit.device)

        selected = torch.full((B, T), -1, dtype=torch.long, device=logit.device)
        logprobs = torch.zeros((B, T), dtype=logit.dtype, device=logit.device)

        # 将来列で更新していく可用性マスク
        avail_mask = mask.clone()

        for t in range(T):
            # done=True の行はこの列の計算を完全スキップ
            active = ~done_bt[:, t]
            if not torch.any(active):
                continue

            l_t = logit[active, :, t]      # [B_act, C]
            m_t = avail_mask[active, :, t] # [B_act, C]

            # 1行でも選択可能がある行だけ計算
            valid_rows = m_t.any(dim=-1)   # [B_act]
            if not torch.any(valid_rows):
                continue

            l_use = l_t[valid_rows]
            m_use = m_t[valid_rows]

            l_use = l_use.masked_fill(~m_use, float("-inf"))
            logp_use = F.log_softmax(l_use, dim=-1)
            probs_use = logp_use.exp()

            dist = torch.distributions.Categorical(probs_use)
            chosen_use = dist.sample()  # [B_valid]

            # グローバルなバッチインデックスへ戻す
            act_idx = active.nonzero(as_tuple=False).squeeze(1)
            choose_idx = act_idx[valid_rows]

            # 結果を反映
            selected[choose_idx, t] = chosen_use
            logprobs[choose_idx, t] = logp_use.gather(1, chosen_use.unsqueeze(1)).squeeze(1)

            # 選ばれたクルーは将来列では使用不可にする
            if t + 1 < T:
                avail_mask[choose_idx, chosen_use, t + 1:] = False

        return selected, logprobs
