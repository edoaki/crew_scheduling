import abc
from typing import Tuple
import torch


def get_decoding_strategy(phase, temperature: float = 1.0, tanh_clipping: float = 10.0):
    if phase == "train":
        decode_strategy = Sampling(temperature=temperature, tanh_clipping=tanh_clipping)
    elif phase in ["val", "test"]:
        decode_strategy = Greedy(temperature=temperature, tanh_clipping=tanh_clipping)
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
        self.logprobs = None

    def post_decoder_hook(self):
        # shape [B, T] を [B] に集約
        if self.logprobs is None:
            return None
        return self.logprobs.sum(dim=1)
       
    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        done: torch.Tensor, # shape [B], bool
    ):  
        selected, logprobs = self._step(logits, mask,done)
        self.actions.append(selected)
        if self.logprobs is None:
            self.logprobs = logprobs
        else:
            # 時間次元に連結
            self.logprobs = torch.cat([self.logprobs, logprobs], dim=-1)
        return selected

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

        # 将来列で更新していく可用性マスク（True=選択可能 を維持）
        avail_mask = mask.clone()

        for t in range(T):
            # この列でまだ処理すべきバッチ
            active = ~done_bt[:, t]
            if not torch.any(active):
                continue

            l_t = logit[active, :, t]         # [B_act, C]
            m_t = avail_mask[active, :, t]    # [B_act, C]

            # 少なくとも1つ選択可能な行だけ処理
            valid_rows = m_t.any(dim=-1)      # [B_act]
            if not torch.any(valid_rows):
                continue

            l_use = l_t[valid_rows]           # [B_valid, C]
            m_use = m_t[valid_rows]           # [B_valid, C]

            # 禁止手は -inf にして温度を適用（Categorical(logits=...) を使う）
            masked_logits = l_use.masked_fill(~m_use, float("-inf"))
            if self.temperature != 1.0:
                masked_logits = masked_logits / self.temperature

            dist = torch.distributions.Categorical(logits=masked_logits)  # [B_valid]
            chosen = self.choose(dist)                                     # [B_valid]
            chosen_logp = dist.log_prob(chosen)                            # [B_valid]

            # グローバルインデックスへ戻す
            act_idx = active.nonzero(as_tuple=False).squeeze(1)            # [B_act]
            choose_idx = act_idx[valid_rows]                                # [B_valid]

            # 結果を反映
            selected[choose_idx, t] = chosen
            logprobs[choose_idx, t] = chosen_logp

            # 選ばれたクルーは将来列では使用不可にする
            if t + 1 < T:
                avail_mask[choose_idx, chosen, t + 1:] = False
    
        return selected, logprobs


class Sampling(DecodingStrategy):
    name = "sampling"

    def __init__(
        self,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
    ):
        super(Sampling, self).__init__(temperature, tanh_clipping)

    def choose(self,dist):
        return dist.sample()
    
class Greedy(DecodingStrategy):
    name = "greedy"
    
    def __init__(
        self,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
    ):
        super(Greedy, self).__init__(temperature, tanh_clipping)

    def choose(self,dist):
        return dist.probs.argmax(dim=-1)