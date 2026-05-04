from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from multi_agent.config import MultiAgentConfig
from multi_agent.utils import safe_softmax

RESIDUAL_WEIGHT: float = 0.3
RESIDUAL_MLP_HIDDEN: int = 64


def _random_mask(emb_dim: int, num_masked: int, seed: int = 0) -> torch.Tensor:
    """Boolean mask: True = visible, False = masked."""
    rng = np.random.default_rng(seed)
    masked_indices = rng.choice(emb_dim, size=num_masked, replace=False)
    mask = torch.ones(emb_dim, dtype=torch.bool)
    mask[masked_indices] = False
    return mask


class AttentionAgent(nn.Module):

    def __init__(
        self,
        agent_id: str = "agent_0",
        emb_dim: int = 768,
        k: int = 8,
        mask_seed: int = 0,
        temperature: float = 0.3,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.emb_dim = emb_dim
        self.k = k
        self.temperature = temperature
        self.role = role

        # 50% of dimensions visible — gives each agent a distinct view.
        self.register_buffer(
            "mask", _random_mask(emb_dim, emb_dim // 2, seed=mask_seed)
        )

        self.attn_query_proj = nn.Linear(emb_dim * 2, emb_dim, bias=False)
        self.attn_key_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.residual_mlp_hidden = nn.Linear(
            emb_dim * 3, RESIDUAL_MLP_HIDDEN, bias=True
        )
        self.residual_mlp_out = nn.Linear(RESIDUAL_MLP_HIDDEN, 1, bias=False)

        self._init_weights(mask_seed)
        self.wins: int = 0
        self.rounds: int = 0
        self.cum_reward: float = 0.0

    def _init_weights(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        nn.init.normal_(self.attn_query_proj.weight, std=0.05, generator=gen)
        nn.init.normal_(self.attn_key_proj.weight, std=0.05, generator=gen)
        nn.init.normal_(self.residual_mlp_hidden.weight, std=0.03, generator=gen)
        nn.init.zeros_(self.residual_mlp_hidden.bias)
        nn.init.normal_(self.residual_mlp_out.weight, std=0.1, generator=gen)

    def _build_policy_state(
        self, query_emb: torch.Tensor, candidate_reps: torch.Tensor
    ) -> torch.Tensor:
        attn_logits = candidate_reps @ query_emb / self.temperature  # (N,)
        attn_weights = torch.softmax(attn_logits, dim=0)  # (N,)
        global_attn_context = attn_weights @ candidate_reps  # (emb_dim,)
        return torch.cat([global_attn_context, query_emb])  # (emb_dim*2,)

    def score_candidates(
        self,
        query_emb: torch.Tensor,
        candidate_reps: torch.Tensor,
    ) -> torch.Tensor:
        N = candidate_reps.shape[0]

        mask_f = self.mask.float()
        q_masked = query_emb * mask_f
        c_masked = candidate_reps * mask_f.unsqueeze(0)

        policy_state = self._build_policy_state(q_masked, c_masked)

        attn_query = self.attn_query_proj(policy_state)
        attn_keys = self.attn_key_proj(c_masked)
        attn_logits = attn_keys @ attn_query

        state_exp = policy_state.unsqueeze(0).expand(N, -1)
        mlp_in = torch.cat([state_exp, c_masked], dim=1)
        residual_logits = self.residual_mlp_out(
            torch.relu(self.residual_mlp_hidden(mlp_in))
        ).squeeze(-1)

        return attn_logits + RESIDUAL_WEIGHT * residual_logits

    def score_candidates_batch(
        self,
        query_embs: torch.Tensor,
        candidate_reps: torch.Tensor,
    ) -> torch.Tensor:
        B = query_embs.shape[0]
        N = candidate_reps.shape[0]

        mask_f = self.mask.float()
        q_masked = query_embs * mask_f.unsqueeze(0)
        c_masked = candidate_reps * mask_f.unsqueeze(0)

        attn_logits = q_masked @ c_masked.t() / self.temperature
        attn_weights = torch.softmax(attn_logits, dim=1)
        global_attn_context = attn_weights @ c_masked

        policy_state = torch.cat([global_attn_context, q_masked], dim=1)

        attn_query = self.attn_query_proj(policy_state)
        attn_keys = self.attn_key_proj(c_masked)
        attn_logits_score = attn_query @ attn_keys.t()

        state_exp = policy_state.unsqueeze(1).expand(B, N, -1)
        cand_exp = c_masked.unsqueeze(0).expand(B, N, -1)
        mlp_in = torch.cat([state_exp, cand_exp], dim=2)
        residual_logits = self.residual_mlp_out(
            torch.relu(self.residual_mlp_hidden(mlp_in))
        ).squeeze(-1)

        return attn_logits_score + RESIDUAL_WEIGHT * residual_logits

    def propose(
        self,
        query_emb: torch.Tensor,
        candidate_reps: torch.Tensor,
        candidate_ids: list[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        N = candidate_reps.shape[0]
        k = min(self.k, N)

        scores = self.score_candidates(query_emb, candidate_reps)
        probs = safe_softmax(scores, self.temperature, dim=0)
        top_k_indices = torch.multinomial(probs, k, replacement=False)
        top_k_ids = [candidate_ids[i] for i in top_k_indices.tolist()]
        return top_k_ids, top_k_indices, scores

    def reset_hidden(self) -> None:
        """No-op — no recurrent state. Kept for API symmetry with CosineActor."""


class CosineActor(nn.Module):
    def __init__(
        self,
        agent_id: str = "cosine",
        emb_dim: int = 768,
        k: int = 8,
        temperature: float = 0.3,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.emb_dim = emb_dim
        self.k = k
        self.temperature = temperature
        self.role = role
        self.wins: int = 0
        self.rounds: int = 0
        self.cum_reward: float = 0.0

    def score_candidates(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        with torch.no_grad():
            scores = candidate_embs @ query_emb
        return scores, None

    def score_candidates_batch(
        self,
        query_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return query_embs @ candidate_embs.t()

    def propose(
        self,
        query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        candidate_ids: list[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        N = candidate_embs.shape[0]
        k = min(self.k, N)
        scores, _ = self.score_candidates(query_emb, candidate_embs)
        probs = safe_softmax(scores, self.temperature, dim=0)
        top_k_indices = torch.multinomial(probs, k, replacement=False)
        top_k_ids = [candidate_ids[i] for i in top_k_indices.tolist()]
        return top_k_ids, top_k_indices, scores

    def reset_hidden(self) -> None:
        """No-op — no recurrent state."""


class AgentPopulation:
    def __init__(self, config: MultiAgentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        roles = config.agent_roles or {}
        self.agents: list[nn.Module] = []
        for i in range(config.num_agents):
            agent_id = f"agent_{i}"
            agent = AttentionAgent(
                agent_id=agent_id,
                emb_dim=config.emb_dim,
                k=config.k,
                mask_seed=i * 100,
                temperature=config.temperature,
                role=roles.get(agent_id),
            ).to(self.device)
            self.agents.append(agent)

        cosine = CosineActor(
            agent_id="cosine",
            emb_dim=config.emb_dim,
            k=config.k,
            temperature=config.temperature,
            role=roles.get("cosine"),
        ).to(self.device)
        self.agents.append(cosine)

    def reset_all_hidden(self) -> None:
        for agent in self.agents:
            agent.reset_hidden()

    def embeddings_to_device(self, embeddings: np.ndarray) -> torch.Tensor:
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)
