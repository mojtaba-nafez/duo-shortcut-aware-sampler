import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nvib_layer import NVIB

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

from dataclasses import dataclass


@dataclass
class Config:
    nvib_prior_mu: float = None
    nvib_prior_var: float = None
    nvib_prior_log_alpha: float = None
    nvib_prior_log_alpha_stdev: float = None

    # optional fields (safe defaults)
    nvib_delta: float = 0.0
    nvib_alpha_tau: float = -45.0
    nvib_mu_tau: float = 1.0
    nvib_stdev_tau: float = 0.1
    nvib_learnable_prior: bool = False
    attention_dropout: float = 0.0



def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)




def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v, layer_number=-1, remove_self_attn=False):
  b, s, h, d = q.shape  # [B, S, H, D]

  # (B, H, S, S) attention logits mask
  diag_mask = torch.zeros((s, s), device=q.device, dtype=q.dtype)
  diag_mask.fill_(0.0)

  # if layer_number > 6 and layer_number <= 11:
  if remove_self_attn and (layer_number > 6):
  # if remove_self_attn:
      diag_mask.fill_diagonal_(-torch.inf)

  # expand to [B, H, S, S]
  attn_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(b, h, s, s)

  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),  # [B, H, S, D]
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=False
  )

  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]

def apply_rotary_pos_emb_sep(x, cos, sin):
    """
    x: [B, H, S, D]
    cos/sin from Rotary()
    """

    # Original:
    # [1, S, 3, 1, D]

    cos = cos[:, :, 0, :, :]   # [1, S, 1, D]
    sin = sin[:, :, 0, :, :]   # [1, S, 1, D]

    # -> [1, 1, S, D]
    cos = cos.permute(0, 2, 1, 3)
    sin = sin.permute(0, 2, 1, 3)

    return (x * cos) + (rotate_half(x) * sin)

class DDiT_NVIB_Block(nn.Module):
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1, layer_number=0):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN

    self.norm1 = LayerNorm(dim)
    # self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    # self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout
    self.layer_number = layer_number 

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()


    assert dim % n_heads == 0

    self.n_kv_heads = n_heads
    self.head_dim = dim // n_heads
    self.scaling = self.head_dim ** -0.5
    self.dim = dim

    config = Config()
    self.nvib = NVIB(
        size_in=dim,
        size_out=dim,
        prior_mu=config.nvib_prior_mu,
        prior_var=config.nvib_prior_var,
        prior_log_alpha=config.nvib_prior_log_alpha,
        prior_log_alpha_stdev=config.nvib_prior_log_alpha_stdev,
        delta=config.nvib_delta,
        nheads=n_heads,
        alpha_tau=config.nvib_alpha_tau,
        mu_tau=config.nvib_mu_tau,
        stdev_tau=config.nvib_stdev_tau,
        learnable_prior=config.nvib_learnable_prior,
    )
    self._last_nvib_outputs = None
    self.attention_dropout = config.attention_dropout

     # Store NVIB outputs so we can compute KL loss after forward pass
    self.kl_gaussian = None
    self.kl_dirichlet = None
    
    self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.attn_out = nn.Linear(n_heads * self.head_dim, dim, bias=False)


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c=None, remove_self_attn=False, kl_loss=False, use_trained_scaling_factor=False, activate_nvib_noise=False):
    batch_size, seq_len = x.shape[0], x.shape[1]
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      # self.adaLN_modulation(c): (128, 1536)
      # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
      # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    z, pi, mu, logvar, alpha, mask = self.nvib(
            encoder_output=x,
            batch_first=True,
            logging=self.training,
            use_trained_scaling_factor=use_trained_scaling_factor,
            activate_nvib_noise=activate_nvib_noise
    )
    if self.training or kl_loss:
      self.kl_gaussian, self.kl_dirichlet = self.get_kl_loss(z, pi, mu, logvar, alpha, mask)
    input_for_kv = z
    
    query_states = self.q_proj(x)
    key_states = self.k_proj(input_for_kv)
    value_states = self.v_proj(input_for_kv)
    query_states = query_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
    cos, sin = rotary_cos_sin

    k_prior = key_states[:, :, :1, :]   # (B, n_kv_heads, 1, head_dim) - Prior
    k_seq = key_states[:, :, 1:, :]     # (B, n_kv_heads, T, head_dim) - Sequence
    query_states = apply_rotary_pos_emb_sep(query_states, cos.to(query_states.dtype), sin.to(query_states.dtype))
    k_seq = apply_rotary_pos_emb_sep(k_seq, cos.to(k_seq.dtype),sin.to(k_seq.dtype))
    key_states = torch.cat([k_prior, k_seq], dim=2)  # (B, n_kv_heads, Nl, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    pi_clamped = torch.clamp(pi, min=torch.finfo(pi.dtype).tiny)
    if self.training or use_trained_scaling_factor:
      exp_scale =  0.2
    else:
      exp_scale = 0.04
    log_pi = torch.log(pi_clamped).permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, Nl)
    l2_norm = (torch.norm(input_for_kv, dim=-1, keepdim=True) ** 2)  # (B, Nl, 1)
    l2_norm = l2_norm.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 1, Nl)
    scale_factor = 1.0 / (2.0 * math.sqrt(self.head_dim))
    attn_weights = attn_weights + log_pi - (scale_factor * l2_norm * exp_scale)

    if remove_self_attn and self.layer_number > 6:
        idx = torch.arange(seq_len, device=attn_weights.device)
        attn_weights[:, :, idx, idx + 1] = -torch.inf

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, n_heads, head_dim)
    x = attn_output.view(batch_size, seq_len, self.dim)  # (B, T, C)

    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x),
                                None,
                                gate_msa,
                                x_skip,
                                self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x
  
  
  
  def get_kl_loss(self, z, pi, mu, logvar, alpha, mask):
    # Compute KL divergence for Gaussian component
    # kl_gaussian expects: mu, logvar, alpha in (Nl, B, *) format
    kl_gaussian = self.nvib.kl_gaussian(
        mu=mu.transpose(0, 1),           # (Nl, B, C)
        logvar=logvar.transpose(0, 1),   # (Nl, B, C)
        alpha=alpha.transpose(0, 1),     # (Nl, B, 1)
        mask=mask,    # (B, Nl) - stays as is
    )  # Returns: (B,)
    
    # Compute KL divergence for Dirichlet component  
    # kl_dirichlet expects: alpha in (Nl, B, *) format
    kl_dirichlet = self.nvib.kl_dirichlet(
        alpha=alpha.transpose(0, 1),     # (Nl, B, 1)
        mask=mask,    # (B, Nl) - stays as is
    )  # Returns: (B,)
    
    return kl_gaussian, kl_dirichlet

  def get_kl_div(self) -> dict[str, torch.Tensor]:
    if self.kl_gaussian is None or self.kl_dirichlet is None:
        raise RuntimeError(
            "KL loss not available. Run forward() in training mode first to compute KL losses."
        )
    return self.kl_gaussian, self.kl_dirichlet


if __name__ == "__main__":

    block = DDiT_NVIB_Block(dim=1024, adaLN=True, n_heads=16, cond_dim=128)

    # x.shape, rotary_cos_sin[0].shape, c.shape: torch.Size([1, 512, 1024]) torch.Size([1, 512, 3, 1, 64]) torch.Size([1, 128])
    block.train()
    x = torch.randn(2, 512, 1024)
    rotary_cos_sin = (
        torch.randn(1, 512, 3, 1, 64),
        torch.randn(1, 512, 3, 1, 64),
    )  
    c = torch.randn(2, 128)
    out = block(x, rotary_cos_sin, c, kl_loss=True)
    print("Forward pass successful.", out.shape)
    print("KL Gaussian:", block.kl_gaussian.shape)
    print("KL Dirichlet:", block.kl_dirichlet.shape)