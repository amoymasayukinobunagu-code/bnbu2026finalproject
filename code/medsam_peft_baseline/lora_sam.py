"""
SAMed LoRA wrapper — adapted from hitachinsk/SAMed/sam_lora_image_encoder.py

Applies LoRA (rank r) to the qkv attention layers in SAM's image encoder ViT blocks.
Only Q and V projections get LoRA adaptation (following the original SAMed paper).

Key difference from original SAMed:
- We use the STANDARD segment_anything (pip-installed), not SAMed's modified version
- We keep prompt-based inference (box prompt via SamPredictor)
- LoRA is applied ONLY to image encoder; prompt_encoder + mask_decoder are fully trained
  (matching SAMed's original approach)

Checkpoint format: saves only LoRA weights (w_As, w_Bs) + prompt_encoder + mask_decoder
"""
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """Wraps SAM's qkv linear with LoRA low-rank update on Q and V.

    Original SAM: self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    LoRA: q_new = q + B_q @ A_q,  v_new = v + B_v @ A_v
    """

    def __init__(self, qkv: nn.Module, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, N, 3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Applies LoRA to SAM image encoder's attention qkv layers.

    Args:
        sam_model: a Sam model instance
        r: LoRA rank (default 4, following SAMed paper)
        lora_layer: which ViT blocks to apply LoRA (default: all)
    """

    def __init__(self, sam_model: Sam, r: int = 4, lora_layer=None):
        super().__init__()
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.w_As = []
        self.w_Bs = []

        # Freeze image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA to each ViT block's qkv
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str):
        """Save LoRA weights + prompt_encoder + mask_decoder state."""
        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)
        # Count trainable params
        trainable = sum(t.numel() for t in a_tensors.values()) + sum(t.numel() for t in b_tensors.values())
        print(f"Saved LoRA checkpoint: {filename} (LoRA params: {trainable:,})")

    def load_lora_parameters(self, filename: str):
        """Load LoRA weights + prompt_encoder + mask_decoder state."""
        state_dict = torch.load(filename, map_location='cpu')
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            w_A_linear.weight = Parameter(state_dict[saved_key])
        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            w_B_linear.weight = Parameter(state_dict[saved_key])

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        # Load prompt encoder
        pe_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        pe_values = [state_dict[k] for k in pe_keys]
        sam_dict.update({k: v for k, v in zip(pe_keys, pe_values)})
        # Load mask decoder
        md_keys = [k for k in sam_keys if 'mask_decoder' in k]
        md_values = [state_dict[k] for k in md_keys]
        sam_dict.update({k: v for k, v in zip(md_keys, md_values)})
        self.sam.load_state_dict(sam_dict)
        print(f"Loaded LoRA checkpoint: {filename}")

    @property
    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
