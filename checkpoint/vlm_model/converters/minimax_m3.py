import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from checkpoint.common.converter import Converter
from checkpoint.common.hf_to_dcp import hf_to_dcp_sharded
from checkpoint.common.dcp_utils import append_state_dict_to_dcp
from checkpoint.common.permissions import set_directory_permissions


class MiniMaxM3Converter(Converter):
    """Converter for MiniMax-M3 HF checkpoints to MindSpeed-MM FSDP2 DCP."""

    expert_pattern = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$"
    )
    gate_up_patterns = (
        (
            re.compile(r"^language_model\.model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.weight$"),
            "model.language_model.layers.{layer_idx}.mlp.gate_up_proj.weight",
        ),
        (
            re.compile(
                r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.shared_experts\.(gate_proj|up_proj)\.weight$"
            ),
            "model.language_model.layers.{layer_idx}.mlp.shared_experts.gate_up_proj.weight",
        ),
    )
    key_mapping = (
        (re.compile(r"^language_model\.model\.(.*)\.self_attn\.index_q_proj\."), r"model.language_model.\1.self_attn.indexer.q_proj."),
        (re.compile(r"^language_model\.model\.(.*)\.self_attn\.index_k_proj\."), r"model.language_model.\1.self_attn.indexer.k_proj."),
        (re.compile(r"^language_model\.model\.(.*)\.self_attn\.index_q_norm\."), r"model.language_model.\1.self_attn.indexer.q_norm."),
        (re.compile(r"^language_model\.model\.(.*)\.self_attn\.index_k_norm\."), r"model.language_model.\1.self_attn.indexer.k_norm."),
        (re.compile(r"^language_model\.model\.(.*)\.block_sparse_moe\.gate\."), r"model.language_model.\1.mlp.gate."),
        (
            re.compile(r"^language_model\.model\.(.*)\.block_sparse_moe\.e_score_correction_bias"),
            r"model.language_model.\1.mlp.gate.e_score_correction_bias",
        ),
        (re.compile(r"^language_model\.model\.(.*)\.block_sparse_moe\.shared_experts\."), r"model.language_model.\1.mlp.shared_experts."),
        (re.compile(r"^language_model\.lm_head\."), "lm_head."),
        (re.compile(r"^language_model\.model\."), "model.language_model."),
        (re.compile(r"^vision_tower\.vision_model\.embeddings\.patch_embedding\."), "model.vision_tower.embeddings.proj."),
        (re.compile(r"^vision_tower\.vision_model\.encoder\.layers\."), "model.vision_tower.layers."),
        (re.compile(r"^vision_tower\.vision_model\."), "model.vision_tower."),
        (re.compile(r"^multi_modal_projector\."), "model.multi_modal_projector."),
        (re.compile(r"^patch_merge_mlp\."), "model.multi_modal_projector.merge_"),
    )

    @classmethod
    def convert_key(cls, key: str) -> str:
        for pattern, replacement in cls.key_mapping:
            new_key, n_subs = pattern.subn(replacement, key)
            if n_subs > 0:
                return new_key
        return key

    @classmethod
    def _match_gate_up_pair(cls, key: str) -> tuple[str, str] | None:
        for pattern, target_template in cls.gate_up_patterns:
            match = pattern.match(key)
            if match is None:
                continue
            layer_idx, weight_name = match.groups()
            return target_template.format(layer_idx=layer_idx), weight_name
        return None

    @staticmethod
    def _is_weight_asset(path: Path) -> bool:
        if path.name == "model.safetensors.index.json":
            return True
        return path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".distcp"}

    @classmethod
    def _copy_model_assets(cls, hf_dir: str, dcp_dir: str) -> None:
        src_root = Path(hf_dir)
        dst_root = Path(dcp_dir)
        dst_root.mkdir(parents=True, exist_ok=True)
        for src in src_root.iterdir():
            if src.name.startswith(".") or cls._is_weight_asset(src):
                continue
            dst = dst_root / src.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    @classmethod
    def _build_weight_indexes(cls, hf_dir: str, hf_prefix: str = ""):
        # Build a map of layer_idx -> expert_idx -> weight_name -> (filename, raw_key)
        # where filename is the safetensors filename and raw_key is the key in that file.
        # this is to ensure that we can merge the experts across HF shards.
        index_path = Path(hf_dir) / SAFE_WEIGHTS_INDEX_NAME
        if not index_path.exists():
            raise RuntimeError(
                "MiniMax-M3 expert conversion requires model.safetensors.index.json so split experts can be "
                "merged across HF shards."
            )

        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        expert_index = defaultdict(lambda: defaultdict(dict))
        gate_up_index = defaultdict(dict)
        for raw_key, filename in weight_map.items():
            key = raw_key.replace(hf_prefix, "", 1) if hf_prefix and raw_key.startswith(hf_prefix) else raw_key
            match = cls.expert_pattern.match(key)
            if match is not None:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                weight_name = match.group(3)
                expert_index[layer_idx][expert_idx][weight_name] = (filename, raw_key)
                continue

            matched = cls._match_gate_up_pair(key)
            if matched is None:
                continue
            target_key, weight_name = matched
            gate_up_index[target_key][weight_name] = (filename, raw_key)
        return expert_index, gate_up_index

    @staticmethod
    def _load_hf_tensor(hf_dir: str, filename: str, key: str) -> torch.Tensor:
        with safe_open(Path(hf_dir) / filename, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    @classmethod
    def _build_expert_layer_state_dict(
        cls,
        hf_dir: str,
        layer_idx: int,
        layer_parts: Dict[int, Dict[str, tuple[str, str]]],
        num_experts: int,
        dcp_prefix: str,
    ) -> Dict[str, torch.Tensor]:
        """
        M3的 MoE expert 参数是为了支持npu gemm算子, 需要把w2 堆叠, 以及把 w1/w3  concat 后堆叠。

        HF expert 通常是：
            experts.{expert_idx}.w1.weight
            experts.{expert_idx}.w2.weight
            experts.{expert_idx}.w3.weight

        目标 格式是：
            experts.gate_up_proj = concat(w1, w3) 后再按 expert 堆叠
            experts.down_proj    = w2 按 expert 堆叠
        """
        for expert_idx in range(num_experts):
            parts = layer_parts.get(expert_idx)
            if parts is None or set(parts) != {"w1", "w2", "w3"}:
                raise RuntimeError(f"Layer {layer_idx} expert {expert_idx} is missing one of w1/w2/w3.")

        first = layer_parts[0]
        w1 = cls._load_hf_tensor(hf_dir, *first["w1"])
        w2 = cls._load_hf_tensor(hf_dir, *first["w2"])
        w3 = cls._load_hf_tensor(hf_dir, *first["w3"])

        if w1.shape != w3.shape:
            raise RuntimeError(f"Layer {layer_idx} expert 0 has mismatched w1/w3 shapes: {w1.shape} vs {w3.shape}.")

        gate_rows = w1.shape[0]
        gate_up = torch.empty(
            num_experts,
            w1.shape[0] + w3.shape[0],
            w1.shape[1],
            dtype=w1.dtype,
            device=w1.device,
        )
        down = torch.empty(num_experts, *w2.shape, dtype=w2.dtype, device=w2.device)
        del w1, w2, w3

        for expert_idx in range(num_experts):
            parts = layer_parts[expert_idx]
            w1 = cls._load_hf_tensor(hf_dir, *parts["w1"])
            w2 = cls._load_hf_tensor(hf_dir, *parts["w2"])
            w3 = cls._load_hf_tensor(hf_dir, *parts["w3"])
            if w1.shape[0] != gate_rows or w3.shape[0] != gate_rows:
                raise RuntimeError(f"Layer {layer_idx} expert {expert_idx} has unexpected w1/w3 shape.")
            gate_up[expert_idx, :gate_rows, :] = w1
            gate_up[expert_idx, gate_rows:, :] = w3
            down[expert_idx] = w2
            del w1, w2, w3

        return {
            f"{dcp_prefix}model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj": gate_up,
            f"{dcp_prefix}model.language_model.layers.{layer_idx}.mlp.experts.down_proj": down,
        }

    @classmethod
    def _build_gate_up_state_dict(
        cls,
        hf_dir: str,
        target_key: str,
        parts: Dict[str, tuple[str, str]],
        dcp_prefix: str,
    ) -> Dict[str, torch.Tensor]:
        if set(parts) != {"gate_proj", "up_proj"}:
            raise RuntimeError(f"{target_key} is missing one of gate_proj/up_proj.")

        gate = cls._load_hf_tensor(hf_dir, *parts["gate_proj"])
        up = cls._load_hf_tensor(hf_dir, *parts["up_proj"])
        if gate.shape != up.shape:
            raise RuntimeError(f"{target_key} has mismatched gate/up shapes: {gate.shape} vs {up.shape}.")

        return {f"{dcp_prefix}{target_key}": torch.cat([gate, up], dim=0)}

    @classmethod
    def _append_gate_up_layers_to_dcp(
        cls,
        hf_dir: str,
        dcp_dir: str,
        gate_up_index,
        dcp_prefix: str,
        start_part_idx: int,
    ) -> None:
        for offset, target_key in enumerate(tqdm(sorted(gate_up_index), desc="Processing gate/up layers")):
            state_dict = cls._build_gate_up_state_dict(
                hf_dir=hf_dir,
                target_key=target_key,
                parts=gate_up_index[target_key],
                dcp_prefix=dcp_prefix,
            )
            append_state_dict_to_dcp(Path(dcp_dir), state_dict, part_idx=start_part_idx + offset)
            del state_dict

    @classmethod
    def _append_expert_layers_to_dcp(
        cls,
        hf_dir: str,
        dcp_dir: str,
        expert_index,
        num_experts: int,
        dcp_prefix: str,
        start_part_idx: int,
    ) -> None:
        for offset, layer_idx in enumerate(tqdm(sorted(expert_index), desc="Processing expert layers")):
            layer_state_dict = cls._build_expert_layer_state_dict(
                hf_dir=hf_dir,
                layer_idx=layer_idx,
                layer_parts=expert_index[layer_idx],
                num_experts=num_experts,
                dcp_prefix=dcp_prefix,
            )
            append_state_dict_to_dcp(Path(dcp_dir), layer_state_dict, part_idx=start_part_idx + offset)
            del layer_state_dict

    def hf_to_dcp(
        self,
        hf_dir: str = "",
        dcp_dir: str = "",
        dcp_prefix: str = "",
        hf_prefix: str = "",
        tie_weight_mapping: Dict[str, str] = None,
        num_workers: int = 0,
        save_model_assets: bool = True,
    ):
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        num_experts = getattr(text_config, "num_local_experts", getattr(text_config, "num_experts", 0))
        if not num_experts:
            raise RuntimeError("Cannot infer num_local_experts/num_experts from MiniMax-M3 config.")

        if save_model_assets:
            self._copy_model_assets(hf_dir, dcp_dir)

        expert_index, gate_up_index = self._build_weight_indexes(hf_dir, hf_prefix)

        def state_dict_convert_func(state_dict):
            converted = {}

            for raw_key, value in state_dict.items():
                key = raw_key.replace(hf_prefix, "", 1) if hf_prefix and raw_key.startswith(hf_prefix) else raw_key
                if self.expert_pattern.match(key) is not None:
                    continue
                if self._match_gate_up_pair(key) is not None:
                    continue

                converted[f"{dcp_prefix}{self.convert_key(key)}"] = value

            if tie_weight_mapping:
                for tgt_weight, src_weight in tie_weight_mapping.items():
                    src_key = f"{dcp_prefix}{src_weight}"
                    if src_key in converted:
                        converted[f"{dcp_prefix}{tgt_weight}"] = converted[src_key]
            return converted

        hf_to_dcp_sharded(
            hf_dir=hf_dir,
            dcp_dir=dcp_dir,
            state_dict_convert_func=state_dict_convert_func,
            num_workers=num_workers,
        )
        base_part_idx = len(list(Path(hf_dir).glob("*.safetensors")))
        self._append_gate_up_layers_to_dcp(
            hf_dir=hf_dir,
            dcp_dir=dcp_dir,
            gate_up_index=gate_up_index,
            dcp_prefix=dcp_prefix,
            start_part_idx=base_part_idx,
        )
        self._append_expert_layers_to_dcp(
            hf_dir=hf_dir,
            dcp_dir=dcp_dir,
            expert_index=expert_index,
            num_experts=num_experts,
            dcp_prefix=dcp_prefix,
            start_part_idx=base_part_idx + len(gate_up_index),
        )
        if save_model_assets:
            set_directory_permissions(Path(dcp_dir))

    @staticmethod
    def dcp_to_hf():
        print("dcp_to_hf is not supported in MiniMaxM3Converter.")

    @staticmethod
    def hf_to_mm():
        pass

    @staticmethod
    def mm_to_hf():
        pass

    @staticmethod
    def resplit():
        pass
