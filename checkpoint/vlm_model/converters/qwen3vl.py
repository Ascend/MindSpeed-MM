from pathlib import Path
from transformers import AutoConfig, AutoProcessor

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict, save_hf_weights, merge_dcp_to_hf_sharded
from checkpoint.common.hf_to_dcp import hf_to_dcp_sharded
from checkpoint.vlm_model.hf_to_mm import load_from_hf, save_by_dcp


class Qwen3VLConverter(Converter):
    """
    A utility class to convert model checkpoints of Qwen3-VL between different formats,
    specifically between Hugging Face (HF) and torch-dcp (DCP) formats.
    
    Supports:
    - HF → DCP conversion
    - DCP → HF merging
    - Placeholder methods for megatron format and resharding operations.
    """
    
    dcp_prefix = "model."
    hf_prefix = ""
    # Mapping for tied weights (used when output head shares weights with input embeddings)
    tie_weight_mapping = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # MoE experts params
    fused_linear_names = ["gate_up_proj", "down_proj"]
    
    def hf_to_dcp(
        self, 
        hf_dir: str = "Qwen3-VL-xxB",        # Input: Path to HF-format model directory
        dcp_dir: str = "Qwen3-VL-xxB-dcp",   # Output: Path to save DCP-format model
        tie_weight: bool = False             # Whether to tie lm_head with embeddings
    ):
        """
        Converts a Hugging Face formatted model checkpoint to torch-dcp format.
        
        Steps:
        1. Load the state dict from HF format.
        2. Optionally tie weights (e.g., share lm_head and embed_tokens weights).
        3. Rename all keys by adding DCP prefix and removing HF prefix.
        4. Save the converted checkpoint in DCP format.
        5. Set proper directory permissions.
        """
        
        def state_dict_convert_func(state_dict):        
            if tie_weight:
                for tgt_weight, src_weight in self.tie_weight_mapping.items():
                    if src_weight in state_dict.keys():
                        state_dict[tgt_weight] = state_dict[src_weight]
                    
            ori_keys = list(state_dict.keys())
            for ori_key in ori_keys:
                value = state_dict.pop(ori_key)
                
                # view experts weight: (expert_num, input_dim, output_dim) -> (expert_num * input_dim, output_dim)
                if any(fused_linear_name in ori_key for fused_linear_name in self.fused_linear_names):
                    value = value.view(-1, value.shape[-1])
                
                new_key = ori_key.replace(self.hf_prefix, self.dcp_prefix, 1) if len(self.hf_prefix) > 0 else f"{self.dcp_prefix}{ori_key}"
                state_dict[new_key] = value
            return state_dict
        
        hf_to_dcp_sharded(
            hf_dir=hf_dir,
            dcp_dir=dcp_dir,
            state_dict_convert_func=state_dict_convert_func
        )
        
    def dcp_to_hf(
        self, 
        load_dir: str = "mm_save_dir/release",     # Input: Directory containing DCP shards
        save_dir: Path = "Qwen3-VL-xxB-hf",         # Output: Directory to save merged HF model
        model_assets_dir: str = "Qwen3-VL-xxB"     # Reference: Original HF model dir (for config/tokenizer)
    ):
        """
        Merges torch-dcp shards and converts them back into standard Hugging Face format.
        
        This is typically used after training or inference in torch-dcp format to export 
        a model that can be easily loaded with Hugging Face Transformers.
        """
        config = AutoConfig.from_pretrained(model_assets_dir)
        num_experts = getattr(config.text_config, "num_experts", None)
        
        def state_dict_convert_func(state_dict):
            state_dict_keys = list(state_dict.keys())

            for key in state_dict_keys:
                # view experts weight: (expert_num * input_dim, output_dim) -> (expert_num, input_dim, output_dim)
                if num_experts and any(fused_linear_name in key for fused_linear_name in self.fused_linear_names):
                    state_dict[key] = state_dict[key].view(num_experts, -1, state_dict[key].shape[-1])
                value = state_dict.pop(key)
                new_key = key.replace(self.dcp_prefix, self.hf_prefix, 1) if key.startswith(self.dcp_prefix) else key
                state_dict[new_key] = value
            
            return state_dict

        merge_dcp_to_hf_sharded(
            load_dir=Path(load_dir),
            save_dir=Path(save_dir),
            model_assets_dir=Path(model_assets_dir),
            select_key_convert_func=lambda key: f"model.{self.dcp_prefix}" + key,
            state_dict_convert_func=state_dict_convert_func
        )
    
    @staticmethod    
    def hf_to_mm():
        pass
    
    @staticmethod
    def mm_to_hf():
        pass
    
    @staticmethod
    def resplit():
        pass