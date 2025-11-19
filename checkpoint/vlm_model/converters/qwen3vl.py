from pathlib import Path

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.common.merge_dcp_to_hf import merge_dcp_to_hf
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
        
        state_dict = load_from_hf(Path(hf_dir))
        
        if tie_weight:
            for tgt_weight, src_weight in self.tie_weight_mapping.items():
                state_dict[tgt_weight] = state_dict[src_weight]
                
        ori_keys = list(state_dict.keys())
        for ori_key in ori_keys:
            state_dict[f"{self.dcp_prefix}{ori_key}"] = state_dict.pop(f"{self.hf_prefix}{ori_key}")
        
        save_by_dcp(state_dict, Path(dcp_dir))
        set_directory_permissions(Path(dcp_dir))
        
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
        
        merge_dcp_to_hf(
            load_dir=load_dir,
            save_dir=save_dir,
            model_assets_dir=model_assets_dir,
            prefix=self.dcp_prefix
        )
        set_directory_permissions(save_dir)
    
    @staticmethod    
    def hf_to_mm():
        pass
    
    @staticmethod
    def mm_to_hf():
        pass
    
    @staticmethod
    def resplit():
        pass