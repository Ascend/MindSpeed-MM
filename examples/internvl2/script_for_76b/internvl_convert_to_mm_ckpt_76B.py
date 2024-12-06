import os
from copy import deepcopy
import re
import stat
import torch
from transformers import AutoModelForCausalLM
import gc

gc.collect()


def load_from_hf(_load_dir):
    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(_load_dir, device_map='cpu', trust_remote_code=True)
    return hf_model


def convert_hg_to_mm(_state_dict, _num_layers):
    new_dict = {}
    for key, value in _state_dict.items():
        new_key = None
        # 权重映射
        if key.startswith('vision_model'):
            new_key = key.replace('vision_model', 'image_encoder.encoder')
            new_key = new_key.replace('norm1', 'input_layernorm')
            new_key = new_key.replace('norm2', 'pre_mlp_layernorm')
            new_key = new_key.replace('attn', 'self_attention')
            new_key = new_key.replace('qkv', 'linear_qkv')
            new_key = new_key.replace('proj', 'linear_proj')
            new_key = new_key.replace('q_norm', 'q_layernorm')
            new_key = new_key.replace('k_norm', 'k_layernorm')
            new_key = new_key.replace('mlp.fc1', 'mlp.linear_fc1')
            new_key = new_key.replace('mlp.fc2', 'mlp.linear_fc2')
           
        elif key.startswith('language_model'):
            new_key = key.replace('language_model', 'text_decoder')
            new_key = new_key.replace('model.embed_tokens', 'embedding.word_embeddings')
            new_key = new_key.replace('model.layers', 'decoder.layers')
            new_key = new_key.replace('self_attn', 'self_attention')

            new_key = new_key.replace('o_proj', 'linear_proj')
            new_key = new_key.replace('q_proj', 'wq')
            new_key = new_key.replace('k_proj', 'wk')
            new_key = new_key.replace('v_proj', 'wv')

            new_key = new_key.replace('gate_proj', 'wg')
            new_key = new_key.replace('up_proj', 'wu')
            new_key = new_key.replace('down_proj', 'linear_fc2')

            new_key = new_key.replace('input_layernorm', 'input_layernorm')
            new_key = new_key.replace('post_attention_layernorm', 'pre_mlp_layernorm')
            new_key = new_key.replace('lm_head', 'output_layer')
            new_key = new_key.replace('model.norm', 'decoder.final_layernorm')
        elif key.startswith('mlp1'):
            new_key = key.replace('mlp1.0', 'image_encoder.projector.norm')
            new_key = new_key.replace('mlp1.1', 'image_encoder.projector.linear_fc1')
            new_key = new_key.replace('mlp1.3', 'image_encoder.projector.linear_fc2')

        # 打印映射过程
        new_dict[new_key] = value

    # 合并wq,wk,wv
    for i in range(_num_layers):
        q_name = f'text_decoder.decoder.layers.{i}.self_attention.wq.weight'
        k_name = f'text_decoder.decoder.layers.{i}.self_attention.wk.weight'
        v_name = f'text_decoder.decoder.layers.{i}.self_attention.wv.weight'
        qkv_name = f'text_decoder.decoder.layers.{i}.self_attention.linear_qkv.weight'
        
        if q_name in new_dict.keys():
            wq = new_dict[q_name]
        if k_name in new_dict.keys():
            wk = new_dict[k_name]
        if v_name in new_dict.keys():
            wv = new_dict[v_name]
        linear_qkv = merge_qkv(wq, wk, wv)
        new_dict[qkv_name] = linear_qkv

        # 移除合并前的权重
        new_dict.pop(q_name)
        new_dict.pop(k_name)
        new_dict.pop(v_name)

        print(f'merge {q_name}、{k_name}、{v_name} to {qkv_name}')

    # 合并wg和wu权重
    for i in range(_num_layers):
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.wg.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.wu.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'

        # 合并 w1 和 w3
        if gate_name in new_dict.keys():
            gate_proj_weight = new_dict[gate_name]
        if up_name in new_dict.keys():
            up_proj_weight = new_dict[up_name]
        linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
        new_dict[fc1_name] = linear_fc1

        # 移除合并前的权重
        new_dict.pop(gate_name)
        new_dict.pop(up_name)

        print(f'merge {gate_name} and {up_name} to {fc1_name}')

    return new_dict


def merge_qkv(wq, wk, wv, hn=64, ng=8):
    dq, d = wq.shape
    dkv = wk.shape[0]
    hh = d // hn
    qkv = torch.zeros(size=[dq + dkv * 2, dq], dtype=wq.dtype, device=wq.device)
    i = 0
    for j in range(hn):
        qkv[i * hh: (i + 1) * hh, :] = wq[j * hh: (j + 1) * hh, :]
        if (j + 1) % ng == 0 and j > 0:
            qkv[(i + 1) * hh: (i + 2) * hh, :] = wk[(j // ng) * hh: (j // ng + 1) * hh, :]
            qkv[(i + 2) * hh: (i + 3) * hh, :] = wv[(j // ng) * hh: (j // ng + 1) * hh, :]
            i = i + 2
        i = i + 1
    return qkv


def save_by_pp(_state_dicts, _save_dir, _latest_checkpointed_iteration='release', _exists_ok=False):
    if os.path.exists(_save_dir):
        if not _exists_ok:
            print(f'save dir: {_save_dir} exists, please check.')
            return
    else:
        os.makedirs(_save_dir)
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(_save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(_latest_checkpointed_iteration)

    if _latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(_latest_checkpointed_iteration)

    if len(_state_dicts) > 1:
        for pp_rank, _state_dict in enumerate(_state_dicts):
            tp_rank = 0
            os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'))
            save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}', 'model_optim_rng.pt')
            save_dict = {}
            save_dict['model'] = _state_dict
            torch.save(save_dict, save_path)
    else:
        _state_dict = _state_dicts[0]
        tp_rank = 0
        os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}'))
        save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}', 'model_optim_rng.pt')
        save_dict = {}
        save_dict['model'] = _state_dict
        torch.save(save_dict, save_path)


def clip_model(hf_model, llm_num_layers, vit_num_layers):
    if len(hf_model.vision_model.encoder.layers) < vit_num_layers:
        raise AssertionError(f"vit's layers num {len(hf_model.vision_model.encoder.layers)} is smaller than {vit_num_layers}")
    if len(hf_model.language_model.model.layers) < llm_num_layers:
        raise AssertionError(f"llm's layers num {len(hf_model.language_model.model.layers)} is smaller than {llm_num_layers}")
    ret_dict = deepcopy(hf_model.state_dict())
    for key, value in hf_model.state_dict().items():
        if key.startswith('vision_model.encoder.layers'):
            res = re.findall(r'vision_model.encoder.layers.(\d+).', key)
            if res:
                layer_idx = eval(res[0])
                if layer_idx >= vit_num_layers:
                    ret_dict.pop(key)
        elif key.startswith('language_model.model.layers'):
            res = re.findall(r'language_model.model.layers.(\d+).', key)
            if res:
                layer_idx = eval(res[0])
                if layer_idx >= llm_num_layers:
                    ret_dict.pop(key)
    return ret_dict
    

def merge_pp_index(llm_num, vit_num, llm_num_layers, vit_num_layers, pp_size):
    if len(llm_num) != pp_size:
        raise AssertionError(f"llm layers num must be divide into {pp_size} stages")
    if len(vit_num) != pp_size:
        raise AssertionError(f"vit layers num must be divide into {pp_size} stages")
    if sum(llm_num) != llm_num_layers:
        raise AssertionError(f"the sum of llm layers number in all stages must be equal {llm_num_layers}")
    if sum(vit_num) != vit_num_layers:
        raise AssertionError(f"the sum of vit layers number in all stages must be equal {vit_num_layers}")
    spilt_method = []
    for a, b in zip(vit_num, llm_num):
        spilt_method.append((a, b))
    return spilt_method


def split_model_by_pp(_state_dict, pp_split):
    if pp_split is None or len(pp_split) <= 1:
        return [_state_dict], {}
    pp_size = len(pp_split)

    vit_range = [0, 0]
    llm_range = [pp_size - 1, pp_size - 1]
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        if vit_num > 0 and pp_rank >= vit_range[1]:
            vit_range[1] = pp_rank
        if llm_num > 0 and pp_rank <= llm_range[0]:
            llm_range[0] = pp_rank
    print(f"vit range: {vit_range[0]}~{vit_range[1]}")
    print(f"llm range: {llm_range[0]}~{llm_range[1]}")
    vit_start_idx = 0
    llm_start_idx = 0
    return_dicts = []
    copy_dict = deepcopy(_state_dict)
    for pp_rank, (vit_num, llm_num) in enumerate(pp_split):
        vit_end_idx = vit_start_idx + vit_num
        llm_end_idx = llm_start_idx + llm_num   
        new_dict = {}
        for key, value in _state_dict.items():
            if key.startswith('image_encoder.encoder.embeddings.'):
                if pp_rank == vit_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.encoder.encoder.layers.'):
                layer_idx = int(key.replace('image_encoder.encoder.encoder.layers.', '').split('.')[0])
                if vit_start_idx <= layer_idx < vit_end_idx and vit_range[0] <= pp_rank <= vit_range[1]:
                    new_idx = layer_idx - vit_start_idx
                    new_key = key.replace(f'image_encoder.encoder.encoder.layers.{layer_idx}', f'image_encoder.encoder.encoder.layers.{new_idx}')
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('image_encoder.projector.'):
                if pp_rank == vit_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.embedding.'):
                if pp_rank == llm_range[0]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.layers.'):
                layer_idx = int(key.replace('text_decoder.decoder.layers.', '').split('.')[0])
                if llm_start_idx <= layer_idx < llm_end_idx and llm_range[0] <= pp_rank <= llm_range[1]:
                    new_idx = layer_idx - llm_start_idx
                    new_key = key.replace(f'text_decoder.decoder.layers.{layer_idx}', f'text_decoder.decoder.layers.{new_idx}')
                    new_dict[new_key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.final_layernorm.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.output_layer.'):
                if pp_rank == llm_range[1]:
                    new_dict[key] = value
                    copy_dict.pop(key)
                
        vit_start_idx = vit_end_idx
        llm_start_idx = llm_end_idx
        return_dicts.append(new_dict)
    return return_dicts, copy_dict


if __name__ == "__main__":
    pp_size = 16
    llm_num_layers = 80
    vit_num_layers = 45
    model_size = 76

    if model_size == 76:
        hg_ckpt_dir = "/path/to/ckpt/InternVL2-Llama3-76B" # huggingface权重目录
    else:
        hg_ckpt_dir = "/path/to/ckpt/InternVL2-8B" # huggingface权重目录
    
    mm_save_dir = f'/path/to/InternVL2-{model_size}B_{vit_num_layers}{llm_num_layers}_pp{pp_size}'

    if pp_size == 1:
        vit_num = [vit_num_layers]
        llm_num = [llm_num_layers]
    elif pp_size == 2:
        vit_num = [1, 0]
        llm_num = [1, 1]
    elif pp_size == 4:
        vit_num = [1, 1, 0, 0]
        llm_num = [0, 1, 1, 1]
    elif pp_size == 16:
        vit_num = [11, 12, 12, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        llm_num = [0, 0, 0, 1, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6]


    pp_split = merge_pp_index(llm_num, vit_num, llm_num_layers, vit_num_layers, pp_size)
    
    hf_model = load_from_hf(_load_dir=hg_ckpt_dir)

    state_dict = clip_model(hf_model, llm_num_layers, vit_num_layers)
    for key, value in state_dict.items():
        print(key, value.shape)

    state_dict = convert_hg_to_mm(state_dict, llm_num_layers)

    state_dicts, remains = split_model_by_pp(state_dict, pp_split)
    if len(remains) > 0:
        raise RuntimeWarning("there are some weight ungrouped")
    
    for rank, state_dict in enumerate(state_dicts):
        for key, value in state_dict.items():
            print(key, value.shape)

    save_by_pp(state_dicts, mm_save_dir, _exists_ok=True)