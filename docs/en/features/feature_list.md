# Feature List

This manual describes the features released in the commercial version of MindSpeed MM.

## Feature Overview

<p style = "display:none">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-citn{background-color:rgba(52, 152, 219, 0.02);border-color:inherit;color:#333;font-weight:bold;text-align:left;
  vertical-align:top}
.tg .tg-whwg{background-color:#F9F9FB;color:#333;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-s7j9{background-color:#F9F9FB;border-color:inherit;color:#333;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-t1fb{background-color:rgba(52, 152, 219, 0.03);border-color:inherit;color:#333;font-weight:bold;text-align:left;
  vertical-align:top}
.tg .tg-3kh5{background-color:#F9F9FB;border-color:inherit;color:#333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-i1fi{background-color:#FFF;border-color:inherit;color:#08C;text-align:left;vertical-align:top}
.tg .tg-fr9f{background-color:#FFF;border-color:inherit;color:#333;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0ijx{background-color:#FFF;border-color:inherit;color:#08C;font-style:italic;text-align:left;vertical-align:top}
.tg .tg-b8up{background-color:rgba(52, 152, 219, 0.02);color:#333;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-jajh{background-color:#FFF;color:#08C;text-align:left;vertical-align:top}
</style>
</p>
<table class="tg" style="undefined;table-layout: fixed; width: 764px"><colgroup>
<col style="width: 167px">
<col style="width: 158px">
<col style="width: 187px">
</colgroup>
<thead>
  <tr>
    <th class="tg-s7j9">Feature Category</th>
    <th class="tg-s7j9">Feature Subcategory</th>
    <th class="tg-s7j9">Feature Name</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-t1fb" rowspan="9">Parallelism Features</td>
    <td class="tg-citn">FSDP2</td>
    <td class="tg-i1fi"><a href="parallel/fsdp2.md">FSDP2</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="2">PP</td>
    <td class="tg-i1fi"><a href="parallel/dynamic_dpcp.md">Dynamic DP/CP switch</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="parallel/virtual_pipeline_parallel.md">Virtual Pipeline Parallel</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="4">Sequence Parallelism</td>
    <td class="tg-i1fi"><a href="parallel/unaligned_ulysses_cp.md">Unaligned Ulysses CP</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="parallel/dit_ring_attention.md">DiT Ring Attention</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="parallel/dit_usp.md">DiT USP</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="parallel/unaligned_sequence_parallel.md">Unaligned Sequence Parallel</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Heterogeneous Parallelism</td>
    <td class="tg-i1fi"><a href="parallel/hetero_parallel.md">Hetero Parallel</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="">tensor-parallel</a></td>
    <td class="tg-i1fi"><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/en/features/tensor-parallel.md">tensor-parallel</a></td>

  </tr>
  <tr>
    <td class="tg-t1fb" rowspan="7">Memory Optimizations</td>
    <td class="tg-citn">Offload</td>
    <td class="tg-i1fi"><a href="memory/async_activation_offload.md">Async Activation Offload</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="3">Tensor Swap</td>
    <td class="tg-i1fi"><a href="memory/swap_core.md">Swap Core</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="optimization/op_replay.md">Op Replay</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="memory/act_stash.md">Act Stash</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="2">Load Balancing</td>
    <td class="tg-0ijx"><a href="memory/online_data_rearrange.md">Online Data Rearrange</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="memory/encoder_dp_balance.md">Encoder DP Balance</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Bucket Reordering</td>
    <td class="tg-i1fi"><a href="memory/bucket_reordering.md">Bucket Reordering</a></td>

  </tr>
  <tr>
    <td class="tg-t1fb" rowspan="7">Optimization Features</td>
    <td class="tg-citn" rowspan="2">Loss Optimization</td>
    <td class="tg-i1fi"><a href="optimization/chunkloss.md">Chunk Loss</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="optimization/vlm_model_loss_calculate_type.md">VLM Model Loss Calculate Type</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="2">Performance Optimizations</td>
    <td class="tg-jajh"><a href="optimization/fpdt.md">FPDT</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="optimization/grad_norm_overlap.md">Grad Norm Overlap</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="3">Optimizer</td>
    <td class="tg-jajh"><a href="optimization/dummy_optimizer.md">Dummy Optimizer</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="optimization/parameter_lr_wd_tuning.md">Parameter LR/WD Tuning</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="optimization/fsdp2_muon_optimizer.md">FSDP2 Muon Optimizer</a></td>

  </tr>
  <tr>
    <td class="tg-whwg" rowspan="5">Training Modes</td>
    <td class="tg-citn">Pre-training</td>
    <td class="tg-jajh"><a href="training_mode/pretrain.md">Pretrain</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="3">Efficient Fine-tuning</td>
    <td class="tg-jajh"><a href="training_mode/lora_finetune.md">LoRA Finetune</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="training_mode/lora_finetune_fsdp2.md">LoRA Finetune with FSDP2</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="training_mode/agentic_sft.md">Agentic SFT</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Layerwise Training</td>
    <td class="tg-jajh"><a href="training_mode/layerwise_disaggregated_training.md">Layerwise Disaggregated Training</a></td>

  </tr>
    <tr>
    <td class="tg-whwg">Training Paradigm</td>
    <td class="tg-citn">RL</td>
    <td class="tg-jajh"><a href="">RLHF</a></td>

  </tr>
  <tr>
    <td class="tg-whwg" rowspan="2">Data Processing</td>
    <td class="tg-citn">Dataset</td>
    <td class="tg-jajh"><a href="data/multimodal_dataset.md">Multimodal Dataset</a></td>

  </tr>
  <tr>
    <td class="tg-citn">SeqPack</td>
    <td class="tg-jajh"><a href="data/seqpack.md">SeqPack</a></td>

  </tr>
  <tr>
    <td class="tg-whwg" rowspan="2">Model Conversion</td>
    <td class="tg-citn">Model Conversion</td>
    <td class="tg-jajh"><a href="../tools/mm_convert.md">MM Convert</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Canonical Model</td>
    <td class="tg-jajh"><a href="other/canonical_model.md">Canonical Model</a></td>

  </tr>
  <tr>
    <td class="tg-whwg">Deterministic Computing</td>
    <td class="tg-citn">Deterministic</td>
    <td class="tg-jajh"><a href="other/deterministic_computing.md">Deterministic Computing</a></td>

  </tr>
  <tr>
    <td class="tg-whwg">Evaluation Tools</td>
    <td class="tg-citn">VBench</td>
    <td class="tg-jajh"><a href="../tools/vbench-evaluate.md">VBench Evaluate</a></td>

  </tr>
</tbody></table>
