# 特性列表

本手册描述MindSpeed MM商用版本中已发布的特性。

## 特性总览

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
    <th class="tg-s7j9">特性大类</th>
    <th class="tg-s7j9">特性子类</th>
    <th class="tg-s7j9">特性名称</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-t1fb" rowspan="10">并行特性</td>
    <td class="tg-citn">FSDP2</td>
    <td class="tg-i1fi"><a href="fsdp2.md">FSDP2</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="2">PP并行</td>
    <td class="tg-i1fi"><a href="dynamic_dpcp.md">动态PP / Dynamic DPCP</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="virtual_pipeline_parallel.md">Virtual Pipeline Parallel</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="4">序列并行</td>
    <td class="tg-i1fi"><a href="unaligned_ulysses_cp.md">Unaligned Ulysses CP</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="dit_ring_attention.md">DiT Ring Attention</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="dit_usp.md">DiT USP</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="unaligned_sequence_parallel.md">Unaligned Sequence Parallel</a></td>

  </tr>
  <tr>
    <td class="tg-citn">异构并行</td>
    <td class="tg-i1fi"><a href="hetero_parallel.md">Hetero Parallel</a></td>

  </tr>
  <tr>
    <td class="tg-citn">自动并行</td>
    <td class="tg-i1fi"><a href="automatic_parallelism_mm.md">Automatic Parallelism</a></td>

  </tr>
  <tr>
    <td class="tg-citn">张量并行</td>
    <td class="tg-i1fi"><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/tensor-parallel.md">tensor-parallel</a></td>

  </tr>
  <tr>
    <td class="tg-t1fb" rowspan="4">显存优化</td>
    <td class="tg-citn">Offload</td>
    <td class="tg-i1fi"><a href="async_activation_offload.md">Async Activation Offload</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="2">负载均衡</td>
    <td class="tg-0ijx"><a href="online_data_rearrange.md">Online Data Rearrange</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="encoder_dp_balance.md">Encoder DP Balance</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Bucket Reordering</td>
    <td class="tg-i1fi"><a href="bucket_reordering.md">Bucket Reordering</a></td>

  </tr>
  <tr>
    <td class="tg-t1fb" rowspan="5">优化特性</td>
    <td class="tg-citn" rowspan="2">损失优化</td>
    <td class="tg-i1fi"><a href="chunkloss.md">Chunk Loss</a></td>

  </tr>
  <tr>
    <td class="tg-i1fi"><a href="vlm_model_loss_calculate_type.md">VLM Model Loss Calculate Type</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="3">性能优化</td>
    <td class="tg-jajh"><a href="fpdt.md">FPDT</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="dummy_optimizer.md">Dummy Optimizer</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="parameter_lr_wd_tuning.md">Parameter LR/WD Tuning</a></td>

  </tr>
  <tr>
    <td class="tg-whwg" rowspan="5">训练模式</td>
    <td class="tg-citn">预训练</td>
    <td class="tg-jajh"><a href="pretrain.md">Pretrain</a></td>

  </tr>
  <tr>
    <td class="tg-citn" rowspan="3">高效微调</td>
    <td class="tg-jajh"><a href="lora_finetune.md">LoRA Finetune</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="lora_finetune_fsdp2.md">LoRA Finetune with FSDP2</a></td>

  </tr>
  <tr>
    <td class="tg-jajh"><a href="agentic_sft.md">Agentic SFT</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Layerwise Training</td>
    <td class="tg-jajh"><a href="layerwise_disaggregated_training.md">Layerwise Disaggregated Training</a></td>

  </tr>
    <tr>
    <td class="tg-whwg">训练范式</td>
    <td class="tg-citn">RL</td>
    <td class="tg-jajh"><a href="">RLHF</a></td>

  </tr>
  <tr>
    <td class="tg-whwg" rowspan="2">数据处理</td>
    <td class="tg-citn">数据集</td>
    <td class="tg-jajh"><a href="multimodal_dataset.md">Multimodal Dataset</a></td>

  </tr>
  <tr>
    <td class="tg-citn">SeqPack</td>
    <td class="tg-jajh"><a href="seqpack.md">SeqPack</a></td>

  </tr>
  <tr>
    <td class="tg-whwg" rowspan="2">模型转换</td>
    <td class="tg-citn">模型转换</td>
    <td class="tg-jajh"><a href="mm_convert.md">MM Convert</a></td>

  </tr>
  <tr>
    <td class="tg-citn">Canonical Model</td>
    <td class="tg-jajh"><a href="canonical_model.md">Canonical Model</a></td>

  </tr>
  <tr>
    <td class="tg-whwg">确定性计算</td>
    <td class="tg-citn">Deterministic</td>
    <td class="tg-jajh"><a href="deterministic_computing.md">Deterministic Computing</a></td>

  </tr>
  <tr>
    <td class="tg-whwg">评估工具</td>
    <td class="tg-citn">VBench</td>
    <td class="tg-jajh"><a href="vbench-evaluate.md">VBench Evaluate</a></td>

  </tr>
</tbody></table>
