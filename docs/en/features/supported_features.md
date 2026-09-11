# Feature List

This table lists the features released in the current version of MindSpeed MM and shows the support status of each feature for the two training backends, **FSDP2** and **MCORE (Megatron)** ("✓" for supported, "×" for not supported).

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
<table class="tg" style="table-layout: fixed; width: 900px"><colgroup>
<col style="width: 130px">
<col style="width: 130px">
<col style="width: 230px">
<col style="width: 90px">
<col style="width: 90px">
</colgroup>
<thead>
  <tr>
    <th class="tg-s7j9">Feature Category</th>
    <th class="tg-s7j9">Feature Subcategory</th>
    <th class="tg-s7j9">Feature Name</th>
    <th class="tg-3kh5">FSDP2</th>
    <th class="tg-3kh5">MCORE</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-t1fb" rowspan="6">Parallelism Features</td>
    <td class="tg-citn">FSDP2</td>
    <td class="tg-i1fi"><a href="fsdp2.md">FSDP2</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">×</td>
  </tr>
  <tr>
    <td class="tg-citn" rowspan="3">Sequence Parallelism</td>
    <td class="tg-i1fi"><a href="unaligned_ulysses_cp.md">Unaligned Ulysses CP</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-i1fi"><a href="dit_ring_attention.md">DiT Ring Attention</a></td>
    <td class="tg-fr9f">×</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-i1fi"><a href="dit_usp.md">DiT USP</a></td>
    <td class="tg-fr9f">×</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-citn">Heterogeneous Parallelism</td>
    <td class="tg-i1fi"><a href="hetero_parallel.md">Hetero Parallel</a></td>
    <td class="tg-fr9f">×</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-citn">Tensor Parallelism</td>
    <td class="tg-i1fi"><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/tensor-parallel.md">tensor-parallel</a></td>
    <td class="tg-fr9f">×</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-t1fb" rowspan="2">Memory Optimization</td>
    <td class="tg-citn">Offload</td>
    <td class="tg-i1fi"><a href="async_activation_offload.md">Async Activation Offload</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-citn">Load Balancing</td>
    <td class="tg-i1fi"><a href="online_data_rearrange.md">Online Data Rearrange</a></td>
    <td class="tg-fr9f">×</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-t1fb" rowspan="2">Optimization Features</td>
    <td class="tg-citn" rowspan="2">Loss Optimization</td>
    <td class="tg-i1fi"><a href="chunkloss.md">ChunkLoss</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">×</td>
  </tr>
  <tr>
    <td class="tg-i1fi"><a href="vlm_model_loss_calculate_type.md">VLM Loss Calculation Types</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-whwg" rowspan="4">Training Mode</td>
    <td class="tg-citn" rowspan="2">Efficient Fine-tuning</td>
    <td class="tg-jajh"><a href="lora_finetune.md">LoRA Fine-Tuning (Mcore Backend)</a></td>
    <td class="tg-fr9f">×</td>
    <td class="tg-fr9f">✓</td>
  </tr>
  <tr>
    <td class="tg-jajh"><a href="lora_finetune_fsdp2.md">LoRA Fine-Tuning (FSDP Backend)</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">×</td>
  </tr>
  <tr>
    <td class="tg-citn">Data Processing</td>
    <td class="tg-jajh"><a href="seqpack.md">SeqPack</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">×</td>
  </tr>
  <tr>
    <td class="tg-citn">Deterministic Computing</td>
    <td class="tg-jajh"><a href="deterministic_computing.md">Deterministic Computing</a></td>
    <td class="tg-fr9f">✓</td>
    <td class="tg-fr9f">✓</td>
  </tr>
</tbody></table>
