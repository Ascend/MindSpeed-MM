"""MagiHuman FSDP2 data plugin package (issue #357).

Plugin root for the MagiHuman precomputed dataset. The YAML
``training.plugin`` list names this directory
(``mindspeed_mm/fsdp/data/datasets/magihuman``); ``import_plugin``
(``mindspeed_mm/fsdp/utils/register.py``) then ``walk_packages`` over it and
imports every submodule, which executes the ``@data_register.register(
"magihuman_precomputed")`` decorator in ``magihuman_precomputed_dataset.py``.

Mirrors the ``ltx2`` datasets package layout (which is importable the same way).
"""
