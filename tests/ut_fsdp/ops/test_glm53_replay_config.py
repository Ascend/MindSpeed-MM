"""Check runtime mHC replay routing without loading NPU dependencies."""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class ReplayConfigTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        path = root / "mindspeed_mm/fsdp/models/glm5_next/modeling_glm5_next.py"
        tree = ast.parse(path.read_text())
        model = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Glm5NextForConditionalGeneration")
        function = next(node for node in model.body
                        if isinstance(node, ast.FunctionDef) and node.name == "overwrite_transformer_config")
        function.decorator_list = []
        self.load_replay = Mock()
        namespace = dict(logger=logging.getLogger(__name__), load_mhc_replay_op=self.load_replay)
        namespace.update(IMPL_EAGER="eager", IMPL_TRITON="triton", IMPL_ASCENDC="ascendc",
                         IMPL_DENSE="dense", IMPL_FOR_KDA=("eager", "triton", "ascendc"),
                         IMPL_FOR_CAUSAL_CONV=("eager", "triton", "ascendc"),
                         IMPL_FOR_DSA=("dense", "sfa"), IMPL_FOR_INDEXER=("eager", "triton"))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        self.overwrite = namespace[function.name]
        self.config = SimpleNamespace(text_config=SimpleNamespace())
        self.scope = SimpleNamespace(cache_ops=["cann_ops_transformer.mhc_pre_sinkhorn.default"])

    def apply(self, recompute=True, reentrant=False, scopes=None):
        plan = SimpleNamespace(use_reentrant=reentrant, op_replay_scopes=scopes or [])
        features = SimpleNamespace(recompute=recompute, recompute_plan=plan)
        return self.overwrite(self.config, SimpleNamespace(), features)

    def test_matching_scope_loads_dispatcher_before_feature_wiring(self):
        result = self.apply(scopes=[self.scope])
        self.assertIs(result, self.config)
        self.assertTrue(result.text_config._mhc_pre_op_replay)
        self.load_replay.assert_called_once_with()

    def test_reentrant_and_disabled_checkpoint_do_not_enable_replay(self):
        for recompute, reentrant in [(False, False), (True, True)]:
            with self.subTest(recompute=recompute, reentrant=reentrant):
                result = self.apply(recompute, reentrant, [self.scope])
                self.assertFalse(result.text_config._mhc_pre_op_replay)
        self.load_replay.assert_not_called()

    def test_unrelated_scope_preserves_unfused_model_path(self):
        scope = SimpleNamespace(cache_ops=["aten.matmul.default"])
        self.assertFalse(self.apply(scopes=[scope]).text_config._mhc_pre_op_replay)
        self.load_replay.assert_not_called()

    def test_reusing_config_clears_previous_runtime_replay_state(self):
        self.apply(scopes=[self.scope])
        self.assertFalse(self.apply().text_config._mhc_pre_op_replay)
        self.assertEqual(self.load_replay.call_count, 1)


if __name__ == "__main__":
    unittest.main()
