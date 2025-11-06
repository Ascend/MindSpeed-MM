import unittest
from unittest.mock import patch, MagicMock
from argparse import Namespace

import torch
import mindspeed.megatron_adaptor

from tests.ut.utils import clear_module


class TestSoraGRPOTrainer(unittest.TestCase):
    """SoraGrpoTrainer is an abstract class and cannot be instantiated; use its subclass FluxGrpoTrainer for instantiation testing."""

    def setUp(self):
        self.mock_dataset_provider = MagicMock()

    def test_init(self):
        # Reload the mindspeed_mm module to activate new patches, otherwise test cases may interfere with each other due to residual states
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state') as initialize_sequence_parallel_state,
            patch('mindspeed_mm.configs.config.mm_extra_args_provider'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(train_valid_test_dataset_provider=self.mock_dataset_provider)
                self.assertEqual(trainer.local_rank, 0)
                self.assertEqual(trainer.rank, 0)
                self.assertEqual(trainer.world_size, 8)
                self.assertEqual(trainer.train_valid_test_dataset_provider, self.mock_dataset_provider)
                self.assertEqual(trainer.optimizer, None)
                self.assertEqual(trainer.device, 0)
                self.assertEqual(trainer.hyper_model, None)
                mock_get_args.assert_called_once()
                initialize_sequence_parallel_state.assert_called_once()

    def test_get_args(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('sys.argv', ['script.py', '--load', 'test.txt'])
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            trainer = FluxGRPOTrainer(self.mock_dataset_provider)
            args = trainer.get_args()
            self.assertEqual(args.dataloader_num_workers, 10)
            self.assertEqual(args.train_batch_size, 1)
            self.assertEqual(args.load, "test.txt")

    def test_get_args_required_args_not_set(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args')
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with self.assertRaises(SystemExit) as cm:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                trainer.get_args()
                self.assertEqual(cm.exception.code, 2)

    def test_sd3_time_shift(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args')
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                sigma_schedule = torch.linspace(1, 0, 17)
                res = trainer.sd3_time_shift(3, sigma_schedule)
                expected = torch.tensor(
                    [1.0000, 0.9783, 0.9545, 0.9286, 0.9000, 0.8684, 0.8333, 0.7941, 0.7500, 0.7000, 0.6429, 0.5769,
                     0.5000, 0.4091, 0.3000, 0.1667, 0.0000])
                self.assertTrue(torch.allclose(res, expected, atol=1e-4))

    def test_gather_tensor(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                input = torch.zeros(3, 4)
                output = trainer.gather_tensor(input)
                self.assertEqual(output.size(), (24, 4))

    def test_assert_eq(self):
        from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
        with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
            FluxGRPOTrainer.assert_eq(1, 1, "test")

    def test_assert_eq_not_equal(self):
        from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
        with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
            with self.assertRaises(AssertionError) as err:
                FluxGRPOTrainer.assert_eq(1, 2, "test")
                self.assertEqual(str(err.exception), "test not equal")

    def test_flux_step(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                config = {
                    "grpo": True,
                    "sde_solver": True,
                    "eta": 0.3,
                    "index": 0,
                }
                model_output = torch.ones(3, 4)
                latents = torch.ones(3, 4)
                sigmas = torch.tensor([0.03, 0.02, 0.01])
                prev_sample = torch.ones(3, 4) / 2
                torch.manual_seed(42)
                prev_sample, pred_original_sample, log_prob = trainer.grpo_step(model_output, latents, sigmas,
                                                                                prev_sample, config)
                exp_prev_sample = torch.tensor([[0.5000, 0.5000, 0.5000, 0.5000],
                                                [0.5000, 0.5000, 0.5000, 0.5000],
                                                [0.5000, 0.5000, 0.5000, 0.5000]])
                exp_pred_original_sample = torch.tensor([[0.9700, 0.9700, 0.9700, 0.9700],
                                                         [0.9700, 0.9700, 0.9700, 0.9700],
                                                         [0.9700, 0.9700, 0.9700, 0.9700]])
                exp_log_prob = torch.tensor([-117.7857, -117.7857, -117.7857])
                self.assertTrue(torch.allclose(prev_sample, exp_prev_sample, atol=1e-4))
                self.assertTrue(torch.allclose(pred_original_sample, exp_pred_original_sample, atol=1e-4))
                self.assertTrue(torch.allclose(log_prob, exp_log_prob, atol=1e-4))

    def test_flux_step_pre_sample_none(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                config = {
                    "grpo": True,
                    "sde_solver": True,
                    "eta": 0.3,
                    "index": 0,
                }
                model_output = torch.ones(3, 4)
                latents = torch.ones(3, 4)
                sigmas = torch.tensor([0.03, 0.02, 0.01])
                torch.manual_seed(42)
                prev_sample, pred_original_sample, log_prob = trainer.grpo_step(model_output, latents, sigmas,
                                                                                None, config)
                exp_prev_sample = torch.tensor([[0.9706, 0.9643, 0.9675, 0.9674],
                                                [0.9268, 0.9549, 1.0267, 0.9413],
                                                [0.9743, 0.9685, 0.9765, 0.9847]])
                exp_pred_original_sample = torch.tensor([[0.9700, 0.9700, 0.9700, 0.9700],
                                                         [0.9700, 0.9700, 0.9700, 0.9700],
                                                         [0.9700, 0.9700, 0.9700, 0.9700]])
                exp_log_prob = torch.tensor([-0.0297, -0.8223, -0.1532])
                self.assertTrue(torch.allclose(prev_sample, exp_prev_sample, atol=1e-4))
                self.assertTrue(torch.allclose(pred_original_sample, exp_pred_original_sample, atol=1e-4))
                self.assertTrue(torch.allclose(log_prob, exp_log_prob, atol=1e-4))

    def test_save_checkpoint_rank_0(self):
        clear_module("mindspeed_mm")
        clear_module("json.dumps")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
            patch('torch.distributed.fsdp.FullyShardedDataParallel'),
            patch('safetensors.torch.save_file') as mock_save_file,
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            try:
                with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                    trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                    transformer = MagicMock()
                    transformer.state_dict = MagicMock()
                    rank = 0
                    output_dir = "sora_grpo_trainer_test"
                    step = 40
                    epoch = 1
                    trainer.save_checkpoint(transformer, rank, output_dir, step, epoch)
                    transformer.state_dict.assert_called_once()
                    mock_save_file.assert_called_once()
            finally:
                import shutil
                shutil.rmtree("sora_grpo_trainer_test")

    def test_save_checkpoint_other_rank(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
            patch('torch.distributed.fsdp.FullyShardedDataParallel'),
            patch('safetensors.torch.save_file') as mock_save_file,
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                transformer = MagicMock()
                transformer.state_dict = MagicMock()
                rank = 1
                output_dir = "sora_grpo_trainer_test"
                step = 40
                epoch = 1
                trainer.save_checkpoint(transformer, rank, output_dir, step, epoch)
                transformer.state_dict.assert_called_once()
                mock_save_file.assert_not_called()

    def test_main_print(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args:
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                trainer.main_print("test test")

    def test_train(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.distributed.init_process_group'),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
            patch('torch.distributed.barrier'),
            patch('torch.distributed.fsdp.FullyShardedDataParallel'),
            patch('mindspeed_mm.tasks.rl.soragrpo.utils.fsdp_util.apply_fsdp_checkpointing'),
            patch('torch.optim.AdamW'),
            patch('diffusers.get_scheduler'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            test_args = Namespace(
                seed=1234,
                save=None,
                load_rank=8,
                fsdp_sharding_strategy="hybrid_full",
                use_cpu_offload=False,
                master_weight_type="fp32",
                gradient_checkpointing=True,
                selective_checkpointing=1.0,
                lr=1e-5,
                weight_decay=0.01,
                lr_scheduler="constant_with_warmup",
                lr_warmup_steps=10,
                lr_num_cycles=1,
                lr_power=1.0,
                sampler_seed=1234,
                train_batch_size=1,
                dataloader_num_workers=10,
                gradient_accumulation_steps=4,
                sp_size=1,
                train_sp_batch_size=1,
                train_iters=5,
                save_interval=1000,
            )
            mock_hyper_model = MagicMock()
            mock_hyper_model.diffuser = MagicMock()
            with (patch.object(FluxGRPOTrainer, 'get_args', return_value=test_args),
                  patch.object(FluxGRPOTrainer, 'train_one_step', return_value=(0.001, 1)) as mock_train_one_step,
                  patch.object(FluxGRPOTrainer, 'model_provider', return_value=mock_hyper_model)):
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                trainer.train()
                self.assertEqual(mock_train_one_step.call_count, 5)

    def test_train_and_train_one_step(self):
        clear_module("mindspeed_mm")
        with (
            patch.dict('os.environ', {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '8',
                'LOCAL_WORLD_SIZE': '8'
            }, clear=True),
            patch('torch.cuda.set_device'),
            patch('torch.cuda.current_device', return_value=0),
            patch(
                'mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.distributed.init_process_group'),
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=8),
            patch('torch.distributed.all_gather'),
            patch('torch.distributed.barrier'),
            patch('torch.distributed.fsdp.FullyShardedDataParallel'),
            patch('torch.distributed.all_reduce'),
            patch('torch.distributed.get_rank', return_value=0),
            patch('mindspeed_mm.tasks.rl.soragrpo.utils.fsdp_util.apply_fsdp_checkpointing'),
            patch('torch.optim.AdamW'),
            patch('diffusers.get_scheduler'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            test_args = Namespace(
                seed=1234,
                save=None,
                load_rank=8,
                fsdp_sharding_strategy="hybrid_full",
                use_cpu_offload=False,
                master_weight_type="fp32",
                gradient_checkpointing=True,
                selective_checkpointing=1.0,
                lr=1e-5,
                weight_decay=0.01,
                lr_scheduler="constant_with_warmup",
                lr_warmup_steps=10,
                lr_num_cycles=1,
                lr_power=1.0,
                sampler_seed=1234,
                train_batch_size=1,
                dataloader_num_workers=10,
                gradient_accumulation_steps=4,
                sp_size=1,
                train_sp_batch_size=1,
                train_iters=5,
                save_interval=1000,
                clip_range=1e-4,
                max_grad_norm=2.0,
                adv_clip_max=5.0
            )
            mock_hyper_model = MagicMock()
            mock_hyper_model.diffuser = MagicMock()

            samples_batched_list = [{
                "timesteps": torch.tensor([[500, 642, 750, 928, 954, 576, 409, 868, 700, 794, 978, 900,
                                            833, 1000, 300]]),
                'latents': torch.rand([1, 15, 2025, 64]),
                'next_latents': torch.rand([1, 15, 2025, 64]),
                'log_probs': torch.tensor([[-0.4984, -0.5021, -0.4993, -0.5006, -0.4955, -0.5030, -0.4986, -0.4998,
                                            -0.5018, -0.5026, -0.4975, -0.4988, -0.5011, -0.4990, -0.4987]]),
                'rewards': torch.rand([1]),
                'image_ids': torch.rand([1, 2025, 3]),
                'text_ids': torch.tensor([[0., 0., 0.]]),
                'encoder_hidden_states': torch.rand([1, 512, 4096]),
                'pooled_prompt_embeds': torch.rand([1, 768]),
                'advantages': torch.rand([1]),
            }] * 12
            train_timesteps = 9
            sigma_schedule = torch.tensor([1.0000, 0.9783, 0.9545, 0.9286, 0.9000, 0.8684, 0.8333, 0.7941, 0.7500,
                                           0.7000, 0.6429, 0.5769, 0.5000, 0.4091, 0.3000, 0.1667, 0.0000])
            perms = torch.tensor([[12, 10, 8, 3, 2, 11, 13, 5, 9, 7, 1, 4, 6, 0, 14],
                                  [6, 7, 10, 12, 11, 2, 13, 4, 9, 0, 5, 3, 8, 1, 14],
                                  [5, 7, 2, 13, 3, 11, 10, 8, 4, 9, 0, 6, 14, 12, 1],
                                  [1, 7, 12, 9, 3, 6, 0, 10, 2, 14, 13, 5, 8, 11, 4],
                                  [8, 13, 5, 0, 6, 14, 2, 10, 7, 12, 1, 3, 11, 9, 4],
                                  [12, 2, 5, 7, 11, 13, 9, 3, 14, 0, 10, 1, 6, 4, 8],
                                  [7, 1, 10, 12, 13, 11, 9, 3, 4, 14, 5, 2, 8, 6, 0],
                                  [6, 11, 14, 9, 10, 13, 5, 8, 7, 12, 2, 0, 3, 1, 4],
                                  [2, 3, 4, 12, 13, 11, 10, 1, 9, 6, 7, 5, 0, 14, 8],
                                  [3, 11, 4, 9, 1, 0, 12, 7, 2, 6, 10, 8, 14, 5, 13],
                                  [13, 10, 14, 1, 6, 9, 0, 3, 5, 2, 4, 7, 11, 8, 12],
                                  [7, 0, 2, 10, 9, 13, 5, 6, 4, 8, 14, 11, 3, 1, 12]])

            with (patch.object(FluxGRPOTrainer, 'get_args', return_value=test_args),
                  patch.object(FluxGRPOTrainer, 'model_provider', return_value=mock_hyper_model),
                  patch.object(FluxGRPOTrainer, 'sample_reference',
                               return_value=(samples_batched_list, train_timesteps, sigma_schedule,
                                             perms)) as mock_sample_reference,
                  patch.object(FluxGRPOTrainer, 'grpo_one_step',
                               return_value=torch.tensor([-0.4985], requires_grad=True)) as mock_grpo_one_step,
                  ):
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                trainer.train()
                # The number of training step executions is correct
                self.assertEqual(mock_sample_reference.call_count, 5)
                # Forward and backward call counts are correct: 12 samples, 9 timesteps, 5 training steps
                self.assertEqual(mock_grpo_one_step.call_count, 12 * 9 * 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
