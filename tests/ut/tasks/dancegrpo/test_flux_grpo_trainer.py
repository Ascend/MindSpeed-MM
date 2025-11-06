import unittest
from unittest.mock import patch, MagicMock, ANY
from argparse import Namespace

import torch
from PIL import Image
import mindspeed.megatron_adaptor

from tests.ut.utils import clear_module


class TestFluxGRPOTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_dataset_provider = MagicMock()

    def test_grpo_one_step(self):
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
            patch('mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.autocast'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with (patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args,
                  patch.object(FluxGRPOTrainer, 'grpo_step',
                               return_value=(torch.randn(1, 2025, 64), torch.randn(1, 2025, 64),
                                             torch.tensor([-0.4985]))) as mock_grpo_step,
                  ):
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                trainer.hyper_model = MagicMock()
                trainer.args = MagicMock()
                mock_transformers = MagicMock()
                trainer.hyper_model.diffuser = mock_transformers
                mock_timesteps = torch.tensor(
                    [[500, 642, 750, 928, 954, 576, 409, 868, 700, 794, 978, 900, 833, 1000, 300]])
                sample = {
                    "latents": torch.randn(1, 15, 2025, 64),
                    "next_latents": torch.randn(1, 15, 2025, 64),
                    "encoder_hidden_states": torch.randn(1, 512, 4096),
                    "pooled_prompt_embeds": torch.randn(1, 768),
                    "text_ids": torch.tensor([[0., 0., 0.]]),
                    "image_ids": torch.randn(1, 2025, 3),
                    "timesteps": mock_timesteps,
                }
                perm = torch.tensor(12)
                sigma_schedule = torch.tensor([1.0000, 0.9783, 0.9545, 0.9286, 0.9000, 0.8684, 0.8333, 0.7941, 0.7500,
                                               0.7000, 0.6429, 0.5769, 0.5000, 0.4091, 0.3000, 0.1667, 0.0000])
                index = 0
                log_prob = trainer.grpo_one_step(sample, perm, sigma_schedule, index)
                # The return value is correct.
                self.assertTrue(torch.allclose(log_prob, torch.tensor([-0.4985]), atol=1e-4))

                mock_transformers.assert_called_with(
                    hidden_states=ANY,
                    encoder_hidden_states=ANY,
                    timestep=ANY,
                    guidance=ANY,
                    txt_ids=ANY,
                    pooled_projections=ANY,
                    img_ids=ANY,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )

    def test_sample_reference(self):
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
            patch('mindspeed_mm.tasks.rl.soragrpo.utils.parallel_states.initialize_sequence_parallel_state'),
            patch('mindspeed_mm.configs.config.merge_mm_args'),
            patch('torch.autocast'),
            patch('diffusers.image_processor.VaeImageProcessor.postprocess',
                  return_value=[Image.new('RGB', (720, 720)), Image.new('RGB', (720, 720)),
                                Image.new('RGB', (720, 720)), Image.new('RGB', (720, 720))]),
            patch('torch.amp.autocast'),
            patch('torch.distributed.get_rank', return_value=0),
            patch('builtins.open'),
        ):
            from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer
            with (patch.object(FluxGRPOTrainer, 'get_args', return_value=Namespace(sp_size=1)) as mock_get_args,
                  patch.object(FluxGRPOTrainer, 'grpo_step',
                               return_value=(torch.randn(4, 2025, 64), torch.randn(4, 2025, 64),
                                             torch.tensor([-0.4990, -0.4997, -0.4996, -0.5004]))) as mock_grpo_step,
                  patch.object(FluxGRPOTrainer, 'sd3_time_shift', return_value=torch.tensor(
                      [1.0000, 0.9783, 0.9545, 0.9286, 0.9000, 0.8684, 0.8333, 0.7941, 0.7500,
                       0.7000, 0.6429, 0.5769, 0.5000, 0.4091, 0.3000, 0.1667, 0.0000])) as mock_sd3_time_shift,
                  ):
                trainer = FluxGRPOTrainer(self.mock_dataset_provider)
                trainer.args = MagicMock()
                trainer.args.use_group = True
                trainer.args.w = 720
                trainer.args.h = 720
                trainer.args.t = 1
                trainer.args.sampling_steps = 16
                trainer.args.sample_batch_size = 4
                trainer.args.init_same_noise = True
                trainer.args.shift = 3.0
                trainer.args.use_hpsv2 = True
                trainer.args.num_generations = 12

                trainer.hyper_model = MagicMock()
                trainer.hyper_model.diffuser = MagicMock()
                trainer.hyper_model.ae = MagicMock()
                mock_reward_model = MagicMock()
                trainer.hyper_model.reward = {
                    'model': mock_reward_model,
                    'processor': MagicMock(),
                    'preprocess_val': MagicMock(),
                }
                mock_reward_model.return_value = {
                    'image_features': torch.randn(1, 1024),
                    'text_features': torch.randn(1, 1024),
                    'logit_scale': torch.tensor(100.0299)
                }

                trainer.device = 'cpu'

                encoder_hidden_states = torch.randn(12, 512, 4096)
                pooled_prompt_embeds = torch.randn(12, 768)
                text_ids = torch.tensor([[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.]])
                caption = ['A giraffe and a zebra in a dirt field.', 'A giraffe and a zebra in a dirt field.',
                           'A giraffe and a zebra in a dirt field.', 'A giraffe and a zebra in a dirt field.',
                           'A giraffe and a zebra in a dirt field.', 'A giraffe and a zebra in a dirt field.',
                           'A giraffe and a zebra in a dirt field.', 'A giraffe and a zebra in a dirt field.',
                           'A giraffe and a zebra in a dirt field.', 'A giraffe and a zebra in a dirt field.',
                           'A giraffe and a zebra in a dirt field.', 'A giraffe and a zebra in a dirt field.']
                mock_dataloader = iter([(encoder_hidden_states, pooled_prompt_embeds, text_ids, caption)])
                samples_batched_list, train_timesteps, sigma_schedule, perms = trainer.sample_reference(mock_dataloader)
                self.assertEqual(len(samples_batched_list), 144)
                self.assertEqual(samples_batched_list[0]['timesteps'].shape, torch.Size([1, 15]))
                self.assertEqual(samples_batched_list[0]['latents'].shape, torch.Size([1, 15, 2025, 64]))
                self.assertEqual(samples_batched_list[0]['next_latents'].shape, torch.Size([1, 15, 2025, 64]))
                self.assertEqual(samples_batched_list[0]['log_probs'].shape, torch.Size([1, 15]))
                self.assertEqual(samples_batched_list[0]['rewards'].shape, torch.Size([1]))
                self.assertEqual(samples_batched_list[0]['image_ids'].shape, torch.Size([1, 2025, 3]))
                self.assertEqual(samples_batched_list[0]['text_ids'].shape, torch.Size([1, 3]))
                self.assertEqual(samples_batched_list[0]['encoder_hidden_states'].shape, torch.Size([1, 512, 4096]))
                self.assertEqual(samples_batched_list[0]['pooled_prompt_embeds'].shape, torch.Size([1, 768]))
                self.assertEqual(samples_batched_list[0]['advantages'].shape, torch.Size([1]))
                self.assertEqual(train_timesteps, 1)
                exp_sigma_schedule = torch.tensor(
                    [1.0000, 0.9783, 0.9545, 0.9286, 0.9000, 0.8684, 0.8333, 0.7941, 0.7500,
                     0.7000, 0.6429, 0.5769, 0.5000, 0.4091, 0.3000, 0.1667, 0.0000])
                self.assertTrue(torch.allclose(sigma_schedule, exp_sigma_schedule, atol=1e-4))
                self.assertEqual(perms.shape, torch.Size([144, 15]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
