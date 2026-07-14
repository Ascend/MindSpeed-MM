from __future__ import annotations

from dataclasses import replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.transformer_args import TransformerArgs


def block_forward(
    self,
    vx: torch.Tensor | None,
    ax: torch.Tensor | None,
    video_enabled: bool = True,
    audio_enabled: bool = True,
    video_timesteps: torch.Tensor | None = None,
    audio_timesteps: torch.Tensor | None = None,
    video_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    audio_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    video_context: torch.Tensor | None = None,
    video_context_mask: torch.Tensor | None = None,
    audio_context: torch.Tensor | None = None,
    audio_context_mask: torch.Tensor | None = None,
    video_prompt_timestep: torch.Tensor | None = None,
    audio_prompt_timestep: torch.Tensor | None = None,
    video_self_attention_mask: torch.Tensor | None = None,
    audio_self_attention_mask: torch.Tensor | None = None,
    video_cross_scale_shift_timestep: torch.Tensor | None = None,
    video_cross_gate_timestep: torch.Tensor | None = None,
    audio_cross_scale_shift_timestep: torch.Tensor | None = None,
    audio_cross_gate_timestep: torch.Tensor | None = None,
    video_cross_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    audio_cross_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    video_self_attn_perturbation_mask: torch.Tensor | None = None,
    video_self_attn_all_perturbed: bool = False,
    audio_self_attn_perturbation_mask: torch.Tensor | None = None,
    audio_self_attn_all_perturbed: bool = False,
    video_cross_attn_perturbation_mask: torch.Tensor | None = None,
    video_cross_attn_skip_all: bool = False,
    audio_cross_attn_perturbation_mask: torch.Tensor | None = None,
    audio_cross_attn_skip_all: bool = False,
    **_: object,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if vx is None and ax is None:
        raise ValueError("At least one of vx or ax must be provided")

    run_vx = vx is not None and video_enabled and vx.numel() > 0
    run_ax = ax is not None and audio_enabled and ax.numel() > 0

    run_a2v = run_vx and (ax is not None and ax.numel() > 0)
    run_v2a = run_ax and (vx is not None and vx.numel() > 0)

    if run_vx:
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], video_timesteps, slice(0, 3)
        )
        norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
        del vshift_msa, vscale_msa

        vx_msa_out = self.attn1(
            norm_vx,
            pe=video_positional_embeddings,
            mask=video_self_attention_mask,
            perturbation_mask=video_self_attn_perturbation_mask,
            all_perturbed=video_self_attn_all_perturbed,
        )
        vx = vx + vx_msa_out * vgate_msa
        del vgate_msa, norm_vx, vx_msa_out

        vx = vx + self._apply_text_cross_attention(
            vx,
            video_context,
            self.attn2,
            self.scale_shift_table,
            getattr(self, "prompt_scale_shift_table", None),
            video_timesteps,
            video_prompt_timestep,
            video_context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )

    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio_timesteps, slice(0, 3)
        )
        norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
        del ashift_msa, ascale_msa

        ax_msa_out = self.audio_attn1(
            norm_ax,
            pe=audio_positional_embeddings,
            mask=audio_self_attention_mask,
            perturbation_mask=audio_self_attn_perturbation_mask,
            all_perturbed=audio_self_attn_all_perturbed,
        )
        ax = ax + ax_msa_out * agate_msa
        del agate_msa, norm_ax, ax_msa_out

        ax = ax + self._apply_text_cross_attention(
            ax,
            audio_context,
            self.audio_attn2,
            self.audio_scale_shift_table,
            getattr(self, "audio_prompt_scale_shift_table", None),
            audio_timesteps,
            audio_prompt_timestep,
            audio_context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )

    if run_a2v or run_v2a:
        vx_pre_av = vx
        ax_pre_av = ax

        if run_a2v and not video_cross_attn_skip_all:
            scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video_cross_scale_shift_timestep,
                video_cross_gate_timestep,
                slice(0, 2),
            )
            a2v_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v)
            del scale_ca_video_a2v, shift_ca_video_a2v

            scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio_cross_scale_shift_timestep,
                audio_cross_gate_timestep,
                slice(0, 2),
            )
            a2v_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v)
            del scale_ca_audio_a2v, shift_ca_audio_a2v

            vx = vx + (
                self.audio_to_video_attn(
                    a2v_vx_scaled,
                    context=a2v_ax_scaled,
                    pe=video_cross_positional_embeddings,
                    k_pe=audio_cross_positional_embeddings,
                )
                * gate_out_a2v
                * video_cross_attn_perturbation_mask
            )
            del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

        if run_v2a and not audio_cross_attn_skip_all:
            scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio_cross_scale_shift_timestep,
                audio_cross_gate_timestep,
                slice(2, 4),
            )
            v2a_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a)
            del scale_ca_audio_v2a, shift_ca_audio_v2a

            scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video_cross_scale_shift_timestep,
                video_cross_gate_timestep,
                slice(2, 4),
            )
            v2a_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a)
            del scale_ca_video_v2a, shift_ca_video_v2a

            ax = ax + (
                self.video_to_audio_attn(
                    v2a_ax_scaled,
                    context=v2a_vx_scaled,
                    pe=audio_cross_positional_embeddings,
                    k_pe=video_cross_positional_embeddings,
                )
                * gate_out_v2a
                * audio_cross_attn_perturbation_mask
            )
            del gate_out_v2a, v2a_ax_scaled, v2a_vx_scaled

        del vx_pre_av, ax_pre_av

    if run_vx:
        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, vx.shape[0], video_timesteps, slice(3, 6)
        )
        vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
        vx = vx + self.ff(vx_scaled) * vgate_mlp
        del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

    if run_ax:
        ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio_timesteps, slice(3, 6)
        )
        ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
        ax = ax + self.audio_ff(ax_scaled) * agate_mlp
        del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

    return vx, ax


def _process_transformer_blocks(
    self,
    video: TransformerArgs | None,
    audio: TransformerArgs | None,
    perturbations: BatchedPerturbationConfig | None,
) -> tuple[TransformerArgs | None, TransformerArgs | None]:
    if perturbations is None:
        batch_size = (video or audio).x.shape[0]
        perturbations = BatchedPerturbationConfig.empty(batch_size)

    for block_idx, block in enumerate(self.transformer_blocks):
        if video is not None:
            video = self.block_input_processor(
                video,
                perturbations,
                block_idx,
                self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
            )
        if audio is not None:
            audio = self.block_input_processor(
                audio,
                perturbations,
                block_idx,
                self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
            )

        vx, ax = block(
            vx=video.x if video is not None else None,
            ax=audio.x if audio is not None else None,
            video_enabled=video.enabled if video is not None else False,
            audio_enabled=audio.enabled if audio is not None else False,
            video_timesteps=video.timesteps if video is not None else None,
            audio_timesteps=audio.timesteps if audio is not None else None,
            video_positional_embeddings=video.positional_embeddings if video is not None else None,
            audio_positional_embeddings=audio.positional_embeddings if audio is not None else None,
            video_context=video.context if video is not None else None,
            video_context_mask=video.context_mask if video is not None else None,
            audio_context=audio.context if audio is not None else None,
            audio_context_mask=audio.context_mask if audio is not None else None,
            video_prompt_timestep=video.prompt_timestep if video is not None else None,
            audio_prompt_timestep=audio.prompt_timestep if audio is not None else None,
            video_self_attention_mask=video.self_attention_mask if video is not None else None,
            audio_self_attention_mask=audio.self_attention_mask if audio is not None else None,
            video_cross_scale_shift_timestep=video.cross_scale_shift_timestep if video is not None else None,
            video_cross_gate_timestep=video.cross_gate_timestep if video is not None else None,
            audio_cross_scale_shift_timestep=audio.cross_scale_shift_timestep if audio is not None else None,
            audio_cross_gate_timestep=audio.cross_gate_timestep if audio is not None else None,
            video_cross_positional_embeddings=video.cross_positional_embeddings if video is not None else None,
            audio_cross_positional_embeddings=audio.cross_positional_embeddings if audio is not None else None,
            video_self_attn_perturbation_mask=video.self_attn_perturbation_mask if video is not None else None,
            video_self_attn_all_perturbed=video.self_attn_all_perturbed if video is not None else False,
            audio_self_attn_perturbation_mask=audio.self_attn_perturbation_mask if audio is not None else None,
            audio_self_attn_all_perturbed=audio.self_attn_all_perturbed if audio is not None else False,
            video_cross_attn_perturbation_mask=video.cross_attn_perturbation_mask if video is not None else None,
            video_cross_attn_skip_all=video.cross_attn_skip_all if video is not None else False,
            audio_cross_attn_perturbation_mask=audio.cross_attn_perturbation_mask if audio is not None else None,
            audio_cross_attn_skip_all=audio.cross_attn_skip_all if audio is not None else False,
        )

        if video is not None:
            video = replace(video, x=vx)
        if audio is not None:
            audio = replace(audio, x=ax)

    return video, audio


def model_forward(
    self,
    video: Modality | None,
    audio: Modality | None,
    perturbations: BatchedPerturbationConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not self.model_type.is_video_enabled() and video is not None:
        raise ValueError("Video is not enabled for this model")
    if not self.model_type.is_audio_enabled() and audio is not None:
        raise ValueError("Audio is not enabled for this model")

    video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
    audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
    video_out, audio_out = self._process_transformer_blocks(
        video=video_args,
        audio=audio_args,
        perturbations=perturbations,
    )

    vx = (
        self._process_output(
            self.scale_shift_table,
            self.norm_out,
            self.proj_out,
            video_out.x,
            video_out.embedded_timestep,
        )
        if video_out is not None
        else None
    )
    ax = (
        self._process_output(
            self.audio_scale_shift_table,
            self.audio_norm_out,
            self.audio_proj_out,
            audio_out.x,
            audio_out.embedded_timestep,
        )
        if audio_out is not None
        else None
    )
    return vx, ax
