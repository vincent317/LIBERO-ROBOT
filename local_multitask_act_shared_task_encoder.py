#!/usr/bin/env python3
"""Local ACT variant with one shared task embedding across VAE encoder, encoder, and decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from local_lerobot_act import bootstrap_lerobot_act


bootstrap_lerobot_act()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import (  # noqa: E402
    ACT,
    ACTPolicy,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE  # noqa: E402


@PreTrainedConfig.register_subclass("act_task_id_shared_encoder")
@dataclass
class ACTTaskIDSharedEncoderConfig(ACTConfig):
    num_task_ids: int = 1
    use_task_id_conditioning: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.num_task_ids < 1:
            raise ValueError(f"`num_task_ids` must be >= 1, got {self.num_task_ids}.")


class TaskConditionedACTSharedEncoder(ACT):
    def __init__(self, config: ACTTaskIDSharedEncoderConfig):
        super().__init__(config)
        self.config = config
        self.task_id_embed = nn.Embedding(config.num_task_ids, config.dim_model)

        n_1d_tokens = 1  # latent
        if self.config.use_task_id_conditioning:
            n_1d_tokens += 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        if self.config.use_vae:
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.use_task_id_conditioning:
                num_input_token_encoder += 1
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.vae_encoder_pos_enc = create_sinusoidal_pos_embedding(
                num_input_token_encoder, config.dim_model
            ).unsqueeze(0)

    def _get_task_ids(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.use_task_id_conditioning:
            return None
        task_ids = batch["task_index"]
        if task_ids.ndim > 1:
            task_ids = task_ids.squeeze(-1)
        return task_ids.long()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and ACTION in batch and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        task_ids = self._get_task_ids(batch)

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = torch.repeat_interleave(self.vae_encoder_cls_embed.weight.unsqueeze(0), batch_size, dim=0)
            vae_encoder_input = [cls_embed]
            if task_ids is not None:
                vae_encoder_input.append(self.task_id_embed(task_ids).unsqueeze(1))
            if self.config.robot_state_feature:
                vae_encoder_input.append(self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1))
            vae_encoder_input.append(self.vae_encoder_action_input_proj(batch[ACTION]))
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            vae_prefix_tokens = 1
            if task_ids is not None:
                vae_prefix_tokens += 1
            if self.config.robot_state_feature:
                vae_prefix_tokens += 1
            cls_joint_is_pad = torch.full(
                (batch_size, vae_prefix_tokens),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if task_ids is not None:
            encoder_in_tokens.append(self.task_id_embed(task_ids))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = cam_features.permute(2, 3, 0, 1).reshape(-1, cam_features.shape[0], cam_features.shape[1])
                cam_pos_embed = cam_pos_embed.permute(2, 3, 0, 1).reshape(-1, cam_pos_embed.shape[0], cam_pos_embed.shape[1])
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        if task_ids is not None:
            decoder_in = decoder_in + self.task_id_embed(task_ids).to(decoder_in.dtype).unsqueeze(0)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class ACTTaskIDSharedEncoderPolicy(ACTPolicy):
    config_class = ACTTaskIDSharedEncoderConfig
    name = "act_task_id_shared_encoder"

    def __init__(self, config: ACTTaskIDSharedEncoderConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = TaskConditionedACTSharedEncoder(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()
