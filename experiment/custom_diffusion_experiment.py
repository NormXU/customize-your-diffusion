# -*- coding:utf-8 -*-
# email:
# create: 2021/7/2

import os
import json
import torch
import time
import munch
import itertools
import torch.nn.functional as F
from collections import OrderedDict
from metrics.meter import AverageMeter
from base.common_util import get_absolute_file_path
from base.driver import logger
from networks import get_network
from base.torch_utils.dl_util import freeze_params, unfreeze_params
from networks.custom_diffusion.model_utils import create_custom_diffusion
from mydatasets import get_dataset
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline
from networks.mydiffusers.stable_diffusion import StableDiffusionPipeline
from experiment.base_experiment import BaseExperiment
from diffusers.configuration_utils import FrozenDict


class CustomDiffusionExperiment(BaseExperiment):

    def __init__(self, config):
        config = self._init_config(config)
        self.experiment_name = config["name"]
        self.args = munch.munchify(config)
        self.init_device(config)
        self.init_random_seed(config)
        self.init_model(config)
        self.noise_scheduler = self.init_noise_scheduler(config)
        self.init_dataset(config)
        self.init_trainer_args(config)
        self.init_predictor_args(config)
        self.set_device()
        self.prepare_accelerator()

    """
        Main Block
    """

    def predict(self, **kwargs):
        self.model.eval()
        schedule_config_pth = os.path.join(self.args.model.pretrained_model_name_or_path,
                                           "scheduler/scheduler_config.json")
        predictor_args = self.args.predictor
        height, width = predictor_args.height, predictor_args.width
        if predictor_args.torch_dtype == "fp16":
            weight_dtype = torch.float16
        elif predictor_args.torch_dtype == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
        with open(schedule_config_pth, "r", encoding="utf-8") as reader:
            text = reader.read()
        scheduler_config = FrozenDict(json.loads(text))
        pipe = StableDiffusionPipeline(
            vae=self.model.vae.to(weight_dtype),
            text_encoder=self.model.text_encoder.to(weight_dtype),
            tokenizer=self.model.tokenizer,
            unet=self.model.unet.to(weight_dtype),
            scheduler=DPMSolverMultistepScheduler.from_config(scheduler_config),
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None)
        if not os.path.exists(predictor_args.save_dir):
            os.mkdir(predictor_args.save_dir)
        generated_batch = predictor_args.generated_batch if hasattr(predictor_args, "generated_batch") else 1
        for idx, (prompt, n_prompt) in enumerate(zip(predictor_args.prompt * generated_batch,
                                                     predictor_args.negative_prompt * generated_batch)):
            image = pipe(prompt, height=height, width=width,
                         guidance_scale=predictor_args.scale, negative_prompt=n_prompt,
                         num_inference_steps=predictor_args.num_inference_steps).images[0]
            image.save("{}/{}.png".format(predictor_args.save_dir, idx))

    def train(self, **kwargs):
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        global_step = 0
        global_eval_step = 0
        ni = 0
        # Keep vae in eval model as we don't train these
        self.model.vae.eval()
        for epoch in range(self.args.trainer.epochs):
            self.model.unet.train()
            if self.train_text_encoder or self.modifier_token is not None:
                self.model.text_encoder.train()
            self.model.zero_grad(set_to_none=True)
            for i, batch in enumerate(self.train_data_loader):
                start = time.time()
                train_batch_size = len(batch)
                ni = i + len(self.train_data_loader) * epoch  # number integrated batches (since train start)
                with self.gradient_accumulate_scope(self.model.unet):
                    loss = self._step_forward(batch)
                    grad_norm = self._step_backward(loss)

                    if self.modifier_token is not None:
                        # Zero out the gradients for all token embeddings except the newly added
                        self._zero_out_embeddings_grad()

                    # update model states
                    if self.accelerator is None:
                        if ((i + 1) % self.args.trainer.grad_accumulate == 0) or (
                                (i + 1) == len(self.train_data_loader)):
                            self._step_optimizer()
                            self._step_scheduler(global_step)
                    else:
                        self._step_optimizer()
                        self._step_scheduler(global_step)
                        # Gather the losses across all processes for logging (if we use distributed training).
                        loss = self.accelerator.gather(loss.repeat(train_batch_size)).mean()
                loss_meter.update(loss.item(), self.args.datasets.train.batch_size)
                norm_meter.update(grad_norm)
                batch_time.update(time.time() - start)
                global_step += 1
                global_eval_step = self._print_step_log(epoch, global_step, global_eval_step,
                                                        loss_meter, norm_meter,
                                                        batch_time, ni)

            if self.args.trainer.scheduler_by_epoch:
                self.scheduler.step()
            global_eval_step = self._print_epoch_log(epoch, global_step, global_eval_step, loss_meter, ni)

            if self.args.trainer.max_train_steps is not None and global_step >= self.args.trainer.max_train_steps:
                break

        model_config_path = self._train_post_process()
        if self.args.device.is_master:
            self.writer.close()
        return {
            'best_model_path': self.args.trainer.best_model_path,
            'model_config_path': model_config_path,
        }

    def _step_forward(self, batch, is_train=True, **kwargs):
        batch = {k: v.to(self.args.device.device_id) for k, v in batch.items()}
        # Convert images to latent space
        latents = self.model.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.model.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            mask = torch.chunk(batch["mask"], 2, dim=0)[0]
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.args.loss.prior_loss_weight * prior_loss
        else:
            mask = batch["mask"]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
        return loss

    def _step_backward(self, loss, **kwargs):
        params_to_clip = (
            itertools.chain(self.model.unet.parameters(), self.model.text_encoder.parameters())
            if self.train_text_encoder
            else self.model.unet.parameters()
        )
        if self.args.model.mixed_precision_flag:
            self.mixed_scaler.scale(loss).backward()
            self.mixed_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, self.args.trainer.grad_clip)
        else:
            if self.accelerator is not None:
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(params_to_clip, self.args.trainer.grad_clip)
            else:
                loss = loss / self.args.trainer.grad_accumulate
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip,
                                                           self.args.trainer.grad_clip)
        return grad_norm

    def _zero_out_embeddings_grad(self):
        # Zero out the gradients for all token embeddings except the newly added
        # embeddings for the concept, as we only want to optimize the concept embeddings
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            grads_text_encoder = self.model.text_encoder.module.get_input_embeddings().weight.grad
        else:
            grads_text_encoder = self.model.text_encoder.get_input_embeddings().weight.grad
        modifier_token_id = self.model.tokenizer.added_tokens_encoder[self.modifier_token]
        # Get the index for tokens that we want to zero the grads for
        index_grads_to_zero = torch.arange(len(self.model.tokenizer)) != modifier_token_id
        grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

    """
        initialization
    """

    def init_noise_scheduler(self, config):
        model_args = config["model"]
        type = model_args["noise_scheduler"].get("type", "DDPMScheduler")
        noise_scheduler = eval(type).from_pretrained(model_args["pretrained_model_name_or_path"],
                                                     subfolder="scheduler")
        return noise_scheduler

    def prepare_accelerator(self):
        if self.accelerator is not None:
            if self.args.model.train_text_encoder or self.modifier_token is not None:
                self.model.unet, self.model.text_encoder, self.optimizer, self.train_data_loader, self.scheduler = \
                    self.accelerator.prepare(self.model.unet, self.model.text_encoder, self.optimizer,
                                             self.train_data_loader, self.scheduler)
            else:
                self.model.unet, self.optimizer, self.train_data_loader, self.scheduler = self.accelerator.prepare(
                    self.model.unet, self.optimizer, self.train_data_loader, self.scheduler)

    def init_model(self, config):
        model_args = config["model"]
        unfreeze_unet = model_args["unfreeze_unet"]
        # init SD model
        self.model = get_network(model_args)  # stable diffusion freezes vae and text encoder by default

        logger.info("unfreeze {} parameters in u-net".format(unfreeze_unet))
        self.model.unet = create_custom_diffusion(self.model.unet, unfreeze_unet)

        modifier_token_id = []
        initializer_token_id = []
        self.modifier_token = model_args["placeholder_token"]
        self.train_text_encoder = model_args["train_text_encoder"]
        initializer_token = model_args["initializer_token"]
        if self.modifier_token is not None:
            logger.info("ready to train a token {} embedding".format(self.modifier_token))
            # Adding a modifier token which is optimized
            # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
            modifier_token = self.modifier_token.split('+')
            initializer_token = initializer_token.split('+')
            if len(modifier_token) > len(initializer_token):
                raise ValueError("You must specify + separated initializer token for each modifier token.")
            for modifier_token, initializer_token in zip(modifier_token,
                                                         initializer_token[:len(modifier_token)]):
                # Add the placeholder token in tokenizer
                num_added_tokens = self.model.tokenizer.add_tokens(modifier_token)
                if num_added_tokens == 0:
                    raise ValueError(
                        f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                        " `modifier_token` that is not already in the tokenizer."
                    )

                # Convert the initializer_token, placeholder_token to ids
                token_ids = self.model.tokenizer.encode([initializer_token], add_special_tokens=False)
                logger.info("initialize {} embedding feature with {}".format(modifier_token, token_ids))
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError("The initializer token must be a single token.")

                initializer_token_id.append(token_ids[0])
                modifier_token_id.append(self.model.tokenizer.convert_tokens_to_ids(modifier_token))

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.model.text_encoder.resize_token_embeddings(len(self.model.tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = self.model.text_encoder.get_input_embeddings().weight.data
            for (x, y) in zip(modifier_token_id, initializer_token_id):
                token_embeds[x] = token_embeds[y]

            # Freeze all parameters except for the token embeddings in text encoder
            params_to_freeze = itertools.chain(
                self.model.text_encoder.text_model.encoder.parameters(),
                self.model.text_encoder.text_model.final_layer_norm.parameters(),
                self.model.text_encoder.text_model.embeddings.position_embedding.parameters(),
            )
            freeze_params(params_to_freeze)
        else:
            logger.info("skip to train text embedding")
            if self.train_text_encoder:
                logger.info("unfreeze all parameters in text encoder")
                unfreeze_params(self.model.text_encoder.parameters())

        # load pretrained weights
        if "model_path" in model_args and model_args['model_path'] is not None:
            model_path = get_absolute_file_path(model_args['model_path'])
            self.load_model(model_path)

    def set_device(self):
        if self.accelerator is not None:
            self.model.to(self.args.device.device_id)
        elif self.args.device.device_id.type != 'cpu':
            torch.cuda.set_device(self.args.device.device_id.index)
            self.model.to(self.args.device.device_id)

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.model.vae.to(dtype=self.weight_dtype)
        if not self.train_text_encoder or self.modifier_token is None:
            self.model.text_encoder.to(dtype=self.weight_dtype)

    def init_dataset(self, config):
        if 'datasets' in config and config.get('phase', 'train') != 'predict':
            dataset_args = config.get("datasets")
            train_data_loader_args = dataset_args.get("train")
            self.with_prior_preservation = train_data_loader_args['dataset'].get('with_prior_preservation', False)
            if config.get('phase', 'train') == 'train':
                train_data_loader_args['dataset'].update(
                    {"tokenizer": self.model.tokenizer,
                     "device": self.args.device.device_id,
                     "pretrain_model_or_path": self.args.model.pretrained_model_name_or_path
                     })
                train_data_loader_args['collate_fn'].update(
                    {"with_prior_preservation": self.with_prior_preservation,
                     "tokenizer": self.model.tokenizer})
                self.train_dataset = get_dataset(train_data_loader_args['dataset'])
                self.train_data_loader = self._get_data_loader_from_dataset(self.train_dataset,
                                                                            train_data_loader_args,
                                                                            phase='train')
                logger.info("success init train data loader len:{} ".format(len(self.train_data_loader)))

    """
        Tool Functions
    """

    def load_model(self, checkpoint_path, strict=True, **kwargs):
        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            for k, v in state_dict.items():
                getattr(self.model, k).load_state_dict(v)
                logger.info("success load {}:{}".format(k, checkpoint_path))

    def save_model(self, checkpoint_path):
        net = self.model.unet
        text_encoder = self.model.text_encoder
        save_state_dict = OrderedDict()
        if self.accelerator is not None:
            save_state_dict['unet'] = self.accelerator.unwrap_model(net).state_dict()
            save_state_dict['text_encoder'] = self.accelerator.unwrap_model(text_encoder).state_dict()
            self.accelerator.save(save_state_dict, checkpoint_path)
        elif self.args.device.is_master:
            save_state_dict['unet'] = net.state_dict()
            save_state_dict['text_encoder'] = text_encoder.state_dict()
            torch.save(save_state_dict, checkpoint_path)
        logger.info("model successfully saved to {}".format(checkpoint_path))

    def _print_step_log(self, epoch, global_step, global_eval_step, loss_meter, norm_meter, batch_time, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.device.is_master and self.args.trainer.print_freq > 0 and global_step % self.args.trainer.print_freq == 0:
            message = "experiment:{}; train, (epoch: {}, steps: {}, lr:{:e}, step_mean_loss:{}," \
                      " average_loss:{}), time, (train_step_time: {:.5f}s, train_average_time: {:.5f}s);" \
                      "(grad_norm_mean: {:.5f}, grad_norm_step: {:.5f})". \
                format(self.experiment_name, epoch, global_step, current_lr,
                       loss_meter.val, loss_meter.avg, batch_time.val, batch_time.avg, norm_meter.avg,
                       norm_meter.val)
            logger.info(message)
            if self.writer is not None:
                self.writer.add_scalar("{}_train/lr".format(self.experiment_name), current_lr, global_step)
                self.writer.add_scalar("{}_train/step_loss".format(self.experiment_name), loss_meter.val, global_step)
                self.writer.add_scalar("{}_train/average_loss".format(self.experiment_name), loss_meter.avg,
                                       global_step)
        if global_step > 0 and self.args.trainer.save_step_freq > 0 and self.args.device.is_master and \
                global_step % self.args.trainer.save_step_freq == 0:
            if not (self.args.trainer.save_best and not (
                    self.args.trainer.save_best and loss_meter.avg < self.args.trainer.best_eval_result)) or \
                    (self.args.trainer.save_best and self.args.trainer.best_eval_result == -1):
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_avg_loss{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                self.save_model(checkpoint_path)
                self.args.trainer.best_eval_result = loss_meter.avg
                self.args.trainer.best_model_path = checkpoint_path
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and self.args.device.is_master and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            if not self.args.trainer.save_best or \
                    (self.args.trainer.save_best and loss_meter.avg < self.args.trainer.best_eval_result) or \
                    (self.args.trainer.save_best and self.args.trainer.best_eval_result == -1):
                checkpoint_name = "{}_epoch{}_step{}_lr{:e}_avg_loss{:.5f}.pth".format(
                    self.experiment_name, epoch, global_step, current_lr, loss_meter.avg)
                checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
                self.save_model(checkpoint_path)
                self.args.trainer.best_eval_result = loss_meter.avg
                self.args.trainer.best_model_path = checkpoint_path
