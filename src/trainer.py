import copy
import time

import numpy
import ray
import torch

import models


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if not self.config.train_on_gpu:
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(replay_buffer.get_batch.remote())
            self_supervised = self.training_step < self.config.self_supervised_steps
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
                reconstruction_loss,
                consistency_loss,
                reward_prediction_error,
                value_prediction_error,
            ) = self.update_weights(batch, self_supervised)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "consistency_loss": consistency_loss,
                    "reward_prediction_error": reward_prediction_error,
                    "value_prediction_error": value_prediction_error,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch, self_supervised):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, num_unroll_steps+1, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        # Generate predictions
        value, reward, policy_logits, hidden_state, reconstruction = self.model.initial_inference(
            observation_batch[:, 0].squeeze(1)
        )
        predictions = [(value, reward, policy_logits, None, reconstruction)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state, reconstruction = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits, hidden_state, reconstruction))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        # Compute losses
        value_loss, reward_loss, policy_loss, reconstruction_loss, consistency_loss = (0, 0, 0, 0, 0)
        value, reward, policy_logits, _, reconstruction = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, _, current_policy_loss, current_reconstruction_loss, _ = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            reconstruction,
            None,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
            observation_batch[:, 0],
            None,
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        reconstruction_loss += current_reconstruction_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )

        reward_prediction_error = []
        value_prediction_error = []

        for i in range(1, len(predictions)):
            target_hidden_state = self.model.representation(
                observation_batch[:, i].squeeze(1)
            ).detach()
            value, reward, policy_logits, hidden_state, reconstruction = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
                current_reconstruction_loss,
                current_consistency_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                reconstruction,
                hidden_state,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
                observation_batch[:, i],
                target_hidden_state,
            )

            reward_prediction_error.append(current_reward_loss.mean().item())
            value_prediction_error.append(current_value_loss.mean().item())

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reconstruction_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_consistency_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss
            reconstruction_loss += current_reconstruction_loss
            consistency_loss += current_consistency_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = (value_loss * self.config.value_loss_weight
                + reward_loss
                + policy_loss
                + reconstruction_loss * self.config.reconstruction_loss_weight
                + consistency_loss * self.config.consistency_loss_weight)

        # For self-supervised pre-training, we should ignore all losses produced directly
        # or indirectly by the environment rewards
        if self_supervised:
            loss = (reconstruction_loss * self.config.reconstruction_loss_weight
                    + consistency_loss * self.config.consistency_loss_weight)

        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
            reconstruction_loss.mean().item(),
            consistency_loss.mean().item(),
            reward_prediction_error,
            value_prediction_error,
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        reconstruction,
        hidden_state,
        target_value,
        target_reward,
        target_policy,
        target_observation,
        target_hidden_state,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        reconstruction_loss = torch.square(reconstruction - target_observation).flatten(start_dim=1).mean(1)
        consistency_loss = None
        if hidden_state != None and target_hidden_state != None:
            consistency_loss = torch.square(hidden_state - target_hidden_state).flatten(start_dim=1).mean(1)
        return value_loss, reward_loss, policy_loss, reconstruction_loss, consistency_loss
