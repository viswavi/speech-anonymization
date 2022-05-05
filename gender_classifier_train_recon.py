#!/usr/bin/env python3

"""
Daniel's comment:

retraining the external sex classifier with reconstructed features (mel filter banks)
To run:

python gender_classifier_train_recon.py \
    speechbrain_configs/gender_classifier_recon.yaml \
    --device cuda
"""
import sys
import torch
import os
import speechbrain as sb
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

sys.path.append("speechbrain/recipes/LibriSpeech")
from librispeech_prepare import prepare_librispeech  # noqa
from models.FullyConnected import FullyConnectedAutoencoder # hard-coded, best model so far


# 1.  # Dataset prep (parsing Librispeech)


# Brain class for speech enhancement training
class GenderBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings, and predictions
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats, lens)
        predictions = self.modules.classifier(embeddings)

        return predictions

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs
        with torch.autograd.profiler.profile(use_cuda=False) as prof:
        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
            if stage == sb.Stage.TRAIN:
                if hasattr(self.modules, "env_corrupt"):
                    wavs_noise = self.modules.env_corrupt(wavs, lens)
                    wavs = torch.cat([wavs, wavs_noise], dim=0)
                    lens = torch.cat([lens, lens])

                if hasattr(self.hparams, "augmentation"):
                    wavs = self.hparams.augmentation(wavs, lens)

            # Feature extraction and normalization
            feats = self.modules.compute_features(wavs)
            feats = self.modules.mean_var_norm(feats, lens)



        # run inference with our trained CycleGAN to reconstruct features, which are then used
        # to (re) train the gender classifier

            with torch.no_grad():
                recon_feats, sex_logits = self.modules.model(feats)
        print(prof)



        # return feats, lens
        return recon_feats, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        _, lens = batch.sig
        gender, _ = batch.gender_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            gender = torch.cat([gender, gender], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function
        loss = sb.nnet.losses.nll_loss(predictions, gender, lens)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, gender, lens, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, gender, lens)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing([self.optimizer], epoch, stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    datasets = [train_data, valid_data, test_data]

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("gender")
    @sb.utils.data_pipeline.provides("gender", "gender_encoded")
    def label_pipeline(gender):
        yield gender
        gender_encoded = label_encoder.encode_label_torch(gender)
        yield gender_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "gender_encoded"],
    )

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="gender",
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    sb.utils.distributed.run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["data_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": hparams["train_csv"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects "train", "valid", and "test".
    train_data, valid_data, test_data = dataio_prepare(hparams)

    hparams["embedding_model"].eval()
    hparams["embedding_model"].to(run_opts["device"])


    # Initialize the Brain object to prepare for training.
    gender_brain = GenderBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        # opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # initialize CycleGAN model from checkpoint
    recon_model = FullyConnectedAutoencoder(hparams["convae_feature_dim"], hparams["batch_size"])
    recon_model = recon_model.to(gender_brain.device)
    recon_model.eval()
    gender_brain.modules["model"] = recon_model

    hparams["model"].append(gender_brain.modules["model"])

    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    gender_brain.fit(
        epoch_counter=gender_brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = gender_brain.evaluate(
        test_set=test_data,
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
