"""
Running instructions:
python speechbrain_convae_train.py \
    speechbrain_configs/convae.yaml \
    --device cpu \
    --model_type [convae / fcae] \
    --folder <path_to_output_dir>
"""

#!/usr/bin/env python3

import os
from queue import Full
import sys
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.train_logger import TensorboardLogger
from models.ConvAutoEncoder import ConvAutoencoder, FullyConnectedAutoencoder, SmallConvAutoencoder
from models.SpeechBrain_ASR import ASR
from gender_classifier_train import GenderBrain
#from mutual_information.MILoss import *
#import visualization

logger = logging.getLogger(__name__)

#import visualization
sys.path.append("speechbrain/recipes/LibriSpeech")
# 1.  # Dataset prep (parsing Librispeech)
from librispeech_prepare import prepare_librispeech  # noqa


# Define training procedure
class SexAnonymizationTraining(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward pass through the model
        return self.modules.ConvAE(feats)

    def compute_objectives(self, predictions, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        reconstructed_speech, sex_logits = predictions

        batch = batch.to(sa_brain.device)

        sex_label = batch.gender
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        utility_loss = 0.0
        if self.hparams.utility_loss_weight > 0:
            orig_enc_out, orig_prob = self.asr_brain.get_predictions(feats, wav_lens, tokens_bos, batch, do_ctc=False)
            recon_enc_out, recon_prob = self.asr_brain.get_predictions(reconstructed_speech.reshape(self.hparams.batch_size, reconstructed_speech.shape[2], reconstructed_speech.shape[1]), wav_lens, tokens_bos, batch, do_ctc=False)
            utility_loss = self.hparams.loss_utility(recon_enc_out, orig_enc_out)

        recon_loss = self.hparams.loss_reconstruction(reconstructed_speech, feats)
        sex_loss = self.hparams.loss_sex_classification(sex_logits, torch.tensor(sex_label))
        #mi_loss = self.hparams.loss_mutual_information(reconstructed_speech, sex_logits, batch)

        loss = (
            self.hparams.recon_loss_weight * recon_loss
            + self.hparams.sex_loss_weight * sex_loss
            + self.hparams.utility_loss_weight * utility_loss
            #+ self.hparams.mi_loss_weight * mi_loss
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            # compute the accuracy of the sex prediction
            self.sex_classification_acc.append(sex_logits.unsqueeze(1), sex_label.unsqueeze(1), torch.tensor(sex_label.shape[0], device=sex_logits.device).unsqueeze(0))

            # Evaluation: performing classification by externally trained sex classifier
            embeddings_extern = self.modules.embedding_model(feats, wav_lens)
            sex_logits_extern = self.modules.external_classifier(embeddings_extern)
            self.sex_classification_acc_extern.append(sex_logits_extern.unsqueeze(1), sex_label.unsqueeze(1),
                                               torch.tensor(sex_label.shape[0], device=sex_logits.device).unsqueeze(0))

            if self.hparams.model_type == "convae":
                reconstructed_speech = reconstructed_speech.reshape(reconstructed_speech.shape[0], reconstructed_speech.shape[2], reconstructed_speech.shape[1])


            if stage == sb.Stage.VALID:
                recon_enc_out, recon_prob = self.asr_brain.get_predictions(reconstructed_speech, wav_lens, tokens_bos, batch, do_ctc=False)
                orig_enc_out, orig_prob = self.asr_brain.get_predictions(feats, wav_lens, tokens_bos, batch, do_ctc=False)
        
                cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
                self.utility_similarity_aggregator.append(cos_sim(recon_enc_out.view(recon_enc_out.shape[0], -1), orig_enc_out.view(orig_enc_out.shape[0], -1)))

            else:
                enc_out, predictions = self.asr_brain.get_predictions(reconstructed_speech, wav_lens, tokens_bos, batch, do_ctc=True)
                recon_enc_out, recon_prob, _, _, _, _, = enc_out
                ids, predicted_words, target_words = predictions
                
                enc_out, predictions = self.asr_brain.get_predictions(feats, wav_lens, tokens_bos, batch, do_ctc=True)
                orig_enc_out, orig_prob, _, _, _, _, = enc_out
                o_ids, o_predicted_words, o_target_words = predictions
                
                print(predicted_words)
                print(target_words)
                print(o_predicted_words)
                self.wer_metric.append(ids, predicted_words, target_words)

                cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
                self.utility_similarity_aggregator.append(cos_sim(recon_enc_out.view(recon_enc_out.shape[0], -1), orig_enc_out.view(orig_enc_out.shape[0], -1)))

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        # self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            if not hasattr(self, "recon_loss"):
                self.recon_loss = [[]]
            else:
                self.recon_loss.append([])
            self.sex_classification_acc = self.hparams.sex_classification_acc()
            self.sex_classification_acc_extern = self.hparams.sex_classification_acc_extern()
            self.utility_similarity_aggregator = self.hparams.utility_similarity_aggregator()
            if stage == sb.Stage.TEST:
                self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.sex_classification_acc.summarize()
            stage_stats["ACC_external"] = self.sex_classification_acc_extern.summarize()
            stage_stats["Utility_Retention"] = self.utility_similarity_aggregator.summarize()

            if stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            current_epoch = self.hparams.epoch_counter.current
            
        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "Utility_Retention": stage_stats["Utility_Retention"], "epoch": epoch},
                max_keys=["Utility_Retention"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            # self.checkpointer.save_and_keep_only(
            #     meta={"Utility_Retention": 1.1, "epoch": epoch},
            #     max_keys=["Utility_Retention"],
            #     num_to_keep=1,
            # )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return
        current_epoch = self.hparams.epoch_counter.current
        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            print("resetting")
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=False)
        self.hparams.model.eval()


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
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    # converting to binary class labels for easy consumption
    sex_string_to_int = {'M':0, 'F':1}
    for item in train_data.data.items():
        item[1]['gender'] = sex_string_to_int[item[1]['gender']]
    for item in valid_data.data.items():
        item[1]['gender'] = sex_string_to_int[item[1]['gender']]
    for testset in test_datasets:
        for item in test_datasets[testset].data.items():
            item[1]['gender'] = sex_string_to_int[item[1]['gender']]

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "gender"],
    )
    return train_data, valid_data, test_datasets, tokenizer


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    #tensorboard_logger = TensorboardLogger()

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
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

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)

    if hparams["model_type"] == "convae":
        model = ConvAutoencoder(hparams["convae_feature_dim"])
    else:
        model = FullyConnectedAutoencoder(hparams["convae_feature_dim"], hparams["batch_size"])

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    #run_on_main(hparams["pretrainer"].collect_files)
    #hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    sa_brain = SexAnonymizationTraining(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    sa_brain.acc_metric = []

    model = model.to(sa_brain.device)

    sa_brain.modules['ConvAE'] = model

    hparams["model"].append(sa_brain.modules['ConvAE'])
    
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected(device=run_opts["device"])
    sa_brain.asr_brain = ASR(
        modules=hparams["asr_modules"],
        hparams=hparams,
        run_opts=run_opts,
        )
    sa_brain.asr_brain.tokenizer = hparams["tokenizer"]
    sa_brain.asr_brain.tokenizer.Load("PretrainedASR/tokenizer.ckpt")
    hparams["asr_model"].load_state_dict(torch.load("PretrainedASR/asr.ckpt"))
    #hparams["normalize"].load_state_dict(torch.load("pretrained_models/asr-transformer-transformerlm-librispeech/normalizer.ckpt"))
    hparams["lm_model"].load_state_dict(torch.load("PretrainedASR/lm.ckpt"))

    print("done loading")

    # #Training
    sa_brain.fit(
        sa_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )


    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        sa_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        sa_brain.evaluate(
            test_datasets[k],
            max_key="Utility_Retention",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )

    # recon_loss_averages = []
    # for epoch_losses in sa_brain.recon_loss:
    #     epoch_loss_values = [v.item() for v in epoch_losses]
    #     recon_loss_averages.append(sum(epoch_loss_values) / len(epoch_loss_values))
    # output_folder = hparams["output_folder"]
    # plot_path = os.path.join(output_folder, "learning_curve.png")
    # visualization.draw_lines(recon_loss_averages, "Epoch", "Avg. Recon. Loss", "Learning Curve", plot_path)
    # print(f"Wrote reconstruction error learning curve to {plot_path}")

