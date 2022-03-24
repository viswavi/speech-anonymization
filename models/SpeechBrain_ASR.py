#!/usr/bin/env python3

import os
import sys
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, feats, wav_lens, tokens_bos, batch, stage, do_ctc=True):
        """Forward computations from the waveform batches to the output probabilities."""

        current_epoch = self.hparams.epoch_counter.current

        #current_epoch = self.hparams.epoch_counter.current
        #feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )
        if not do_ctc:
            return enc_out, pred
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
            ]

        return enc_out, pred, p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        (enc_out, pred, p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage == sb.Stage.TEST:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            return ids, predicted_words, target_words
            
        return loss

    def evaluate_batch(self, feats, wav_lens, tokens_bos, batch, stage, do_ctc=True):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(feats, wav_lens, tokens_bos, batch, stage=stage, do_ctc=do_ctc)
            if do_ctc:
                return predictions, self.compute_objectives(predictions, batch, stage=sb.Stage.TEST)
            return predictions
        

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()
        self.hparams.asr_model.eval()

    def get_predictions(self, feats, wav_lens, tokens_bos, batch, do_ctc=False):
        self.on_evaluate_start()
        return self.evaluate_batch(feats, wav_lens, tokens_bos, batch, sb.Stage.TEST, do_ctc)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]