"""
Running instructions:
python speechbrain_convae_train.py \
    speechbrain_configs/convae.yaml \
    --device cpu \
    --model_type [convae / fcae] \
    --folder <path_to_output_dir>
"""

#!/usr/bin/env python3

from hyperpyyaml import load_hyperpyyaml
import librosa
import logging
import numpy as np
import os
from pathlib import Path
from queue import Full
from scipy.io.wavfile import write
import speechbrain as sb
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.train_logger import TensorboardLogger
from sympy import FU
import sys
from tqdm.contrib import tqdm
import torch
from torch.utils.data import DataLoader

from models.ConvAutoEncoder import ConvAutoencoder, FullyConnectedAutoencoder
from models.SpeechBrain_ASR import ASR
#from mutual_information.MILoss import *
#import visualization

logger = logging.getLogger(__name__)

#import visualization
from speechbrain_convae_train import dataio_prepare, SexAnonymizationTraining
sys.path.append("speechbrain/recipes/LibriSpeech")
# 1.  # Dataset prep (parsing Librispeech)
from librispeech_prepare import prepare_librispeech  # noqa

sys.path.append("espnet")
from utils.convert_fbank_to_wav import griffin_lim, logmelspc_to_linearspc


def save_wav (wav, path):
        wav *= 32767 / max (0.01, np.max(np.abs(wav)))
        write (path, 16000, wav.astype(np.int16))

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

    state_dict = torch.load("results/fc_vae_60_epochs_fixed_batched/8886/save/CKPT+2022-03-25+11-50-47+00/model.ckpt")
    for key in list(state_dict.keys()):
        if key.startswith("0."):
            key_fixed = key[2:]
            state_dict[key_fixed] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict)
    hparams["modules"]["ConvAE"] = model

    # Trainer initialization
    sa_brain = SexAnonymizationTraining(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Testing
    generated_audio_directory = os.path.join(hparams["output_folder"], "generated_speech")
    os.makedirs(generated_audio_directory, exist_ok=True)
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        sa_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )

        test_set = test_datasets[k]

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs = {"ckpt_prefix": None}
            test_set = sa_brain.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )

        progressbar = True
        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                all_features, (have_padded, pad), fbank_feats = sa_brain.compute_forward(batch, stage=sb.Stage.TEST)
                generated_fbank_features = all_features[0]
                fs = 16000
                n_mels = 80
                n_fft = 400
                n_shift = n_fft/4
                win_length = n_fft
                window = "hamming"
                iters = 100
                fmin=0
                fmax=8000
                for batch_item_idx in range(len(generated_fbank_features)):
                    utt_id = batch.id[batch_item_idx]
                    if utt_id != "2830-3980-0076":
                        continue
                    recon_mspc = generated_fbank_features[batch_item_idx].cpu().detach().numpy().T
                    print("test")
                    recon_recov = librosa.feature.inverse.mel_to_audio(M=recon_mspc,
                                                                 sr=fs,
                                                                 n_fft=n_fft,
                                                                 window=window,
                                                                 n_iter=iters,
                                                                 fmin=fmin,
                                                                 fmax=fmax,
                                                                 power=2.0,
                                                                 norm='slaney')

                    audio_path = os.path.join(generated_audio_directory, utt_id + "_reconstructed.wav")
                    save_wav(recon_recov, audio_path)
                    print(f"Wrote audio file to {audio_path}")


                    orig_mspc = fbank_feats[batch_item_idx].cpu().detach().numpy().T
                    print("test")
                    orig_recov = librosa.feature.inverse.mel_to_audio(M=orig_mspc,
                                                                 sr=fs,
                                                                 n_fft=n_fft,
                                                                 window=window,
                                                                 n_iter=iters,
                                                                 fmin=fmin,
                                                                 fmax=fmax,
                                                                 power=2.0,
                                                                 norm='slaney')

                    audio_path = os.path.join(generated_audio_directory, utt_id + "_orig.wav")
                    save_wav(orig_recov, audio_path)
                    print(f"Wrote audio file to {audio_path}")
