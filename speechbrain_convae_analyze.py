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
#import visualization
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#import visualization
from speechbrain_convae_train import dataio_prepare, SexAnonymizationTraining
sys.path.append("speechbrain/recipes/LibriSpeech")
# 1.  # Dataset prep (parsing Librispeech)
from librispeech_prepare import prepare_librispeech  # noqa


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
    hparams_file, run_opts, overrides  = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    with open("./speechbrain_configs/evaluator_inference.yaml") as fin:
        hparams_eval = load_hyperpyyaml(fin)

    for mod in hparams_eval['modules']:
        hparams_eval['modules'][mod].to(run_opts['device'])
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

    state_dict = torch.load("results/fullyconn_updatedsexclassifier_recon1.0_l1_2_60_epoch_adam_lr_1.0/8886/save/CKPT+2022-03-23+20-59-04+00//model.ckpt")
    breakpoint()
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