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
from models.ConvAutoEncoder import ConvAutoencoder
from models.FullyConnected import FullyConnectedAutoencoder
from models.SpeechBrain_ASR import ASR
from gender_classifier_train import GenderBrain
#import visualization
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import pyworld as pw
from tqdm import tqdm
#import visualization

from speechbrain_convae_train import SexAnonymizationTraining
logger = logging.getLogger(__name__)

#import visualization
sys.path.append("speechbrain/recipes/LibriSpeech")
# 1.  # Dataset prep (parsing Librispeech)
from librispeech_prepare import prepare_librispeech  # noqa


def dataio_prepare_pitch_adjust(hparams):
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
        pitch_adjusted_directory = "/home/ec2-user/LibriSpeech_pitch_adjusted"
        basename = os.path.basename(wav)
        pitch_adjusted_file = os.path.join(pitch_adjusted_directory, basename)

        # Pitch-normalize audio file and write to disk
        wav_data, sr = sf.read(wav)
        f0, sp, ap  = pw.wav2world(wav_data, sr)
        voiced_idx = np.where(f0 != 0)
        voiced = f0[voiced_idx]
        voiced = np.maximum(0, (voiced - np.mean(voiced)) + 500)
        f0[voiced_idx] = voiced
        rec = pw.synthesize(f0, sp, ap, sr)
        sf.write(pitch_adjusted_file, rec, sr)

        sig = sb.dataio.dataio.read_audio(pitch_adjusted_file)
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
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare_pitch_adjust(hparams)

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

    test_dataloader = sa_brain.make_dataloader(test_datasets['test-clean'],
                                               sb.Stage.TEST,
                                               ckpt_prefix="dataloader-",
                                               # **hparams["test_dataloader_opts"]
                                               **{'batch_size': 1})

    sa_brain.on_stage_start(sb.Stage.TEST, 0)

    sa_brain.sex_classification_acc_extern = sa_brain.hparams.sex_classification_acc()
    for test_batch in tqdm(test_dataloader):
        wavs, wav_lens = test_batch.sig
        sex_label = test_batch.gender
        feats = sa_brain.hparams.compute_features(wavs)
        current_epoch = sa_brain.hparams.epoch_counter.current
        feats = sa_brain.modules.normalize(feats, wav_lens, epoch=current_epoch)
        sex_label = sex_label.to(sa_brain.device)
        feats = feats.to(sa_brain.device)
        wav_lens = wav_lens.to(sa_brain.device)
        with torch.no_grad():
            sex_logits_extern, score, index = sa_brain.external_classifier_model.classify_batch_feats(feats, wav_lens)
        sa_brain.sex_classification_acc_extern.append(sex_logits_extern.unsqueeze(0), sex_label.unsqueeze(0),
                                            torch.tensor(sex_label.shape[0], device=sex_logits_extern.device).unsqueeze(0))

    print("external classification ACC on reconstructed feats = ")
    breakpoint()
    print(sa_brain.sex_classification_acc_extern.summarize())
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
    pretrained_ASR_path = hparams["pretrained_lm_tokenizer_path"]
    sa_brain.asr_brain.tokenizer.Load(os.path.join(pretrained_ASR_path, "tokenizer.ckpt"))
    hparams["asr_model"].load_state_dict(torch.load(os.path.join(pretrained_ASR_path, "asr.ckpt")))
    #hparams["normalize"].load_state_dict(torch.load("pretrained_models/asr-transformer-transformerlm-librispeech/normalizer.ckpt"))
    hparams["lm_model"].load_state_dict(torch.load(os.path.join(pretrained_ASR_path, "lm.ckpt")))

    hparams_eval["classifier"].load_state_dict(torch.load("results/gender_classifier/1230/save/CKPT+2022-04-19+09-12-03+00/classifier.ckpt"))
    hparams_eval["embedding_model"].load_state_dict(
        torch.load("results/gender_classifier/1230/save/CKPT+2022-04-19+09-12-03+00/embedding_model.ckpt"))
    # hparams_eval["normalizer"].load_state_dict(
    #     torch.load("results/gender_classifier/1230/save/trained_external_classifier_ckpt/normalizer.ckpt"))

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

