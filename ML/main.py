import sys
sys.path.append(".")
sys.path.append("..")
import os
from typing import Literal

from ML.Core.trainer import Trainer
import torch
from ML import ml_utils
from Utilities import utils

if __name__ == '__main__':
    data_name: Literal['ours', 'prssm'] = 'prssm'
    train_percent = 0.75
    val_percent = 0.1
    intersect = 1
    n_epochs = 30
    seed = 3407
    criterion = 'L1Loss'
    regularization_factor = 10
    optimizer = 'Adam'
    patience = 10
    patience_tolerance = 0.005
    features_norm = 'zscore'
    features_global_norm = True
    targets_norm = 'identity'
    targets_global_norm = True
    flip_history = True
    use_hard_drive = False

    seq2seq_name = 'Seq2seq'
    mlp_name = 'Mlp'
    rnn_name = 'Rnn'
    ltsf_linear_name = os.path.join('LTSF', 'Linear')
    ltsf_nlinear_name = os.path.join('LTSF', 'NLinear')
    ltsf_informer_name = os.path.join('LTSF', 'Informer')
    ltsf_transformer_name = os.path.join('LTSF', 'Transformer')
    ltsf_autoformer_name = os.path.join('LTSF', 'Autoformer')
    ltsf_fedformer_name = os.path.join('LTSF', 'FedFormer')

   
    forces_path = "./Data/forces_clean.pt" if data_name == "ours" else "./Data/forces_prssm.pt"
    kinematics_path = "./Data/kinematics_clean.pt" if data_name == "ours" else "./Data/kinematics_prssm.pt"
    forces, kinematics = torch.load(forces_path), torch.load(kinematics_path)
    breakpoint()
    if isinstance(forces, list) and isinstance(kinematics, list):
        input_size, output_size = forces[0].shape[-1], kinematics[0].shape[-1]
    else:
        input_size, output_size = forces.shape[-1], kinematics.shape[-1]

    model_args_key = 'model_args'
    feature_lags = [512 if data_name == 'ours' else 256]
    batch_sizes = [512]
    target_lags = [1]
    embedding_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    hidden_sizes = [40, 50, 60, 70, 80, 90, 100, 110]
    label_lens = [0]
    layers = [1]
    bidirs = [False]
    dropouts = [0.1]
    activations = ['gelu']
    output_attentions = [False]
    embed_types = [3]
    factors = [1]
    n_heads = [2]
    e_layers = [1]
    d_layers = [2]
    distil = [True]
    moving_avg = [25]
    fedformer_version = ['Fourier']
    fedformer_mode_select = ['random']
    fedformer_n_modes = [32]
    individual = [False]
    use_adl = [True]
    complexify = [False]
    gate = [True]
    multidim_fft = [False]
    concat_adl = [False]
    per_freq_layer = [True, False]
    csd = [False]
    freq_thresholds = [200]
    
    seq2seq_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lags=feature_lags, batch_size=batch_sizes),
        model_args=dict(target_lags=target_lags, enc_embedding_size=embedding_sizes, enc_hidden_size=hidden_sizes,
                        enc_num_layers=layers, enc_bidirectional=bidirs, dec_output_size=[output_size],
                        use_adl=use_adl, concat_adl=concat_adl, complexify=complexify, gate=gate,
                        multidim_fft=multidim_fft, dropout=dropouts, freq_threshold=freq_thresholds,
                        per_freq_layer=per_freq_layer, cross_spectrum_density=csd),
        model_shared_pairs={'dec_hidden_size': 'enc_hidden_size', 'dec_embedding_size': 'enc_embedding_size'},
        model_args_key=model_args_key
    )
    transformer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(batch_size=batch_sizes),
        model_args=dict(feature_lags=feature_lags, target_lags=target_lags, label_len=label_lens,
                        output_attention=output_attentions,
                        enc_in=[input_size], d_model=[16], dropout=[0.05], dec_in=[output_size],
                        embed_type=embed_types, factor=factors, e_layers=e_layers, activation=activations,
                        n_heads=n_heads, d_layers=[1], c_out=[output_size]),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    informer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lags=feature_lags, batch_size=batch_sizes),
        model_args=dict(target_lags=target_lags, label_len=label_lens, output_attention=output_attentions,
                        enc_in=[input_size], d_model=hidden_sizes, dropout=dropouts, dec_in=[output_size],
                        embed_type=embed_types, factor=factors, e_layers=e_layers, activation=activations,
                        n_heads=n_heads, d_layers=d_layers, c_out=[output_size], distil=distil),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    autoformer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(feature_lags=feature_lags, batch_size=batch_sizes),
        model_args=dict(target_lags=target_lags, label_len=label_lens, output_attention=output_attentions,
                        enc_in=[input_size], d_model=[8], dropout=[0.05], dec_in=[input_size],
                        embed_type=embed_types, factor=factors, e_layers=[2], activation=activations,
                        n_heads=n_heads, d_layers=d_layers, c_out=[input_size], moving_avg=moving_avg,
                        output_size=[output_size]),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    fedformer_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(batch_size=batch_sizes),
        model_args=dict(feature_lags=feature_lags, target_lags=target_lags, label_len=label_lens,
                        output_attention=output_attentions, enc_in=[input_size], d_model=[40], dropout=[0.05],
                        dec_in=[input_size], embed_type=embed_types, e_layers=[1], activation=activations,
                        n_heads=[8], d_layers=[2], c_out=[input_size], moving_avg=moving_avg,
                        version=fedformer_version, mode_select=fedformer_mode_select, modes=fedformer_n_modes,
                        output_size=[output_size]),
        model_shared_pairs={'d_ff': 'd_model'},
        model_args_key=model_args_key
    )
    linear_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(batch_size=batch_sizes),
        model_args=dict(feature_lags=feature_lags, target_lags=target_lags, n_features=[input_size],
                        individual=individual,
                        output_size=[output_size]),
        model_args_key=model_args_key
    )
    nlinear_params = ml_utils.generate_hyperparam_combinations(
        global_args=dict(batch_size=batch_sizes),
        model_args=dict(feature_lags=feature_lags, target_lags=target_lags, n_features=[input_size],
                        individual=individual,
                        output_size=[output_size]),
        model_args_key=model_args_key
    )

    used_hyperparams = seq2seq_params  # TODO: <---------------------------- CHANGE THIS TO RUN DIFFERENT MODELS
    model_class_name = seq2seq_name    # TODO: <---------------------------- CHANGE THIS TO RUN DIFFERENT MODELS

    for i, hparams in enumerate(used_hyperparams):
        print('=' * 20 + f' Hyperparams iter #{i}/{len(used_hyperparams)} ' + '=' * 20)
        f_lags = hparams['feature_lags'] if 'feature_lags' in hparams else hparams[model_args_key]['feature_lags']
        t_lags = hparams['target_lags'] if 'target_lags' in hparams else hparams[model_args_key]['target_lags']
        input_dim = (hparams['batch_size'], f_lags, input_size)
        if model_class_name == seq2seq_name: hparams[model_args_key]['input_dim'] = input_dim

        trainer = Trainer(
            exp_name=f"{'ADL' if 'use_adl' in hparams[model_args_key] and hparams[model_args_key]['use_adl'] else ''}[{data_name},T={t_lags}",

            features_path=forces_path,
            targets_path=kinematics_path,
            train_percent=train_percent,
            val_percent=val_percent,
            feature_win=input_dim[1],
            target_win=t_lags,
            intersect=intersect,
            batch_size=hparams['batch_size'],
            model_class_name=model_class_name,
            model_args=hparams[model_args_key],
            optimizer_name=optimizer,
            criterion_name=criterion,
            patience=patience,
            patience_tolerance=patience_tolerance,
            n_epochs=n_epochs,
            seed=seed,
            features_norm_method=features_norm,
            features_global_normalizer=features_global_norm,
            targets_norm_method=targets_norm,
            targets_global_normalizer=targets_global_norm,
            flip_history=flip_history,
            regularization_factor=regularization_factor,
            hyperparams=utils.flatten_dict(hparams)
        )
        trainer.fit()
        _, loss_df = trainer.predict(data_name, f'test', False)
