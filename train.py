# -*- coding:utf-8 -*-
from __future__ import print_function

from utils import get_model_config
from utils import get_training_config
from utils import load, save, prepare_train_dev_data
from utils import create_vocab_tables, create_vocab_file, get_batch

from model.word_model import Seq2SeqModel
import argparse
import time
import yaml
import tensorflow as tf
import numpy as np
import os
from datetime import datetime


def setup_workpath(workspace):
    for p in ['data', 'nn_models', 'results']:
        wp = "{}/{}".format(workspace, p)
        if not os.path.exists(wp):
            os.mkdir(wp)


def add_summary(summary_writer, global_step, tag, value):
    """
    Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def parse_args():
    '''
    Parse Seq2seq with attention arguments.
    '''
    parser = argparse.ArgumentParser(description="Run seq2seq training.")

    parser.add_argument('--config', nargs='?',
                        default='./configs/config_seq2seqAttn_beamsearch.yaml',
                        help='Configuration file for model specifications')

    return parser.parse_args()


def main(args):

    # loading configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)["configuration"]

    # set up workspace
    work_space = config["workspace"]
    tf_board = config["tf_board"]
    setup_workpath(work_space)
    name = config["Name"]

    # Construct or load embeddings
    print("Initializing embeddings ...")
    vocab_size = config["embeddings"]["vocab_size"]
    embed_size = config["embeddings"]["embed_size"]

    vocab_file = '%s/data/%s-%s' % (work_space, "vocab", vocab_size)
    print("\tDone.")

    # Build the model and compute losses
    (enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
     attn_num_units, dec_num_layers, dec_num_units, dec_cell_type,
     state_pass, infer_max_iter, l2_regularize, learning_rate
     ) = get_model_config(config)

    (train_s_file, train_t_file, dev_s_file, dev_t_file, max_length,
     gpu_fraction, gpu_id, checkpoint_every, max_checkpoints, print_every,
     train_steps, is_beam_search, batch_size, beam_size) = get_training_config(config)

    print("Building model architecture ...")
    train_model = Seq2SeqModel(
        mode='train', model_name=name,
        vocab_size=vocab_size, embedding_size=embed_size,
        enc_num_layers=enc_num_layers, enc_num_units=enc_num_units, enc_cell_type=enc_cell_type, enc_bidir=enc_bidir,
        attn_num_units=attn_num_units,
        dec_num_layers=dec_num_layers, dec_num_units=dec_num_units, dec_cell_type=dec_cell_type,
        batch_size=batch_size, beam_search=is_beam_search, beam_size=beam_size, infer_max_iter=infer_max_iter,
        l2_regularize=l2_regularize, learning_rate=learning_rate, max_to_keep=max_checkpoints)

    print("\tDone.")

    logdir = '%s/nn_models/' % work_space
    restore_from = '%s/nn_models/' % work_space

    is_overwritten_training = logdir != restore_from  # 判断两个文件件是否相同

    # Set up session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=gpu_id)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)

    # tensorbord
    train_writer = tf.summary.FileWriter(tf_board + 'train/', sess.graph)
    test_writer = tf.summary.FileWriter(tf_board + 'test/', sess.graph)

    try:
        saved_global_step = load(train_model.saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except Exception:
        print("Something went wrong while restoring checkpoint. "
              "Training is terminated to avoid the overwriting.")
        raise

    # ##### Training #####
    # Load data
    print("Loading data ...")

    # Load vocabularies.
    if os.path.exists(vocab_file):
        vocab_table, reverse_vocab_table = create_vocab_tables(vocab_file)
    else:
        create_vocab_file(train_s_file, train_t_file,
                          dev_s_file, dev_t_file,
                          vocab_file, vocab_size)
        vocab_table, reverse_vocab_table = create_vocab_tables(vocab_file)

    train_set, dev_set = prepare_train_dev_data(train_s_file, train_t_file,
                                                dev_s_file, dev_t_file,
                                                vocab_table, max_length)

    # Training
    last_saved_step = saved_global_step
    num_steps = saved_global_step + train_steps
    losses = []
    steps = []

    print("Start training ...")
    try:
        for step in range(saved_global_step + 1, num_steps):
            start_time = time.time()

            batch = get_batch(train_set, max_length, batch_size)

            loss_value = train_model.train(sess, batch)

            losses.append(loss_value)
            duration = (time.time() - start_time)
            if step % print_every == 0 and step != 0:
                # train perplexity
                t_perp = train_model.compute_perplexity(sess, batch)
                add_summary(train_writer, step, 'train perplexity', t_perp)

                # eval perplexity
                dev_str = ""
                if dev_set is not None:
                    eval_batch = get_batch(dev_set, max_length, batch_size)

                    eval_perp = train_model.compute_perplexity(sess, eval_batch)
                    add_summary(test_writer, step, 'eval perplexity', eval_perp)
                    dev_str += "val_prep: {:.3f}\n".format(eval_perp)

                steps.append(step)
                info = 'step {:d}, loss = {:.6f},perp: {:.3f}\n{}({:.3f} sec/step)'
                print(info.format(step, loss_value, t_perp, dev_str, duration))

            if step % checkpoint_every == 0:
                save(train_model.saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C so save message is on its own line.
        print()

    finally:
        if step > last_saved_step:
            save(train_model.saver, sess, logdir, step)


if __name__ == "__main__":
    args = parse_args()
    main(args)
