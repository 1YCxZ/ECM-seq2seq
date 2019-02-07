# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import math

from tensorflow.contrib.seq2seq import tile_batch, BahdanauAttention,\
    BeamSearchDecoder, GreedyEmbeddingHelper, BasicDecoder, dynamic_decode, AttentionWrapper
import numpy as np
from encoder import build_encoder
from cell import create_rnn_cell


PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


class Seq2SeqModel():
    def __init__(self, mode,
                       model_name,
                       vocab_size, embedding_size,
                       enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
                       attn_num_units,
                       dec_num_layers, dec_num_units, dec_cell_type,
                       batch_size, beam_search, beam_size, infer_max_iter,
                       l2_regularize, learning_rate, max_to_keep=100, max_gradient_norm=5.0):

        self.mode = mode
        self.model_name = model_name
        # embedding 参数
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # encoder 参数
        self.enc_num_layers = enc_num_layers
        self.enc_num_units = enc_num_units
        self.enc_cell_type = enc_cell_type
        self.enc_bidir = enc_bidir
        # attention 参数
        self.attn_num_units = attn_num_units
        # decoder 参数
        self.dec_num_layers = dec_num_layers
        self.dec_num_units = dec_num_units
        self.dec_cell_type = dec_cell_type
        # 杂项
        self.batch_size = batch_size
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.l2_regularize = l2_regularize
        self.infer_max_iter = infer_max_iter
        self.learning_rate = learning_rate
        self.max_to_keep = max_to_keep
        self.max_gradient_norm = max_gradient_norm
        # build model
        self.build_model()

    def init_embedding(self, vocab_size, embed_size, dtype=tf.float32,
                       initializer=None, initial_values=None,
                       ):
        """
        embeddings:
            initialize trainable embeddings or load pretrained from files
        """
        if initial_values:
            embedding = tf.Variable(initial_value=initial_values,
                                    name="embedding", dtype=dtype)
        else:
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            embedding = tf.Variable(
                initializer(shape=(vocab_size, embed_size)),
                name="embedding", dtype=dtype)

        return embedding

    def build_model(self):
        print('building model... ...')
        with tf.variable_scope('seq2seq_placeholder'):
            self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
            self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
            self.decoder_targets = tf.placeholder(tf.int32, [None, None], name="decoder_targets")
            self.decoder_targets_masks = tf.placeholder(tf.float32, [None, None], name="mask")
            self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
            self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")
            self.max_target_sequence_length = tf.reduce_max(self.decoder_length, name='max_target_len')

        with tf.variable_scope('seq2seq_embedding'):
            self.embedding = self.init_embedding(self.vocab_size, self.embedding_size)

        with tf.variable_scope('seq2seq_encoder'):
            encoder_outputs, encoder_states = build_encoder(
                self.embedding, self.encoder_inputs, self.encoder_length,
                self.enc_num_layers, self.enc_num_units, self.enc_cell_type,
                bidir=self.enc_bidir)

        with tf.variable_scope('seq2seq_decoder'):
            encoder_length = self.encoder_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_states = tile_batch(encoder_states, multiplier=self.beam_size)
                encoder_length = tile_batch(encoder_length, multiplier=self.beam_size)

            attention_mechanism = BahdanauAttention(num_units=self.attn_num_units,
                                                    memory=encoder_outputs,
                                                    memory_sequence_length=encoder_length)

            decoder_cell = create_rnn_cell(self.dec_num_layers, self.dec_num_units, self.dec_cell_type)
            decoder_cell = AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                              attention_layer_size=self.dec_num_units, name='Attention_Wrapper')

            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_states)

            output_layer = tf.layers.Dense(self.vocab_size,
                                           use_bias=False,
                                           name='output_projection')

            if self.mode == 'train':
                decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_length,
                                                                    name='training_helper')

                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                   helper=training_helper,
                                                                   initial_state=decoder_initial_state,
                                                                   output_layer=output_layer)

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.max_target_sequence_length)

                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)

                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets,
                                                             weights=self.decoder_targets_masks)

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            elif self.mode == 'infer':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * SOS_ID
                end_token = EOS_ID

                if self.beam_search:
                    inference_decoder = BeamSearchDecoder(cell=decoder_cell,
                                                          embedding=self.embedding,
                                                          start_tokens=start_tokens,
                                                          end_token=end_token,
                                                          initial_state=decoder_initial_state,
                                                          beam_width=self.beam_size,
                                                          output_layer=output_layer)
                else:
                    decoding_helper = GreedyEmbeddingHelper(embedding=self.embedding,
                                                            start_tokens=start_tokens,
                                                            end_token=end_token)

                    inference_decoder = BasicDecoder(cell=decoder_cell,
                                                     helper=decoding_helper,
                                                     initial_state=decoder_initial_state,
                                                     output_layer=output_layer)

                decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=self.infer_max_iter)
                if self.beam_search:
                    infer_outputs = decoder_outputs.predicted_ids  # [batch_size, decoder_targets_length, beam_size]
                    self.infer_outputs = tf.transpose(infer_outputs, [0, 2, 1])  # [batch_size, beam_size, decoder_targets_length]
                else:
                    self.infer_outputs = decoder_outputs.sample_id  # [batch_size, decoder_targets_length]

        # =================================4, 保存模型
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.max_to_keep)

    def train(self, sess, batch):
        feed_dict = {
            self.encoder_inputs: batch[0],
            self.decoder_inputs: batch[1],
            self.decoder_targets: batch[2],
            self.decoder_targets_masks: batch[3],
            self.encoder_length: batch[4],
            self.decoder_length: batch[5]
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def eval(self, sess, batch):
        feed_dict = {
            self.encoder_inputs: batch[0],
            self.decoder_inputs: batch[1],
            self.decoder_targets: batch[2],
            self.decoder_targets_masks: batch[3],
            self.encoder_length: batch[4],
            self.decoder_length: batch[5]
        }
        loss = sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):
        feed_dict = {
            self.encoder_inputs: batch[0],
            self.encoder_length: batch[1]
        }
        predict = sess.run([self.infer_outputs], feed_dict=feed_dict)
        return predict

    def compute_perplexity(self, sess, batch):
        # 计算困惑度
        feed_dict = {
            self.encoder_inputs: batch[0],
            self.decoder_inputs: batch[1],
            self.decoder_targets: batch[2],
            self.decoder_targets_masks: batch[3],
            self.encoder_length: batch[4],
            self.decoder_length: batch[5]
        }
        loss = sess.run(self.loss, feed_dict=feed_dict)
        perplexity = math.exp(float(loss))
        return perplexity


