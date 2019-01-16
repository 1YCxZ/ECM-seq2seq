# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

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
                       l2_regularize, learning_rate, max_to_keep=100):

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
        #=================================1, 定义模型的placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.decoder_targets_masks = tf.placeholder(tf.bool, [None, None], name="mask")
        self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
        self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")
        #=================================2, embedding
        with tf.variable_scope('seq2seq_embedding'):
            self.embedding = self.init_embedding(self.vocab_size, self.embedding_size)
        #=================================3, 定义模型的encoder部分
        with tf.variable_scope('seq2seq_encoder'):
            encoder_outputs, encoder_states = build_encoder(
                self.embedding, self.encoder_inputs, self.encoder_length,
                self.enc_num_layers, self.enc_num_units, self.enc_cell_type,
                bidir=self.enc_bidir)
        # =================================4, 定义模型的decoder部分
        with tf.variable_scope('seq2seq_decoder'):
            encoder_length = self.encoder_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_states = tile_batch(encoder_states, multiplier=self.beam_size)
                encoder_length = tile_batch(encoder_length, multiplier=self.beam_size)

            #定义要使用的attention机制。
            attention_mechanism = BahdanauAttention(num_units=self.attn_num_units,
                                                    memory=encoder_outputs,
                                                    memory_sequence_length=encoder_length)

            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = create_rnn_cell(self.dec_num_layers, self.dec_num_units, self.dec_cell_type)
            decoder_cell = AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                              attention_layer_size=self.dec_num_units, name='MyAttention_Wrapper')
            #如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_states)

            output_layer = tf.layers.Dense(self.vocab_size,
                                           use_bias=False,
                                           name='output_projection')

            if self.mode == 'train':
                decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

                decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
                    decoder_cell, decoder_inputs_embedded,
                    sequence_length=self.decoder_length,
                    initial_state=decoder_initial_state,
                    scope='decoder',
                    dtype=tf.float32,
                    swap_memory=True)

                # 这里加variable_scope是为了配合官方api
                with tf.variable_scope('decoder'):
                    train_logits = output_layer(decoder_outputs)

                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=train_logits, labels=self.decoder_targets)

                losses = tf.boolean_mask(losses, self.decoder_targets_masks)
                self.reduced_loss = tf.reduce_mean(losses)
                self.CE = tf.reduce_sum(losses)  # cross entropy

                if self.l2_regularize is None:
                    loss = self.reduced_loss
                else:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not ('bias' in v.name)])

                    total_loss = self.reduced_loss + self.l2_regularize * l2_loss
                    loss = total_loss

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                trainable = tf.trainable_variables()  # 返回所有可供训练的参数
                self.train_op = optimizer.minimize(loss, var_list=trainable)

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
        _, loss = sess.run([self.train_op, self.reduced_loss], feed_dict=feed_dict)
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
        loss = sess.run([self.reduced_loss], feed_dict=feed_dict)
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

        CE_words = sess.run(self.CE, feed_dict=feed_dict)
        N_words = np.sum(batch[3])
        return np.exp(CE_words / N_words)


