# -*- coding:utf-8 -*-
import tensorflow as tf
import math

from tensorflow.contrib.seq2seq import tile_batch, BahdanauAttention,\
    GreedyEmbeddingHelper, BasicDecoder, dynamic_decode,TrainingHelper
from ECM_Attention import ECMWrapper
from encoder import build_encoder
from cell import create_rnn_cell
from ECM_beamsearch import ECMBeamSearchDecoder

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


class ECMModel():
    def __init__(self, mode,
                       model_name,
                       vocab_size, embedding_size,
                       enc_num_layers, enc_num_units, enc_cell_type, enc_bidir,
                       attn_num_units,
                       dec_num_layers, dec_num_units, dec_cell_type,
                       emo_cat_emb_size, emo_internal_memory_units, num_emotion,
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
        # ECM 参数
        self.emo_cat_emb_size = emo_cat_emb_size  # emotion category embedding
        self.emo_internal_memory_units = emo_internal_memory_units  # emotion memory size
        self.num_emotion = num_emotion  # num of emotions
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
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.decoder_targets_masks = tf.placeholder(tf.bool, [None, None], name="mask")
        self.encoder_length = tf.placeholder(tf.int32, [None], name="encoder_length")
        self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")
        # ECM placeholder
        self.choice_qs = tf.placeholder(tf.float32, [None, None], name="choice")
        self.emo_cat = tf.placeholder(tf.int32, [None], name="emotion_category")
        self.max_target_sequence_length = tf.reduce_max(self.decoder_length, name='max_target_len')

        with tf.variable_scope('seq2seq_embedding'):
            self.embedding = self.init_embedding(self.vocab_size, self.embedding_size)
            # create emotion category embeddings
            emo_initializer = tf.contrib.layers.xavier_initializer()
            emo_cat_embeddings = tf.get_variable("emo_cat_embeddings",
                                                 [self.num_emotion, self.emo_cat_emb_size],
                                                 initializer=emo_initializer,
                                                 dtype=tf.float32)
            self.emo_internal_memory_embedding = tf.get_variable("emo_internal_memory_embedding",
                                                                 [self.num_emotion, self.emo_internal_memory_units],
                                                                 initializer=emo_initializer,
                                                                 dtype=tf.float32)
            self.emo_cat_embs = tf.nn.embedding_lookup(emo_cat_embeddings, self.emo_cat)

        with tf.variable_scope('seq2seq_encoder'):
            encoder_outputs, encoder_states = build_encoder(
                self.embedding, self.encoder_inputs, self.encoder_length,
                self.enc_num_layers, self.enc_num_units, self.enc_cell_type,
                bidir=self.enc_bidir)

        with tf.variable_scope('seq2seq_decoder'):
            encoder_length = self.encoder_length
            emo_cat = self.emo_cat
            emo_cat_embs = self.emo_cat_embs
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_states = tile_batch(encoder_states, multiplier=self.beam_size)
                encoder_length = tile_batch(encoder_length, multiplier=self.beam_size)
                emo_cat = tile_batch(emo_cat, multiplier=self.beam_size)
                emo_cat_embs = tile_batch(emo_cat_embs, multiplier=self.beam_size)

            attention_mechanism = BahdanauAttention(num_units=self.attn_num_units,
                                                    memory=encoder_outputs,
                                                    memory_sequence_length=encoder_length)

            decoder_cell = create_rnn_cell(self.dec_num_layers, self.dec_num_units, self.dec_cell_type)

            self.read_g = tf.layers.Dense(
                self.emo_internal_memory_units, use_bias=False, name="internal_read_gate")

            self.write_g = tf.layers.Dense(
                self.emo_internal_memory_units, use_bias=False, name="internal_write_gate")

            decoder_cell = ECMWrapper(cell=decoder_cell,
                                      attention_mechanism=attention_mechanism,
                                      emo_cat_embs=emo_cat_embs,  # emotion category embedding
                                      emo_cat=emo_cat,  # emotion category
                                      emo_internal_memory_units=self.emo_internal_memory_units,  # emotion memory size
                                      emo_internal_memory_embedding=self.emo_internal_memory_embedding,  # num of emotions
                                      read_gate=self.read_g,
                                      write_gate=self.write_g,
                                      attention_layer_size=self.dec_num_units,
                                      name='ECMWrapper')

            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_states)

            output_layer = tf.layers.Dense(self.vocab_size,
                                           use_bias=False,
                                           name='output_projection')

            # ECM external memory module
            emo_output_layer = tf.layers.Dense(
                self.vocab_size, use_bias=False, name="emo_output_projection")

            emo_choice_layer = tf.layers.Dense(
                1, use_bias=False, name="emo_choice_alpha")

            if self.mode == 'train':
                decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_inputs)

                training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                                 sequence_length=self.decoder_length,
                                                 name='training_helper')

                training_decoder = BasicDecoder(cell=decoder_cell,
                                                helper=training_helper,
                                                initial_state=decoder_initial_state,
                                                )
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                self.decoder_outputs, self.final_state, self.final_sequence_length = dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.max_target_sequence_length)

                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                self.decoder_logits_train = tf.identity(self.decoder_outputs.rnn_output)

                with tf.variable_scope('decoder'):
                    self.generic_logits = output_layer(self.decoder_logits_train)
                    self.emo_ext_logits = emo_output_layer(self.decoder_logits_train)
                    self.alphas = tf.nn.sigmoid(emo_choice_layer(self.decoder_logits_train))
                    self.int_M_emo = self.final_state.internal_memory

                g_probs = tf.nn.softmax(self.generic_logits) * (1 - self.alphas)
                e_probs = tf.nn.softmax(self.emo_ext_logits) * self.alphas
                train_log_probs = tf.log(g_probs + e_probs)

                # compute losses
                self.alphas = tf.squeeze(self.alphas, axis=-1)
                self.g_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.generic_logits, labels=self.decoder_targets) - tf.log(1 - self.alphas)

                self.e_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.emo_ext_logits, labels=self.decoder_targets) - tf.log(self.alphas)

                losses = self.g_losses * (1 - self.choice_qs) + self.e_losses * self.choice_qs

                # alpha and internal memory regularizations
                self.alpha_reg = tf.reduce_mean(self.choice_qs * -tf.log(self.alphas))
                self.int_mem_reg = tf.reduce_mean(tf.norm(self.int_M_emo + 1e-7, axis=1))

                losses = tf.boolean_mask(losses, self.decoder_targets_masks)
                self.loss = tf.reduce_mean(losses) + self.alpha_reg + self.int_mem_reg

                # prepare for perlexity computations
                CE = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=train_log_probs, labels=self.decoder_targets)
                CE = tf.boolean_mask(CE, self.decoder_targets_masks)
                self.CE = tf.reduce_mean(CE)

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            elif self.mode == 'infer':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * SOS_ID
                end_token = EOS_ID

                inference_decoder = ECMBeamSearchDecoder(cell=decoder_cell,
                                                         embedding=self.embedding,
                                                         start_tokens=start_tokens,
                                                         end_token=end_token,
                                                         initial_state=decoder_initial_state,
                                                         beam_width=self.beam_size,
                                                         output_layer=output_layer,
                                                         emo_output_layer=emo_output_layer,
                                                         emo_choice_layer=emo_choice_layer)

                decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=self.infer_max_iter)

                infer_outputs = decoder_outputs.predicted_ids  # [batch_size, decoder_targets_length, beam_size]
                self.infer_outputs = tf.transpose(infer_outputs, [0, 2, 1])  # [batch_size, beam_size, decoder_targets_length]

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.max_to_keep)

    def train(self, sess, batch):
        feed_dict = {
            self.encoder_inputs: batch[0],
            self.decoder_inputs: batch[1],
            self.decoder_targets: batch[2],
            self.decoder_targets_masks: batch[3],
            self.encoder_length: batch[4],
            self.decoder_length: batch[5],
            self.choice_qs: batch[6],
            self.emo_cat: batch[7]
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
            self.decoder_length: batch[5],
            self.choice_qs: batch[6],
            self.emo_cat: batch[7]
        }
        loss = sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):
        feed_dict = {
            self.encoder_inputs: batch[0],
            self.encoder_length: batch[1],
            self.emo_cat: batch[2]
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
            self.decoder_length: batch[5],
            self.choice_qs: batch[6],
            self.emo_cat: batch[7]
        }
        loss = sess.run(self.CE, feed_dict=feed_dict)
        perplexity = math.exp(float(loss))
        return perplexity



