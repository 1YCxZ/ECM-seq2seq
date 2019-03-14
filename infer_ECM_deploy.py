from utils import get_ecm_model_config
from utils import get_ecm_infer_batch, get_ecm_infer_config
from utils import load, prepare_ecm_infer_data, create_vocab_tables

from model.ECM_model import ECMModel
import argparse
import yaml
import tensorflow as tf
import numpy as np
import time
from pprint import pprint
import utils

# p_map = {
#     0: 'other',
#     1: 'like',
#     2: 'Sadness',
#     3: 'Disgust',
#     4: 'Anger',
#     5: 'Happiness'
# }

# p_map = {
#     0: 'other',
#     1: 'pos',
#     2: 'neg',
# }

p_map = {
    0: 'no',
    1: 'happy',
    2: 'angry',
    3: 'sad',
    4: 'surprise',
}

def parse_args():
    '''
    Parse Seq2seq with attention arguments.
    '''
    parser = argparse.ArgumentParser(description="Run seq2seq inference.")

    parser.add_argument('--config', nargs='?',
                        default='./configs/ecm_config.yaml',
                        help='Configuration file for model specifications')

    return parser.parse_args()


def token_to_str(tokens, reverse_vocab_table):
    tokens = list(tokens)

    word_list = [reverse_vocab_table[id] for id in tokens if id > 0]
    sentence = " ".join(word_list)
    return sentence.encode('utf-8')


def main(args):
    # loading configurations
    with open(args.config) as f:
        config = yaml.safe_load(f)["configuration"]

    work_space = config["workspace"]
    infer_model_dir = '%s/infer_model' % work_space
    vocab_size = config["embeddings"]["vocab_size"]
    vocab_file = '%s/data/%s-%s' % (work_space, "vocab", vocab_size)

    (is_beam_search, beam_size, batch_size,
     infer_source_file, infer_target_file, infer_emotion_category_file, infer_source_max_length,
     output_path, output_choice_path, gpu_fraction, gpu_id) = get_ecm_infer_config(config)

    # Set up session
    gpu_fraction = config["training"]["gpu_fraction"]
    gpu_id = config["training"]["gpu_id"]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=gpu_id)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            gpu_options=gpu_options))

    print("loading model ...")
    ecm_model = tf.saved_model.loader.load(sess, tf.saved_model.tag_constants.SERVING, infer_model_dir)
    # print('global_variables:\n')
    # glob_var = tf.global_variables()
    # pprint(glob_var)

    # print('all nodes:')
    # all_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # pprint(all_nodes)

    encoder_inputs = sess.graph.get_tensor_by_name('seq2seq_placeholder/encoder_inputs:0')
    encoder_length = sess.graph.get_tensor_by_name('seq2seq_placeholder/encoder_length:0')
    emo_cat = sess.graph.get_tensor_by_name('seq2seq_placeholder/emotion_category:0')
    infer_outputs = sess.graph.get_tensor_by_name('seq2seq_decoder/infer_outputs:0')
    print("\tDone.")


    # ##### Inference #####
    # Load data
    print("Loading inference data ...")

    # Load vocabularies.
    vocab_table, reverse_vocab_table = create_vocab_tables(vocab_file)

    src_dataset = prepare_ecm_infer_data(infer_source_file, infer_emotion_category_file,
                                         vocab_table, max_length=infer_source_max_length)
    print("\tDone.")

    # Inference
    print("Start inferring ...")
    final_result = []

    for ith in range(int(len(src_dataset) / batch_size)):
        start = ith
        end = ith + 1
        batch = get_ecm_infer_batch(src_dataset, start, end, infer_source_max_length)

        sentence = token_to_str(batch[0][0], reverse_vocab_table)
        # target = token_to_str(tgt_dataset[ith], reverse_vocab_table)
        emo_category = p_map[batch[2][0]]

        start_time = time.time()

        feed_dict = {
            encoder_inputs: batch[0],
            encoder_length: batch[1],
            emo_cat: batch[2]
        }
        result = sess.run([infer_outputs], feed_dict=feed_dict)

        duration =round((time.time() - start_time), 3)
        print("sentence:%s, cost:%s s" % (ith, duration))

        # res = "src:{}  emotion:{}\ntgt:{}\n".format(sentence, emo_category, target)
        res = "src:{}  emotion:{}\n".format(sentence, emo_category)
        if is_beam_search is True:
            for idx, i in enumerate(result[0][0]):
                reply = token_to_str(i, reverse_vocab_table)
                res += "\tpred %s:%s\n" % (idx, reply)
            res += "\n"
        else:
            reply = result[0][0]
            reply = token_to_str(reply, reverse_vocab_table)
            res += "\tpred:%s\n\n" % reply
        print(res)
        final_result.append(res)

    with open(config["inference"]["output_path"], 'w') as f:
        for i in final_result:
            f.write(i+'\n')
    print("\tDone.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
