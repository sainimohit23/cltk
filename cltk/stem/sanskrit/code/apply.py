'''
apply a trained model to a text => Create its padapatha form.
'''
import tensorflow as tf
import helper_functions
import data_loader
import os

def split_sandhi(text, config):
    with data_loader.DataLoader('../data/input', config, load_data_into_ram=False, load_data = False) as data:
        graph_pred = tf.Graph()
        with graph_pred.as_default():
            with tf.Session(graph=graph_pred) as sess:
                # Restore saved values
                print('\nRestoring...')
                model_dir = model_dir = os.path.normpath( os.path.join(os.getcwd(), config['model_directory']) )
                tf.saved_model.loader.load(
                    sess,
                    [tf.saved_model.tag_constants.SERVING],
                    model_dir
                )
                print('Ok')
                
                x_ph = graph_pred.get_tensor_by_name('inputs:0')
                split_cnts_ph = graph_pred.get_tensor_by_name('split_cnts:0')
                dropout_ph = graph_pred.get_tensor_by_name('dropout_keep_prob:0')
                seqlen_ph  = graph_pred.get_tensor_by_name('seqlens:0')
                predictions_ph = graph_pred.get_tensor_by_name('predictions:0')
                
                return helper_functions.analyze_text(text, '/out.txt', predictions_ph, x_ph, split_cnts_ph, seqlen_ph, dropout_ph, data, sess, verbose=True)