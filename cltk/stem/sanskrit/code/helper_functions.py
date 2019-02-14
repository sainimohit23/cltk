import numpy as np
import sys
from cltk.stem.sanskrit.code import defines

def analyze_text(text, path_out, predictions_ph, x_ph, split_cnts_ph, seqlen_ph, dropout_ph, loader, session, verbose = False):
    '''
    Apply a trained model to a text.
    
    The xxx_ph parameters are the placeholders that are fed + the prediction.
    Required to make the function compatible with train and application settings.
    '''
    if verbose==True:
        print('Analyzing text ...' )
    seqs,lens,splitcnts,lines_orig = loader.load_external_text(text)
    if seqs is None:
        print('Something went wrong while loading text \n' )
        return
    
    final_str = ''
    batch_size = 500
    start = 0
    P = None
    while True:
        end = min(start+batch_size, seqs.shape[0])
        if end<=start:
            break
        if verbose==True:
            sys.stdout.write(' lines {0} => {1}\r'.format(start,end) ); sys.stdout.flush();
        p = session.run(predictions_ph, feed_dict = {
            x_ph:seqs[start:end,:],
            split_cnts_ph:splitcnts[start:end,:,:],
            seqlen_ph:lens[start:end],
            dropout_ph:1.0
            })
        if P is None:
            P = p
        else:
            P = np.concatenate([P,p], axis=0)
        start = end
    if verbose==True:
        print('')
    ''' decode and write to the file '''
    for i in range(P.shape[0]):
        pred_sym_seq = [loader.deenc_output.get_sym(x) for x in P[i, :lens[i] ] ] # skip the last symbol
        pred_str = ''
        for p_s, o_s in zip(pred_sym_seq[1:], lines_orig[i][1:] ):
            if p_s==defines.SYM_IDENT:
                pred_str+=o_s
            elif p_s==defines.SYM_SPLIT:
                pred_str+=o_s + '-'
            else:
                pred_str+=p_s
        pred_str = loader.internal_transliteration_to_unicode(pred_str).replace('- ', ' ').replace('= ', ' ')
        pred_str = pred_str + u' '
        final_str += pred_str    
    return final_str
        