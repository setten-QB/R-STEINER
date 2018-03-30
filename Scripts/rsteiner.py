"""
R-STEINER
"""
import numpy as np
import pandas as pd
from datetime import datetime as dt
from multiprocessing import Pool
import sys
import subprocess
import platform
from sklearn.externals import joblib
import settenQBmodule as sq


def evaluate_rna_fold(seq_list):
    """
    calculate the secondary structure energy of RNA
    """
    RNAfold_Path = "RNAfold"  # path of RNAfold
    paramFile_Path = "../Parameters/rna_turner2004.par"
    Convert_to_minus_dG = True
    seq_txt = ''
    for i in range(len(seq_list)):
        if seq_list[i] == '':
            seq_list[i] = 'NNN'
        seq_txt = seq_txt + seq_list[i] + '\n'
    if platform.system() == 'Windows':
        p = subprocess.Popen(
            [RNAfold_Path, '--noPS', "--paramFile="+paramFile_Path],
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
            )
    else:
        p = subprocess.Popen(
            [RNAfold_Path, '--noPS', "--paramFile="+paramFile_Path],
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
    tmp = p.communicate(seq_txt.encode())[0].decode()
    evaluated_list = []
    if platform.system() == 'Windows':
        tmp = tmp.split(')\r\n')
    else:
        tmp = tmp.split(')\n')
    for j in range(len(seq_list)):
        re_each = tmp[j].split()
        dG = float(re_each[-1].replace('(', '').replace(')', ''))
        dG = round(dG, 2)
        if dG == 0:
            evaluated_list.append(dG)
        elif True:
            evaluated_list.append(-dG)
        else:
            evaluated_list.append(dG)
    return evaluated_list


def random_seq_generator(num_seq):
    """
    argument is the number of sequences
    """
    # greater than 20 and lowe than 47
    seq_lengths = np.random.randint(20, 47, num_seq)
    utr5s = []  # 5'UTR list
    for seq_len in seq_lengths:
        ''' generate sequences '''
        seq_arr = np.random.choice(['A', 'C', 'G', 'T'], seq_len)
        utr5 = 'GG'  # initial codon
        for codon in seq_arr:
            ''' joint strings '''
            utr5 += codon
        utr5s.append(utr5)
    return utr5s


if __name__ == "__main__":
    """
    parameters
    """
    processn = 20
    iteration = 10**(6)  # the number of generated sequences
    tdatetime = dt.now()  # get time
    tstr = tdatetime.strftime('%Y%m%d%H%M')  # time for model nam
    seq_cds = "XXX"  # given CDS sequence
    seq_3utr = "XXX"  # given 3'UTR sequence
    k = 5  # the number of mRNA sequences yielded by R-STEINER
    """ paths of predictors (change xxx to your model numbers) """
    path_rfr_hs = "../Models/Random-Forest_HS-{0}".format("xxx")
    path_gbr_hs = "../Models/Gradient-Boosting_HS-{0}".format("xxx")
    path_xgbr_hs = "../Models/XGBoosting_HS-{0}".format("xxx")
    path_rfr_con = "../Models/Random-Forest_Con-{0}".format("xxx")
    path_gbr_con = "../Models/Gradient-Boosting_Con-{0}".format("xxx")
    path_xgbr_con = "../Models/XGBoosting_Con-{0}".format("xxx")

    """
    make features of CDS and 3UTR
    """
    def counter_5utr(seq_5utrs):
        '''
        basic function to make counter feature
        '''
        ngrams = sq.ngram_list(3)  # n-grams whose lengths are less than three
        rcnts = []  # list to contain the counters of n-grams on each 5UTRs
        for seq_5utr in seq_5utrs:
            # counter of n-grams in one 5UTR
            cnts = [seq_5utr.count(ngram) for ngram in ngrams]
            rcnts.append(cnts)
        return rcnts
    ngrams = sq.ngram_list(3)
    # n-grams whose lengths are three
    ncodons = [ngram for ngram in sq.ngram_list(3) if len(ngram) == 3]
    cols_cds = [i+'.CDS' for i in ncodons]
    cols_3utr = [i+'.3UTR' for i in ngrams]

    """ count feature """
    cnt_cds = [seq_cds.count(ncodon) for ncodon in ncodons]
    cnt_cds = pd.Series(cnt_cds, index=cols_cds)
    cnt_3utr = [seq_3utr.count(ngram) for ngram in ngrams]
    cnt_3utr = pd.Series(cnt_3utr, index=cols_3utr)

    """ length feature """
    length_cds = len(seq_cds)
    length_3utr = len(seq_3utr)

    """ secondary structure energy feature """
    se_cds = evaluate_rna_fold([seq_cds])
    se_3utr = evaluate_rna_fold([seq_3utr])

    """
    make 5UTR sequences
    """
    seq_5utrs = random_seq_generator(iteration)
    mrnas = pd.DataFrame(seq_5utrs)
    mrnas.reshape((-1, 1))
    mrnas.columns = ["5UTR"]

    """
    make feature of 5UTR
    """
    """ Make count feature """
    step = int(len(seq_5utrs)/processn)
    pseq_5utrs = [
        seq_5utrs[i: i+step] for i in range(0, len(seq_5utrs), step)]
    p = Pool(processn)  # make process for multi-processing
    ngrams = sq.ngram_list(3)
    cols_5utr = [i+'.5UTR' for i in ngrams]
    cnts_5utr = p.map(counter_5utr, pseq_5utrs)
    p.close()
    cnts_5utr = sq.flatten(cnts_5utr)
    cnts_5utr = pd.DataFrame(cnts_5utr, columns=cols_5utr)

    """ make length feature """
    length_5utrs = [len(seq) for seq in seq_5utrs]

    """ make secondary structure energy feature """
    p = Pool(processn)
    se_5utr = p.map(evaluate_rna_fold, pseq_5utrs)
    p.close()
    se_5utr = sq.flatten(se_5utr)

    """
    make feature matarix
    """
    # n-grams whose lengths are three
    ncodons = [ngram for ngram in sq.ngram_list(3) if len(ngram) == 3]
    """ count feature """
    cnts_cds = pd.DataFrame([cnt_cds for i in range(len(cnts_5utr))])
    cnts_3utr = pd.DataFrame([cnt_3utr for i in range(len(cnts_5utr))])

    """ length feature """
    length_5utr = pd.DataFrame(length_5utrs, columns=['length_5UTR'])
    lengths_cds = pd.DataFrame(
        length_cds * np.ones(len(length_5utrs)), columns=['length_CDS'])
    lengths_3utr = pd.DataFrame(
        length_3utr * np.ones(len(length_5utrs)), columns=['length_3UTR'])
    length = pd.concat([lengths_3utr, length_5utr, lengths_cds], axis=1)

    """ secondary structure energy """
    se_5utr = pd.DataFrame(se_5utr, columns=['secondaryE.5UTR'])
    ses_cds = pd.DataFrame(
        se_cds * np.ones(len(length_5utrs)), columns=['secondaryE.CDS'])
    ses_3utr = pd.DataFrame(
        se_3utr * np.ones(len(length_5utrs)), columns=['secondaryE.3UTR'])

    """ make feature dataframe """
    feature = pd.concat(
        [cnts_5utr, cnts_cds, cnts_3utr, length,
         se_5utr, ses_cds, ses_3utr],
        axis=1)
    feature.to_pickle(
        '../Data/feature/maked-mRNA-{0}-feature.pickle'.format(tstr))
    """ numpy array versiong """
    x_arr = feature.values

    """
    prediction
    """
    """ import models """
    rfr_hs = joblib.load(path_rfr_hs)
    gbr_hs = joblib.load(path_gbr_hs)
    xgbr_hs = joblib.load(path_xgbr_hs)
    rfr_con = joblib.load(path_rfr_con)
    gbr_con = joblib.load(path_gbr_con)
    xgbr_con = joblib.load(path_xgbr_con)

    """" predictions """
    prediction_rfrhs_fluc = rfr_hs.predict(x_arr)
    prediction_gbrhs_fluc = gbr_hs.predict(x_arr)
    prediction_xgbrhs_fluc = xgbr_hs.predict(x_arr)

    prediction_rfrcon_fluc = rfr_con.predict(x_arr)
    prediction_gbrcon_fluc = gbr_con.predict(x_arr)
    prediction_xgbrcon_fluc = xgbr_con.predict(x_arr)

    predictions = pd.DataFrame([
        prediction_rfrhs_fluc, prediction_gbrhs_fluc, prediction_xgbrhs_fluc,
        prediction_rfrcon_fluc, prediction_gbrcon_fluc, prediction_xgbrcon_fluc
        ]).T
    predictions.columns = [
        'RFR-HS', 'GBR-HS', 'XGB-HS', 'RFR-Con', 'GBR-Con', 'XGB-Con']
    prediction = predictions.mean(axis=1)  # mean of all predictor
    prediction = pd.DataFrame(prediction).reshape((-1, 1))
    prediction.columns = ["prediction"]
    prediction = pd.concat([mrnas, prediction], axis=1)  # result dataframe
    # sort by predicted PR-value
    prediction = prediction.sort_values(by="prediction", ascending=False)
    prediction = prediction.iloc[0:k, :]  # select top-k sequences
    prediction.to_csv("../Data/result_R-STEINER.csv")
