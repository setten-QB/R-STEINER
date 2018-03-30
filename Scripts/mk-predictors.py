"""
This script make predictors of PR-value from RNA sequence data.
"""
import numpy as np
import pandas as pd
from datetime import datetime as dt
import pickle
from multiprocessing import Pool
import sys
import subprocess
import platform
import subprocess
import platform
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.externals import joblib
import scipy.stats
import sklearn as sl
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
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
        elif seq_list[i] == 'N':
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
        dG = float(re_each[-1].replace('(','').replace(')',''))
        dG = round(dG, 2)
        if dG == 0:
            evaluated_list.append(dG)
        elif True:
            evaluated_list.append(-dG)
        else:
            evaluated_list.append(dG)
    return evaluated_list


if __name__ == "__main__":
    """
    parameters
    """
    processn = 4  # the number of processor you can use
    n = 3  # fix
    # incert the path of sequencial data in HS condition
    hs_sequence_path = "xxx"
    hs_PR_path = "yyy"  # incert the path of PR-value data in HS condition
    # you should incert the path of sequencial data in Con condition
    con_sequence_path = "yyy"
    con_PR_path = "yyy"  # incert the path of PR-value data in HS condition
    """ model number """
    tdatetime = dt.now()  # get time
    tstr = tdatetime.strftime('%Y%m%d%H%M')  # time for model nam

    """
    make cnt feature in HS condition
    """
    df = pd.read_pickle(hs_sequence_path)
    y = pd.read_pickle(hs_PR_path)
    ngrams = sq.ngram_list(n)  # n-grams whose lengths are less than three
    # n-grams whose lengths are three
    ncodons = [ngram for ngram in sq.ngram_list(3) if len(ngram) == 3]

    def cnter_5utr(name_list):
        '''
        count the n-gram appearence in CDS.
        name_list means the list containing the Name of mRNA
        and this function returns the list containing the list of counters in
        ngram_list.
        '''
        rcnts = []  # list to contain the counters of n-grams on each 5'UTR
        for name in name_list:
            ''' loop in Name '''
            seqs = df['5UTR'][name]  # 5'UTRs which have same name
            # list containing the series of coutners of n-grams
            cnts = [seqs.count(ngram) for ngram in ngrams]
            rcnts.append(cnts)
        return rcnts

    def cnter_cds(name_list):
        '''
        count the n-gram appearence in CDS.
        name_list means the list containing the Name of mRNA
        and this function returns the list containing the list of counters in
        ngram_list.
        '''
        rcnts = []  # list to contain the counters of n-grams on each CDS
        for name in name_list:
            ''' loop in Name '''
            seqs = df['CDS'][name]  # CDSs which have same name
            # list containing the series of coutners of n-grams
            cnts = [seqs.count(ngram) for ngram in ncodons]
            rcnts.append(cnts)
        return rcnts

    def cnter_3utr(name_list):
        '''
        count the n-gram appearence in CDS.
        name_list means the list containing the Name of mRNA
        and this function returns the list containing the list of counters in
        ngram_list.
        '''
        rcnts = []  # list to contain the counters of n-grams on each 3'UTR
        for name in name_list:
            ''' loop in Name '''
            seqs = df['3UTR'][name]  # 3'UTRs which have same name
            # list containing the series of coutners of n-grams
            cnts = [seqs.count(ngram) for ngram in ngrams]
            rcnts.append(cnts)
        return rcnts

    names = df.index  # index of originl dataframe
    step = int(len(names)/processn)
    indexs = [names[i: i+step] for i in range(0, len(names), step)]
    p = Pool(processn)  # make process for multi-processing

    cols_5utr = [i+'.5UTR' for i in ngrams]
    cols_cds = [i+'.CDS' for i in ncodons]
    cols_3utr = [i+'.3UTR' for i in ngrams]

    """ 5UTR """
    cnts_5utr = p.map(cnter_5utr, indexs)
    cnts_5utr = sq.flatten(cnts_5utr)
    cnts_5utr = pd.DataFrame(cnts_5utr, index=names, columns=cols_5utr)
    cnts_5utr.to_pickle(
        '../Data/feature/cnts_3-gram-5utr-{0}.pickle'.format("HS"))

    """ CDS """
    cnts_cds = p.map(cnter_cds, indexs)
    cnts_cds = sq.flatten(cnts_cds)
    cnts_cds = pd.DataFrame(cnts_cds, index=names, columns=cols_cds)
    cnts_cds.to_pickle(
        '../Data/feature/cnts_3-gram-cds-{0}.pickle'.format("HS"))

    """ 3UTR """
    cnts_3utr = p.map(cnter_3utr, indexs)
    cnts_3utr = sq.flatten(cnts_3utr)
    cnts_3utr = pd.DataFrame(cnts_3utr, index=names, columns=cols_3utr)
    cnts_3utr.to_pickle(
        '../Data/feature/cnts_3-gram-3utr-{0}.pickle'.format("HS"))

    """
    make length features in HS condition
    """
    length_5utr = [len(seq) for seq in df['5UTR']]
    length_cds = [len(seq) for seq in df['CDS']]
    length_3utr = []
    for i in df['3UTR']:
        '''
        Exception because of containing N
        5UTR doesn't contain N.
        '''
        if i == 'N':
            length_3utr.append(0)
        else:
            length_3utr.append(len(i))
    length_3utr = pd.Series(length_3utr, name='3UTR')
    length = pd.DataFrame(
        {'length_5UTR': length_5utr,
         'length_CDS': length_cds,
         'length_3UTR': length_3utr},
        index=df.index)
    length.to_pickle('../Data/feature/length-{0}.pickle'.format("HS"))

    """
    make free energy features in HS condition
    """
    """ 5UTR """
    ''' divide list for multi-processing '''
    list_5utr = list(df['5UTR'])
    step = int(len(list_5utr)/processn)
    indexs = [list_5utr[i: i+step] for i in range(0, len(list_5utr), step)]
    ''' multi-processing '''
    p = Pool(processn)
    se_5utr = p.map(evaluate_rna_fold, indexs)
    se_5utr = sq.flatten(se_5utr)
    p.close()
    se_5utr_out = pd.DataFrame(
        se_5utr, index=df.index, columns=['secondaryE.5UTR'])
    se_5utr_out.to_pickle(
        '../Data/feature/secondaryE-{0}-5utr.pickle'.format("HS"))

    """ CDS """
    ''' divide list for multi-processing '''
    list_cds = list(df['5UTR'])
    step = int(len(list_cds)/processn)
    indexs = [list_cds[i: i+step] for i in range(0, len(list_cds), step)]
    ''' calculate secondary energy with multi-processing '''
    p = Pool(processn)
    se_cds = p.map(evaluate_rna_fold, indexs)
    se_cds = sq.flatten(se_cds)
    p.close()
    se_cds_out = pd.DataFrame(
        se_cds, index=df.index, columns=['secondaryE.CDS'])
    se_cds_out.to_pickle(
        '../Data/feature/secondaryE-{0}-cds.pickle'.format("HS"))

    """ 3UTR """
    ''' divide list for multi-processing '''
    list_3utr = list(df['3UTR'])
    step = int(len(list_3utr)/processn)
    indexs = [list_3utr[i: i+step] for i in range(0, len(list_3utr), step)]
    ''' calculate secodnary energy with multi-processing '''
    p = Pool(processn)
    se_3utr = p.map(evaluate_rna_fold, indexs)
    se_3utr = sq.flatten(se_3utr)
    p.close()
    se_3utr_out = pd.DataFrame(
        se_3utr, index=df.index, columns=['secondaryE.3UTR'])
    se_3utr_out.to_pickle(
        '../Data/feature/secondaryE-{0}-3utr.pickle'.format("HS"))

    """
    make cnt feature in Con condition
    """
    df = pd.read_pickle(con_sequence_path)
    y = pd.read_pickle(con_PR_path)
    ngrams = sq.ngram_list(n)  # n-grams whose lengths are less than three
    # n-grams whose lengths are three
    ncodons = [ngram for ngram in sq.ngram_list(3) if len(ngram) == 3]

    def cnter_5utr(name_list):
        '''
        count the n-gram appearence in CDS.
        name_list means the list containing the Name of mRNA
        and this function returns the list containing the list of counters in
        ngram_list.
        '''
        rcnts = []  # list to contain the counters of n-grams on each 5'UTR
        for name in name_list:
            ''' loop in Name '''
            seqs = df['5UTR'][name]  # 5'UTRs which have same name
            # list containing the series of coutners of n-grams
            cnts = [seqs.count(ngram) for ngram in ngrams]
            rcnts.append(cnts)
        return rcnts

    def cnter_cds(name_list):
        '''
        count the n-gram appearence in CDS.
        name_list means the list containing the Name of mRNA
        and this function returns the list containing the list of counters in
        ngram_list.
        '''
        rcnts = []  # list to contain the counters of n-grams on each CDS
        for name in name_list:
            ''' loop in Name '''
            seqs = df['CDS'][name]  # CDSs which have same name
            # list containing the series of coutners of n-grams
            cnts = [seqs.count(ngram) for ngram in ncodons]
            rcnts.append(cnts)
        return rcnts

    def cnter_3utr(name_list):
        '''
        count the n-gram appearence in CDS.
        name_list means the list containing the Name of mRNA
        and this function returns the list containing the list of counters in
        ngram_list.
        '''
        rcnts = []  # list to contain the counters of n-grams on each 3'UTR
        for name in name_list:
            ''' loop in Name '''
            seqs = df['3UTR'][name]  # 3'UTRs which have same name
            # list containing the series of coutners of n-grams
            cnts = [seqs.count(ngram) for ngram in ngrams]
            rcnts.append(cnts)
        return rcnts

    names = df.index  # index of originl dataframe
    step = int(len(names)/processn)
    indexs = [names[i: i+step] for i in range(0, len(names), step)]
    p = Pool(processn)  # make process for multi-processing

    cols_5utr = [i+'.5UTR' for i in ngrams]
    cols_cds = [i+'.CDS' for i in ncodons]
    cols_3utr = [i+'.3UTR' for i in ngrams]

    """ 5UTR """
    cnts_5utr = p.map(cnter_5utr, indexs)
    cnts_5utr = sq.flatten(cnts_5utr)
    cnts_5utr = pd.DataFrame(cnts_5utr, index=names, columns=cols_5utr)
    cnts_5utr.to_pickle(
        '../Data/feature/cnts_3-gram-5utr-{0}.pickle'.format("Con"))

    """ CDS """
    cnts_cds = p.map(cnter_cds, indexs)
    cnts_cds = sq.flatten(cnts_cds)
    cnts_cds = pd.DataFrame(cnts_cds, index=names, columns=cols_cds)
    cnts_cds.to_pickle(
        '../Data/feature/cnts_3-gram-cds-{0}.pickle'.format("Con"))

    """ 3UTR """
    cnts_3utr = p.map(cnter_3utr, indexs)
    cnts_3utr = sq.flatten(cnts_3utr)
    cnts_3utr = pd.DataFrame(cnts_3utr, index=names, columns=cols_3utr)
    cnts_3utr.to_pickle(
        '../Data/feature/cnts_3-gram-3utr-{0}.pickle'.format("Con"))

    """
    make length features in Con condition
    """
    length_5utr = [len(seq) for seq in df['5UTR']]
    length_cds = [len(seq) for seq in df['CDS']]
    length_3utr = []
    for i in df['3UTR']:
        '''
        Exception because of containing N
        5UTR doesn't contain N.
        '''
        if i == 'N':
            length_3utr.append(0)
        else:
            length_3utr.append(len(i))
    length_3utr = pd.Series(length_3utr, name='3UTR')
    length = pd.DataFrame(
        {'length_5UTR': length_5utr,
         'length_CDS': length_cds,
         'length_3UTR': length_3utr},
        index=df.index)
    length.to_pickle('../Data/feature/length-{0}.pickle'.format("Con"))

    """
    make free energy features in Con condition
    """
    """ 5UTR """
    ''' divide list for multi-processing '''
    list_5utr = list(df['5UTR'])
    step = int(len(list_5utr)/processn)
    indexs = [list_5utr[i: i+step] for i in range(0, len(list_5utr), step)]
    ''' multi-processing '''
    p = Pool(processn)
    se_5utr = p.map(evaluate_rna_fold, indexs)
    se_5utr = sq.flatten(se_5utr)
    p.close()
    se_5utr_out = pd.DataFrame(
        se_5utr, index=df.index, columns=['secondaryE.5UTR'])
    se_5utr_out.to_pickle(
        '../Data/feature/secondaryE-{0}-5utr.pickle'.format("Con"))

    """ CDS """
    ''' divide list for multi-processing '''
    list_cds = list(df['5UTR'])
    step = int(len(list_cds)/processn)
    indexs = [list_cds[i: i+step] for i in range(0, len(list_cds), step)]
    ''' calculate secondary energy with multi-processing '''
    p = Pool(processn)
    se_cds = p.map(evaluate_rna_fold, indexs)
    se_cds = sq.flatten(se_cds)
    p.close()
    se_cds_out = pd.DataFrame(
        se_cds, index=df.index, columns=['secondaryE.CDS'])
    se_cds_out.to_pickle(
        '../Data/feature/secondaryE-{0}-cds.pickle'.format("Con"))

    """ 3UTR """
    ''' divide list for multi-processing '''
    list_3utr = list(df['3UTR'])
    step = int(len(list_3utr)/processn)
    indexs = [list_3utr[i: i+step] for i in range(0, len(list_3utr), step)]
    ''' calculate secodnary energy with multi-processing '''
    p = Pool(processn)
    se_3utr = p.map(evaluate_rna_fold, indexs)
    se_3utr = sq.flatten(se_3utr)
    p.close()
    se_3utr_out = pd.DataFrame(
        se_3utr, index=df.index, columns=['secondaryE.3UTR'])
    se_3utr_out.to_pickle(
        '../Data/feature/secondaryE-{0}-3utr.pickle'.format("Con"))

    """
    make predictor by random forest in HS condition
    """
    """ import datasets """
    cnts_5utr = pd.read_pickle('../Data/feature/cnts_3-gram-5utr-HS.pickle')
    cnts_cds = pd.read_pickle('../Data/feature/cnts_3-gram-cds-HS.pickle')
    cnts_3utr = pd.read_pickle('../Data/feature/cnts_3-gram-3utr-HS.pickle')
    length = pd.read_pickle('../Data/feature/length-HS.pickle')
    se_5utr = pd.read_pickle('../Data/feature/secondaryE-HS-5utr.pickle')
    se_cds = pd.read_pickle('../Data/feature/secondaryE-HS-cds.pickle')
    se_3utr = pd.read_pickle('../Data/feature/secondaryE-HS-3utr.pickle')
    y = pd.read_pickle('../Data/feature/y-HS.pickle')

    def mk_rfr(feature, y):
        """ make feature dataframe for learning """
        x_arr = feature.values
        y_arr = y.values
        """ split dataset """
        np.random.seed(10)
        sampler = np.random.permutation(len(x_arr))
        test_size = 5446  # 25% of dataset
        train_index = sampler[2*test_size:]
        vali_index = sampler[test_size: 2*test_size]
        test_index = sampler[:test_size]
        ''' training set '''
        x_train = x_arr[train_index, :]
        y_train = y_arr[train_index]
        ''' validation set '''
        x_vali = x_arr[vali_index, :]
        y_vali = y_arr[vali_index]
        ''' test set '''
        x_test = x_arr[test_index, :]
        y_test = y_arr[test_index]

        '''
        Tune n_estimators
        '''
        paras = range(10, 100, 10)
        err_train = []
        err_vali = []

        for para in paras:
            rfr = RandomForestRegressor(
                n_estimators=para, n_jobs=processn, random_state=15)
            rfr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(rfr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(rfr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        n_esti = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                n_esti = paras[i+1]
        print("Turning n_estimators...")
        print(err)
        print('selected n_estimator:{0}'.format(n_esti))
        err.plot()
        plt.savefig("Turning-HS_Random-Forest_n_estimators-{0}".format(tstr))

        '''
        Tune max_depth
        '''
        paras = range(1, 32, 5)
        err_train = []
        err_vali = []

        for para in paras:
            rfr = RandomForestRegressor(
                n_estimators=n_esti,
                max_depth=para, n_jobs=processn, random_state=15)
            rfr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(rfr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(rfr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        maxd = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                maxd = paras[i+1]
        print("Turning max_depth...")
        print(err)
        print('selected max_depth:{0}'.format(maxd))
        err.plot()
        plt.savefig("Turning-HS_Random-Forest_max_depth-{0}".format(tstr))

        """
        Grid Searech Cross Validation
        """
        params = {'n_estimators': range(max([1, n_esti-10]), n_esti+10, 5),
                  'max_depth': range(max([1, maxd-5]), maxd+5, 1)}
        rfr = RandomForestRegressor(random_state=15)
        gsrfr = GridSearchCV(rfr, params, n_jobs=processn)
        gsrfr.fit(x_train, y_train)
        print(
            'validation MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_vali), y_vali)))
        print(
            'test MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_test), y_test)))
        joblib.dump(gsrfr, '../Models/Random-Forest_HS-{0}'.format(tstr))
        return 0

    """ using cnt, length and secondaryEs (i.e. using all of feature) """
    feature = pd.concat([cnts_5utr, cnts_cds, cnts_3utr, length,
                         se_5utr, se_cds, se_3utr],
                        axis=1)
    mk_rfr(feature, y)

    """
    make predictor by gradient boosting in HS condition
    """
    def mk_gbr(feature, y):
        """ make feature dataframe for learning """
        x_arr = feature.values
        y_arr = y.values
        """ split dataset """
        np.random.seed(10)
        sampler = np.random.permutation(len(x_arr))
        test_size = 5446  # 25% of dataset
        train_index = sampler[2*test_size:]
        vali_index = sampler[test_size: 2*test_size]
        test_index = sampler[:test_size]
        ''' training set '''
        x_train = x_arr[train_index, :]
        y_train = y_arr[train_index]
        ''' validation set '''
        x_vali = x_arr[vali_index, :]
        y_vali = y_arr[vali_index]
        ''' test set '''
        x_test = x_arr[test_index, :]
        y_test = y_arr[test_index]

        '''
        Tune n_estimators
        '''
        paras = range(35, 51, 5)
        err_train = []
        err_vali = []

        for para in paras:
            gbr = GradientBoostingRegressor(
                n_estimators=para, n_jobs=processn, random_state=15)
            gbr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(gbr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(gbr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        n_esti = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                n_esti = paras[i+1]
        print("Turning n_estimators...")
        print(err)
        print('selected n_estimator:{0}'.format(n_esti))
        err.plot()
        plt.savefig(
            "Turning-HS_Gradient-Boosting_n_estimators-{0}".format(tstr))

        '''
        Tune max_depth
        '''
        paras = range(9, 15, 1)
        err_train = []
        err_vali = []

        for para in paras:
            rfr = RandomForestRegressor(
                n_estimators=n_esti,
                max_depth=para, n_jobs=processn, random_state=15)
            rfr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(rfr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(rfr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        maxd = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                maxd = paras[i+1]
        print("Turning max_depth...")
        print(err)
        print('selected max_depth:{0}'.format(maxd))
        err.plot()
        plt.savefig("Turning-HS_Gradient-Boosting_max_depth-{0}".format(tstr))

        """
        Grid Searech Cross Validation
        """
        params = {'n_estimators': range(max([1, n_esti-10]), n_esti+10, 5),
                  'max_depth': range(max([1, maxd-5]), maxd+5, 1)}
        gbr = GradientBoostingRegressor(random_state=15)
        gsgbr = GridSearchCV(gbr, params, n_jobs=processn)
        gsgbr.fit(x_train, y_train)
        print(
            'validation MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_vali), y_vali)))
        print(
            'test MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_test), y_test)))
        joblib.dump(gsgbr, '../Models/Gradient-Boosting_HS-{0}'.format(tstr))
        return 0

    mk_gbr(feature, y)

    """
    build the predictor by XGBoost
    """
    def mk_xgbr(feature, y):
        """ parameters """
        tdatetime = dt.now()  # get time
        tstr = tdatetime.strftime('%Y%m%d%H%M')  # time for model nam
        print("Model number: {0}".format(tstr))
        """ make feature dataframe for learning """
        x_arr = feature.values
        y_arr = y.values
        """ split dataset """
        np.random.seed(10)
        sampler = np.random.permutation(len(x_arr))
        test_size = 5446  # 25% of dataset
        train_index = sampler[2*test_size:]
        vali_index = sampler[test_size: 2*test_size]
        test_index = sampler[:test_size]
        ''' training set '''
        x_train = x_arr[train_index, :]
        y_train = y_arr[train_index]
        ''' validation set '''
        x_vali = x_arr[vali_index, :]
        y_vali = y_arr[vali_index]
        ''' test set '''
        x_test = x_arr[test_index, :]
        y_test = y_arr[test_index]

        '''
        Tune n_estimators
        '''
        paras = [2**i for i in range(2, 16)]
        err_train = []
        err_vali = []

        for para in paras:
            xgbr = xgb.XGBRegressor(
                n_estimators=para, seed=15, nthread=processn)
            xgbr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(
                mean_squared_error(xgbr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(xgbr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        n_esti = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                n_esti = paras[i+1]
        print("Turning n_estimators...")
        print(err)
        print('selected n_estimator:{0}'.format(n_esti))
        err.plot()
        plt.savefig(
            "../Figure/Turning-HS_XGBoosting-n_estimators-{0}.pdf".format(
                tstr))

        '''
        Tune max_depth
        '''
        paras = range(1, 47, 5)
        err_train = []
        err_vali = []

        for para in paras:
            xgbr = xgb.XGBRegressor(
                n_estimators=n_esti,
                max_depth=para, nthread=processn, seed=15)
            xgbr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(
                mean_squared_error(xgbr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(xgbr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        maxd = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                maxd = paras[i+1]
        print("Turning max_depth...")
        print(err)
        print('selected max_depth:{0}'.format(maxd))
        err.plot()
        plt.savefig(
            "../Figure/Turning-HS_XGBoosting-max_depth-{0}.pdf".format(
                tstr))

        """
        Grid Searech Cross Validation
        """
        params = {'n_estimators': range(max([1, n_esti-50]), n_esti+50, 10),
                  'max_depth': range(max([1, maxd-5]), maxd+5, 1)}
        xgbr = xgb.XGBRegressor(seed=15)
        gsxgbr = GridSearchCV(xgbr, params, n_jobs=processn)
        gsxgbr.fit(x_train, y_train)
        print(
            'validation MSE:{0}'.format(
                mean_squared_error(gsxgbr.predict(x_vali), y_vali)))
        print(
            'test MSE:{0}'.format(
                mean_squared_error(gsxgbr.predict(x_test), y_test)))
        joblib.dump(gsxgbr, '../Models/XGBoosting_HS-{0}.pkl'.format(tstr))
        return 0

    """ using cnt, length and secondaryEs (i.e. using all of feature) """
    feature = pd.concat([cnts_5utr, cnts_cds, cnts_3utr, length,
                         se_5utr, se_cds, se_3utr],
                        axis=1)
    mk_xgbr(feature, y)

    """
    make predictor by random forest in Con condition
    """
    """ import datasets """
    cnts_5utr = pd.read_pickle('../Data/feature/cnts_3-gram-5utr-Con.pickle')
    cnts_cds = pd.read_pickle('../Data/feature/cnts_3-gram-cds-Con.pickle')
    cnts_3utr = pd.read_pickle('../Data/feature/cnts_3-gram-3utr-Con.pickle')
    length = pd.read_pickle('../Data/feature/length-Con.pickle')
    se_5utr = pd.read_pickle('../Data/feature/secondaryE-Con-5utr.pickle')
    se_cds = pd.read_pickle('../Data/feature/secondaryE-Con-cds.pickle')
    se_3utr = pd.read_pickle('../Data/feature/secondaryE-Con-3utr.pickle')
    y = pd.read_pickle('../Data/feature/y-Con.pickle')

    def mk_rfr(feature, y):
        """ make feature dataframe for learning """
        x_arr = feature.values
        y_arr = y.values
        """ split dataset """
        np.random.seed(10)
        sampler = np.random.permutation(len(x_arr))
        test_size = 5446  # 25% of dataset
        train_index = sampler[2*test_size:]
        vali_index = sampler[test_size: 2*test_size]
        test_index = sampler[:test_size]
        ''' training set '''
        x_train = x_arr[train_index, :]
        y_train = y_arr[train_index]
        ''' validation set '''
        x_vali = x_arr[vali_index, :]
        y_vali = y_arr[vali_index]
        ''' test set '''
        x_test = x_arr[test_index, :]
        y_test = y_arr[test_index]

        '''
        Tune n_estimators
        '''
        paras = range(10, 100, 10)
        err_train = []
        err_vali = []

        for para in paras:
            rfr = RandomForestRegressor(
                n_estimators=para, n_jobs=processn, random_state=15)
            rfr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(rfr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(rfr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        n_esti = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                n_esti = paras[i+1]
        print("Turning n_estimators...")
        print(err)
        print('selected n_estimator:{0}'.format(n_esti))
        err.plot()
        plt.savefig("Turning-Con_Random-Forest_n_estimators-{0}".format(tstr))

        '''
        Tune max_depth
        '''
        paras = range(1, 32, 5)
        err_train = []
        err_vali = []

        for para in paras:
            rfr = RandomForestRegressor(
                n_estimators=n_esti,
                max_depth=para, n_jobs=processn, random_state=15)
            rfr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(rfr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(rfr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        maxd = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                maxd = paras[i+1]
        print("Turning max_depth...")
        print(err)
        print('selected max_depth:{0}'.format(maxd))
        err.plot()
        plt.savefig("Turning-Con_Random-Forest_max_depth-{0}".format(tstr))

        """
        Grid Searech Cross Validation
        """
        params = {'n_estimators': range(max([1, n_esti-10]), n_esti+10, 5),
                  'max_depth': range(max([1, maxd-5]), maxd+5, 1)}
        rfr = RandomForestRegressor(random_state=15)
        gsrfr = GridSearchCV(rfr, params, n_jobs=processn)
        gsrfr.fit(x_train, y_train)
        print(
            'validation MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_vali), y_vali)))
        print(
            'test MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_test), y_test)))
        joblib.dump(gsrfr, '../Models/Random-Forest_Con-{0}'.format(tstr))
        return 0

    """ using cnt, length and secondaryEs (i.e. using all of feature) """
    feature = pd.concat([cnts_5utr, cnts_cds, cnts_3utr, length,
                         se_5utr, se_cds, se_3utr],
                        axis=1)
    mk_rfr(feature, y)

    """
    make predictor by gradient boosting in HS condition
    """
    def mk_gbr(feature, y):
        """ make feature dataframe for learning """
        x_arr = feature.values
        y_arr = y.values
        """ split dataset """
        np.random.seed(10)
        sampler = np.random.permutation(len(x_arr))
        test_size = 5446  # 25% of dataset
        train_index = sampler[2*test_size:]
        vali_index = sampler[test_size: 2*test_size]
        test_index = sampler[:test_size]
        ''' training set '''
        x_train = x_arr[train_index, :]
        y_train = y_arr[train_index]
        ''' validation set '''
        x_vali = x_arr[vali_index, :]
        y_vali = y_arr[vali_index]
        ''' test set '''
        x_test = x_arr[test_index, :]
        y_test = y_arr[test_index]

        '''
        Tune n_estimators
        '''
        paras = range(35, 51, 5)
        err_train = []
        err_vali = []

        for para in paras:
            gbr = GradientBoostingRegressor(
                n_estimators=para, n_jobs=processn, random_state=15)
            gbr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(gbr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(gbr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        n_esti = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                n_esti = paras[i+1]
        print("Turning n_estimators...")
        print(err)
        print('selected n_estimator:{0}'.format(n_esti))
        err.plot()
        plt.savefig(
            "Turning-Con_Gradient-Boosting_n_estimators-{0}".format(tstr))

        '''
        Tune max_depth
        '''
        paras = range(9, 15, 1)
        err_train = []
        err_vali = []

        for para in paras:
            rfr = RandomForestRegressor(
                n_estimators=n_esti,
                max_depth=para, n_jobs=processn, random_state=15)
            rfr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(mean_squared_error(rfr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(rfr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        maxd = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                maxd = paras[i+1]
        print("Turning max_depth...")
        print(err)
        print('selected max_depth:{0}'.format(maxd))
        err.plot()
        plt.savefig("Turning-Con_Gradient-Boosting_max_depth-{0}".format(tstr))

        """
        Grid Searech Cross Validation
        """
        params = {'n_estimators': range(max([1, n_esti-10]), n_esti+10, 5),
                  'max_depth': range(max([1, maxd-5]), maxd+5, 1)}
        gbr = GradientBoostingRegressor(random_state=15)
        gsgbr = GridSearchCV(gbr, params, n_jobs=processn)
        gsgbr.fit(x_train, y_train)
        print(
            'validation MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_vali), y_vali)))
        print(
            'test MSE:{0}'.format(
                mean_squared_error(rfr.predict(x_test), y_test)))
        joblib.dump(gsgbr, '../Models/Gradient-Boosting_Con-{0}'.format(tstr))
        return 0

    mk_gbr(feature, y)

    """
    build the predictor by XGBoost
    """
    def mk_xgbr(feature, y):
        """ parameters """
        tdatetime = dt.now()  # get time
        tstr = tdatetime.strftime('%Y%m%d%H%M')  # time for model nam
        print("Model number: {0}".format(tstr))
        """ make feature dataframe for learning """
        x_arr = feature.values
        y_arr = y.values
        """ split dataset """
        np.random.seed(10)
        sampler = np.random.permutation(len(x_arr))
        test_size = 5446  # 25% of dataset
        train_index = sampler[2*test_size:]
        vali_index = sampler[test_size: 2*test_size]
        test_index = sampler[:test_size]
        ''' training set '''
        x_train = x_arr[train_index, :]
        y_train = y_arr[train_index]
        ''' validation set '''
        x_vali = x_arr[vali_index, :]
        y_vali = y_arr[vali_index]
        ''' test set '''
        x_test = x_arr[test_index, :]
        y_test = y_arr[test_index]

        '''
        Tune n_estimators
        '''
        paras = [2**i for i in range(2, 16)]
        err_train = []
        err_vali = []

        for para in paras:
            xgbr = xgb.XGBRegressor(
                n_estimators=para, seed=15, nthread=processn)
            xgbr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(
                mean_squared_error(xgbr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(xgbr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        n_esti = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                n_esti = paras[i+1]
        print("Turning n_estimators...")
        print(err)
        print('selected n_estimator:{0}'.format(n_esti))
        err.plot()
        plt.savefig(
            "../Figure/Turning-Con_XGBoosting-n_estimators-{0}.pdf".format(
                tstr))

        '''
        Tune max_depth
        '''
        paras = range(1, 47, 5)
        err_train = []
        err_vali = []

        for para in paras:
            xgbr = xgb.XGBRegressor(
                n_estimators=n_esti,
                max_depth=para, nthread=processn, seed=15)
            xgbr.fit(x_train, y_train)
            ''' MSE '''
            err_train.append(
                mean_squared_error(xgbr.predict(x_train), y_train))
            err_vali.append(mean_squared_error(xgbr.predict(x_vali), y_vali))
        err = pd.DataFrame(
            {'train': err_train, 'validation': err_vali}, index=paras)

        maxd = paras[0]
        for i in range(len(paras)-1):
            j = paras[i]
            j_next = paras[i+1]
            if err['validation'][j_next] < err['validation'][j]:
                maxd = paras[i+1]
        print("Turning max_depth...")
        print(err)
        print('selected max_depth:{0}'.format(maxd))
        err.plot()
        plt.savefig(
            "../Figure/Turning-Con_XGBoosting-max_depth-{0}.pdf".format(
                tstr))

        """
        Grid Searech Cross Validation
        """
        params = {'n_estimators': range(max([1, n_esti-50]), n_esti+50, 10),
                  'max_depth': range(max([1, maxd-5]), maxd+5, 1)}
        xgbr = xgb.XGBRegressor(seed=15)
        gsxgbr = GridSearchCV(xgbr, params, n_jobs=processn)
        gsxgbr.fit(x_train, y_train)
        print(
            'validation MSE:{0}'.format(
                mean_squared_error(gsxgbr.predict(x_vali), y_vali)))
        print(
            'test MSE:{0}'.format(
                mean_squared_error(gsxgbr.predict(x_test), y_test)))
        joblib.dump(gsxgbr, '../Models/XGBoosting_Con-{0}.pkl'.format(tstr))
        return 0

    """ using cnt, length and secondaryEs (i.e. using all of feature) """
    feature = pd.concat([cnts_5utr, cnts_cds, cnts_3utr, length,
                         se_5utr, se_cds, se_3utr],
                        axis=1)
    mk_xgbr(feature, y)
