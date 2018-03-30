'''
This script contains the original functions.
'''
import itertools
import numpy as np
import pandas as pd
from scipy import stats
import re
import copy


def zero_padding(oharray):
    '''
    This method do the zero padding.
    The argument oharray is the one-hot-arrrays which is the list
    containing the arrays which is the one-hot-array whose rows
    are the one-hot-vector.
    '''
    # the list containing the length of sequences
    tmp = [i.shape[0] for i in oharray]
    tmp1 = []  # list for containing the zero-paddided one-hot-array
    maximum_len = max(tmp)  # the maximum value of length
    for i in oharray:
        ''' Add the 0 array to the elements of oharray '''
        # zeromatrix to add the one-hot-array
        zerom = np.zeros(
            (maximum_len-i.shape[0], i.shape[1]), dtype=np.float32)
        tmp1.append(np.r_[i, zerom])
    return np.array(tmp1, dtype=np.float32)


def ngram_list(n):
    '''
    This function generate the n-gram from permutaion of 'ACGT'.
    '''
    ''' generate the permutation of n strings '''
    ngrams = []
    for i in range(1, n+1):
        ''' make the permutation of i<=n sequences '''
        # the list containing permutaion of i strings
        tmp = itertools.product('ACGT', repeat=i)
        ngrams.extend(tmp)
    ''' join the strings '''
    ngrams = [''.join(i) for i in ngrams]  # list containing the n-grams
    return ngrams


def ncodon_df(df, codons, cname, m):
    '''
    mRNA配列のデータフレームdfとn-codonのリストを引数にとり，
    n-codonの頻度データフレームを出力する．
    cnameはmRNA配列が含まれている列名．
    関数内で並列処理を行っている．並列のプロセス数は引数mで変更する．
    '''
    from multiprocessing import Pool
    codons_array = np.array(df[cname])  # データフレームをarrayに変換
    tmp_len = len(codons)  # 配列のサンプル数
    step = int(tmp_len/m)  # 並列処理用インデックスのステップ
    # 並列処理用インデックス
    indexs = [range(tmp_len)[x:x+step] for x in range(0, tmp_len, step)]

    def multifunc(indexl):
        '''
        並列処理用の関数.
        与えられたインデックスのmRNA配列のn-codon頻度行列用のリストを返す．
        '''
        frequencies = []
        for i in indexl:
            ''' i番目のmRNAでのn-codonの頻度をカウントする '''
            frequence_codons = [codons_array[i].count(j) for j in codons]
            # i番目のmRNAのn-codon頻度を，全体のデータフレーム用リストに追加
            frequencies.append(frequence_codons)
        return frequencies

    p = Pool(m)  # 並列処理用のプロセス生成
    frequence_mat = p.map(multifunc, indexs)  # indexsごとに並列処理
    tmp = frequence_mat[0]  # 頻度配列生成用リストの初期値
    for i in range(1, len(frequence_mat)):
        ''' 頻度配列生成用リスト作成 '''
        tmp = tmp + frequence_mat[i]
    tmp = np.array(tmp)
    tmp = pd.DataFrame(tmp, index=df.index, columns=codons)
    return tmp


def part_frequency(sequence, m, ngram_list):
    '''
    配列をm個に等分し、その配列内でのn-gramリストに含まれるn-gramの出現頻度を計算する
    sequenceはstring型
    '''
    ''' devide the sequence into m-parts '''
    part_len = int(len(sequence)/m)
    if m > len(sequence):
        ''' mが文字列よりも長い場合は、文字列全体を部分文字列とする '''
        sub_sequences = [sequence]
    else:
        sub_sequences = [sequence[i: i+part_len] for i in range(
            0, len(sequence), part_len)]
    ''' 部分文字列でのn-gramの出現回数(forが外側から回ることに注意) '''
    frequencies = [i.count(j) for i in sub_sequences for j in ngram_list]
    return frequencies


def cs(seq1, seq2):
    '''
    This function find the common sequence between seq1 and se2.
    '''
    if len(seq1) > len(seq2):
        ''' the length of sequence1 is less than one of sequence2 '''
        tmp_seq = seq1
        seq1 = seq2
        seq2 = tmp_seq
    cseq = []  # for keeping the common sequences
    for length in range(len(seq1), 0, -1):
        ''' length is the length of substring '''
        for p0 in range(len(seq1) - length + 1):
            ''' p0 is the position of starting the substring '''
            substr = seq1[p0:(p0 + length)]
            if substr in seq2:
                cseq.append(substr)
    return cseq


def multinomial_sampling(names, probs):
    '''
    names is the results and probs is probabilitys of each results.
    In my work, names mean the list of ids of one name.
    '''
    choices = np.random.multinomial(1, probs)
    choice = np.where(choices == 1)[0][0]  # index of samplig
    name = names[choice]
    return name


def mkprobs(probs):
    '''
    make probability list which containing sum of the list equal to 1.
    The argument is the list containing percents of each event
    '''
    probs_tmp = np.array(probs)
    max_posi = np.argmax(probs_tmp)  # position of argmax
    min_posi = np.argmin(probs_tmp)  # position of argmin
    if sum(probs) < 1:
        probs[min_posi] += 1 - sum(probs)
    elif sum(probs) > 1:
        probs[max_posi] -= sum(probs) - 1
    return probs


def flatten(nested_list):
    """ 2重のリストをフラットにする関数 """
    return [e for inner_list in nested_list for e in inner_list]


def seq_to_sentence(string, n):
    '''
    translate mRNA sequence to sentence containing space
    '''
    strings = [string[i: i+n] for i in range(0, len(string), n)]
    sentence = ' '.join(strings)
    return sentence


def cnt_cdn(cdn, seqs):
    '''
    This function counts the appearance of codon in sequence.
    The argument seq is the series of RNA sequences.
    '''
    ''' counter of appearence of cdn '''
    cnt = sum([len(re.findall(cdn, seq+'n')) for seq in seqs])
    return cnt


def rm_continuous_strings(string, pattern):
    '''
    The argument is the sequence of 5'UTR, CDS or 3'UTR.
    The string splited by continuous strings,
    for example AAA... or CCC..., are returned.
    '''
    strings = re.split(pattern, string)
    while (True):
        try:
            strings.remove("")
        except:
            break
    sentence = ' '.join(strings)  # the sentence joined strings
    return (sentence, strings)
