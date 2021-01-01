import math
import json
import argparse
from collections import defaultdict

from utils import read_corpus, as_bigram_tag

# Argparse setting
parser = argparse.ArgumentParser(description="세종 말뭉치 품사/단어별 빈도수 산출")

# Training argument
parser.add_argument('--data_path', type=str, default='data/corpus_type1_all.txt')
parser.add_argument('--num_lines', type=int, default=-1)
parser.add_argument('--save_path', type=str, default='data/trained_corpus_type1.json')

def _to_log_prob(pos2words, transition, bos):
    """계산의 편의를 위해 곱셈을 덧셈으로 변경"""

    # 품사별 단어 등장 확률 (로그)
    base = {pos:sum(words.values()) for pos, words in pos2words.items()}
    pos2words_ = {pos:{word:math.log(count/base[pos]) for word, count in words.items()}
                      for pos, words in pos2words.items()}

    # transition 확률 (로그)
    base = defaultdict(int)
    for pos0_pos1, count in transition.items():
        base[pos0_pos1.split("_")[0]] += count
    transition_ = {pos:math.log(count/base[pos0_pos1.split("_")[0]]) for pos, count in transition.items()}

    # 시작 확률
    base = sum(bos.values())
    bos_ = {pos:math.log(count/base) for pos, count in bos.items()}

    return pos2words_, transition_, bos_

def train(corpus):
    """말뭉치를 불러온 후 품사별 단어 빈도, 품사간 연속 빈도, 시작 빈도를 계산하여 리턴"""
    pos2words = defaultdict(lambda: defaultdict(int))
    trans = defaultdict(int)
    bos = defaultdict(int)

    # sent = [(word, tag), (word, tag), ...]
    for sent in corpus:

        # Count pos to words frequencies
        for word, pos in sent:
            pos2words[pos][word] += 1

        # Count transition frequencies
        for bigram in as_bigram_tag(sent):
            trans[bigram] += 1

        # Count beginning pos frequencies
        bos[sent[0][1]] += 1

        # Count EOS frequencies
        trans['_'.join([sent[-1][1], 'EOS'])] += 1

    pos2words_, transition_, bos_ = _to_log_prob(pos2words, trans, bos)
    trained = dict()
    trained['emission'] = pos2words_
    trained['transition'] = transition_
    trained['begin'] = bos_

    return trained

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data_path
    num_lines = args.num_lines
    save_path = args.save_path

    # 데이터 로드
    print("Data Loading...")
    corpus = read_corpus(data_path, num_lines)
    trained = train(corpus)

    print("Data Saving...")
    with open(save_path, 'w') as f:
        json.dump(trained, f)

    print("Save finished.")
