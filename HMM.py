import math
from collections import defaultdict

from utils import read_corpus, as_bigram_tag

def _to_log_prob(pos2words, transition, bos):
    """계산의 편의를 위해 곱셈을 덧셈으로 변경"""
    
    # 품사별 단어 등장 확률 (로그)
    base = {pos:sum(words.values()) for pos, words in pos2words.items()}
    pos2words_ = {pos:{word:math.log(count/base[pos]) for word, count in words.items()}
                      for pos, words in pos2words.items()}
    
    # transition 확률 (로그)
    base = defaultdict(int)
    for (pos0, pos1), count in transition.items():
        base[pos0] += count
    transition_ = {pos:math.log(count/base[pos[0]]) for pos, count in transition.items()}
    
    # 시작 확률
    base = sum(bos.values())
    bos_ = {pos:math.log(count/base) for pos, count in bos.items()}
    
    return pos2words_, transition_, bos_

class TrainedHMM:
    def __init__(self, pos2words, transition, bos):
        self.pos2words = pos2words
        self.transition = transition
        self.bos = bos
        
        self.unknown_penalty = -15 # 모르는 단어에 대한 페널티 (-infty의 대체값)
        self.eos_tag = 'EOS'
        
    def sentence_log_prob(self, sent):
        # emission probability
        log_prob = sum(
            (self.pos2words.get(t, {}).get(w, self.unknown_penalty)
            for w, t in sent)
        )
        
        # bos
        log_prob += self.bos.get(sent[0][1], self.unknown_penalty)
        
        # transition prob
        bigrams = [(t0, t1) for (_, t0), (_, t1) in zip(sent, sent[1:])]
        log_prob += sum(
            (self.transition.get(bigram, self.unknown_penalty)
            for bigram in bigrams)
        )
        
        # eos
        log_prob += self.transition.get(
            (sent[-1][1], self.eos_tag), self.unknown_penalty
        )
        
        # length normalization
        log_prob /= len(sent)
        
        return log_prob
    
    
if __name__ == '__main__':

    DATA_PATH = 'data/corpus_type3_all.txt'
    corpus = read_corpus(DATA_PATH)

    pos2words = defaultdict(lambda: defaultdict(int)) # 품사별로 단어 빈도 기록
    trans = defaultdict(int) # transition 기록
    bos = defaultdict(int) # 문장이 시작할 떄 state 횟수 기록

    # sent = [(word, tag), (word, tag), ...]
    for sent in corpus:

        # generation prob
        for word, pos in sent:
            pos2words[pos][word] += 1

        # transition prob
        for bigram in as_bigram_tag(sent):
            trans[bigram] += 1

        # begin prob (BOS -> tag)
        bos[sent[0][1]] += 1

        # end prob (tag -> EOS)
        trans[(sent[-1][1], 'EOS')] += 1

    pos2words_, trans_, bos_ = _to_log_prob(pos2words, trans, bos)
    trained_hmm = TrainedHMM(pos2words_, trans_, bos_)
    
    candidates = [
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')],
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Noun')],
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')],
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    ]

    for sent in candidates:    
        print('\n{}'.format(sent))
        print(trained_hmm.sentence_log_prob(sent))