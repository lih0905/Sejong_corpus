import json
import argparse

from soynlp.lemmatizer import lemma_candidate

# Argparse setting
parser = argparse.ArgumentParser(description="세종 말뭉치 이용 품사 분석 입력")

# argument
parser.add_argument('--json_path', type=str, default='data/trained_corpus_type1.json')
parser.add_argument('--text', type=str, required=True)

def load_from_json(json_path):
    """훈련된 json 데이터 불러오기"""
    with open(json_path, 'r') as f:
        trained = json.load(f)
    emission = trained['emission']
    transition = trained['transition']
    begin = trained['begin']

    transition = {tuple(k.split("_")):v for k, v in transition.items()}
    return emission, transition, begin

def ford_list(E, V, S, T):
    """포드 알고리즘 구현 - log를 취한 확률이므로 longest path 찾도록 구현"""

    ## Initialize ##
    # (max weight + 1) * num of nodes
    inf = (min((weight for from_, to_, weight in E)) - 1) * len(V)

    # distance
    d = {node:0 if node == S else inf for node in V}
    # previous node
    prev = {node:None for node in V}

    ## Iteration ##
    # preventing infinite loop
    for _ in range(len(V)):
        # for early stop
        changed = False
        for u, v, Wuv in E:
            d_new = d[u] + Wuv
            if d_new > d[v]:
                d[v] = d_new
                prev[v] = u
                changed = True
        if not changed:
            break

    # Checking negative cycle loop
    for u, v, Wuv in E:
        if d[u] + Wuv > d[v]:
            raise ValueError('Cycle exists')

    # Finding path
    prev_ = prev[T]
    if prev_ == S:
        return {'paths':[[prev_, S][::-1]], 'cost':d[T]}

    path = [T]
    while prev_ != S:
        path.append(prev_)
        prev_ = prev[prev_]
    path.append(S)

    return path[::-1], d[T]

class HMMTagger:
    def __init__(self, emission, transition, begin):
        self.emission = emission
        self.transition = transition
        self.begin = begin
        self._max_word_len = max(
            len(w) for words in emission.values() for w in words
        )
        self._min_emission = min(
            s for words in emission.values() for s in words.values()) - 0.05
        self._min_transition = min(transition.values()) - 0.05

    def tag(self, sentence):
        # lookup & generate graph
        links, bos, eos = self._generate_link(sentence)
        graph = self._add_weight(links)

        # find optimal path
        nodes = {node for edge in graph for node in edge[:2]}
        path, cost = ford_list(graph, nodes, bos, eos)
        pos = self._flatten(path)

        # infering tag of unknown words
        pos = self._inference_unknown(pos)

        # post processing
        pos = self._postprocessing(pos)

        return pos

    def _sentence_lookup(self, sentence):
        """문장을 어절 단위로 분해"""
        sent = []
        for eojeol in sentence.split():
            sent += self._eojeol_lookup(eojeol, offset=len(sent))
        #print(sent)
        return sent

    def _eojeol_lookup(self, eojeol, offset=0):
        """어절/품사로 들어온 입력을 분리하여 단어/품사/품사/시작/끝 형태의
        튜플로 반환"""
        n = len(eojeol)
        pos = [[] for _ in range(n)]
        for b in range(n):
            for r in range(1, self._max_word_len+1):
                e = b + r
                if e > n:
                    continue
                surface = eojeol[b:e]
                for tag in self._get_pos(surface):
                    pos[b].append((surface, tag, tag, b+offset, e+offset))
                # 용언 분리 시도
                for i in range(1, r+1):
                    suffix_len = r - i
                    try:
                        lemmas = self._lemmatize(surface, i)
                        if lemmas:
                            for morphs, tag0, tag1 in lemmas:
                                pos[b].append((morphs, tag0, tag1, b+offset, e+offset))
                    except:
                        continue
        return pos

    def _get_pos(self, sub):
        """주어진 어휘가 속하는 품사 전체를 리턴 """
        tags = []
        for tag, words in self.emission.items():
            if sub in words:
                tags.append(tag)
        return tags

    def _lemmatize(self, word, i):
        """주어진 용언을 어간/어미로 분리"""
        l = word[:i]
        r = word[i:]
        lemmas = []
        len_word = len(word)
        for l_, r_ in lemma_candidate(l, r):
            word_ = l_ + ' + ' + r_
            if (l_ in self.emission['Verb']) and (r_ in self.emission['Eomi']):
                lemmas.append((word_, 'Verb', 'Eomi'))
            if (l_ in self.emission['Adjective']) and (r_ in self.emission['Eomi']):
                lemmas.append((word_, 'Adjective', 'Eomi'))
        return lemmas

    def _generate_link(self, sentence):

        def get_nonempty_first(sent, end, offset=0):
            """offset으로 부터 가장 먼저 등장하는 단어 인덱스 반환"""
            for i in range(offset, end):
                if sent[i]:
                    return i
            return offset

        chars = sentence.replace(' ', '')
        sent = self._sentence_lookup(sentence)
        n_char = len(sent) + 1

        # EOS 등록
        eos = ('EOS', 'EOS', 'EOS', n_char-1, n_char)
        sent.append([eos])

        # 첫 단어가 등장하는 인덱스가 0보다 크면
        # 그 인덱스까지를 Unk로 등록
        i = get_nonempty_first(sent, n_char)
        if i > 0:
            sent[0].append((chars[:i], 'Unk', 'Unk', 0, i))

        links = []
        for words in sent[:-1]:
            for word in words:
                begin = word[3]
                end = word[4]
                # 현재 단어의 끝점에서 시작하는 단어가 없는 경우는
                # 그 끝점 이후 처음으로 등장하는 단어의 시작점까지를
                # Unk로 등록
                if not sent[end]:
                    b = get_nonempty_first(sent, n_char, end)
                    unk = (chars[end:b], 'Unk', 'Unk', end, b)
                    links.append((word, unk))
                # 아닌 경우는 단어들을 edge로 등록
                else:
                    for adjacent in sent[end]:
                        links.append((word, adjacent))

        # Unk로 시작하는 edge 등록
        unks = {to_node for _, to_node in links if to_node[1] == 'Unk'}
        for unk in unks:
            for adjacent in sent[unk[3]]:
                links.append((unk, adjacent))

        # BOS 등록
        bos = ('BOS', 'BOS', 'BOS', 0, 0)
        for word in sent[0]:
            links.append((bos, word))
        links = sorted(links, key=lambda x:(x[0][3], x[1][4]))

        return links, bos, eos

    def _add_weight(self, links):
        """링크된 점들 사이의 edge를 만들어 그래프 생성"""

        def weight(from_node, to_node):
            morphs = to_node[0].split(' + ')

            # 첫 단어의 점수
            w = self.emission.get(to_node[1], {}).get(morphs[0], self._min_emission)
            w += self.transition.get((from_node[2], to_node[1]), self._min_transition)

            # 두번째 단어의 점수 (용언의 어미 부분)
            if len(morphs) == 2:
                w += self.emission.get(to_node[2], {}).get(morphs[1], self._min_emission)
                w += self.transition.get((from_node[2], to_node[2]), self._min_transition)
            return w

        graph = []
        for from_node, to_node in links:
            edge = (from_node, to_node, weight(from_node, to_node))
            graph.append(edge)
        return graph

    def _flatten(self, path):
        """용언을 어간+어미 형태로 표현된 부분을 두 개로 분리"""
        pos = []
        for word, tag0, tag1, b, e in path:
            morphs = word.split(' + ')
            pos.append((morphs[0], tag0))
            if len(morphs) == 2:
                pos.append((morphs[1], tag1))
        return pos

    def _inference_unknown(self, pos):
        pos_ = []
        for i, pos_i in enumerate(pos[:-1]):
            if not (pos_i[1] == 'Unk'):
                pos_.append(pos_i)
                continue

            # Unk 토큰인 경우 추론 프로세스
            # 이전 토큰의 품사 정보 반영
            if i == 1:
                tag_prob = self.begin.copy()
            else:
                tag_prob = {
                    tag:prob for (prev_tag, tag), prob in self.transition.items()
                    if prev_tag == pos[i-1][1]
                }

            # 이후 토큰의 품사 정보 반영
            for (tag, next_tag), prob in self.transition.items():
                if next_tag == pos[i+1][1]:
                    tag_prob[tag] = tag_prob.get(tag, 0) + prob

            # 이렇게 찾아지지 않은 품사이면 명사로 예측
            if not tag_prob:
                infered_tag = 'Noun'
            else:
                infered_tag = sorted(tag_prob, key=lambda x:-tag_prob[x])[0]
            pos_.append((pos_i[0], infered_tag))

        return pos_ + pos[-1:]

    def _postprocessing(self, pos):
        """BOS, EOS 제거 """
        return pos[1:-1]

    def add_user_dictionary(self, word, tag, score):
        if not (tag in self.emission):
            self.emission[tag] = {word: score}
        else:
            self.emission[tag][word] = score

if  __name__ == '__main__':
    args = parser.parse_args()
    text = args.text
    json_path = args.json_path
    
    emission, transition, begin = load_from_json(json_path)

    #print("transition", transition)
    hmm_tagger = HMMTagger(emission, transition, begin)
    #print(hmm_tagger.tag('tt도예시였다'))
    print(hmm_tagger.tag(text))
