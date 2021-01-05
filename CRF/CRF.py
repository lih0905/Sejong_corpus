import json
from collections import namedTuple

import pycrfsuite

Feature = namedTuple('Feature', 'idx count')

bos = 'BOS'
eos = 'EOS'

class AbstractFeatureTransformer:
    """Feature transformer가 상속하는 abstract class"""
    def __call__(self, sentence):
        return self.sentence_to_xy(sentence)

    def sentence_to_xy(self, sentence):
        """Feature transformer

        input:
            sentence (List(tuple)) : [(word, tag), (word, tag),...]
        """

        words, tags = zip(*sentence)
        words_ = tuple((bos, *words, eos))
        tags_ = tuple((bos, *tags, eos))

        encoded_sentence = self.potential_function(words_, tags_)
        return  encoded_sentence, tags

    def potential_function(self, words_, tags_):
        n = len(tags_) - 2 # except bos & eos
        sentence_ = [self.to_feature(words_, tags_, i) for i in range(1, n+1)]
        return sentence_

    def to_feature(self, sentence):
        raise NotImplemented

class BaseFeatureTransformer(AbstractFeatureTransformer):
    """현재 단어, 현재 단어와 앞 단어의 품사,
       앞 단어와 현재 단어, 앞 단어와 현재 단어 + 앞 단어의 품사,
       앞/뒤에 등장한 단어, 앞/뒤에 등장한 단어와 앞 단어의 품사를
       feature로 이용하는 potential_function
    """
    def __init__(self):
        super().__init__()

    def to_feature(self, words_, tags_, i):
        features = [
            'x[0]=%s' % words_[i],
            'x[0]=%s, y[-1]=%s' % (words_[i], tags_[i-1]),
            'x[-1:0]=%s-%s' % (words_[i-1], words_[i]),
            'x[-1:0]=%s-%s, y[-1]=%s' %  (words_[i-1], words_[i], tags_[i-1]),
            'x[-1,1]=%s-%s' % (words_[i-1], words_[i+1]),
            'x[-1,1]=%s-%s, y[-1]=%s' % (words_[i-1], words_[i+1], tags_[i-1])
        ]
        return features

class HMMStyleFeatureTransformer(AbstractFeatureTransformer):
    """HMM이 이용하는 정보들을 이용한 CRF"""
    def __init__(self):
        super().__init__()

    def to_feature(self, words_, tags_, i):
        features = [
            'x[0]=%s' % words_[i],
        ]
        return features

class Trainer:
    def __init__(self, corpus=None, sentence_to_xy=None, min_count=3,
                 l2_cost=1.0, l1_cost=1.0, max_iter=300, verbose=True,
                 scan_batch_size=200000):
        self.sentence_to_xy = sentence_to_xy
        self.min_count = min_count
        self.l2_cost = l2_cost
        self.l1_cost = l1_cost
        self.max_iter = max_iter
        self.verbose = verbose
        self.scan_batch_size = scan_batch_size

        if corpus is not None:
            self.train(corpus)

    def scan_features(self, sentences, sentence_to_xy, min_count=2):
        """최소 등장 횟수 이하의 feature는 cut하는 메서드"""
        def trim(count, min_count):
            counter = {
                feature:count for feature, count in counter.items()
                # 최소 등장 횟수 넘거나 단어 자체만 기억
                if (count >= min_count) or (feature[:4] == 'x[0]' and not ', ' in feature)
            }
            return counter

        counter = {}

        for i, sentence in enumerate(sentences):
            # 문장을 feature로 변환
            sentence_, _ = sentence_to_xy(sentence)
            for features in sentence_:
                for feature in features:
                    counter[feature] = counter.get(feature, 0) + 1


        # 등장수 적은 feature를 삭제
        counter = trim(counter, min_count)
        return counter

    def train(self, sentences):
        features = self.scan_features(
            sentences, self.sentence_to_xy,
            self.min_count, self.scan_batch_size)

        # feature encoder
        self._features = {
            # feature의 idx와 count 기록
            feature:Feature(idx, count) for idx, (feature, count) in
            enumerate(sorted(features.items(), key=lambda x:-x[1]
                             ))
        }

        # feature id decoder
        self._idx2feature = [
            feature for feature in sorted(
                self._features, key=lambda x:self._features[x].idx)]

        self._train_pycrfsuite(sentences)
        self._parse_coefficients()

    def _train_pycrfsuite(self, sentences):
        trainer = pycrfsuite.Trainer(verbose=self.verbose)
        for i, sentence in enumerate(sentences):
            # transform sentence to features
            sentence_, _ = self.sentence_to_xy(sentence)
            # use only conformed feature
            x = [[xij for xij in xi if xij in self._features] for xi in x]
            trainer.append(x, y)

        # set pycrfsuite parameters
        params = {
            'feature.minfreq':max(0, self.min_count),
            'max_iterations':max(1, self.max_iter),
            'c1':max(0, self.l1_cost),
            'c2':max(0, self.l2_cost)
        }

        # do train
        trainer.set_params(params)
        trainer.train('temporal_model')

    def _parse_coefficients(self):
        # load pycrfsuite trained model
        tagger = pycrfsuite.Tagger()
        tagger.open('temporal_model')

        # state feature coefficient
        debugger = tagger.info()
        self.state_features = debugger.state_features

        # transition coefficient
        self.transitions = debugger.trasitions

    def _save_as_json(self, json_path):
        # concatenate key that formed as tuple of str
        state_features_json = {
            ' -> '.join(state_feature):coef
            for state_feature, coef in self.state_fatures.items()
        }

        # concatenate key that formed as tuple of str
        transitions_json = {
            ' -> '.join(transition):coef
            for transition, coef in self.transitions.items()
        }

        # re-group parameters
        params = {
            'state_features': state_features_json,
            'transitions': transitions_json,
            'idx2feature': self._idx2feature,
            'features': self._features
        }

        # save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

class TrainedCRFTagger:
    def __init__(self, model_path=None, coefficients=None,
                 feature_transformer=None, verbose=False):

        self.feature_transformer = feature_transformer
        self.verbose = verbose
        if model_path:
            self._load_from_json(model_path)

    def _load_from_json(self, json_path, marker=' -> '):
        with open(json_path, encoding='utf-8') as f:
            model = json.load(f)

        # parse transition
        self._transitions = {
            tuple(trans.split(marker)):coef
            for trans, coef in model['transitions'].items()
        }

        # parse state features
        self._state_features = {
            tuple(feature.split(marker)): coef
            for feature, coef in model['state_feature'].items()
        }

    def score(self, sentence):

        # feature transform
        sentence_, tags = self.feature_transformer(sentence)
        score = 0

        # transition weight
        for s0, s1 in zip(tags, tags[1:]):
            score += self.transitions.get((s0, s1), 0)

        # state feature weight
        for features, tag in zip(sentence_, tags):
            for feature in features:
                coef = self.state_features.get((feature, tag), 0)
                score += coef

        return score



