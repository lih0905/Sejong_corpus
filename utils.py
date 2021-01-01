from typing import List

def read_corpus(txt_path: str, num_lines: int = 0) -> List[tuple]:
    """클린 처리된 세종 코퍼스를 문장 단위로 다음과 같은 형태로 리턴하는 함수

        [('프랑스', 'Noun'),
          ('의', 'Josa'),
          ('세계적', 'Noun'),
          ('이', 'Adjective'),
          ('ㄴ', 'Eomi'),
          ('의상', 'Noun'),
          ('디자이너', 'Noun'),
          ('엠마누엘', 'Noun'),
          ('웅가로', 'Noun'),
          ('가', 'Josa'),
          ('실내', 'Noun'),
          ('장식용', 'Noun'),
          ('직물', 'Noun'),
          ('디자이너', 'Noun'),
          ('로', 'Josa'),
          ('나서', 'Verb'),
          ('었다', 'Eomi')]

    args:
        txt_path (str) : 전처리된 세종 코퍼스 txt 파일의 위치
        num_lines (int) : 세종 코퍼스에서 불러올 라인수. 0을 입력하면 전체를 불러온다.
    """

    def conv_sent_into_forms(sent_list: List) -> List[tuple]:
        tag_infos = []
        for word in sent_list:
            tag_info = word.split('\t')[1]
            tag_info = tag_info.split(' + ')
            tag_infos.extend(tag_info)

        sents = []
        for tag in tag_infos:
            tag = tag.replace('\n','')
            word = ''.join(tag.split('/')[:-1])
            pos = tag.split('/')[-1]
            sents.append((word, pos))

        return sents

    with open(txt_path, 'r') as f:
        if num_lines:
            raw_sents = f.readlines()[:num_lines]
        else:
            raw_sents = f.readlines()

    corpus = []
    sent = []

    for s in raw_sents:
        if s == '\n':
            if sent:
                corpus.append(conv_sent_into_forms(sent))
                sent = []
            else:
                continue
        else:
            sent.append(s)

    return corpus


def as_bigram_tag(wordpos):
    """문장 sent에서 tag만 취하여 이를 bigram으로 묶음"""
    poslist = [pos for _, pos in wordpos]
    return ["_".join([pos0,pos1]) for pos0, pos1 in zip(poslist, poslist[1:])]

