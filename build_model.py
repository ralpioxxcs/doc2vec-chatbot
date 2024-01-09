import sys

import pandas as pd

from nltk.corpus import stopwords

from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

from khaiii import KhaiiiApi

if len(sys.argv) != 2:
  print("질문을 입력하세요")
  sys.exit()

#
# 문장 토크나이징, 필터에 맞는 토큰만 넣음
#
def tokenize_only_noun(sentence):
  filter = ['NNG', 'NNP', 'OL', 'VA', 'VV', 'VXV']
  tokenized = []
  for word in KhaiiiApi().analyze(sentence):
    for morph in word.morphs:
      if str(morph).split("/")[1] in filter:
        tokenized.append(str(morph).split("/")[0])

  return tokenized


# load as data frame
df = pd.read_csv("faq.csv")

tokens_faqs = []
for i in range(df.__len__()):
  questions = df['questions'][i]

  # 각 질문 문장에 대한 형태소 분석 수행
  tokenized = tokenize_only_noun(questions)

  # 순번화
  tokens_faqs.append([tokenized, i])

# 태그 생성
tagged_faqs = [TaggedDocument(d, [int(c)]) for d, c in tokens_faqs]

# 모델 생성
d2v_model = doc2vec.Doc2Vec(
    vector_size=200,
    # alpha=0.025,
    # min_alpha=0.025,
    hs=1,
    negative=0,
    dm=0,
    #window=3,
    dbow_words=1,
    min_count=2,
    workers=8,
    seed=0,
    epochs=20)

d2v_model.build_vocab(tagged_faqs)

for epoch in range(50):
  d2v_model.train(tagged_faqs,
                  total_examples=d2v_model.corpus_count,
                  epochs=d2v_model.epochs)
  d2v_model.alpha -= 0.0025
  d2v_model.min_alpha = d2v_model.alpha

d2v_model.save('doc2vec.model')

#
# 테스트
#
test_question = sys.argv[1]

token_test_question = []
for word in KhaiiiApi().analyze(test_question):
  token_test_question.extend([str(m) for m in word.morphs])

topn = 5
test_vector = d2v_model.infer_vector(token_test_question)
result = d2v_model.dv.most_similar([test_vector], topn=topn)

for i in range(topn):
  print("[{}]: {} ({})".format(i + 1, df['answer'][result[i][0]],
                               result[i][1]))
