import pickle

with open("./corpus_info.pkl", 'rb') as f:
    corpus_info = pickle.load(f)

# chunk_id에서 문서 이름 부분만 추출해서 중복 제거
doc_names = set([item['chunk_id'].split('_chunk_')[0] for item in corpus_info])

print("✅ 현재 DB에 임베딩된 문서 목록:")
for i, name in enumerate(doc_names, 1):
    print(f"{i}. {name}")