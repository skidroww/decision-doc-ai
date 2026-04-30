import pickle

with open("./corpus_info2.pkl", 'rb') as f:
    corpus_info = pickle.load(f)

doc_info = {}
for item in corpus_info:
    doc_id = item['chunk_id'].split('_chunk_')[0]
    title = item["injected_text"].split("[사건명: ")[1].split("]")[0]
    doc_info[doc_id] = title

print("✅ 현재 DB 문서 목록:")
for i, (doc_id, title) in enumerate(doc_info.items(), 1):
    print(f"{i}. {doc_id} | {title}")