import pickle
import chromadb
from sentence_transformers import SentenceTransformer


print("검색 엔진 및 DB 로딩 중...")
embedding_model = SentenceTransformer('BAAI/bge-m3')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="ftc_resolutions")

with open("./bm25_index.pkl", 'rb') as f:
    bm25 = pickle.load(f)
with open("./corpus_info.pkl", 'rb') as f:
    corpus_info = pickle.load(f)

print("로딩 완료! 검색을 시작합니다.\n")


def hybrid_search(query: str, target_company: str, top_k: int = 5):
    # [1] Vector Search 
    query_embedding = embedding_model.encode([query], normalize_embeddings=True).tolist()
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=20,
        where={"company": target_company}
    )
    vector_ids = vector_results['ids'][0]

    # [2] Keyword Search (BM25)
    # 인덱싱할 때 띄어쓰기 기준으로 토큰화했으므로 동일하게 처리
    tokenized_query = query.split() 
    bm25_scores = bm25.get_scores(tokenized_query)
    
    ## 점수 상위 20개의 인덱스를 뽑아 chunk_id로 변환
    #bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    #bm25_ids = [corpus_info[i]['chunk_id'] for i in bm25_top_indices]

    # BM25 결과에서도 타겟 기업의 문서만 필터링해서 뽑아냅니다.
    bm25_filtered = []
    for i, score in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True):
        #  corpus_info[i]["metadata"]["company"] == target_company -> 더 좋은방법. 나중에 시도
        if corpus_info[i]["injected_text"].find(target_company) != -1: # 타겟 기업이 포함된 청크만
            bm25_filtered.append(corpus_info[i]['chunk_id'])
        if len(bm25_filtered) >= 20:
            break

    # [3] RRF (Reciprocal Rank Fusion) 알고리즘으로 두 결과의 순위 합산
    k = 60
    rrf_scores = {}
    
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
    for rank, doc_id in enumerate(bm25_filtered):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # [4] 합산된 점수를 기준으로 내림차순 정렬
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 K개(5개)의 chunk_id만 추출
    top_k_ids = [doc_id for doc_id, score in sorted_rrf[:top_k]]
    
    # [5] (매우 중요) 대회 규정: 무조건 5개를 반환해야 함 (부족할 경우 채워넣기)
    if len(top_k_ids) < 5:
        #print("경고: 검색된 결과가 5개 미만입니다. 강제로 5개를 채웁니다.")
        for item in corpus_info:
            if target_company in item['injected_text'] and item['chunk_id'] not in top_k_ids:
                top_k_ids.append(item['chunk_id'])
            if len(top_k_ids) == 5:
                break

    return top_k_ids


if __name__ == "__main__":
    #test_query = "한국토종닭협회가 부과받은 최종 과징금액은 얼마이며, 납부 기한은 언제까지인가요?"
    test_query = "과징금 납부 기한"
    target_company = "(사)한국토종닭협회"

    print(f"질문: {target_company}가 {test_query}\n")
    
    results = hybrid_search(test_query, target_company)
    
    print("-" * 80)
    print(f"최종 추출된 Top-5 Chunk IDs (총 {len(results)}개):")


    for i, chunk_id in enumerate(results, 1):
        matched_info = next((item for item in corpus_info if item['chunk_id'] == chunk_id), None)

        if matched_info:
            preview = matched_info["injected_text"][:].replace("\n", " ") + "..."
            print(f"{i}. {chunk_id} \n  미리보기: {preview}\n")
        else:
            print(f"{i}. {chunk_id} \n  미리보기:(정보를 찾을 수 없음))\n")
        
    print("-" * 80)