import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt 
from kiwipiepy import Kiwi


print("검색 엔진 및 DB 로딩 중...")
embedding_model = SentenceTransformer('BAAI/bge-m3')
#okt = Okt() 
kiwi = Kiwi()


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="ftc_resolutions")

with open("./bm25_index_kiwi.pkl", 'rb') as f:
    bm25 = pickle.load(f)
with open("./corpus_info.pkl", 'rb') as f:
    corpus_info = pickle.load(f)


STOPWORDS = {
    "있다", "하다", "되다", "위하다", "통하다",
    "경우", "사항", "내용", "관련"
}

def tokenize_kiwi(text):
    tokens = kiwi.tokenize(text)
    
    return [
        t.form for t in tokens
        if (t.tag.startswith('N') or t.tag.startswith('V'))
        and len(t.form) > 1
        and t.form not in STOPWORDS
    ]

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

    
    tokenized_query = tokenize_kiwi(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # BM25 결과에서도 타겟 기업의 문서만 필터링해서 뽑아냅니다.
    bm25_filtered = []
    for i, score in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True):
        # [개선 2] 단순 문자열 검색(find)이 아닌 메타데이터 완전 일치 필터링
        # 주의: 이 코드가 작동하려면 embedding2.py에서 corpus_info를 저장할 때 metadata도 함께 저장해야 합니다.
        if corpus_info[i].get("metadata", {}).get("company") == target_company: 
            bm25_filtered.append(corpus_info[i]['chunk_id'])
        if len(bm25_filtered) >= 20:
            break

    # [3] RRF (Reciprocal Rank Fusion) 알고리즘으로 두 결과의 순위 합산
    k = 60
    rrf_scores = {}
    
    # [개선 4] Vector와 BM25의 가중치(Weight) 조정
    # 문맥 이해가 중요한 행정 문서 특성상 Vector의 비중을 더 높게 설정 (예: Vector 0.7, BM25 0.3)
    vector_weight = 0.7
    bm25_weight = 0.3
    
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight * (1 / (k + rank + 1))
        
    for rank, doc_id in enumerate(bm25_filtered):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight * (1 / (k + rank + 1))

    # [4] 합산된 점수를 기준으로 내림차순 정렬
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 K개(5개)의 chunk_id만 추출
    top_k_ids = [doc_id for doc_id, score in sorted_rrf[:top_k]]
    
    # [5] (매우 중요) 대회 규정: 무조건 5개를 반환해야 함 (부족할 경우 채워넣기)
    if len(top_k_ids) < 5:
        for item in corpus_info:
            # 여기도 메타데이터 기준으로 안전하게 fallback 처리
            if item.get("metadata", {}).get("company") == target_company and item['chunk_id'] not in top_k_ids:
                top_k_ids.append(item['chunk_id'])
            if len(top_k_ids) == 5:
                break

    return top_k_ids


if __name__ == "__main__":
    #test_query = "과징금 납부 기한"
    #target_company = "(사)한국토종닭협회"
    test_query = "타이어뱅크가 대리점에게 적용한 '이월재고 차감' 기준 중, 제조일이 48개월 초과된 일반 타이어(D등급)의 차감 비율은 공장도가격의 몇 퍼센트이며, 공정위가 이 사건과 관련하여 부과한 최종 과징금액은 얼마인가요?"
    target_company = "타이어뱅크(주)"
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
    # 1-80-46-79-74