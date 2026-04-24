import os
import json
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
#from konlpy.tag import Mecab
from konlpy.tag import Okt
from typing import List, Dict, Any
from tqdm import tqdm

# ==========================================
# 1. 설정 및 초기화
# ==========================================
DATA_DIR = "./data/공개본 의결서"  # hybrid.json과 metadata.json이 있는 폴더
DB_DIR = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index.pkl"
CORPUS_INFO_PATH = "./corpus_info.pkl" # BM25와 매핑할 원본 데이터

# 임베딩 모델 로드 (BGE-M3 추천)
print("임베딩 모델 로딩 중...")
embedding_model = SentenceTransformer('BAAI/bge-m3')

# 형태소 분석기 로드
#mecab = Mecab()
okt = Okt()

# ChromaDB 로컬 영구 저장소 초기화
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(    
    name="ftc_resolutions",
    metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
)

# ==========================================
# 2. 데이터 파싱 및 메타데이터 주입 (테스트용으로 수정됨)
# ==========================================
def load_and_inject_data(data_dir: str, max_files: int = 10) -> List[Dict[str, Any]]:
    processed_chunks = []
    
    # 1. 디렉토리에서 metadata.json 파일만 모두 찾기
    all_files = [f for f in os.listdir(data_dir) if f.endswith("_metadata.json")]
    
    # 2. [핵심] 테스트를 위해 설정한 갯수(10개)만큼만 자르기
    test_files = all_files[:max_files]
    print(f"💡 테스트 모드: 전체 {len(all_files)}개 문서 중 {len(test_files)}개만 처리합니다.")
    
    for filename in test_files:
        base_name = filename.replace("_metadata.json", "")
        meta_path = os.path.join(data_dir, filename)
        hybrid_path = os.path.join(data_dir, f"{base_name}_hybrid.json")
        
        if not os.path.exists(hybrid_path):
            continue
            
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        with open(hybrid_path, 'r', encoding='utf-8') as f:
            hybrid_data = json.load(f)
            
        title = meta_data.get("의결서제목", "제목없음")
        
        피심인_list = meta_data.get("피심인정보", [])
        if 피심인_list:
            company = 피심인_list[0].get("피심인기업명", "기업명없음")
            violation = 피심인_list[0].get("위반유형", "유형없음")
        else:
            company, violation = "기업명없음", "유형없음"

        for i, chunk in enumerate(hybrid_data):
            chunk_id = chunk["metadata"]["chunk_id"]
            original_text = chunk.get("page_content", "")
            
            injected_text = f"[사건명: {title}] [피심인기업명: {company}] [위반유형: {violation}]\n본문: {original_text}"
            
            processed_chunks.append({
                "chunk_id": chunk_id,
                "injected_text": injected_text,
                "original_text": original_text,
                "metadata": {
                    "title": title,
                    "company": company,
                    "violation": violation
                }
            })
            
    return processed_chunks

# ==========================================
# 3. 임베딩, DB 저장 및 인덱스 생성 (수정된 버전)
# ==========================================
def build_indices():
    print("데이터 로드 및 메타데이터 주입 시작...")
    chunks = load_and_inject_data(DATA_DIR)
    print(f"총 {len(chunks)}개의 청크가 준비되었습니다.")

    ids = []
    documents = []
    metadatas = []
    tokenized_corpus = []
    
    # 1. 데이터 준비 및 BM25 토큰화 (tqdm 추가로 진행률 확인)
    for item in tqdm(chunks, desc="데이터 전처리 및 BM25 토큰화"):
        ids.append(item["chunk_id"])
        documents.append(item["injected_text"])
        metadatas.append(item["metadata"])
        
        # [수정됨] Okt 형태소 분석기가 너무 느리므로 임시로 띄어쓰기 기반 분할 적용
        # 나중에 Mecab 환경이 구축되면 tokens = mecab.morphs(item["injected_text"]) 로 복구하세요.
        tokens = item["injected_text"].split()
        tokenized_corpus.append(tokens)

    # 2. ChromaDB에 벡터 저장 (배치 처리 및 bge-m3 직접 임베딩)
    BATCH_SIZE = 128  # 한 번에 처리할 청크 개수 (메모리가 부족하면 64로 줄이세요)
    
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="ChromaDB 임베딩 및 저장"):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]

        # [수정됨] 로드해둔 bge-m3 모델로 텍스트를 직접 벡터화 (매우 중요)
        batch_embeddings = embedding_model.encode(batch_docs, normalize_embeddings=True).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings, # 벡터값을 명시적으로 전달
            documents=batch_docs,
            metadatas=batch_meta
        )

    # 3. BM25 인덱스 생성
    print("BM25 인덱스 구축 중...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
        
    corpus_info = [{"chunk_id": c["chunk_id"], "injected_text": c["injected_text"]} for c in chunks]
    with open(CORPUS_INFO_PATH, 'wb') as f:
        pickle.dump(corpus_info, f)
        
    print("모든 인덱싱 과정이 완료되었습니다.")

if __name__ == "__main__":
    build_indices()