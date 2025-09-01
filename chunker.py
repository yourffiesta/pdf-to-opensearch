import time
from typing import Optional, List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core import Document

# 전역 캐시 - 모델을 한 번만 로드
_cached_embed_model = None
_cached_model_name = None

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", use_cache: bool = True):
    """
    임베딩 모델을 가져오는 함수 (캐싱 지원)
    
    빠른 모델 옵션들:
    - "sentence-transformers/all-MiniLM-L6-v2" (기본) - 매우 빠름, 적당한 품질
    - "sentence-transformers/paraphrase-MiniLM-L6-v2" - 빠름, 좋은 품질
    - "BAAI/bge-small-en-v1.5" - 빠름, 좋은 품질 (영어)
    - "BAAI/bge-m3" (원본) - 느림, 최고 품질
    """
    global _cached_embed_model, _cached_model_name
    
    if use_cache and _cached_embed_model is not None and _cached_model_name == model_name:
        return _cached_embed_model
    
    print(f"🔄 임베딩 모델 로딩 중: {model_name}")
    start_time = time.time()
    
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
        device="cpu"  # GPU 사용 시 "cuda"로 변경
    )
    
    load_time = time.time() - start_time
    print(f"✅ 모델 로딩 완료 ({load_time:.2f}초)")
    
    if use_cache:
        _cached_embed_model = embed_model
        _cached_model_name = model_name
    
    return embed_model

def split_text_into_chunks_fast(
    text: str, 
    buffer_size: int = 2, 
    breakpoint_percentile_threshold: int = 95,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chunk_size: int = 1000,
    overlab_size: int = 200,
)-> list[dict]:
    """
    빠른 의미기반 청킹을 수행하는 함수 (최적화된 버전)
    
    Args:
        text: 청킹할 텍스트
        buffer_size: 의미 유사성을 평가할 때 함께 그룹화할 문장의 수 (기본값: 2)
        breakpoint_percentile_threshold: 코사인 비유사성의 백분위수 임계값 (기본값: 95)
        model_name: 사용할 임베딩 모델 (빠른 모델 기본값)
        max_chunk_size: 하이브리드 방식에서 1차 분할 크기
        use_hybrid: 하이브리드 방식 사용 여부 (길이 기반 + 의미 기반)
    """
    # 텍스트 전처리 - 불필요한 공백 제거
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    
    print("🔀 하이브리드 방식 사용: 길이 기반 1차 분할 + 의미 기반 2차 분할")
    return _hybrid_chunking(text, buffer_size, breakpoint_percentile_threshold, model_name, max_chunk_size, overlab_size)

def _hybrid_chunking(
    text: str, 
    buffer_size: int, 
    breakpoint_percentile_threshold: int, 
    model_name: str,
    max_chunk_size: int,
    overlab_size:int
) -> List[dict]:
    """하이브리드 청킹: 길이 기반 1차 분할 후 의미 기반 2차 분할"""
    
    # 1차: 길이 기반 분할
    sentence_splitter = SentenceSplitter(chunk_size=max_chunk_size, chunk_overlap=overlab_size)
    initial_chunks = sentence_splitter.split_text(text)
    
    print(f"📏 1차 길이 기반 분할: {len(initial_chunks)}개 청크")
    
    if len(initial_chunks) <= 1:
        return initial_chunks
    
    # 2차: 각 청크에 대해 의미 기반 분할
    embed_model = get_embedding_model(model_name)
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold
    )
    
    final_chunks = []
    for i, chunk in enumerate(initial_chunks):
        print(f"🔍 2차 의미 기반 분할 ({i+1}/{len(initial_chunks)})")
        
        if len(chunk) < 200:  # 너무 작은 청크는 그대로 유지
            final_chunks.append(chunk)
            continue
            
        document = Document(text=chunk)
        nodes = parser.get_nodes_from_documents([document])
        final_chunks.extend([node.text for node in nodes])
    
    return final_chunks


def benchmark_models(sample_text: str):
    """다양한 모델의 성능을 벤치마크하는 함수"""
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # 가장 빠름
        "sentence-transformers/paraphrase-MiniLM-L6-v2",  # 빠름
        "BAAI/bge-small-en-v1.5",  # 중간
        "BAAI/bge-m3"  # 느림하지만 고품질
    ]
    
    results = {}
    for model in models:
        print(f"\n🧪 벤치마크: {model}")
        start_time = time.time()
        try:
            chunks = split_text_into_chunks_fast(
                sample_text, 
                model_name=model,
                use_hybrid=True
            )
            elapsed = time.time() - start_time
            results[model] = {
                "time": elapsed,
                "chunks": len(chunks),
                "success": True
            }
            print(f"✅ 완료: {elapsed:.2f}초, {len(chunks)}개 청크")
        except Exception as e:
            results[model] = {
                "time": None,
                "chunks": 0,
                "success": False,
                "error": str(e)
            }
            print(f"❌ 실패: {e}")
    
    return results