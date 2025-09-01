# PDF to OpenSearch Pipeline

PDF 문서를 처리하여 OpenSearch에 업로드하는 파이프라인입니다.

## 개요

이 파이프라인은 다음 3단계로 구성됩니다:

1. **PDF 처리** (`pdf_processor.py`) - PDF를 텍스트/이미지 기반으로 분류하여 변환
2. **의미적 청킹** (`semantic_chunker.py`) - 변환된 텍스트를 의미 단위로 분할
3. **OpenSearch 업로드** (`uploader.py`) - 청크 데이터를 OpenSearch에 업로드

## 사용법

### 1. PDF 처리 (pdf_processor.py)

PDF 파일을 텍스트 기반/이미지 기반으로 자동 분류하여 변환합니다.

#### 기본 사용법
```bash
# Gemini API 키 지정
python pdf_processor.py document.pdf --output_dir output --gemini_api_key YOUR_API_KEY
```

#### 환경 변수
```bash
export GEMINI_API_KEY="your_gemini_api_key"
export environment="production"  # CLI 모드 활성화
```

#### 출력 파일
- `*.md` - 마크다운 형식의 변환 결과
- `*.json` - 구조화된 JSON 데이터
- `imgs/` - 추출된 이미지 파일들

### 2. 의미적 청킹 (semantic_chunker.py)

변환된 마크다운 파일을 의미 단위로 분할합니다.

#### 기본 사용법
```python
from semantic_chunker import main

# 단일 파일 처리
main("/path/to/document.md")
```

#### 출력 파일
- `*.files/document_name_chunks.md` - 청크별 마크다운
- `*.files/document_name_chunks.txt` - `<chunk>` 태그로 감싼 청크

#### 청킹 설정
```python
# 청킹 파라미터 조정
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 임베딩 모델
overlab_size = 0  # 청크 간 겹침 크기
```

### 3. OpenSearch 업로드 (uploader.py)

청크 데이터를 OpenSearch에 업로드합니다.

#### 기본 사용법
```bash
# 단일 파일 업로드
python uploader.py --file .files/document_chunks.txt --batch-size 100

# 디렉토리 내 모든 파일 업로드
python uploader.py --directory .files --pattern "*_chunks.txt" --batch-size 100

# 기존 문서 삭제 없이 업로드
python uploader.py --file document_chunks.txt --no-delete
```

#### 고급 옵션
```bash
# 커스텀 인덱스 및 임베딩 모델 설정
python uploader.py \
  --directory .files \
  --index-name "my-custom-index" \
  --embedding-model "cohere.embed-multilingual-v3" \
  --embedding-dimension 1024 \
  --batch-size 50
```

#### 환경 설정
AWS SSO LOGIN

## 전체 파이프라인 실행

### 1단계: PDF 처리
```bash
python pdf_processor.py /path/to/pdfs --output_dir output_v2
```

### 2단계: 청킹
```python
# output_v2 디렉토리의 모든 .md 파일을 청킹
python semantic_chunker.py
```

### 3단계: OpenSearch 업로드
```bash
python uploader.py --directory .files --pattern "*_chunks.txt"
```

## 📊 출력 데이터 구조

### 청크 파일 형식
```
<chunk>
청크 내용...
[page_index: 15]
[URL: https://example.com/image.png]
</chunk>
```

### OpenSearch 문서 구조
```json
{
  "metadata": {
    "source_type": "PDF",
    "source_title": "농업기술길잡이-고추",
    "crop_name": "고추",
    "chunk_sequence": 1,
    "page_number": 16,
    "image_urls": ["https://example.com/image.png"]
  },
  "chunk_text_previous": "이전 청크 내용...",
  "chunk_text_current": "현재 청크 내용...",
  "chunk_text_next": "다음 청크 내용..."
}
```

## ⚙️ 설정 및 의존성

### 필수 패키지
```bash
pip install pymupdf pandas camelot-py opensearch-py boto3
```

### 주요 의존성 모듈
```bash
# PDF 처리
PyMuPDF==1.26.3          # PDF 텍스트/이미지 추출
camelot-py==1.0.0        # 표 추출
pandas==2.2.3            # 데이터 처리
numpy==1.26.4            # 수치 계산

# 이미지 처리
Pillow==11.1.0           # 이미지 처리
opencv-python-headless==4.12.0.88  # 컴퓨터 비전

# AI/ML
sentence-transformers==3.0.1  # 의미적 청킹
transformers==4.48.1     # 자연어 처리
torch==2.5.1             # 딥러닝 프레임워크
google-genai==1.25.0

# OpenSearch/AWS
opensearch-py==2.7.1     # OpenSearch 클라이언트
boto3==1.35.92           # AWS SDK
aiobotocore==2.14.0      # 비동기 AWS 클라이언트

# 유틸리티
python-dotenv==1.0.1     # 환경변수 관리
requests==2.32.3         # HTTP 요청
tqdm==4.67.1             # 진행률 표시
```

### API 키 설정
- **Gemini API**: 이미지 기반 PDF 처리용
- **AWS 자격 증명**: OpenSearch 및 S3 접근용

### 권장 사양
- **메모리**: 8GB 이상
- **저장공간**: PDF 크기의 3-5배
- **네트워크**: 안정적인 인터넷 연결 (API 호출용)

## 문제 해결

### 일반적인 오류
1. **Gemini API 키 누락**: 이미지 기반 PDF 처리 실패
2. **메모리 부족**: 대용량 PDF 처리 시 청킹 단계에서 오류
3. **네트워크 오류**: OpenSearch 업로드 실패

### 로그 확인
```bash
# 상세 로그 확인
export PDF_CONVERTER_DEBUG=true
python pdf_processor.py document.pdf
```