"""
PDF 변환 파이프라인을 위한 공통 유틸리티 함수들과 상수 정의
"""
import os
import json
import boto3
import re
from typing import Optional, Dict, Any
import unicodedata

# --- 공통 상수 ---
DEFAULT_S3_BUCKET = "gl-data-cdn-storage"
DEFAULT_CDN_URL = "https://ddht7b7vmidcm.cloudfront.net/"
DEFAULT_CHUNK_SIZE = 3
DEFAULT_CONCURRENCY_LIMIT = 5

# 캡션 탐지를 위한 정규식 패턴들
CAPTION_PATTERNS = {
    "table": r"\s*[<(\[]?\s*(표|Table)\s*([\d.-]+)",
    "figure": r"\s*[<(\[]?\s*(그림|Figure|Fig\.|그래프|차트|Chart)\s*([\d.-]+)"
}


class S3Manager:
    """S3 업로드/다운로드를 관리하는 클래스"""
    
    def __init__(self, bucket_name: str = DEFAULT_S3_BUCKET):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
    
    def upload_file(self, local_path: str, s3_key: str) -> Optional[str]:
        """
        로컬 파일을 S3에 업로드하고 S3 키를 반환
        
        Args:
            local_path: 업로드할 로컬 파일 경로
            s3_key: S3에서 사용할 키
            
        Returns:
            업로드 성공시 S3 키, 실패시 None
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"Successfully uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except Exception as e:
            print(f"[Warn] Failed to upload {local_path} to S3: {e}")
            return None
    
    def upload_file_with_pdf_structure(self, local_path: str, pdf_name: str) -> Optional[str]:
        """
        PDF 구조에 맞는 S3 키로 파일을 업로드
        
        Args:
            local_path: 업로드할 로컬 파일 경로
            pdf_name: PDF 파일명 (확장자 제외)
            
        Returns:
            업로드 성공시 S3 키, 실패시 None
        """
        s3_key = os.path.join("farming_info/publicdata/", pdf_name, os.path.basename(local_path))

        return self.upload_file(local_path, s3_key)


def calculate_iou(box_a: list, box_b: list) -> float:
    """
    두 경계 상자의 IoU(Intersection over Union)를 계산
    
    Args:
        box_a: [x1, y1, x2, y2] 형태의 첫 번째 박스
        box_b: [x1, y1, x2, y2] 형태의 두 번째 박스
        
    Returns:
        0.0 ~ 1.0 사이의 IoU 값
    """
    # 교집합 영역 계산
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    
    # 교집합이 없는 경우
    if x_b <= x_a or y_b <= y_a:
        return 0.0
    
    # 교집합 면적
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    
    # 각 박스의 면적
    box_a_area = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    
    # 합집합 면적
    union_area = box_a_area + box_b_area - inter_area
    
    # IoU 계산
    return float(inter_area / union_area) if union_area > 0 else 0.0


def ensure_directory_exists(dir_path: str) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(dir_path, exist_ok=True)


def get_safe_filename(text: str, max_length: int = 50) -> str:
    """
    파일명에 안전한 문자열로 변환
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        
    Returns:
        파일명으로 사용 가능한 안전한 문자열
    """
    safe_text = "".join(c for c in text if c.isalnum() or c in ' -').strip()
    safe_text = safe_text.replace(" ", "_")
    return safe_text[:max_length] if safe_text else "untitled"


def parse_json_response(response_text: str) -> Dict[Any, Any]:
    """
    응답 텍스트에서 JSON을 파싱 (```json ... ``` 형태 포함)
    
    Args:
        response_text: 파싱할 응답 텍스트
        
    Returns:
        파싱된 JSON 딕셔너리
        
    Raises:
        ValueError: JSON 파싱 실패시
    """
    import re
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # ```json ... ``` 형태의 응답 처리
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            return json.loads(json_match.group(1))
        else:
            raise ValueError(f"Cannot parse JSON from response: {response_text[:100]}...")


def generate_cdn_url(s3_key: str, cdn_base_url: str = DEFAULT_CDN_URL) -> str:
    """
    S3 키로부터 CDN URL 생성
    
    Args:
        s3_key: S3 객체 키
        cdn_base_url: CDN 베이스 URL
        
    Returns:
        완성된 CDN URL
    """
    return os.path.join(cdn_base_url, s3_key)


def save_json_file(data: Dict[Any, Any], file_path: str) -> None:
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터
        file_path: 저장할 파일 경로
    """
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON file saved to: {file_path}")


def save_markdown_file(content: str, file_path: str) -> None:
    """
    마크다운 내용을 파일로 저장
    
    Args:
        content: 마크다운 내용
        file_path: 저장할 파일 경로
    """
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Markdown file saved to: {file_path}")


def safe_save_pixmap(pix, output_path: str) -> bool:
    """
    PyMuPDF Pixmap을 안전하게 PNG 파일로 저장
    
    Args:
        pix: fitz.Pixmap 객체
        output_path: 저장할 파일 경로
        
    Returns:
        저장 성공 여부
    """
    import fitz
    
    ensure_directory_exists(os.path.dirname(output_path))
    
    try:
        # CMYK나 다른 색상 공간을 RGB로 변환
        if pix.n - pix.alpha > 3:  # CMYK (4채널) 또는 그 이상
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            rgb_pix.save(output_path)
            rgb_pix = None
        elif pix.n - pix.alpha == 1:  # 그레이스케일
            pix.save(output_path)
        elif pix.n - pix.alpha != 3:  # RGB가 아닌 다른 색상 공간
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            rgb_pix.save(output_path)
            rgb_pix = None
        else:  # RGB
            pix.save(output_path)
        
        return True
        
    except Exception as save_error:
        print(f"[Warning] Failed to save pixmap with original format: {save_error}")
        try:
            # 대안: 강제로 RGB 변환
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            rgb_pix.save(output_path)
            rgb_pix = None
            return True
        except Exception as alt_error:
            print(f"[Error] Failed to save pixmap: {alt_error}")
            return False


def safe_pixmap_to_png_bytes(pix) -> bytes:
    """
    PyMuPDF Pixmap을 안전하게 PNG 바이트로 변환
    
    Args:
        pix: fitz.Pixmap 객체
        
    Returns:
        PNG 바이트 데이터 또는 None
    """
    import fitz
    
    try:
        # CMYK나 다른 색상 공간을 RGB로 변환
        if pix.n - pix.alpha > 3:  # CMYK 또는 그 이상
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = rgb_pix.tobytes("png")
            rgb_pix = None
            return img_bytes
        else:
            return pix.tobytes("png")
            
    except Exception as convert_error:
        print(f"[Warning] Failed to convert pixmap to PNG, trying RGB conversion: {convert_error}")
        try:
            rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = rgb_pix.tobytes("png")
            rgb_pix = None
            return img_bytes
        except Exception as alt_error:
            print(f"[Error] Failed to convert pixmap to PNG: {alt_error}")
            return None
        
def normalize_NFC(text: str) -> str:
    """
    NFC 정규화 적용
    
    Args:
        text: 정규화할 텍스트
    """
    return unicodedata.normalize('NFC', text)


class PDFProcessorConfig:
    """PDF 프로세서 설정을 관리하는 클래스"""
    
    def __init__(
        self,
        output_dir: str = "output",
        gemini_api_key: Optional[str] = None,
        s3_bucket: str = DEFAULT_S3_BUCKET,
        cdn_url: str = DEFAULT_CDN_URL,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ):
        self.output_dir = output_dir
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.s3_bucket = s3_bucket
        self.cdn_url = cdn_url
        self.concurrency_limit = concurrency_limit
        self.chunk_size = chunk_size
        
        # 출력 디렉토리 생성
        ensure_directory_exists(self.output_dir)
        ensure_directory_exists(os.path.join(self.output_dir, "imgs"))
    
    def get_s3_manager(self) -> S3Manager:
        """S3Manager 인스턴스 반환"""
        return S3Manager(self.s3_bucket) 
    

def starts_with_list_item(text):
    patterns = [
        r'^\d+\.\s',           # 1. , 2. , 10. 
        r'^\d+\)\s',           # 1) , 2) , 10) 
        r'^\(\d+\)\s',         # (1) , (2) , (10) 
        r'^[가-하]\)\s',       # 가) , 나) , 다) 
        r'^\([가-하]\)\s',     # (가) , (나) , (다) 
        r'^[A-Z]\)\s',         # A) , B) , Z) 
        r'^\([A-Z]\)\s',       # (A) , (B) , (Z) 
        r'^[a-z]\)\s',         # a) , b) , z) 
        r'^\([a-z]\)\s',       # (a) , (b) , (z) 
        r'^[가-하]\.\s',       # 가. , 나. , 다. 
        r'^[A-Z]\.\s',         # A. , B. , Z. 
        r'^[a-z]\.\s'          # a. , b. , z. 
    ]
    return any(re.match(pattern, text) for pattern in patterns)