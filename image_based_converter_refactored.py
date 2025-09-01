import re
import fitz  # PyMuPDF
import asyncio
import os
import json
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_fixed
from collections import defaultdict

# 공통 유틸리티 import
from utils import (
    PDFProcessorConfig, S3Manager, calculate_iou, get_safe_filename,
    parse_json_response, generate_cdn_url, save_json_file, save_markdown_file,
    DEFAULT_CHUNK_SIZE, safe_pixmap_to_png_bytes
)

# Gemini API 호출을 위한 기본 프롬프트
KO_USER_PROMPT = """
당신은 PDF 문서를 분석하여 JSON 형식으로 출력하는 AI입니다.

[Goal]
PDF 문서를 페이지 순서, 그리고 페이지 내 위에서 아래 순서로 스캔하여 모든 요소를 찾고, 아래 [JSON Output Schema]를 엄격히 준수하는 JSON 객체를 생성하세요.

[PDF Info]
{{bbox_list}}

[Key Rules]
1.  단어 중간의 불필요한 줄 바꿈을 제거하여 문법적으로 자연스러운 텍스트를 생성합니다.
2.  각 페이지의 마지막 `paragraph`가 문법적으로 끝나지 않은 경우, `is_incomplete` 필드를 `true`로 설정합니다.
3.  header와 footer는 결과에 포함하지 않습니다.
4.  이미지 처리 방법 (type이 "image"일 경우에만 다음 규칙 적용)
  - [PDF Info]의 `bbox_list`(x1, y1, x2, y2)에서 실제 이미지로 판단되는 항목을 `image_bbox` 필드에 포함합니다.
  - 캡션이 있는 이미지만 처리합니다. (skip if no caption)
  - 여러 개의 이미지가 하나의 캡션과 연관된 경우, 동일한 캡션을 가진 별도의 `image` 타입 요소를 각각 생성합니다.
  - Concisely formulate image descriptions within approximately 5 tokens in the content field.

[JSON Output Schema]
아래 JSON Schema 정의를 반드시 준수하여 최종 결과를 생성해야 합니다.

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": { "type": "string", "description": "요소의 유형", "enum": ["sub_title", "paragraph", "table", "image", "etc"] },
          "page_index": { "type": "integer", "description": "zero based physical page index"},
          "content": { "type": "string", "description": "추출된 텍스트 또는 마크다운 형식의 테이블" },
          "caption": { "type": "string", "description": "이미지나 테이블의 캡션" },
          "image_bbox": { 
            "type": "array",
            "description": "입력 bbox 좌표중 실제 이미지로 판단된 bbox 좌표 (x1, y1, x2, y2)",
            "items": { "type": "number" },
            "minItems": 4,
            "maxItems": 4
          },
          "is_incomplete": { "type": "boolean", "description": "문장이 문법적으로 미완성일 경우 true" },
          "original_type": { "type": "string", "description": "type이 'etc'일 경우, 원래 요소의 유형" }
        },
        "required": ["type", "page_index"]
      }
    }
  },
  "required": ["data"]
}"""


class ImageBasedConverter:
    """
    이미지 기반 PDF를 Gemini API를 사용하여 분석하고,
    결과를 JSON 및 마크다운으로 변환하는 클래스.
    PdfProcessor 인터페이스에 맞게 설계됨.
    """
    
    def __init__(self, output_dir: str = "output", api_key: str = None, concurrency_limit: int = 5):
        """
        PdfProcessor 인터페이스에 맞는 생성자
        
        Args:
            output_dir: 출력 디렉토리
            api_key: Gemini API 키 (None이면 환경변수에서 가져옴)
            concurrency_limit: 동시 처리 제한
        """
        self.config = PDFProcessorConfig(
            output_dir=output_dir,
            gemini_api_key=api_key,
            concurrency_limit=concurrency_limit
        )
        
        if not self.config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not provided or set in environment.")
            
        self.client = genai.Client(api_key=self.config.gemini_api_key)
        self.semaphore = asyncio.Semaphore(self.config.concurrency_limit)
        self.s3_manager = self.config.get_s3_manager()

    async def convert(self, pdf_path: str):
        """
        PdfProcessor 인터페이스에 맞는 메인 변환 함수
        
        Args:
            pdf_path: 변환할 PDF 파일 경로
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return

        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"Opening PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return
        
        print("Splitting PDF into chunks...")
        pdf_chunks = self._split_pdf(doc, DEFAULT_CHUNK_SIZE)
        drawing_rect_dict = self._get_drawing_rects(doc)

        # 청크별로 drawing rect 그룹핑
        grouped_rect_list = self._group_drawing_rects_by_chunk(drawing_rect_dict)

        tasks = [
            self._process_chunk(chunk_bytes, base_index, grouped_rect_list.get(base_index, []))
            for chunk_bytes, base_index in pdf_chunks
        ]
        
        print(f"Processing {len(tasks)} chunks concurrently...")
        results = await asyncio.gather(*tasks)
        
        print("Merging results...")
        # 실패한 None 결과 필터링
        successful_results = [res for res in results if res is not None]
        final_json = self._merge_results(successful_results)

        # 이미지 저장 및 S3 업로드
        print("Saving images based on JSON data...")
        self._save_images_from_json(doc, final_json, base_filename)
        
        # 결과 저장
        self._save_results(final_json, base_filename)
        
        doc.close()

    def test(self, pdf_path: str):
        """
        PdfProcessor 인터페이스에 맞는 메인 변환 함수
        
        Args:
            pdf_path: 변환할 PDF 파일 경로
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return

        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"Opening PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return
        
        with open(os.path.join(self.config.output_dir, f"{base_filename}.json"), "r") as f:
            final_json = json.load(f)

        # 이미지 저장 및 S3 업로드
        # print("Saving images based on JSON data...")
        # self._save_images_from_json(doc, final_json, base_filename)
        
        # 결과 저장
        self._save_results(final_json, base_filename)
        
        doc.close()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _call_gemini_api(self, pdf_chunk_bytes: bytes, prompt_text: str) -> dict:
        """Gemini API를 호출하고 결과를 반환 (재시도 로직 포함)"""
        prompt_parts = [
            types.Part.from_text(text=prompt_text),
            types.Part.from_bytes(data=pdf_chunk_bytes, mime_type="application/pdf")
        ]
        contents = [types.Content(role="user", parts=prompt_parts)]
        config = types.GenerateContentConfig(response_mime_type="application/json")
        
        print("Sending chunk to Gemini API...")
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )
        print("Received response from Gemini API.")
        
        try:
            result_json = parse_json_response(response.text)
            return result_json
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            raise e
            
    def _get_drawing_rects_from_page(self, page: fitz.Page, threshold: int = 20) -> dict:
        """페이지에서 드로잉 rect 정보 추출"""
        input_bbox_list = []

        page_width_threshold = page.rect.width - threshold * 2
        page_height_threshold = page.rect.height - threshold * 2

        for item in page.get_drawings():
            for sub in item['items']:
                if sub[0] == 're':
                    rect = sub[1]
                    if (rect.width > threshold and rect.height > threshold and
                        rect.width < page_width_threshold and rect.height < page_height_threshold):
                        input_bbox_list.append([
                            round(rect.x0, 1), round(rect.y0, 1), 
                            round(rect.x1, 1), round(rect.y1, 1)
                        ])

        for img in page.get_image_info():
            bbox = img['bbox']
            input_bbox_list.append([
                round(bbox[0], 1), round(bbox[1], 1), 
                round(bbox[2], 1), round(bbox[3], 1)
            ])
        
        if input_bbox_list:
            return {
                "pdf_width": round(page.rect.width, 1),
                "pdf_height": round(page.rect.height, 1),
                "bbox_list": input_bbox_list
            }
        return {}
    
    def _get_drawing_rects(self, doc: fitz.Document, threshold: int = 20) -> dict:
        """문서 전체에서 드로잉 rect 정보 추출"""
        drawing_rects = {}
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            rect_info = self._get_drawing_rects_from_page(page, threshold)
            if rect_info:
                drawing_rects[page_index] = rect_info
        return drawing_rects

    def _group_drawing_rects_by_chunk(self, drawing_rect_dict: dict) -> dict:
        """드로잉 rect를 청크별로 그룹핑"""
        grouped_rect_list = {}
        for index, item in drawing_rect_dict.items():
            base_index = (index // DEFAULT_CHUNK_SIZE) * DEFAULT_CHUNK_SIZE
            group_item_list = grouped_rect_list.get(base_index, [])
            item["page_index"] = index - base_index
            group_item_list.append(item)
            grouped_rect_list[base_index] = group_item_list
        return grouped_rect_list

    async def _process_chunk(self, pdf_chunk_bytes: bytes, base_page_index: int, drawing_rect_list: list) -> dict:
        """단일 PDF 청크를 처리하고 페이지 인덱스를 재조정"""
        async with self.semaphore:
            try:
                prompt_text = KO_USER_PROMPT.replace("{{bbox_list}}", json.dumps(drawing_rect_list))
                result_json = await self._call_gemini_api(pdf_chunk_bytes, prompt_text)

                # 페이지 인덱스 재조정
                if 'data' in result_json:
                    for item in result_json['data']:
                        if 'page_index' in item:
                            item['page_index'] += base_page_index
                
                return result_json

            except Exception as e:
                print(f"Error processing chunk starting at page {base_page_index}: {e}")
                return None

    def _split_pdf(self, doc: fitz.Document, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list:
        """PDF를 지정된 크기의 청크로 분할"""
        chunks = []
        for i in range(0, doc.page_count, chunk_size):
            start_page = i
            end_page = min(i + chunk_size, doc.page_count)
            
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
            
            pdf_chunk_bytes = chunk_doc.tobytes()
            chunks.append((pdf_chunk_bytes, start_page))
            chunk_doc.close()
            
        return chunks

    def _merge_results(self, results: list) -> dict:
        """여러 청크의 결과(JSON)를 하나로 병합"""
        final_json = {"data": []}
        for res in results:
            if res and 'data' in res:
                final_json["data"].extend(res['data'])
        
        # 최종적으로 페이지 인덱스 순으로 정렬
        final_json["data"].sort(key=lambda x: x.get('page_index', 0))
        return final_json

    def _save_single_image(self, index: int, image_item: dict, doc: fitz.Document, iou_threshold: float = 0.9) -> str:
        """JSON 항목 하나를 기반으로 이미지를 저장하고 S3에 업로드"""
        try:
            page_index = image_item.get("page_index")
            if page_index is None:
                print(f"Warning: Image item missing 'page_index', skipping.")
                return None

            json_bbox = image_item.get("image_bbox")
            if not json_bbox:
                print(f"Warning: Image item on page {page_index} missing 'image_bbox', skipping.")
                return None

            page = doc.load_page(page_index)
            image_infos = page.get_image_info(xrefs=True)
            
            # 가장 일치하는 이미지 찾기
            best_match = None
            max_iou = 0
            for img_info in image_infos:
                iou = calculate_iou(json_bbox, img_info['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    best_match = img_info

            # 파일명 생성
            caption = image_item.get('caption', '').strip()
            safe_caption = get_safe_filename(caption, 30)
            img_filename = f"page_{page_index}_{safe_caption}_{index}.png"
            img_path = os.path.join(self.config.output_dir, "imgs", img_filename)

            # 이미지 추출 및 저장
            img_bytes = None
            if max_iou > iou_threshold and best_match:
                print(f"Found matching image on page {page_index} with IoU {max_iou:.2f}. Extracting directly.")
                img_data = doc.extract_image(best_match['xref'])
                img_bytes = img_data["image"]
            else:
                print(f"No direct match found (max IoU: {max_iou:.2f}). Cropping page {page_index} by image_bbox.")
                rect = fitz.Rect(json_bbox)
                pix = page.get_pixmap(clip=rect, dpi=200)
                
                # 안전한 PNG 변환
                img_bytes = safe_pixmap_to_png_bytes(pix)
                pix = None
                
                if img_bytes is None:
                    print(f"[Error] Failed to convert cropped image to PNG on page {page_index}")
                    return None

            if img_bytes:
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"Image saved to {img_path}")
                return img_path

        except Exception as e:
            print(f"Error processing image on page {page_index}: {e}")
            return None

    def _save_images_from_json(self, doc: fitz.Document, json_data: dict, pdf_name: str):
        """JSON 데이터와 PDF 문서를 기반으로 모든 이미지 항목을 저장하고 S3에 업로드"""
        if 'data' not in json_data:
            return
        
        img_items = [item for item in json_data['data'] if item.get('type') == 'image']
        for i, item in enumerate(img_items):
            local_file_path = self._save_single_image(i, item, doc)
            if local_file_path:
                # S3 업로드
                s3_key = self.s3_manager.upload_file_with_pdf_structure(local_file_path, pdf_name)
                if s3_key:
                    item['s3_key'] = s3_key
                    item['local_file_path'] = local_file_path

    def _format_json_to_markdown(self, json_data: dict) -> str:
        """PDF 추출 결과 JSON을 RAG 임베딩에 최적화된 마크다운 텍스트로 변환"""
        if not json_data or 'data' not in json_data or not json_data['data']:
            return ""

        chunk_parts = []
        formatted_chunks = []
        prev_is_incomplete = False
        last_item = json_data['data'][-1]
        write_page_index = -1
        prev_item = None
        
        for item in json_data['data']:
            item_type = item.get('type')
            page_index = item.get('page_index', 'N/A')
            content = item.get('content', '')
            caption = item.get('caption')
            s3_key = item.get('s3_key')
            is_incomplete = item.get('is_incomplete', False)

            # 직전에 완료되지 않은 문구가 있었다면.
            if prev_is_incomplete:
                if prev_item['type'] == item_type: # 이전에 완료된거랑 다음 시작이랑 타입이 같아야만...
                    content = prev_item['content'] + content
                    item['content'] = content
                else: # 타입이 다르면 중간에 개행처리만 한번 넣어준다.
                    content = prev_item['content'] + "\n" + content
                    item['content'] = content

            if item == last_item:  # 마지막 항목이면 무조건 False!
                is_incomplete = False

            # 완료되지 않은 문구에 대한 처리는 문단, 표, 제목만.
            # 완료되지 않은 경우 다음 item 에 처리를 위임한다.
            if is_incomplete and item_type in ('paragraph', 'table', 'sub_title'):
                prev_item = item
                prev_is_incomplete = is_incomplete
                continue
            else:
                is_incomplete = False

            chunk_parts = []

            # page_index 정보 추가
            if write_page_index != page_index:
                chunk_parts.append(f"[page_index: {page_index}]")
                write_page_index = page_index

            if item_type == 'sub_title':
                chunk_parts.append(f"## {content}")
            elif item_type == 'paragraph':
                chunk_parts.append(content)
            elif item_type == 'table':
                if caption:
                    chunk_parts.append(f"**표: {caption}**")
                chunk_parts.append(content)
            elif item_type == 'image':
                if s3_key:
                    chunk_parts.append(f"**그림: {caption if caption else 'no caption'}**")
                    chunk_parts.append(f"URL: {generate_cdn_url(s3_key, self.config.cdn_url)}")
                    if content:
                        chunk_parts.append(f"내용: {content}")
            elif item_type == 'etc':
                original_type = item.get('original_type', '기타')
                chunk_parts.append(f"[{original_type.upper()}]: {content}")
            else:
                chunk_parts.append(content)
            
            formatted_chunks.append("\n".join(chunk_parts))
            formatted_chunks.append("\n\n")
            prev_is_incomplete = is_incomplete

        return "".join(formatted_chunks)

    def _save_results(self, final_json: dict, base_filename: str):
        """최종 결과를 JSON과 마크다운 파일로 저장"""
        # JSON 파일 저장
        json_output_path = os.path.join(self.config.output_dir, f"{base_filename}.json")
        save_json_file(final_json, json_output_path)
        
        # 마크다운 파일 저장
        print("Formatting JSON to Markdown...")
        markdown_content = self._format_json_to_markdown(final_json)
        md_output_path = os.path.join(self.config.output_dir, f"{base_filename}.md")
        save_markdown_file(markdown_content, md_output_path) 