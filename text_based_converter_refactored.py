import fitz  # PyMuPDF
import camelot
import pandas as pd
import os
import re
import json
from collections import defaultdict, Counter

# 공통 유틸리티 import
from utils import (
    PDFProcessorConfig, S3Manager, CAPTION_PATTERNS, 
    get_safe_filename, generate_cdn_url, save_json_file, save_markdown_file,
    ensure_directory_exists, safe_save_pixmap, starts_with_list_item
)

# Pandas 출력 옵션 (디버깅용)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)


class TextBasedConverter:
    """
    텍스트 기반 PDF를 분석하고 마크다운 및 JSON으로 변환하는 클래스.
    PdfProcessor 인터페이스에 맞게 설계됨.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        PdfProcessor 인터페이스에 맞는 생성자
        
        Args:
            output_dir: 출력 디렉토리
        """
        self.config = PDFProcessorConfig(output_dir=output_dir)
        self.s3_manager = self.config.get_s3_manager()
        
        # 컴파일된 정규식 패턴들
        self.caption_patterns = {
            key: re.compile(pattern, re.IGNORECASE) 
            for key, pattern in CAPTION_PATTERNS.items()
        }

    def convert(self, pdf_path: str):
        """
        PdfProcessor 인터페이스에 맞는 메인 변환 함수 - 새로운 3단계 플로우
        
        Args:
            pdf_path: 변환할 PDF 파일 경로
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return

        print(f"Starting PDF conversion with new 3-step flow...")
        
        # 1단계: 전체 문서 메타데이터 분석
        print("Step 1: Analyzing document metadata...")
        all_document_elements = self._analyze_document_layout(doc)
        print(f"Found {len(all_document_elements)} elements in document")
        
        # 2단계: 미디어 콘텐츠 추출 (필요한 페이지만 선택적 로드)
        print("Step 2: Extracting media contents...")
        media_elements = self._extract_contents(doc, all_document_elements, pdf_path)
        
        # 3단계: 미디어 콘텐츠와 영역이 겹치지 않는 text 요소 추출
        text_elements = []
        for tel in all_document_elements:
            if tel['type'] in ['text', 'title', 'etc']:
                if not any(mel['bbox'].intersects(tel['bbox']) for mel in media_elements if tel['page_index'] == mel['page_index']):
                    text_elements.append(tel)

        media_elements = [mel for mel in media_elements if mel['type'] != 'caption']

        # 4단계: 텍스트 요소와 미디어 요소 통합
        final_elements = text_elements + media_elements
        
        # 5단계: 후처리 - 전역 문단 그룹화 및 최종 출력 생성
        print("Step 3: Processing document-wide paragraph grouping and generating output...")
        final_markdown, final_json_elements = self._process_and_generate_output_new(final_elements)

        # 결과 저장
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self._save_results_new(final_markdown, final_json_elements, pdf_name)
        
        doc.close()
        print(f"✅ PDF conversion completed: {pdf_name}")

    def _process_and_generate_output_new(self, all_final_elements: list) -> tuple:
        """새로운 방식의 전체 문서 요소를 대상으로 문단 그룹화, 정렬, 최종 출력 생성"""
        
        all_final_elements.sort(key=lambda x: x["sort_key"])

        rearranged_elements = self._rearrange_elements(all_final_elements)
        json_elements = self._prepare_json_elements(rearranged_elements)
        final_markdown = self._generate_final_markdown(rearranged_elements)
        

        return final_markdown, json_elements

    def _group_text_blocks_into_paragraphs_global_new(self, text_elements: list) -> list:
        """전체 문서 텍스트 요소를 문단으로 그룹화 (페이지 경계 고려)"""
        import pandas as pd
        import fitz
        
        if not text_elements:
            return []

        # 정렬 기준: (page_index, y0)로 정렬
        sorted_blocks = sorted(text_elements, key=lambda el: el['sort_key'])

        # 단일 라인 높이 추정을 위한 분석
        single_line_heights = []
        for block in sorted_blocks:
            bbox = block["bbox"]
            text_lines = block["content"].splitlines()
            if len(text_lines) == 1 and bbox.height > 0:
                single_line_heights.append(bbox.height)
            elif len(text_lines) > 1 and bbox.height > 0:
                single_line_heights.append(bbox.height / len(text_lines))
        
        median_line_height = pd.Series(single_line_heights).median() if single_line_heights else 12
        paragraph_vertical_threshold = median_line_height * 0.8  # 조금 더 관대한 임계값
        indent_threshold = 5

        paragraphs = []
        if not sorted_blocks:
            return paragraphs

        current_paragraph_text = sorted_blocks[0]["content"].strip()
        # bbox 복사본 생성하여 원본 보호
        current_paragraph_bbox = fitz.Rect(sorted_blocks[0]["bbox"])
        current_paragraph_type = sorted_blocks[0]["type"]
        current_page_index = sorted_blocks[0]["page_index"]
        # 페이지 범위 추적
        paragraph_pages = [current_page_index]

        for i in range(1, len(sorted_blocks)):
            block = sorted_blocks[i]
            prev_block = sorted_blocks[i-1]

            # 페이지가 바뀌었는지 확인
            page_changed = block["page_index"] != prev_block["page_index"]
            
            if page_changed:
                is_same_type = block["type"] == prev_block["type"]
                # 페이지가 바뀐 경우: 더 관대한 문단 연결 조건
                prev_text_ends_complete = current_paragraph_text.rstrip().endswith(('.', '!', '?', '。', '！', '？'))
                next_text_starts_new = (
                    block["bbox"].x0 > (prev_block["bbox"].x0 + indent_threshold)  # 들여쓰기
                )
                
                is_new_paragraph = is_same_type and (prev_text_ends_complete or next_text_starts_new)
            else:
                # 같은 페이지 내에서는 기존 로직 사용
                vertical_gap = block["bbox"].y0 - prev_block["bbox"].y1
                is_new_by_gap = vertical_gap > paragraph_vertical_threshold
                is_new_by_indent = block["bbox"].x0 > (prev_block["bbox"].x0 + indent_threshold)
                is_new_paragraph = is_new_by_gap or is_new_by_indent

            if is_new_paragraph:
                # 현재 문단 완료
                paragraphs.append({
                    "type": current_paragraph_type,
                    "bbox": current_paragraph_bbox,  # 이미 복사본이므로 안전
                    "content": current_paragraph_text,
                    "page_index": current_page_index,  # 시작 페이지
                    "page_range": paragraph_pages,  # 포함된 모든 페이지
                    "sort_key": (current_page_index, current_paragraph_bbox.y0)
                })
                
                # 새 문단 시작
                current_paragraph_text = block["content"].rstrip("\n")
                current_paragraph_bbox = fitz.Rect(block["bbox"])  # 새로운 복사본 생성
                current_paragraph_type = block["type"]
                current_page_index = block["page_index"]
                paragraph_pages = [current_page_index]
            else:
                # 기존 문단에 연결
                if page_changed:
                    # 페이지가 바뀌었지만 같은 문단인 경우, 공백 하나만 추가
                    current_paragraph_text += " " + block["content"].rstrip("\n")
                    # 새 페이지 추가
                    if block["page_index"] not in paragraph_pages:
                        paragraph_pages.append(block["page_index"])
                else:
                    # 같은 페이지 내 연결
                    current_paragraph_text += " " + block["content"].rstrip("\n")
                
                # bbox 확장 (복사본이므로 원본에 영향 없음)
                current_paragraph_bbox.include_rect(block["bbox"])

        # 마지막 문단 추가
        paragraphs.append({
            "type": current_paragraph_type,
            "bbox": current_paragraph_bbox,  # 복사본이므로 안전
            "content": current_paragraph_text,
            "page_index": current_page_index,  # 시작 페이지
            "page_range": paragraph_pages,  # 포함된 모든 페이지
            "sort_key": (current_page_index, current_paragraph_bbox.y0)
        })

        return paragraphs

    def _analyze_indentation_within_body(self, doc: fitz.Document, body_rect: fitz.Rect) -> dict:
        """본문 영역 내의 텍스트를 분석하여 가장 흔한 두 개의 들여쓰기 값을 찾습니다."""
        all_x_coords = []
        for page in doc:
            # 본문 영역에 포함되는 블록만 필터링
            body_blocks = [b for b in page.get_text("blocks") if fitz.Rect(b[:4]).intersects(body_rect)]
            for block in body_blocks:
                # x0 좌표를 소수점 첫째 자리까지 반올림하여 그룹화
                all_x_coords.append(round(block[0], 1))

        # 가장 흔한 x-좌표 2개 찾기
        indent_counts = Counter(all_x_coords)
        most_common_indents = indent_counts.most_common(2)

        start_indent = None
        continuation_indent = None

        if len(most_common_indents) == 2:
            indent1, count1 = most_common_indents[0]
            indent2, count2 = most_common_indents[1]
            
            continuation_indent = min(indent1, indent2)
            start_indent = max(indent1, indent2)

        elif len(most_common_indents) == 1:
            continuation_indent = most_common_indents[0][0]
        
        return {"start": start_indent, "continuation": continuation_indent}

    def _analyze_document_font_info(self, doc: fitz.Document) -> dict:
        """전체 문서의 폰트 정보를 분석하여 가장 많이 사용되는 size, font를 plain text 기준으로 설정"""
        font_info_lengths = {}
        
        # 전체 문서를 순회하면서 모든 텍스트의 폰트 정보 및 텍스트 길이 수집
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text_info = page.get_text('dict')

            for block in text_info['blocks']:
                if block['type'] == 0 and block.get('lines', []):
                    for line in block['lines']:
                        for span in line.get('spans', []):
                            text = span.get('text', '').strip()
                            # 텍스트가 있는 span만 고려
                            if text:
                                font_info = (span['size'], span['font'], span['bbox'][3] - span['bbox'][1])
                                font_info_lengths[font_info] = font_info_lengths.get(font_info, 0) + len(text)

        # 텍스트 길이가 가장 긴 폰트 정보 찾기
        if font_info_lengths:
            most_common_font_info_key = max(font_info_lengths, key=font_info_lengths.get)
            plain_size, plain_font, plain_height = most_common_font_info_key
            return {
                'plain_size': plain_size, 
                'plain_font': plain_font,
                'plain_height': plain_height,
                # 'all_font_counts': font_info_lengths
            }
        return {
            'plain_size': 10.0, 
            'plain_font': 'unknown',
            'plain_height': 12,
            # 'all_font_counts': Counter()
        }

    def _classify_text_by_font(self, span: dict, font_info: dict) -> str:
        """폰트 정보를 기반으로 텍스트를 text, title, etc로 분류"""
        span_size = span.get('size', 0)
        span_font = span.get('font', '')
        
        plain_size = font_info['plain_size']
        plain_font = font_info['plain_font']
        
        # text: 가장 많이 사용되는 size, font와 정확히 일치
        if span_size == plain_size: # and span_font == plain_font:
            return "text"
        
        # title: text와 다르지만 사이즈가 같거나 큰 경우
        elif span_size > plain_size:
            return "title"
        
        # etc: 그 외의 모든 경우
        else:
            return "etc"

    def _extract_all_text_blocks_with_classification(self, doc: fitz.Document, font_info: dict) -> list:
        """전체 문서에서 모든 텍스트 블록을 추출하고 분류"""
        all_text_blocks = []
        
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text_info = page.get_text('dict')
            
            for block in text_info['blocks']:
                if block['type'] == 0 and block.get('lines', []):
                    # 블록의 전체 텍스트 수집
                    span_text_list = []
                    last_span = None
                    # block['lines']를 y0 기준으로 정렬
                    # block['lines'].sort(key=lambda line: (line['bbox'][1], line['bbox'][0]))  # y0 기준으로 정렬 후 x0 기준으로 정렬
                    
                    for line in block['lines']:
                        span_list = line.get('spans', [])
                        # span_list.sort(key=lambda span: span['bbox'][0]) # x0 기준으로 정렬
                        for span in span_list:
                            if span.get('text', '').strip():
                                span_text_list.append(span['text'])
                                last_span = span
                    
                    if span_text_list and last_span:
                        text_type = self._classify_text_by_font(span, font_info)

                        all_text_blocks.append({
                            'page_index': page_index,
                            'bbox': fitz.Rect(block['bbox']),
                            'text': " ".join(span_text_list),
                            'type': text_type,
                            'font_size': span.get('size', 0),
                            'font_name': span.get('font', '')
                        })
        
        return all_text_blocks

    def _detect_header_footer_from_text_bounds(self, all_text_blocks: list, doc: fitz.Document) -> dict:
        """text 타입 블록들의 경계를 기준으로 여백 영역을 계산"""
        text_blocks = [block for block in all_text_blocks if block['type'] == 'text']
        
        page_height = doc[0].rect.height if doc else 800
        page_width = doc[0].rect.width if doc else 600
        
        if not text_blocks:
            # text 블록이 없으면 기본값 반환
            return {
                'header_y_max': 0, 
                'footer_y_min': page_height, 
                'left_margin_x_max': 0, 
                'right_margin_x_min': page_width
            }
        
        # text 블록들의 경계 계산
        min_x0 = min(block['bbox'].x0 for block in text_blocks)
        max_x1 = max(block['bbox'].x1 for block in text_blocks)
        min_y0 = min(block['bbox'].y0 for block in text_blocks)
        max_y1 = max(block['bbox'].y1 for block in text_blocks)
        
        # 여백 추가 (픽셀 단위)
        margin_padding = 10  # 여백 패딩
        
        header_y_max = max(0, min_y0 - margin_padding)
        footer_y_min = min(page_height, max_y1 + margin_padding)
        left_margin_x_max = max(0, min_x0 - margin_padding)
        right_margin_x_min = min(page_width, max_x1 + margin_padding)
        
        return {
            'header_y_max': header_y_max,
            'footer_y_min': footer_y_min,
            'left_margin_x_max': left_margin_x_max,
            'right_margin_x_min': right_margin_x_min,
        }

    def _detect_header_footer_from_etc_texts(self, all_text_blocks: list, doc: fitz.Document) -> dict:
        """etc로 분류된 텍스트 중에서 header/footer/right_margin 패턴을 감지하고 좌표를 추출"""
        etc_blocks = [block for block in all_text_blocks if block['type'] in ('title', 'etc')]
        
        if not etc_blocks:
            return {'header_y_max': 0, 'footer_y_min': doc[0].rect.height if doc else 0, 'left_margin_x_max': 0, 'right_margin_x_min': doc[0].rect.width if doc else 0}
        
        header_candidates = defaultdict(list)
        footer_candidates = defaultdict(list)
        right_margin_candidates = defaultdict(list)
        left_margin_candidates = defaultdict(list)
        
        page_height = doc[0].rect.height if doc else 800
        page_width = doc[0].rect.width if doc else 600
        header_margin_ratio = 0.15
        footer_margin_ratio = 0.85
        right_margin_ratio = 0.85
        left_margin_ratio = 0.15
        
        # etc 텍스트들을 페이지별로 그룹화하고 header/footer/right_margin/left_margin 후보 찾기
        for block in etc_blocks:
            page_index = block['page_index']
            bbox = block['bbox']
            text = block['text']
            
            # 숫자를 제거한 패턴으로 그룹화 (페이지 번호 등을 일반화)
            pattern = re.sub(r'\d+', '', text).strip()
            
            if bbox.y1 < page_height * header_margin_ratio:
                header_candidates[pattern].append((page_index + 1, bbox))
            elif bbox.y0 > page_height * footer_margin_ratio:
                footer_candidates[pattern].append((page_index + 1, bbox))
            elif bbox.x1 < page_width * left_margin_ratio:
                left_margin_candidates[pattern].append((page_index + 1, bbox))
            elif bbox.x0 > page_width * right_margin_ratio:
                right_margin_candidates[pattern].append((page_index + 1, bbox))
            
        # 반복되는 패턴 식별
        y_min_occurrence = max(len(doc) // 10, 3)
        x_min_occurrence = 3
        
        # Header y_max 계산
        header_y_max = 0
        valid_header_patterns = []
        for pattern, occurrences in header_candidates.items():
            if len(occurrences) >= y_min_occurrence:
                valid_header_patterns.append((pattern, occurrences))
        
        if valid_header_patterns:
            # 가장 많이 등장하는 패턴을 선택 (출현 빈도 기준)
            most_frequent_pattern = max(valid_header_patterns, key=lambda x: len(x[1]))
            pattern, occurrences = most_frequent_pattern
            # 해당 패턴에서 가장 많이 등장하는 y1 좌표를 선택
            y1_counts = Counter(bbox.y1 for _, bbox in occurrences)
            most_common_y1 = y1_counts.most_common(1)[0][0] if y1_counts else 0
            header_y_max = most_common_y1
        
        # Footer y_min 계산
        footer_y_min = page_height
        valid_footer_patterns = []
        for pattern, occurrences in footer_candidates.items():
            if len(occurrences) >= y_min_occurrence:
                valid_footer_patterns.append((pattern, occurrences))
        
        if valid_footer_patterns:
            # 가장 많이 등장하는 패턴을 선택 (출현 빈도 기준)
            most_frequent_pattern = max(valid_footer_patterns, key=lambda x: len(x[1]))
            pattern, occurrences = most_frequent_pattern
            # 해당 패턴에서 가장 많이 등장하는 y0 좌표를 선택
            y0_counts = Counter(bbox.y0 for _, bbox in occurrences)
            most_common_y0 = y0_counts.most_common(1)[0][0] if y0_counts else page_height
            footer_y_min = most_common_y0

        # Left margin x_max 계산
        left_margin_x_max = 0
        valid_left_margin_patterns = []
        for pattern, occurrences in left_margin_candidates.items():
            if len(occurrences) >= x_min_occurrence:
                valid_left_margin_patterns.append((pattern, occurrences))
        
        if valid_left_margin_patterns:
            # 가장 많이 등장하는 패턴을 선택 (출현 빈도 기준)
            most_frequent_pattern = max(valid_left_margin_patterns, key=lambda x: len(x[1]))
            pattern, occurrences = most_frequent_pattern
            # 해당 패턴에서 가장 많이 등장하는 x1 좌표를 선택
            x1_counts = Counter(bbox.x1 for _, bbox in occurrences)
            most_common_x1 = x1_counts.most_common(1)[0][0] if x1_counts else 0
            left_margin_x_max = most_common_x1
        
        # Right margin x_min 계산
        right_margin_x_min = page_width
        valid_right_margin_patterns = []
        for pattern, occurrences in right_margin_candidates.items():
            if len(occurrences) >= x_min_occurrence:
                valid_right_margin_patterns.append((pattern, occurrences))
        
        if valid_right_margin_patterns:
            # 가장 많이 등장하는 패턴을 선택 (출현 빈도 기준)
            most_frequent_pattern = max(valid_right_margin_patterns, key=lambda x: len(x[1]))
            pattern, occurrences = most_frequent_pattern
            # 해당 패턴에서 가장 많이 등장하는 x0 좌표를 선택
            x0_counts = Counter(bbox.x0 for _, bbox in occurrences)
            most_common_x0 = x0_counts.most_common(1)[0][0] if x0_counts else page_width
            right_margin_x_min = most_common_x0
        
        return {
            'header_y_max': header_y_max,
            'footer_y_min': footer_y_min,
            'left_margin_x_max': left_margin_x_max,
            'right_margin_x_min': right_margin_x_min,
        }

    def _detect_captions_from_etc_texts(self, all_text_blocks: list) -> dict:
        """etc로 분류된 텍스트 중에서 정규식 패턴으로 caption을 감지"""
        etc_blocks = [block for block in all_text_blocks if block['type'] == 'etc']
        
        detected_captions = {"table": [], "figure": []}
        
        for block in etc_blocks:
            text = block['text']
            
            # 정규식 패턴 매칭
            for caption_type, pattern in self.caption_patterns.items():
                match = pattern.search(text)
                if match:
                    detected_captions[caption_type].append({
                        "page_index": block['page_index'],
                        "bbox": block["bbox"],
                        "text": text.strip(),
                        "number": match.group(2) if len(match.groups()) >= 2 else ""
                    })
                    break  # 하나의 블록은 하나의 캡션만 가진다고 가정
        
        return detected_captions

    def _analyze_document_layout(self, doc: fitz.Document) -> list:
        """전체 문서를 분석하여 모든 요소의 메타데이터를 통합 리스트로 반환"""
        
        # 1. 전체 문서의 폰트 정보 분석 (가장 많이 사용되는 size, font를 plain text로 설정)
        font_info = self._analyze_document_font_info(doc)
        
        # 2. 모든 텍스트 블록을 추출하고 분류 (text, title, etc)
        all_text_blocks = self._extract_all_text_blocks_with_classification(doc, font_info)
        
        # 3. text 블록들의 경계 기준으로 여백 영역 계산
        header_footer_info = self._detect_header_footer_from_text_bounds(all_text_blocks, doc)
        
        
        # 4. etc 텍스트 중에서 caption 감지
        detected_captions = self._detect_captions_from_etc_texts(all_text_blocks)
        
        # 5. 통합된 요소 리스트 생성
        all_document_elements = []
        
        # 5-1. 텍스트 요소 추가 (header/footer/right_margin 제외, caption 제외)
        header_y_max = header_footer_info['header_y_max']
        footer_y_min = header_footer_info['footer_y_min']
        left_margin_x_max = header_footer_info['left_margin_x_max']
        right_margin_x_min = header_footer_info['right_margin_x_min']
        
        # # 캡션 bbox들을 수집 (텍스트에서 제외하기 위해)
        # caption_bboxes_by_page = {}
        # for caption_type in detected_captions.values():
        #     for caption in caption_type:
        #         page_idx = caption['page_index']
        #         if page_idx not in caption_bboxes_by_page:
        #             caption_bboxes_by_page[page_idx] = []
        #         caption_bboxes_by_page[page_idx].append(caption["bbox"])
        
        for text_block in all_text_blocks:
            bbox = text_block['bbox']
            page_idx = text_block['page_index']
            
            # header/footer/right_margin 영역 제외
            if bbox.y1 <= header_y_max or bbox.y0 >= footer_y_min or bbox.x0 >= right_margin_x_min or bbox.x1 <= left_margin_x_max:
                continue
                
            # # 같은 페이지의 캡션과만 겹치는지 확인
            # page_caption_bboxes = caption_bboxes_by_page.get(page_idx, [])
            # is_overlapping = any(bbox.intersects(cap_bbox) for cap_bbox in page_caption_bboxes)
            # if is_overlapping:
            #     continue
                
            # # etc 타입은 제외 (일반적으로 노이즈)
            # if text_block['type'] == 'etc':
            #     continue
                
            all_document_elements.append({
                "type": text_block['type'],  # "text" or "title" or 'etc'
                "page_index": text_block['page_index'],
                "bbox": bbox,
                "content": text_block['text'],
                "sort_key": (text_block['page_index'], bbox.y0)
            })
        
        # 5-2. 드로잉/이미지 메타데이터 추가 (페이지 재로드 최적화)
        pages_data = {}  # 페이지별 데이터 캐싱
        
        for page_index in range(len(doc)):
            # 페이지 로드 및 캐싱
            if page_index not in pages_data:
                page = doc.load_page(page_index)
                pages_data[page_index] = {
                    'drawings': self._absorb_small_rects(page.cluster_drawings()),
                    'images': page.get_images(full=True),
                    'page_obj': page
                }
            
            page_data = pages_data[page_index]
            
            # 드로잉 요소 메타데이터
            for i, drawing in enumerate(page_data['drawings']):
                all_document_elements.append({
                    "type": "drawing_meta",
                    "page_index": page_index,
                    "bbox": drawing,
                    "content": {"drawing_id": i},
                    "sort_key": (page_index, drawing.y0)
                })
            
            # 이미지 요소 메타데이터
            for i, img_info in enumerate(page_data['images']):
                try:
                    img_bbox = page_data['page_obj'].get_image_bbox(img_info)
                    all_document_elements.append({
                        "type": "image_meta",
                        "page_index": page_index,
                        "bbox": img_bbox,
                        "content": {"img_info": img_info, "img_xref": img_info[0]},
                        "sort_key": (page_index, img_bbox.y0)
                    })
                except:
                    continue  # 이미지 정보 추출 실패시 스킵
        
        # 5-3. 캡션 메타데이터 추가
        for caption_type, captions in detected_captions.items():
            for caption in captions:
                all_document_elements.append({
                    "type": f"{caption_type}_caption",  # "table_caption" or "figure_caption"
                    "page_index": caption['page_index'],
                    "bbox": caption['bbox'],
                    "content": {
                        "text": caption['text'],
                        "number": caption.get('number', ''),
                        "caption_type": caption_type
                    },
                    "sort_key": (caption['page_index'], caption['bbox'].y0)
                })
        
        # 6. 전체 정렬 (페이지 순서, y좌표 순서)
        all_document_elements.sort(key=lambda x: x['sort_key'])
        
        # 메타데이터도 함께 반환 (후처리에서 필요할 수 있음)
        self._doc_metadata = {
            "font_info": font_info,
            "header_footer_info": header_footer_info,
            # "detected_captions": detected_captions
        }
        
        print(f"📊 Text analysis summary:")
        print(f"  - Total text blocks found: {len(all_text_blocks)}")
        print(f"  - Plain text blocks: {len([t for t in all_text_blocks if t['type'] == 'text'])}")
        print(f"  - Title blocks: {len([t for t in all_text_blocks if t['type'] == 'title'])}")
        print(f"  - ETC blocks: {len([t for t in all_text_blocks if t['type'] == 'etc'])}")
        print(f"  - Final included text elements: {len([e for e in all_document_elements if e['type'] in ['text', 'title']])}")
        print(f"  - Pages with text: {sorted(set([t['page_index'] for t in all_text_blocks]))}")
        print(f"  - Pages in final elements: {sorted(set([e['page_index'] for e in all_document_elements]))}")
        
        # 페이지별 텍스트 분포 확인
        pages_with_included_text = set([e['page_index'] for e in all_document_elements if e['type'] in ['text', 'title']])
        all_pages = set(range(len(doc)))
        missing_text_pages = all_pages - pages_with_included_text
        if missing_text_pages:
            print(f"  ⚠️  Pages without included text: {sorted(missing_text_pages)}")
        
        def json_format(data):
            return str(data)
        
        
        # 디버깅을 위해 all_document_elements와 self._doc_metadata를 JSON 파일로 저장
        debug_dir = os.path.join(self.config.output_dir, "debug")
        ensure_directory_exists(debug_dir)
        
        try:
            # all_document_elements 저장
            elements_path = os.path.join(debug_dir, "all_document_elements.json")
            with open(elements_path, 'w', encoding='utf-8') as f:
                # fitz.Rect 객체는 직렬화할 수 없으므로 문자열로 변환
                serializable_elements = []
                for element in all_document_elements:
                    element_copy = element.copy()
                    if 'bbox' in element_copy and isinstance(element_copy['bbox'], fitz.Rect):
                        element_copy['bbox'] = str(element_copy['bbox'])
                    serializable_elements.append(element_copy)
                json.dump(serializable_elements, f, ensure_ascii=False, indent=2, default=json_format)
            
            # self._doc_metadata 저장
            metadata_path = os.path.join(debug_dir, "doc_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                # 직렬화할 수 없는 객체 처리
                serializable_metadata = {}
                for key, value in self._doc_metadata.items():
                    # if key == 'detected_captions':
                    #     serializable_captions = {}
                    #     for caption_type, captions in value.items():
                    #         serializable_captions[caption_type] = []
                    #         for caption in captions:
                    #             caption_copy = caption.copy()
                    #             if 'bbox' in caption_copy and isinstance(caption_copy['bbox'], fitz.Rect):
                    #                 caption_copy['bbox'] = str(caption_copy['bbox'])
                    #             serializable_captions[caption_type].append(caption_copy)
                    #     serializable_metadata[key] = serializable_captions
                    # else:
                    serializable_metadata[key] = value
                json.dump(serializable_metadata, f, ensure_ascii=False, indent=2, default=json_format)
            
            print(f"✅ 디버그 정보가 저장되었습니다: {debug_dir}")
        except Exception as e:
            print(f"⚠️ 디버그 정보 저장 중 오류 발생: {e}")
        
        return all_document_elements

    def _extract_contents(self, doc: fitz.Document, all_document_elements: list, pdf_path: str) -> list:
        """메타데이터 기반으로 미디어 파일들을 실제로 추출하고 저장"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 페이지별로 그룹화하여 효율적으로 처리
        pages_to_process = {}
        for element in all_document_elements:
            if element['type'] in ['image_meta', 'drawing_meta', 'table_caption', 'figure_caption']:
                page_index = element['page_index']
                if page_index not in pages_to_process:
                    pages_to_process[page_index] = []
                pages_to_process[page_index].append(element)
        
        # 실제 파일 저장을 위한 처리된 요소들
        processed_elements = []
        
        for page_index, page_elements in pages_to_process.items():                
            page = doc.load_page(page_index)
            page_num = page_index + 1
            
            print(f"Extracting media content from page {page_num}/{len(doc)}...")
            
            # 페이지의 요소들을 타입별로 분류
            images = [el for el in page_elements if el['type'] == 'image_meta']
            drawings = [el for el in page_elements if el['type'] == 'drawing_meta']
            table_captions = [el for el in page_elements if el['type'] == 'table_caption']
            figure_captions = [el for el in page_elements if el['type'] == 'figure_caption']

            # 같은 페이지의 etc 텍스트 수집 (y축 레이블 텍스트 포함)
            page_etc_rects = []
            page_text_rects = []
            for element in all_document_elements:
                if (element['page_index'] == page_index and
                    element.get('content')):
                    if element['type'] in ('etc', 'title'):
                        page_etc_rects.append(element['bbox'])
                    else:
                        page_text_rects.append(element['bbox'])
            
            # 1. 표(Table) 처리
            if table_captions and drawings:
                for caption_element in table_captions:
                    caption_bbox = caption_element['bbox']
                    caption_text = caption_element['content']['text']
                    drawing_elements = []

                    for drawing_element in drawings:
                        drawing_rect = drawing_element['bbox']
                        is_overlapping = any(
                            drawing_rect == el['bbox'] for el in processed_elements if el['page_index'] == page_index
                        )
                        if not is_overlapping:
                            drawing_elements.append(drawing_element)

                    # 가장 가까운 드로잉 찾기
                    if drawing_elements:
                        closest_drawing_element = min(
                            drawing_elements,
                            key=lambda d: self._calculate_closest_distance_to_outline(
                                caption_bbox, d['bbox']
                            )
                        )
                        closest_drawing = closest_drawing_element['bbox']
                        
                        # 캡션과 drawing 간의 거리가 15픽셀 이내인지 확인
                        distance = self._calculate_closest_distance_to_outline(caption_bbox, closest_drawing)
                        if distance > 15:
                            continue  # 거리가 너무 멀면 건너뛰기
                        
                        # Camelot으로 표 추출
                        x0, y0, x1, y1 = closest_drawing
                        camelot_y1_top = page.rect.height - y0
                        camelot_y2_bottom = page.rect.height - y1
                        table_area_str = f"{x0},{camelot_y1_top},{x1},{camelot_y2_bottom}"
                        
                        try:
                            tables = camelot.read_pdf(pdf_path, pages=str(page_num), 
                                                   flavor='stream', table_areas=[table_area_str])
                            if tables.n > 0:
                                for table in tables:
                                    processed_elements.append({
                                        "type": "caption",
                                         "page_index": page_index,
                                        "bbox": caption_bbox,
                                        "content": caption_text,
                                        "sort_key": (page_index, caption_bbox.y0)
                                    })
                                    processed_elements.append({
                                        "type": "table",
                                        "page_index": page_index,
                                        "bbox": closest_drawing,
                                        "caption": caption_text,
                                        "content": table.df,
                                        "sort_key": (page_index, closest_drawing.y0)
                                    })
                        except Exception as e:
                            print(f"[Warning] Page {page_num}: Camelot failed for table near '{caption_text}'. Error: {e}")
            
            # 2. 이미지 처리
            for image_element in images:
                pix = None
                closest_caption = None

                try:
                    img_info = image_element['content']['img_info']
                    img_xref = image_element['content']['img_xref']
                    bbox = image_element['bbox']
                    
                    # 가장 가까운 figure caption 찾기
                    page_figure_captions = [c for c in figure_captions]
                    if page_figure_captions:
                        closest_caption = min(
                            page_figure_captions,
                            key=lambda c: self._calculate_closest_distance_to_outline(c['bbox'], bbox)
                        )
                        caption_text = closest_caption['content']['text']
                    else:
                        caption_text = "NO CAPTION"
                        if not bbox.height or not(0.15 < bbox.width / bbox.height < 9.5):
                            continue # 종횡비 이상치 체크
                    
                    # 파일 저장
                    safe_caption = get_safe_filename(caption_text, 30)
                    unique_filename = f"page_{page_index}_pic_{safe_caption}_{img_xref}.png"
                    output_path = os.path.join(self.config.output_dir, 'imgs', unique_filename)
                    
                    ensure_directory_exists(os.path.dirname(output_path))

                    need_clip = False
                    clip_rect = fitz.Rect(bbox)
                    for etc_bbox in page_etc_rects:
                        if etc_bbox.intersects(bbox):
                            need_clip = True
                            clip_rect.include_rect(etc_bbox)
                            
                    if need_clip:
                        PADDING = 3
                        clip_rect.x0 -= PADDING
                        clip_rect.y0 -= PADDING
                        clip_rect.x1 += PADDING
                        clip_rect.y1 += PADDING
                        clip_rect = clip_rect & page.rect
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect)
                    else:
                        # 이미지 영역 내에 text 가 있는지 체크해서 있으면 캡쳐링으로 저장하도록 수정해야함.!!!!
                        pix = fitz.Pixmap(doc, img_xref)
                    
                    if safe_save_pixmap(pix, output_path):
                        s3_key = self.s3_manager.upload_file_with_pdf_structure(output_path, pdf_name)

                        if closest_caption:
                            processed_elements.append({
                                "type": "caption",
                                "page_index": page_index,
                                "bbox": closest_caption['bbox'],
                                "content": closest_caption['content']['text'],
                                "sort_key": (page_index, closest_caption['bbox'].y0)
                            })
                        processed_elements.append({
                            "type": "image",
                            "page_index": page_index,
                            "bbox": bbox,
                            "caption": caption_text,
                            "filename": unique_filename,
                            "s3_key": s3_key,
                            "sort_key": (page_index, bbox.y0)
                        })
                    else:
                        print(f"[Error] Failed to save image {img_xref}")
                    
                except (ValueError, RuntimeError) as e:
                    print(f"[Warning] Page {page_num}: Failed to process image xref {img_info[0]}. Error: {e}")
            
            # 3. 차트/그림 처리 (caption이 있는 drawing)
            if figure_captions:
                # 이미 처리된 이미지들과 겹치지 않는 drawing들 필터링
                available_drawings = []
                
                for drawing_element in drawings:
                    drawing_rect = drawing_element['bbox']
                    is_overlapping = any(
                        drawing_rect.intersects(el['bbox']) for el in processed_elements if el['page_index'] == page_index
                    )
                    if not is_overlapping and 0.15 < drawing_rect.width / drawing_rect.height < 9.5:
                        available_drawings.append(drawing_element)
                
                for caption_element in figure_captions:
                    if not available_drawings:
                        break
                        
                    caption_bbox = caption_element['bbox']
                    caption_text = caption_element['content']['text']
                    
                    # 스마트한 차트 선택 + 축 레이블 병합
                    drawing_rects = [d['bbox'] for d in available_drawings]
                    
                    # 디버깅 모드에서 drawing 분류 검증
                    debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
                    if debug_mode:
                        print(f"\n🔍 Processing caption: '{caption_text}' on page {page_index + 1}")
                        classification_stats = self._validate_drawing_classification(drawing_rects, page.rect)
                        print(f"Available drawings for chart selection: {len(drawing_rects)}")
                        print(f"Available etc texts for axis labels: {len(page_etc_rects)}")
                    
                    closest_drawing = self._find_chart_with_axis_merge(
                        caption_bbox, drawing_rects, page.rect, etc_text_rects=page_etc_rects,
                        text_rects=page_text_rects
                    )
                    
                    if not closest_drawing:
                        continue
                    
                    # 캡션과 병합된 최종 영역 계산
                    PADDING = 3
                    merged_bbox = fitz.Rect(closest_drawing)
                    merged_bbox.include_rect(caption_bbox)
                    merged_bbox.x0 -= PADDING
                    merged_bbox.y0 -= PADDING
                    merged_bbox.x1 += PADDING
                    merged_bbox.y1 += PADDING
                    
                    # 파일 저장
                    safe_caption = get_safe_filename(caption_text, 30)
                    unique_filename = f"page_{page_index}_chart_{safe_caption}_{len(processed_elements)}.png"
                    output_path = os.path.join(self.config.output_dir, "imgs", unique_filename)
                    
                    try:
                        clip_rect = merged_bbox & page.rect
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect)
                        if not safe_save_pixmap(pix, output_path):
                            print(f"[Error] Failed to save chart image for caption '{caption_text}'")
                            pix = None
                            continue
                        pix = None
                        
                        s3_key = self.s3_manager.upload_file_with_pdf_structure(output_path, pdf_name)
                        
                        processed_elements.append({
                            "type": "caption",
                            "page_index": page_index,
                            "bbox": caption_bbox,
                            "content": caption_text,
                            "sort_key": (page_index, caption_bbox.y0)
                        })
                        processed_elements.append({
                            "type": "image",
                            "page_index": page_index,
                            "bbox": clip_rect,
                            "caption": caption_text,
                            "filename": unique_filename,
                            "s3_key": s3_key,
                            "sort_key": (page_index, clip_rect.y0)
                        })
                        
                        # 사용된 drawing 제거
                        available_drawings = [
                            d for d in available_drawings 
                            if not d['bbox'].intersects(closest_drawing)
                        ]
                        
                    except Exception as e:
                        print(f"[Warning] Page {page_num}: Failed to save chart for caption '{caption_text}'. Error: {e}")
        
        return processed_elements

    def _is_near_enough(self, large_rect: fitz.Rect, small_rect: fitz.Rect, tolerance: int) -> bool:
        """두 rect 간의 거리가 tolerance 이내인지 확인"""
        checked_rect = fitz.Rect(
            large_rect.x0 - tolerance, large_rect.y0 - tolerance, 
            large_rect.x1 + tolerance, large_rect.y1 + tolerance
        )
        return checked_rect.intersects(small_rect)    

    def _absorb_small_rects(self, rects: list, size_threshold: int = 20, distance_threshold: int = 25) -> list:
        """크기가 작고 메인 rect와 가까운 rect들을 흡수하는 로직"""
        if not rects:
            return []
        
        # 1. rect들을 크기 순으로 정렬 (큰 것부터)
        sorted_rects = sorted(rects, key=lambda r: r.width * r.height, reverse=True)
        
        # 2. 작은 rect와 큰 rect 분리
        small_rects = []
        large_rects = []
        
        for rect in sorted_rects:
            if rect.width <= size_threshold or rect.height <= size_threshold:
                small_rects.append(rect)
            else:
                large_rects.append(rect)
        
        # 3. 각 큰 rect에 대해 가까운 작은 rect들 흡수
        absorbed_rects = []
        
        for large_rect in large_rects:
            absorbed_rect = fitz.Rect(large_rect.x0, large_rect.y0, large_rect.x1, large_rect.y1)
            
            # 가까운 작은 rect들 찾기
            nearby_small_rects = []
            for small_rect in small_rects:
                if self._is_near_enough(large_rect, small_rect, distance_threshold):
                    nearby_small_rects.append(small_rect)
            
            # 찾은 작은 rect들을 큰 rect에 병합
            for small_rect in nearby_small_rects:
                absorbed_rect |= small_rect
            
            absorbed_rects.append(absorbed_rect)
        
        return absorbed_rects

    def _merge_drawings_with_text_blocks(self, drawing: fitz.Rect, text_blocks: list, 
                                        padding: int = 10, only_bottom_merge: bool = False) -> fitz.Rect:
        """드로잉 Rect와 텍스트 블록을 병합하는 함수"""
        checked_rect = fitz.Rect(
            drawing.x0 - padding, drawing.y0 - padding, 
            drawing.x1 + padding, drawing.y1 + padding
        )

        # 텍스트 블록을 drawing에서 가까운 순서대로 정렬
        if text_blocks:
            # 각 텍스트 블록과 drawing 사이의 거리 계산
            def calculate_distance(block):
                # drawing과 block_rect 중심점 간의 거리 계산
                drawing_center_x = (drawing.x0 + drawing.x1) / 2
                drawing_center_y = (drawing.y0 + drawing.y1) / 2
                block_center_x = (block.x0 + block.x1) / 2
                block_center_y = (block.y0 + block.y1) / 2
                
                return ((drawing_center_x - block_center_x)**2 + 
                        (drawing_center_y - block_center_y)**2)**0.5
            
            # 거리 기준으로 텍스트 블록 정렬
            text_blocks = sorted(text_blocks, key=calculate_distance)
        is_merged = False
        for block in text_blocks:
            text_rect = fitz.Rect(
                block[0] - padding, block[1] - padding, 
                block[2] + padding, block[3] + padding
            )
            if checked_rect.intersects(text_rect):
                if only_bottom_merge:
                    if text_rect.y1 > drawing.y1:
                        checked_rect.y1 = text_rect.y1
                        is_merged = True
                else:
                    checked_rect.include_rect(text_rect)
                    is_merged = True

        return checked_rect if is_merged else drawing

    def _remove_overlapping_bboxes(self, main_bboxs: list, ref_targets: list) -> list:
        """이미 추출된 요소의 bbox 영역에 속한 drawings 제거"""
        if not ref_targets or not main_bboxs:
            return main_bboxs

        filtered_drawings = []
        for drawing in main_bboxs:
            is_contained = False
            for element_bbox in ref_targets:
                # drawing이 이미 추출된 요소의 bbox 내에 포함되어 있는지 확인
                if element_bbox.contains(drawing):
                    is_contained = True
                    break
            if not is_contained:
                filtered_drawings.append(drawing)
        
        return filtered_drawings

    def _calculate_closest_distance_to_outline(self, caption_bbox: fitz.Rect, object_bbox: fitz.Rect) -> float:
        """캡션이 객체의 어느 위치에 있는지 파악하고, 객체 outline 기준 가장 가까운 거리를 계산"""
        caption_center_x = (caption_bbox.x0 + caption_bbox.x1) / 2
        caption_center_y = (caption_bbox.y0 + caption_bbox.y1) / 2
        
        obj_left, obj_top, obj_right, obj_bottom = object_bbox

        if caption_center_x < obj_left:
            if caption_center_y < obj_top:
                distance = ((obj_left - caption_center_x)**2 + (obj_top - caption_center_y)**2)**0.5
            elif caption_center_y > obj_bottom:
                distance = ((obj_left - caption_center_x)**2 + (caption_center_y - obj_bottom)**2)**0.5
            else:
                distance = obj_left - caption_center_x
        elif caption_center_x > obj_right:
            if caption_center_y < obj_top:
                distance = ((caption_center_x - obj_right)**2 + (obj_top - caption_center_y)**2)**0.5
            elif caption_center_y > obj_bottom:
                distance = ((caption_center_x - obj_right)**2 + (caption_center_y - obj_bottom)**2)**0.5
            else:
                distance = caption_center_x - obj_right
        else:
            if caption_center_y < obj_top:
                distance = obj_top - caption_center_y
            elif caption_center_y > obj_bottom:
                distance = caption_center_y - obj_bottom
            else:
                distance = 0
        
        return distance

    def _is_axis_label_text(self, text_content: str) -> bool:
        """텍스트가 축 레이블인지 판별"""
        if not text_content:
            return False
        
        text = text_content.strip()
        
        # 1. 숫자 패턴 (0-9, 소수점, 음수)
        if re.match(r'^-?\d+(\.\d+)?$', text):
            return True
        
        # 2. 짧은 텍스트 (1-3글자)
        if len(text) <= 3 and text.isalnum():
            return True
        
        # 3. 특수 패턴 (%, ℃, °, 단위 등)
        axis_patterns = [
            r'^\d+%$',  # 50%
            r'^\d+℃$',  # 25℃
            r'^\d+°$',  # 90°
            r'^[A-Z]$',  # A, B, C
            r'^[가-힣]{1,2}$',  # 한글 1-2글자
        ]
        
        for pattern in axis_patterns:
            if re.match(pattern, text):
                return True
        
        # 4. 빈 텍스트나 공백만 있는 경우 제외
        if not text or text.isspace():
            return False
        
        return False

    def _classify_drawing_type(self, drawing: fitz.Rect, page_rect: fitz.Rect, all_drawings: list = None) -> str:
        """drawing의 유형을 분류 (메인 차트 vs 범례 vs 축)"""
        area = drawing.width * drawing.height
        page_area = page_rect.width * page_rect.height
        area_ratio = area / page_area if page_area > 0 else 0
        
        # 종횡비 계산
        aspect_ratio = drawing.width / drawing.height if drawing.height > 0 else 0
        
        # 기본 크기 임계값들
        min_width = 20
        min_height = 20
        
        # 디버깅 정보
        debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
        if debug_mode:
            print(f"  Classifying drawing: {drawing}")
            print(f"    Area ratio: {area_ratio:.4f}, Aspect ratio: {aspect_ratio:.2f}")
            print(f"    Width: {drawing.width:.1f}, Height: {drawing.height:.1f}")
        
        # 1. 명확한 축/범례 패턴 (우선 분류)
        if (area_ratio < 0.003 or  # 매우 작은 면적
            aspect_ratio > 20 or aspect_ratio < 0.05 or  # 극단적인 종횡비
            drawing.width < min_width or drawing.height < min_height):  # 절대 크기가 너무 작음
            if debug_mode:
                print(f"    → axis_or_legend (small/extreme)")
            return "axis_or_legend"
        
        # 2. 상대적 크기 고려 (all_drawings가 제공된 경우)
        if all_drawings:
            # 모든 drawing 중에서 상위 크기에 속하는지 확인
            all_areas = [d.width * d.height for d in all_drawings if d != drawing]
            if all_areas:
                larger_count = sum(1 for area_val in all_areas if area_val > area)
                relative_rank = larger_count / len(all_areas) if all_areas else 0
                
                if debug_mode:
                    print(f"    Relative rank: {relative_rank:.2f} (top {relative_rank*100:.1f}%)")
                
                # 상위 50% 안에 들고 적절한 종횡비를 가지면 메인 차트 후보
                if relative_rank < 0.5 and 0.2 < aspect_ratio < 8.0:
                    if debug_mode:
                        print(f"    → main_chart (relative size)")
                    return "main_chart"
        
        # 3. 절대 기준으로 메인 차트 판단 (더 관대한 기준)
        if (area_ratio > 0.01 and  # 페이지의 1% 이상 (기존 0.02에서 완화)
            0.2 < aspect_ratio < 8.0 and  # 종횡비 범위 확장 (기존 0.3~4.0에서 확장)
            drawing.width > min_width * 2 and  # 최소 너비
            drawing.height > min_height * 2):  # 최소 높이
            if debug_mode:
                print(f"    → main_chart (absolute criteria)")
            return "main_chart"
        
        # 4. 기타 경우
        if debug_mode:
            print(f"    → unknown")
        return "unknown"

    def _score_chart_candidate(self, caption_bbox: fitz.Rect, drawing: fitz.Rect, 
                              drawing_type: str, page_rect: fitz.Rect) -> float:
        """차트 후보의 적합성 점수를 계산"""
        distance = self._calculate_closest_distance_to_outline(caption_bbox, drawing)
        area = drawing.width * drawing.height
        
        # 캡션이 drawing의 아래쪽에 있는지 확인 (일반적인 차트 구조)
        caption_center_y = (caption_bbox.y0 + caption_bbox.y1) / 2
        is_below = caption_center_y > drawing.y1
        
        # 가로폭 유사성 확인
        width_similarity = min(caption_bbox.width, drawing.width) / max(caption_bbox.width, drawing.width)
        
        # 종합 점수 계산
        score = 0
        
        # 거리 점수 (가까울수록 높음, 최대 100점)
        score += max(0, 100 - distance)
        
        # 면적 점수 (적당한 크기가 좋음, 최대 50점)
        page_area = page_rect.width * page_rect.height
        area_ratio = area / page_area if page_area > 0 else 0
        if 0.01 < area_ratio < 0.3:
            score += 50 * min(area_ratio / 0.1, 1.0)
        
        # 위치 관계 점수 (캡션이 아래에 있으면 +50점)
        if is_below:
            score += 50
        
        # 가로폭 유사성 점수 (최대 20점)
        score += width_similarity * 20
        
        # 타입별 보너스 점수
        if drawing_type == "main_chart":
            score += 50  # 메인 차트로 분류되면 큰 보너스
        elif drawing_type == "axis_or_legend":
            score -= 30  # 축이나 범례로 분류되면 감점
        
        return score

    def _find_chart_components_for_caption(self, caption_bbox: fitz.Rect, drawings: list, 
                                         page_rect: fitz.Rect = None) -> fitz.Rect:
        """캡션을 기준으로 스마트한 차트 영역 선택"""
        if not drawings:
            return None
        
        # 기본 페이지 rect 설정 (전달되지 않은 경우)
        if page_rect is None:
            # drawings로부터 대략적인 페이지 크기 추정
            all_x = [d.x0 for d in drawings] + [d.x1 for d in drawings]
            all_y = [d.y0 for d in drawings] + [d.y1 for d in drawings]
            page_rect = fitz.Rect(min(all_x), min(all_y), max(all_x), max(all_y))
        
        # 각 drawing을 분류하고 점수 계산
        candidates = []
        for drawing in drawings:
            drawing_type = self._classify_drawing_type(drawing, page_rect, drawings)
            score = self._score_chart_candidate(caption_bbox, drawing, drawing_type, page_rect)
            distance = self._calculate_closest_distance_to_outline(caption_bbox, drawing)
            
            candidates.append({
                'drawing': drawing,
                'type': drawing_type,
                'score': score,
                'distance': distance,
                'area': drawing.width * drawing.height
            })
        
        # 점수 기준으로 정렬 (높은 점수가 먼저)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 디버깅 정보 출력 (개발 모드에서만)
        debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
        if debug_mode and len(candidates) > 1:
            print(f"Chart selection for caption: {caption_bbox}")
            for i, cand in enumerate(candidates[:3]):  # 상위 3개만 출력
                print(f"  {i+1}. Type: {cand['type']}, Score: {cand['score']:.1f}, "
                      f"Distance: {cand['distance']:.1f}, Area: {cand['area']:.0f}")
        
        # 최고 점수의 drawing 선택
        if candidates:
            best_candidate = candidates[0]
            
            # 최고 점수가 너무 낮으면 (< 50) 기존 방식으로 폴백
            if best_candidate['score'] < 50:
                if debug_mode:
                    print(f"  Low score ({best_candidate['score']:.1f}), using fallback method")
                return min(drawings, key=lambda d: self._calculate_closest_distance_to_outline(caption_bbox, d))
            
            return best_candidate['drawing']
        
        return None

    def _merge_chart_with_axis_labels(self, main_chart: fitz.Rect, all_drawings: list, 
                                    page_rect: fitz.Rect, expansion_ratio: float = 0.3, 
                                    etc_texts: list = None) -> fitz.Rect:
        """
        메인 차트 영역에 관련된 x축, y축 레이블들을 병합 (drawing + etc 텍스트 고려)
        
        Args:
            main_chart: 메인 차트 영역
            all_drawings: 모든 드로잉 영역들
            page_rect: 페이지 전체 영역
            expansion_ratio: 축 레이블 검색 범위 확장 비율
            etc_texts: etc 타입의 텍스트 요소들 (y축 레이블 텍스트 포함 가능)
            
        Returns:
            축 레이블이 병합된 확장된 차트 영역
        """
        if not all_drawings:
            return main_chart
        
        # 메인 차트 주변 검색 영역 설정
        chart_width = main_chart.width
        chart_height = main_chart.height
        
        # x축 레이블 검색 영역 (차트 아래쪽)
        x_axis_search = fitz.Rect(
            main_chart.x0 - chart_width * expansion_ratio,
            main_chart.y1,  # 차트 아래부터
            main_chart.x1 + chart_width * expansion_ratio,
            main_chart.y1 + chart_height * 0.5  # 차트 높이의 50%까지
        )
        
        # y축 레이블 검색 영역 (차트 왼쪽)
        y_axis_search = fitz.Rect(
            main_chart.x0 - chart_width * 0.5,  # 차트 왼쪽으로 차트 폭의 50%까지
            main_chart.y0 - chart_height * expansion_ratio,
            main_chart.x0,  # 차트 왼쪽까지
            main_chart.y1 + chart_height * expansion_ratio
        )
        
        # 페이지 경계 내로 제한
        x_axis_search = x_axis_search & page_rect
        y_axis_search = y_axis_search & page_rect
        
        # 축 레이블 후보들 찾기 (drawing + etc 텍스트)
        axis_candidates = []
        
        # 1. Drawing 기반 축 레이블 찾기
        for drawing in all_drawings:
            # 메인 차트와 겹치는 것은 제외
            if drawing == main_chart or main_chart.intersects(drawing):
                continue
                
            drawing_type = self._classify_drawing_type(drawing, page_rect, all_drawings)
            
            # x축 영역에 있는 축/범례 요소들
            if x_axis_search.intersects(drawing) and drawing_type == "axis_or_legend":
                # x축 레이블은 주로 가로로 길고 차트 아래에 위치
                aspect_ratio = drawing.width / drawing.height if drawing.height > 0 else 0
                if aspect_ratio > 2.0:  # 가로로 긴 형태
                    axis_candidates.append(('x_axis', drawing, 'drawing'))
            
            # y축 영역에 있는 축/범례 요소들  
            elif y_axis_search.intersects(drawing) and drawing_type == "axis_or_legend":
                # y축 레이블은 주로 세로로 길고 차트 왼쪽에 위치
                aspect_ratio = drawing.width / drawing.height if drawing.height > 0 else 0
                if aspect_ratio < 0.5:  # 세로로 긴 형태
                    axis_candidates.append(('y_axis', drawing, 'drawing'))
        
        # 병합된 영역 계산
        merged_chart = fitz.Rect(main_chart)
        
        for axis_type, axis_bbox, source_type in axis_candidates:
            merged_chart.include_rect(axis_bbox)

        # 2. etc 텍스트 기반 축 레이블 찾기 (y축 레이블 텍스트)
        if etc_texts:
            merged_chart = self._merge_drawings_with_text_blocks(merged_chart, etc_texts)
        
            
        # 디버깅 정보 출력
        debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
        if debug_mode and axis_candidates:
            print(f"    Merging axis labels: {len(axis_candidates)} found")
            for axis_type, axis_bbox, source_type in axis_candidates:
                print(f"      {axis_type} ({source_type}): {axis_bbox}")
            print(f"    Original chart: {main_chart}")
            print(f"    Expanded chart: {merged_chart}")
        
        return merged_chart

    def _validate_drawing_classification(self, drawings: list, page_rect: fitz.Rect) -> dict:
        """drawing 분류 결과를 검증하고 통계를 반환"""
        if not drawings:
            return {"main_charts": 0, "axis_or_legend": 0, "unknown": 0, "total": 0}
        
        classification_stats = {"main_charts": 0, "axis_or_legend": 0, "unknown": 0, "total": len(drawings)}
        
        debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
        if debug_mode:
            print(f"\n=== Drawing Classification Validation ===")
            print(f"Total drawings to classify: {len(drawings)}")
        
        for i, drawing in enumerate(drawings):
            drawing_type = self._classify_drawing_type(drawing, page_rect, drawings)
            classification_stats[drawing_type + "s"] = classification_stats.get(drawing_type + "s", 0) + 1
            
            if debug_mode:
                area = drawing.width * drawing.height
                page_area = page_rect.width * page_rect.height
                area_ratio = area / page_area if page_area > 0 else 0
                aspect_ratio = drawing.width / drawing.height if drawing.height > 0 else 0
                
                print(f"Drawing {i+1}: {drawing}")
                print(f"  Type: {drawing_type}")
                print(f"  Area ratio: {area_ratio:.4f}, Aspect ratio: {aspect_ratio:.2f}")
                print(f"  Size: {drawing.width:.1f} x {drawing.height:.1f}")
        
        if debug_mode:
            print(f"\nClassification Summary:")
            print(f"  Main charts: {classification_stats.get('main_charts', 0)}")
            print(f"  Axis/Legend: {classification_stats.get('axis_or_legends', 0)}")
            print(f"  Unknown: {classification_stats.get('unknowns', 0)}")
            print(f"==========================================\n")
        
        return classification_stats

    def _analyze_horizontal_chart_relationship(self, main_chart: fitz.Rect, candidate_chart: fitz.Rect) -> float:
        """수평으로 나란히 있는 차트들의 관계 점수 계산"""
        score = 0
        
        # 1. 수직 정렬 점수 (y좌표가 비슷할수록 높은 점수)
        y_center_diff = abs((main_chart.y0 + main_chart.y1)/2 - (candidate_chart.y0 + candidate_chart.y1)/2)
        y_alignment_score = max(0, 100 - y_center_diff)
        
        # 2. 크기 유사도 점수
        height_ratio = min(main_chart.height, candidate_chart.height) / max(main_chart.height, candidate_chart.height)
        width_ratio = min(main_chart.width, candidate_chart.width) / max(main_chart.width, candidate_chart.width)
        size_similarity_score = (height_ratio + width_ratio) * 50  # 최대 100점
        
        # 3. 수평 거리 점수 (적절한 거리에 있을 때 높은 점수)
        # 좌우 배치에 관계없이 가장 가까운 거리 계산
        horizontal_distance = min(
            abs(candidate_chart.x0 - main_chart.x1),  # candidate가 main 오른쪽에 있는 경우
            abs(candidate_chart.x1 - main_chart.x0)   # candidate가 main 왼쪽에 있는 경우
        )
        ideal_distance = main_chart.width * 0.2  # 차트 너비의 20%를 이상적인 거리로 가정
        distance_score = max(0, 100 - abs(horizontal_distance - ideal_distance))
        
        # 종합 점수 계산 (각 요소별 가중치 적용)
        score = (y_alignment_score * 0.4 +  # 수직 정렬이 가장 중요
                size_similarity_score * 0.3 +  # 크기 유사도도 중요
                distance_score * 0.3)  # 수평 거리도 고려
        
        return score
    
    def _merge_related_charts(self, main_chart: fitz.Rect, all_drawings: list, page_rect: fitz.Rect) -> fitz.Rect:
        """메인 차트와 관련된 다른 차트들을 병합"""
        if not all_drawings:
            return main_chart
            
        merged_chart = fitz.Rect(main_chart)  # 초기 영역은 메인 차트
        used_drawings = {main_chart}  # 이미 사용된 drawing 추적
        
        # 1. 후보 차트들 찾기
        candidates = []
        for drawing in all_drawings:
            if drawing in used_drawings:
                continue
                
            # 차트 타입 확인 (메인 차트와 비슷한 크기의 것들만 고려)
            if self._classify_drawing_type(drawing, page_rect) == "main_chart":
                # 수평 관계 점수 계산
                score = self._analyze_horizontal_chart_relationship(main_chart, drawing)
                if score > 70:  # 임계값 이상인 경우만 후보로 선정
                    candidates.append((drawing, score))
        
        # 2. 점수 기준으로 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 양방향으로 병합 (왼쪽과 오른쪽 모두)
        debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
        
        for candidate, score in candidates:
            # 이미 병합된 영역과 인접한지 확인 (왼쪽 또는 오른쪽)
            is_adjacent = (
                candidate.x0 > merged_chart.x1 or  # 오른쪽에 인접
                candidate.x1 < merged_chart.x0     # 왼쪽에 인접
            )
            
            if is_adjacent:
                if debug_mode:
                    position = "right" if candidate.x0 > merged_chart.x1 else "left"
                    print(f"    Merging {position} chart: {candidate} (score: {score:.1f})")
                
                # 중간 영역도 포함하여 병합
                merged_chart.include_rect(candidate)
                used_drawings.add(candidate)

        # 4. 다 병합하고 나서 다른 차트들과 교차되는 영역 체크하기
        for r in all_drawings:
            if merged_chart not in used_drawings:
                if merged_chart.intersects(r):
                    merged_chart.include_rect(r)
                    used_drawings.add(r)
        
        return merged_chart

    def _find_chart_with_axis_merge(self, caption_bbox: fitz.Rect, drawings: list, 
                                  page_rect: fitz.Rect = None, etc_text_rects: list = None, text_rects: list = None) -> fitz.Rect:
        """
        캡션을 기준으로 차트를 찾고 관련된 축 레이블들과 병합.
        모든 drawings를 후보로 고려하며, _score_chart_candidate에서 위치 관계를 통해 적절한 차트를 선택.
        """
        if not drawings:
            return None

        # 전체 drawings를 대상으로 메인 차트 찾기 (점수 기반 선택)
        main_chart = self._find_chart_components_for_caption(caption_bbox, drawings, page_rect)

        if not main_chart:
            return None
        
        # 캡션과 메인 차트 사이에 텍스트가 있는지 확인
        if text_rects:
            between_area = fitz.Rect(
                min(caption_bbox.x0, main_chart.x0),
                min(caption_bbox.y1, main_chart.y1),
                max(caption_bbox.x1, main_chart.x1),
                max(caption_bbox.y0, main_chart.y0)
            )
            
            debug_mode = os.environ.get("PDF_CONVERTER_DEBUG", "false").lower() == "true"
            if debug_mode:
                print(f"    Checking for text between caption and chart: {between_area}")
            
            for text_rect in text_rects:
                if between_area.intersects(text_rect):
                    if debug_mode:
                        print(f"    Text found between caption and chart, abandoning: {text_rect}")
                    return None
        
        # 관련된 차트들과 병합
        merged_chart = self._merge_related_charts(main_chart, drawings, page_rect or fitz.Rect(0, 0, 600, 800))

        # 축 레이블들과 병합 (drawing + etc 텍스트 고려)
        final_chart = self._merge_chart_with_axis_labels(
            merged_chart, drawings, 
            page_rect or fitz.Rect(0, 0, 600, 800),
            etc_texts=etc_text_rects
        )
        return final_chart

    def _rearrange_elements(self, all_elements: list) -> str:   
        rebuild_all_elements = []
        extra_text_elements = []
        text_bbox_by_page = {}
        prev_item = all_elements[0]

        for i in range(1, len(all_elements)):
            item = all_elements[i]

            item_type = item.get('type')
            page_index = item.get('page_index', 'N/A')
            content = item.get('content', '')
            caption = item.get('caption', '')
            bbox = item.get('bbox', None)

            if page_index not in text_bbox_by_page:
                min_x0 = float('inf')
                min_y0 = float('inf')
                max_x1 = float('-inf')
                max_y1 = float('-inf')
                for sub_item in range(i, len(all_elements)):
                    if page_index == all_elements[sub_item].get('page_index', 'N/A'):
                        if all_elements[sub_item].get('type') == 'text':
                            sub_item_bbox = all_elements[sub_item].get('bbox')

                            min_x0 = min(min_x0, sub_item_bbox.x0)
                            max_x1 = max(max_x1, sub_item_bbox.x1)
                            min_y0 = min(min_y0, sub_item_bbox.y0)
                            max_y1 = max(max_y1, sub_item_bbox.y1)
                    else:
                        break
                text_bbox_by_page[page_index] = (min_x0, min_y0, max_x1, max_y1)

            # 단이 나눠진 상태로 좌측, 우측에 각주들이 나열되는 경우가 있다. 메인 text 영역 외에 있는 text 는 각 페이지의 제대로 마무리된 text 하단에 모아서 배치한다.
            min_tect_x0, min_tect_y0, max_text_x1, max_text_y1 = text_bbox_by_page.get(page_index, (0, 0, 1000, 1000))
            if (bbox.x1 < min_tect_x0 or bbox.x0 > max_text_x1) and bbox.y0 >= min_tect_y0 and bbox.y1 <= max_text_y1:
                extra_text_elements.append(item)
                continue

            if item_type == prev_item['type'] and rebuild_all_elements:
                if item_type == 'text':
                    if not prev_item['content'].rstrip().endswith(('.', '!', '?', '。', '！', '？')) and not starts_with_list_item(content):
                        rebuild_all_elements[-1]['content'] = rebuild_all_elements[-1]['content'] + content
                        continue
                elif item_type == 'table' and prev_item.get('caption', '') == caption:
                    rebuild_all_elements[-1]['content'] = rebuild_all_elements[-1]['content'] + "\n" + content
                    continue

            if extra_text_elements:
                rebuild_all_elements.extend(extra_text_elements)
                extra_text_elements = []

            prev_item = item
            rebuild_all_elements.append(item)
            
        return rebuild_all_elements
        
    def _prepare_json_elements(self, all_elements: list) -> list:
        """JSON 출력을 위한 요소들 준비"""
        serializable_elements = []
        for el in all_elements:
            new_el = el.copy()
            
            # bbox 안전하게 직렬화
            bbox = new_el["bbox"]
            if hasattr(bbox, 'irect'):  # fitz.Rect 객체인 경우
                new_el["bbox"] = [round(c, 2) for c in bbox.irect]
            elif hasattr(bbox, '__iter__') and not isinstance(bbox, str):  # 이미 리스트/튜플인 경우
                new_el["bbox"] = [round(float(c), 2) for c in bbox]
            else:  # 기타 경우
                new_el["bbox"] = [0, 0, 0, 0]  # 기본값
                
            if new_el["type"] == "table":
                new_el["content"] = new_el["content"].to_dict(orient="split")
            # filename, s3_key 등은 이미 문자열이므로 그대로 유지
            serializable_elements.append(new_el)
        return serializable_elements

    def _generate_final_markdown(self, all_elements: list) -> str:
        write_page_index = -1
        formatted_chunks = []

        for i in range(1, len(all_elements)):
            item = all_elements[i]

            item_type = item.get('type')
            page_index = item.get('page_index', 'N/A')
            content = item.get('content', '')
            caption = item.get('caption', '')
            s3_key = item.get('s3_key')
            
            chunk_parts = []
            # page_index 정보 추가
            if write_page_index != page_index:
                chunk_parts.append(f"[page_index: {page_index}]")
                write_page_index = page_index

            if item_type == 'title':
                chunk_parts.append(f"## {content}")
            elif item_type == 'text':
                chunk_parts.append(content)
            elif item_type == 'etc':
                chunk_parts.append(content)
            elif item_type == 'table':
                if isinstance(content, pd.DataFrame) and not content.empty:
                    chunk_parts.append(f"** 표: {caption if caption else 'no caption'} **")
                    chunk_parts.append(content.to_markdown(index=False))
            elif item_type == 'image':
                if s3_key:
                    chunk_parts.append(f"**그림: {caption if caption else 'no caption'}**")
                    cdn_url = generate_cdn_url(s3_key, self.config.cdn_url)
                    chunk_parts.append(f"[URL: {cdn_url}]")
                    if content:
                        chunk_parts.append(f"[내용: {content}]")
            else:
                chunk_parts.append(content)
            
            formatted_chunks.append("\n".join(chunk_parts))
        return "\n\n".join(formatted_chunks)

    def _save_results_new(self, markdown_content: str, json_elements: list, pdf_name: str):
        """새로운 방식으로 최종 결과를 마크다운과 JSON 파일로 저장"""
        # 마크다운 파일 저장
        md_output_path = os.path.join(self.config.output_dir, f"{pdf_name}.md")
        save_markdown_file(markdown_content, md_output_path)

        # JSON 파일 저장
        json_output = {"content": json_elements}
        json_output_path = os.path.join(self.config.output_dir, f"{pdf_name}.json")
        save_json_file(json_output, json_output_path) 