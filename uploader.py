#!/usr/bin/env python3
"""
OpenSearch 문서 업로드 모듈

이 모듈은 PDF에서 추출된 청크 데이터를 OpenSearch에 업로드하는 기능을 제공합니다.
"""

import os
import re
import glob
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from opensearch_loader import OpenSearchLoader
import unicodedata


class DocumentUploader:
    """OpenSearch에 문서를 업로드하는 클래스"""
    
    def __init__(self, 
                 index_name: str):
        """
        DocumentUploader 초기화
        
        Args:
            index_name: OpenSearch 인덱스 이름
        """
        if not index_name:
            raise ValueError("index_name is required")

        self.loader = OpenSearchLoader()
        self.index_name = index_name
    
    def read_chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        청크 파일에서 데이터를 읽어와 파싱합니다.
        
        Args:
            file_path: 청크 파일 경로
            
        Returns:
            파싱된 청크 데이터 리스트
        """
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = []
            page_number = None
            image_urls = []
            chunk_sequence = 1
            
            for line in content.split('\n'):
                if line == '<chunk>':
                    pass
                elif line == '</chunk>':
                    if lines:
                        chunk_data = {
                            'chunk_sequence': chunk_sequence,
                            'content': '\n'.join(lines)
                        }
                        
                        if page_number:
                            chunk_data['page_number'] = page_number
                        
                        if image_urls:
                            chunk_data['image_urls'] = image_urls.copy()
                        
                        chunks.append(chunk_data)
                        chunk_sequence += 1
                        lines = []
                        image_urls = []

                elif re.match(r'\[page_index:\s*(\d+)\]', line):
                    page_number = int(re.match(r'\[page_index:\s*(\d+)\]', line).group(1)) + 1
                elif re.match(r'\[URL:\s*(.*?)\]', line):
                    url = re.match(r'\[URL:\s*(.*?)\]', line).group(1)
                    image_urls.append(url)
                else:
                    lines.append(line)
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
            
        return chunks
    
    def create_base_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        파일 경로에서 기본 메타데이터를 생성합니다.
        
        Args:
            file_path: 청크 파일 경로
            
        Returns:
            기본 메타데이터 딕셔너리
        """
        base_name = os.path.basename(file_path)
        source_title = base_name.replace("_chunks.txt", "")
        crop_name = source_title.replace("농업기술길잡이_", "")
        
        return {
            "source_type": "PDF",
            "source_uri": "https://www.nongsaro.go.kr/portal/ps/psb/psbx/cropEbookMain.ps?menuId=PS65290",
            "source_title": f"농업기술길잡이-{crop_name}",
            "crop_name": crop_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def create_document_list(self, chunks: List[Dict[str, Any]], 
                           base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        청크 데이터를 OpenSearch 문서 형태로 변환합니다.
        
        Args:
            chunks: 청크 데이터 리스트
            base_metadata: 기본 메타데이터
            
        Returns:
            OpenSearch 문서 리스트
        """
        document_list = []
        
        for i in range(len(chunks)):
            chunk = chunks[i]
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

            metadata = base_metadata.copy()
            metadata["chunk_sequence"] = chunk["chunk_sequence"]
            metadata["page_number"] = chunk["page_number"]

            if chunk.get("image_urls"):
                metadata["image_urls"] = chunk["image_urls"]
            
            current_document = {
                "metadata": metadata,
            }
            
            if prev_chunk:
                current_document["chunk_text_previous"] = prev_chunk['content']
            current_document["chunk_text_current"] = chunk['content']
            if next_chunk:
                current_document["chunk_text_next"] = next_chunk['content']

            document_list.append(current_document)
        
        return document_list
    
    def delete_existing_documents(self, metadata_filter: Dict[str, Any]) -> int:
        """
        메타데이터 필터에 일치하는 기존 문서들을 삭제합니다.
        
        Args:
            metadata_filter: 삭제할 문서를 찾기 위한 메타데이터 필터
            
        Returns:
            삭제된 문서 수
        """
        deleted_count = self.loader.delete_documents_by_metadata(
            index_name=self.index_name, 
            metadata_filter=metadata_filter
        )
        print(f"-> API reported {deleted_count} documents for deletion.")
        return deleted_count
    
    def upload_documents(self, document_list: List[Dict[str, Any]], 
                        batch_size: int = 100,
                        local_save_path: str = "default",
                        s3_save_path: str = "default") -> None:
        """
        문서 리스트를 OpenSearch에 업로드합니다.
        
        Args:
            document_list: 업로드할 문서 리스트
            batch_size: 배치 크기
            local_save_path: 로컬 저장 경로
            s3_save_path: S3 저장 경로
        """
        self.loader.insert_document_list(
            self.index_name,
            document_list,
            local_save_path=local_save_path,
            s3_save_path=s3_save_path,
            batch_size=batch_size
        )
    
    def process_single_file(self, file_path: str, 
                          batch_size: int = 100,
                          delete_existing: bool = True) -> None:
        """
        단일 청크 파일을 처리하여 OpenSearch에 업로드합니다.
        
        Args:
            file_path: 청크 파일 경로
            batch_size: 배치 크기
            delete_existing: 기존 문서 삭제 여부
        """
        file_path = unicodedata.normalize('NFC', file_path)
        print(f"Processing file: {file_path}")
        
        # 1. 청크 파일 읽기
        chunks = self.read_chunk_file(file_path)
        if not chunks:
            print(f"No chunks found in {file_path}")
            return
        
        print(f"Found {len(chunks)} chunks")
        
        # 2. 메타데이터 생성
        base_metadata = self.create_base_metadata(file_path)
        
        # 3. 문서 리스트 생성
        document_list = self.create_document_list(chunks, base_metadata)
        
        # 4. 기존 문서 삭제 (옵션)
        if delete_existing:
            metadata_filter = {
                "source_type": base_metadata["source_type"],
                "source_uri": base_metadata["source_uri"],
                "source_title": base_metadata["source_title"],
            }
            self.delete_existing_documents(metadata_filter)
        
        # 5. 새 문서 업로드
        print(f"Uploading {len(document_list)} documents...")
        self.upload_documents(document_list, batch_size=batch_size)
        print(f"Upload completed for {file_path}")
    
    def process_directory(self, directory_path: str, 
                         file_pattern: str = "*_chunks.txt",
                         batch_size: int = 100,
                         delete_existing: bool = True) -> None:
        """
        디렉토리 내의 모든 청크 파일을 처리합니다.
        
        Args:
            directory_path: 처리할 디렉토리 경로
            file_pattern: 파일 패턴 (기본: *_chunks.txt)
            batch_size: 배치 크기
            delete_existing: 기존 문서 삭제 여부
        """
        chunk_files = glob.glob(os.path.join(directory_path, file_pattern))
        chunk_files = sorted(chunk_files)
        
        if not chunk_files:
            print(f"No chunk files found in {directory_path} with pattern {file_pattern}")
            return
        
        print(f"Found {len(chunk_files)} chunk files to process")
        
        for i, file_path in enumerate(chunk_files, 1):
            print(f"\n[{i}/{len(chunk_files)}] Processing: {os.path.basename(file_path)}")
            try:
                self.process_single_file(file_path, batch_size, delete_existing)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload documents to OpenSearch")
    parser.add_argument("--file", "-f", type=str, help="Single chunk file to process")
    parser.add_argument("--directory", "-d", type=str, help="Directory containing chunk files")
    parser.add_argument("--pattern", "-p", type=str, default="*_chunks.txt", 
                       help="File pattern to match (default: *_chunks.txt)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, 
                       help="Batch size for uploading (default: 100)")
    parser.add_argument("--no-delete", action="store_true", 
                       help="Skip deleting existing documents")
    parser.add_argument("--embedding-model", type=str, default="cohere.embed-multilingual-v3",
                       help="Embedding model ID")
    parser.add_argument("--embedding-dimension", type=int, default=1024,
                       help="Embedding dimension")
    parser.add_argument("--index-name", type=str, default="agri-kb-unified-vectors-dev-snapshot-20250716",
                       help="OpenSearch index name")
    
    args = parser.parse_args()
    
    # DocumentUploader 초기화
    uploader = DocumentUploader(
        index_name=args.index_name
    )
    
    delete_existing = not args.no_delete
    
    if args.file:
        # 단일 파일 처리
        uploader.process_single_file(args.file, args.batch_size, delete_existing)
    elif args.directory:
        # 디렉토리 처리
        uploader.process_directory(args.directory, args.pattern, args.batch_size, delete_existing)
    else:
        print("Either --file or --directory must be specified")
        parser.print_help()


if __name__ == "__main__":
    # DocumentUploader 초기화
    uploader = DocumentUploader(
        index_name="agri-kb-unified-vectors-dev-snapshot-20250716"
    )
    uploader.process_directory("/Users/yoonhae/greenlabs/data-labs/.files/", "*_chunks.txt", 100)