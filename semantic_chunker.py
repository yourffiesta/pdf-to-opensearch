import os
import unicodedata
from datetime import datetime, timezone
from document_parser import extract_text
from chunker import split_text_into_chunks_fast

def main(file_path):    
    text = extract_text(file_path)
        
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    overlab_size = 0
    chunks = split_text_into_chunks_fast(text=text, model_name=model_name, overlab_size=overlab_size)
    
    # 파일명 추출 (확장자 제외)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # .files 디렉토리 생성
    os.makedirs('.files', exist_ok=True)
    
    # chunks.md 에 현재 텍스트만 저장
    with open(f'.files/{base_name}_chunks.md', 'w') as f:
        f.write(f'\n\n{"---"}\n\n'.join(chunks))
    
    # chunks.txt 에 <chunk> 태그로 감싸서 저장
    with open(f'.files/{base_name}_chunks.txt', 'w') as f:
        for chunk in chunks:
            f.write(f'<chunk>\n{chunk}\n</chunk>\n\n')
    
if __name__ == '__main__':
    from glob import glob
    dir_path = "/Users/yoonhae/greenlabs/data-labs/experiments/engineering/pipelines/pdf-to-opensearch/output_v2/"
    pdf_files = sorted(glob(os.path.join(dir_path, "*.md")))
    for file_path in pdf_files[1:3]:
        new_file_path = unicodedata.normalize('NFC', file_path)
        base_name = os.path.splitext(os.path.basename(new_file_path))[0]
        main(file_path=new_file_path)
        break




