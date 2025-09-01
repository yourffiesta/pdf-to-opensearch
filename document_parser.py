from llama_index.core import SimpleDirectoryReader
import pymupdf4llm
import os

import kss



def extract_text(file_path: str) -> str:
    """
    파일 확장자에 따라 적절한 방식으로 텍스트를 추출합니다.
    지원: PDF, TXT, DOCX, PPTX (llamaIndex SimpleDirectoryReader 지원 포맷)
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    elif ext in [".docx", ".pptx"]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()
        return "\n\n".join([doc.text for doc in docs])
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


def extract_text_from_pdf(file_path: str) -> str:
    """
    """
    reader = pymupdf4llm.LlamaMarkdownReader()
    docs = reader.load_data(file_path)
    text = "\n\n".join([doc.text_resource.text if doc.text_resource else "" for doc in docs])
    return text

def test():
    file_path = '.files/20250514.pdf'
    text = extract_text(file_path)
    text = text.replace('\n', '')
    for n,sent in enumerate(kss.split_sentences(text)):
        print(f"{n} : "+sent)

    # print(text)

if __name__ == "__main__":
    test()