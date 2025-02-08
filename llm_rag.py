import os
import fitz  # pip install PyMuPDF
from pptx import Presentation  # pip install python-pptx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA
import configparser

# 설정 파일 로드
config = configparser.ConfigParser()
config.read('config.cfg')
openai_api_key = config['openai']['TOKEN']

# vector db
QDRANT_URL = "http://localhost:6333"

"""
PDF -> 텍스트 문서 생성
    - 청킹(Chunking): RecursiveCharacterTextSplitter 활용
    - 메타데이터 추가: {source, page}
"""
def process_pdf(file_path):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text = page.get_text("text").strip()
            chunks = splitter.split_text(text)
            for chunk in chunks:
                metadata = {"source": os.path.basename(file_path), "page": page.number + 1}
                docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

"""
PPT -> 텍스트 문서 생성
    - 청킹: 슬라이드 내 텍스트를 분할
    - 메타데이터 추가: {source, slide}
"""
def process_ppt(file_path):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    prs = Presentation(file_path)
    
    for i, slide in enumerate(prs.slides):
        slide_text = ""
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text += shape.text + "\n"
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            for shape in notes_slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text += shape.text + "\n"
        if slide_text.strip():
            chunks = splitter.split_text(slide_text)
            for chunk in chunks:
                metadata = {"source": os.path.basename(file_path), "slide": i + 1}
                docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

"""
Qdrant 벡터 데이터베이스에 문서 저장
    - HuggingFaceEmbeddings("all-mpnet-base-v2") 활용
"""
def setup_vector_store(documents):
    # process_pdf나 process_ppt은 각각 리스트를 반환하므로, 전체 문서를 하나의 리스트로 합칩니다.
    all_docs = []
    for doc_list in documents:
        all_docs.extend(doc_list)
        
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = Qdrant.from_documents(
        all_docs,
        embeddings,
        collection_name="document_collection",
        url=QDRANT_URL
    )
    return vectorstore

"""
RAG 체인 생성 (벡터 DB + LLM)
"""
def create_retrieval_qa(vectorstore, llm=None): 
    if llm is None:
        # ChatOpenAI를 사용하며 openai_api_key를 전달해야 합니다.
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa

"""
사용자 질의응답 루프
    - 사용자가 입력한 질문에 대해 관련 문서와 답변을 출력.
"""
def query_loop(qa):
    print("문서 기반 질의응답 시스템을 시작합니다. (종료하려면 'exit' 입력)")
    while True:
        query = input("\n🔹 질문을 입력하세요: ")
        if query.lower().strip() == "exit":
            print("프로그램을 종료합니다.")
            break

        result = qa({"query": query})
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        source_str = "General Agent"
        if "내용이 없습니다" in answer or not source_docs: # 답변이 없거나, 관련 문서가 없는 경우
            answer = "해당 질문에 대한 답변을 찾을 수 없습니다."        
            
        else:
            sources = set()
            for doc in source_docs:
                meta = doc.metadata
                if "page" in meta:
                    sources.add(f"{meta['source']} {meta['page']}페이지")
                elif "slide" in meta:
                    sources.add(f"{meta['source']} {meta['slide']}슬라이드")
            source_str = ", ".join(sources)
        if "내용이 없습니다" in answer:
            answer = "해당 질문에 대한 답변을 찾을 수 없습니다."
        print("답변:", answer)
        print("출처:", source_str)

def main():
    # PDF 및 PPT 문서 처리 (예시로 두 개의 PDF 파일 사용)
    documents = [process_pdf(".pdf"), process_pdf(".pdf")]
    # 벡터 DB (Qdrant)에 저장
    vectorstore = setup_vector_store(documents)
    # RAG 체인 생성
    qa = create_retrieval_qa(vectorstore)
    # 질의응답 루프 실행
    query_loop(qa)

if __name__ == '__main__':
    main()
