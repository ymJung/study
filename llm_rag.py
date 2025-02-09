import os
import fitz  # pip install PyMuPDF
from pptx import Presentation  # pip install python-pptx
import cv2 # 
import numpy as np
from PIL import Image
import pytesseract # sudo apt-get install tesseract-ocr # pip install pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.docstore.document import Document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # pip install faiss-cpu

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import configparser
import re

# ============================================================
# 설정 파일 로드 (config.cfg 파일 필요: [openai] TOKEN = <API_KEY>)
# ============================================================
config = configparser.ConfigParser()
config.read('config.cfg')
openai_api_key = config['openai']['TOKEN']


def preprocess_image_for_ocr_alternative(image):
    # PIL 이미지를 OpenCV BGR 이미지로 변환
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)    
    # CLAHE 적용 (국부 대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)    
    # 샤프닝 필터 적용 (텍스트 경계 강화)
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, sharpening_kernel)    
    # 처리된 이미지를 PIL 이미지로 변환하여 반환
    processed_image = Image.fromarray(sharpened)
    return processed_image

def clean_hangul_spacing(text):
    """
    한글 문자 사이에 있는 불필요한 공백을 제거합니다.    
    단, 영어와 숫자, 기호 등은 그대로 둡니다.
    """
    # 한글(가-힣) 뒤에 공백이 있고 그 뒤에 한글이 오는 경우를 찾아서 공백을 제거
    return re.sub(r'(?<=[가-힣])\s+(?=[가-힣])', '', text)


# ============================================================
# PDF 처리: 텍스트 추출, OCR (텍스트 없을 경우), 청킹 및 메타데이터 추가
# agent: deepseek, brochure, 또는 general (기본값)
# ============================================================
def process_pdf(file_path, agent="general"):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text = page.get_text("text").strip()
                # 텍스트가 없는 경우 OCR 적용
                if not text:
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        processed_img = preprocess_image_for_ocr_alternative(img)
                        custom_config = r'--oem 3 --psm 6'
                        
                        # 단어별 OCR 결과(좌표, 텍스트, 신뢰도 등)를 추출
                        ocr_data = pytesseract.image_to_data(
                            processed_img,
                            lang="eng+kor",
                            config=custom_config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # 1. 각 라인(line_num) 별로 단어 정보를 수집 (신뢰도 60 이상)
                        horizontal_gap_threshold = 50  # 단어 사이 가로 간격 임계값 (픽셀)
                        lines_dict = {}  # key: line_num, value: list of (left, width, word)
                        num_boxes = len(ocr_data['text'])
                        for i in range(num_boxes):
                            try:
                                conf = float(ocr_data['conf'][i])
                            except Exception:
                                conf = 0.0
                            if conf > 60 and ocr_data['text'][i].strip():
                                line_num = ocr_data['line_num'][i]
                                if line_num not in lines_dict:
                                    lines_dict[line_num] = []
                                left = ocr_data['left'][i]
                                width = ocr_data['width'][i]
                                word = ocr_data['text'][i].strip()
                                lines_dict[line_num].append((left, width, word))
                        
                        # 2. 각 라인 내에서 단어들을 좌측 좌표 순으로 정렬한 후,
                        #    가로 간격(gap)이 horizontal_gap_threshold 이상이면 다른 그룹으로 분리
                        grouped_lines = []
                        for line in sorted(lines_dict.keys()):
                            words = sorted(lines_dict[line], key=lambda x: x[0])
                            groups = []
                            current_group = []
                            for word_info in words:
                                if not current_group:
                                    current_group.append(word_info)
                                else:
                                    last_word = current_group[-1]
                                    gap = word_info[0] - (last_word[0] + last_word[1])
                                    if gap > horizontal_gap_threshold:
                                        groups.append(" ".join(w for _, _, w in current_group))
                                        current_group = [word_info]
                                    else:
                                        current_group.append(word_info)
                            if current_group:
                                groups.append(" ".join(w for _, _, w in current_group))
                            grouped_lines.append(groups)
                        
                        # 3. 라인별 그룹이 모두 리스트(예: 각 라인에서 컬럼별로 분리됨)
                        #    각 라인의 그룹 수는 다를 수 있으므로, 먼저 최대 그룹 개수를 파악
                        max_cols = max(len(groups) for groups in grouped_lines)
                        
                        # 4. 같은 인덱스(컬럼) 그룹끼리 합쳐서 하나의 구역(region) 텍스트로 만듦
                        merged_regions = []  # 각 요소가 하나의 컬럼에 해당하는 텍스트
                        for col in range(max_cols):
                            region_texts = []
                            for groups in grouped_lines:
                                if col < len(groups):
                                    region_texts.append(groups[col])
                            merged_regions.append(" ".join(region_texts))
                        
                        # 5. 모든 merged region(컬럼)을 하나의 텍스트로 합쳐서 Document 생성
                        page_text = " ".join(merged_regions)                        
                        page_text = clean_hangul_spacing(page_text)                        
                        docs.append(Document(
                            page_content=page_text,
                            metadata={
                                "source": os.path.basename(file_path),
                                "page": page.number + 1,
                                "agent": agent,
                                "doc_type": "PDF"
                            }
                        ))
                        
                    except Exception as ocr_err:
                        print(f"[{os.path.basename(file_path)}] OCR 실패 (페이지 {page.number + 1}): {ocr_err}")
                        text = ""
                else:
                    # 텍스트 레이어가 있는 경우
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": os.path.basename(file_path),
                                "page": page.number + 1,
                                "agent": agent,
                                "doc_type": "PDF"
                            }
                        ))
    except Exception as e:
        print(f"PDF 처리 중 오류 발생 ({file_path}): {e}")
    return docs


# ============================================================
# PPT 처리: 슬라이드 텍스트, 노트 추출, (이미지 OCR 자리) 및 메타데이터 추가
# ============================================================
def process_ppt(file_path, agent="general"):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    try:
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            slide_text = ""
            # 슬라이드 내 텍스트 추출
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text += shape.text + "\n"
                # 이미지 OCR 처리 (예시)
                if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE (상수값 13)
                    try:
                        # 실제 이미지 데이터 추출 및 OCR 처리는 PPTX 내부 구조에 따라 다름.
                        # 필요시 shape.image 또는 관련 API를 활용해 이미지 데이터를 추출 후 OCR 적용
                        pass
                    except Exception as img_err:
                        print(f"[{os.path.basename(file_path)}] PPT 이미지 OCR 오류 (슬라이드 {i+1}): {img_err}")
            # 슬라이드 노트 추출
            if slide.has_notes_slide:
                notes_slide = slide.notes_slide
                for shape in notes_slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_text += shape.text + "\n"
            if slide_text.strip():                
                chunks = splitter.split_text(slide_text)
                for chunk in chunks:
                    metadata = {
                        "source": os.path.basename(file_path),
                        "slide": i + 1,
                        "agent": agent,
                        "doc_type": "PPT"
                    }
                    docs.append(Document(page_content=chunk, metadata=metadata))
    except Exception as e:
        print(f"PPT 처리 중 오류 발생 ({file_path}): {e}")
    return docs

# ============================================================
# 여러 문서 리스트를 합쳐 FAISS 벡터 스토어 생성
# ============================================================
def setup_vector_store(documents):
    all_docs = []
    for doc_list in documents:
        all_docs.extend(doc_list)
    # 한국어 지원이 더 좋은 다국어 모델로 변경
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    return vectorstore

# ============================================================
# RAG 체인 생성 (벡터 스토어 + LLM 연결)
# ============================================================
def create_retrieval_qa(vectorstore, llm=None):
    if llm is None:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa

# ============================================================
# 일반 질문 응답 (retrieval 없이 단순 LLM 활용)
# ============================================================
def create_general_qa(llm=None):
    if llm is None:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    def answer(query):        
        response = llm.invoke({"query": query})
        return {"result": response, "source_documents": []}
    return answer


AGENT_ROUTING_RULES = [
    {
        "agent": "deepseek",
        "keywords": ["강화학습", "deepseek"]
    },
    {
        "agent": "brochure",
        "keywords": ["서비스"]
    },
    {
        "agent": "rule",
        "keywords": ["남녀고용평등법", "회사"]
    }
]
# ============================================================
# 질문 내용에 따른 에이전트 라우팅 (간단한 키워드 기반 예시)
# ============================================================
def route_agent(query):
    """
    쿼리 문자열을 받아서, 미리 정의한 키워드 기반 규칙에 따라 에이전트를 선택합니다.
    해당하는 키워드가 여러 개 일 경우, 우선순위는 AGENT_ROUTING_RULES 리스트의 순서를 따릅니다.
    """
    lower_query = query.lower()
    for rule in AGENT_ROUTING_RULES:
        for keyword in rule["keywords"]:
            if keyword.lower() in lower_query:
                return rule["agent"]
    return "general"

# ============================================================
# 사용자 질의응답 루프: 질문에 따라 적절한 에이전트(RAG 체인 또는 일반 LLM) 선택
# ============================================================
def query_loop(qa_deepseek=None, qa_brochure=None, qa_rule=None, general_qa=None):
    print("문서 기반 질의응답 시스템을 시작합니다. (종료하려면 'exit' 입력)")
    while True:
        query = input("\n🔹 질문을 입력하세요: ")
        if query.lower().strip() == "exit":
            print("프로그램을 종료합니다.")
            break

        # 에이전트 선택 (deepseek / brochure / general)
        agent = route_agent(query)
        if agent == "deepseek":
            qa_chain = qa_deepseek
        elif agent == "brochure":
            qa_chain = qa_brochure
        elif agent == "rule":
            qa_chain = qa_rule
        else:
            qa_chain = None  # general agent

        if qa_chain:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # 관련 문서가 없으면 기본 메시지 출력
            if not source_docs:
                answer = "해당 질문에 대한 답변을 찾을 수 없습니다."
                source_str = "General Agent"
            else:
                sources = set()
                for doc in source_docs:
                    meta = doc.metadata
                    if "page" in meta:
                        sources.add(f"{meta.get('source', '')} {meta.get('page', '')}페이지")
                    elif "slide" in meta:
                        sources.add(f"{meta.get('source', '')} {meta.get('slide', '')}슬라이드")
                source_str = ", ".join(sources)
        else:
            # general agent: retrieval 없이 LLM 단독 사용
            res = general_qa(query)
            answer = res.get("result", "")
            source_str = "General Agent (No Document Source)"
        
        print("답변:", answer)
        print("출처:", source_str)

# ============================================================
# main: 각 문서별 에이전트 지정, 벡터 스토어 및 체인 구성 후 질의응답 루프 실행
# ============================================================
def main():
    # 문서 처리: 에이전트별로 메타데이터에 "agent"를 기록
    docs_deepseek = process_pdf("pdf_file1.pdf", agent="deepseek")
    docs_brochure = process_pdf("pdf_file2.pdf", agent="brochure")
    docs_rule = process_ppt("ppt_file1.pptx", agent="rule")
    # 벡터 DB 구성: 각 에이전트별 별도 벡터 스토어 구축
    vectorstore_deepseek = setup_vector_store([docs_deepseek])
    vectorstore_brochure = setup_vector_store([docs_brochure])
    vectorstore_rule = setup_vector_store([docs_rule])
    
    # RAG 체인 생성
    qa_deepseek = create_retrieval_qa(vectorstore_deepseek)
    qa_brochure = create_retrieval_qa(vectorstore_brochure)
    qa_rule = create_retrieval_qa(vectorstore_rule)
    # 일반 에이전트 (retrieval 미적용)
    general_qa = create_general_qa()
    
    # 사용자 질의응답 루프 실행
    query_loop(qa_deepseek, qa_brochure, qa_rule, general_qa)
    
    
if __name__ == '__main__':
    main() 