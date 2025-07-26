# DocuGenie 🧞‍♂️  
문서 기반 AI 챗봇  

PDF, DOCX, PPTX 파일을 업로드하고 궁금한 점을 대화로 물어보세요!  
한국어로도 자연스럽게 응답합니다.

---

## 📺 데모 영상
[![Demo Video](https://img.youtube.com/vi/HqXMsq-ZIs0/0.jpg)](https://www.youtube.com/watch?v=HqXMsq-ZIs0)


## 🛠️ 주요 기능

- 다양한 문서 업로드 지원 (PDF, DOCX, PPTX)
- 문서 내 정보 기반 대화형 질의응답
- OpenAI GPT-4 기반 자연어 처리
- LangChain 기반 문서 검색 및 응답 처리
- 한국어 완벽 지원

---

## 🧰 사용 기술

- **Python**
- **OpenAI GPT-4**: 챗봇 모델 및 자연어 이해 및 생성
- **LangChain**: 문서 검색(QA) 체인 구성
- **Streamlit**: 프론트엔드 UI 구성
- **FAISS / ChromaDB**: 벡터스토어 기반 검색

---

## 🚀 설치 및 실행 방법

1. **필수 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

2. **Streamlit 앱 실행**
    ```bash
    streamlit run Document_chatbot.py
    ```

3. **웹 브라우저에서 챗봇 사용**
    - 문서 업로드 → OpenAI API 키 입력 → 질문 시작!

---

## ⚙️ 환경 변수

- `OpenAI API Key` (앱 내에서 직접 입력 가능)

---

## 📁 파일 구조

```
├── Document_chatbot.py
├── requirements.txt
└── README.md
```

## 🔗 데모 체험

Streamlit에서 배포된 DocuGenie를 직접 사용 할 수 있습니다!  
👉 [https://docugenie-yfmfyolgms4d4prrtvfqtb.streamlit.app](https://docugenie-yfmfyolgms4d4prrtvfqtb.streamlit.app)

- 최대 200MB 문서 업로드(PDF, DOCX, PPTX 지원)
- OpenAI API Key 입력 후 질문 가능


