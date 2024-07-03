RAG 구현하기 전 읽었을 땐 와닿지 않던 논문인데요. 구현한 뒤 읽어보니 더 많은 인사이트가 되어 기록합니다. 논문 내용은 영어로, 첨언은 한국어도 함께 적었습니다.

This is the review of 

## 📄 Retrieval-Augmented Generation for Large Language Models: A Survey
> Yunfan Gao et al.
> 
> 27 Mar 2024

### Background
1. Rise of the Transformer architecture
   - The Transformer architecture focused on enhancing LMs by incorporating additional knowledge through Pre-Training Models, aiming refining pre-training techs.
2. ChatGPT
  - `ICL` (In-Context Learning)
3. RAG's shift
  - There had been RAG techs, but researches shifted their way:
      - (previous) providing better info. for LLMs
      - ▶️ answer more complex and knowledge-intensive tasks
  - Basic structure of RAG: `Retrieval`, `Generation`, `Augmentation`

### Overview of RAG

#### [TYPES]

1. `Naive RAG` : `"Retrieve-Read generation"` (Indexing → Retrieval → Generation)
   - `Indexing` : cleaning & extracting of raw data → chunking → encoding text and store vectors in vector database
   - `Retrieval` : transform a query into a vector repr → compute similarity score → priotize and retrieve the top K chunks
   - `Generation` : synthesize the query & selected chunks into a coherent prompt → LLM formulate a response
   - **`Drawbacks`**
     - `Retrieval Challenges` : precision, recall 👎 ▶️ misaligned / irrelevant chunks, missing crucial info.
       - 유사도 점수에만 의존해 좋은 검색 성능을 갖는 것은 불가능에 가깝습니다.
       - 저는 하이브리드 서치로 워드 임베딩 + LLM 임베딩을 사용했는데요, 이 역시 정확한 단어의 일치를 보는 것은 아니기 때문에 bm25까지 같이 사용할까 생각 중입니다.
     - `Generation Difficulties`
     - `Augment Hurdles`
       
     이 약점들을 완전히 해소할 수 있는 RAG 기법이 있을지는 모르겠지만.. Advanced RAG에서 훨씬 좋은 결과를 보입니다.
2. Advanced RAG
   - 
3. Modular RAG
   -
---

#### [STEPS]

1. RETRIEVAL
2. GENERATION
3. AUGMENTATION
