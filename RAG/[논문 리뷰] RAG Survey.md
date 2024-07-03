RAG 구현하기 전 읽었을 땐 와닿지 않던 논문인데요. 구현한 뒤 읽어보니 더 많은 인사이트가 되어 기록합니다. 논문 내용은 영어로, 첨언은 한국어도 함께 적었습니다.

## 📄 Retrieval-Augmented Generation for Large Language Models: A Survey
> Yunfan Gao et al.
> 
> 27 Mar 2024

### Background
**1. Rise of the Transformer architecture**
   - The Transformer architecture focused on enhancing LMs by incorporating additional knowledge through Pre-Training Models, aiming refining pre-training techs.
     
**2. ChatGPT**
  - `ICL` (In-Context Learning)
    
**3. RAG's shift**
  - There had been RAG techs, but researches shifted their way:
      - (previous) providing better info. for LLMs
      - ▶️ answer more complex and knowledge-intensive tasks
  - Basic structure of RAG: `Retrieval`, `Generation`, `Augmentation`

### RAG vs. FT
> _The choice between RAG and FT depends on the specific needs for data dynamics, customization, and computational capabilities in the application context._
![image](https://github.com/yuneun92/personal_study/assets/101092482/fb82dc65-6c22-463c-af9c-10ecce4fdfd3)

- `Prompt Engineering`: leverages a model's inherent capabilities with minimum necessity for external knowledge and model adaptation
- `RAG`: ideal for precise information retrieval tasks.
  - 👍 effetive utilization of external knowledge sources with high interpretability.
  - 👎 higher latency
- `FT`: suitable for scenarios requiring replication of specific structures, styles, or formets
  - 👍 deep customization of the model's behavior and styles.
    - reduce hallucination
  - 👎 more static, requiring retraining for updates
    - demands significant computer resources
    - may face challenges with unfamilier data.
    - LLMs struggle to learn new factual information through unsupervised fine-tuning.
    


### Overview of RAG

#### [TYPES]

1. `Naive RAG` : `"Retrieve-Read generation"` (Indexing → Retrieval → Generation)
   - `Indexing` : cleaning & extracting of raw data → chunking → encoding text and store vectors in vector database
   - `Retrieval` : transform a query into a vector repr → compute similarity score → priotize and retrieve the top K chunks
   - `Generation` : synthesize the query & selected chunks into a coherent prompt → LLM formulate a response
   - _**`Drawbacks`**_
     - `Retrieval Challenges` : precision, recall 👎 ▶️ misaligned / irrelevant chunks, missing crucial info.
       - 유사도 점수에만 의존해 좋은 검색 성능을 갖는 것은 불가능에 가깝습니다.
       - 저는 하이브리드 서치로 워드 임베딩 + LLM 임베딩을 사용했는데요, 이 역시 정확한 단어의 일치를 보는 것은 아니기 때문에 bm25까지 같이 사용할까 생각 중입니다.
     - `Generation Difficulties` : hallucination
       - 프롬프트 엔지니어링만으로 한계가 있다고 느껴서, p-tuning → PEFT → Fine Tuning 순으로 테스트해볼 생각입니다. 
     - `Augment Hurdles` : disjointed or incoherent outputs, redundancy
       
     이 약점들을 완전히 해소할 수 있는 RAG 기법이 있을지는 모르겠지만.. Advanced RAG에서 훨씬 좋은 결과를 보입니다.
2. `Advanced RAG` : _Enhance retrieval quality_ ; pre-retrieval and porst-retrieval stretegies
   - `Pre-retrieval process` : oprimize the indexing structure and the original query
     - enhance data granularity(세분화)
     - optimize index structure
     - add metadata
   - `Post-retrieval process` : rerank chunks, context compressing
     - Re-ranking: LlamaIndex, LangChain, HayStack
     - Select essential information: emphasize critical sections, shortening the cotext to be processed
       - 데이터의 품질을 저해하는 토큰이 많을 때, LLM은 답변을 잘 생성해내지 못합니다. 
       
3. `Modular RAG`
   - 💡 **Search Module** :
     1. 특정 시나리오에 적응 - LLM 생성 코드, 쿼리 언어 등.
     2. 멀티 쿼리  - parellel vector search, re-ranking (단어 + 의미 모두 포착할 수 있도록)
   - 💡 **Memory Module** :
     1. LLM에게 주는 데이터가 반복적인 자기 보완 과정에서도 기존의 데이터 분포와 유사하도록 함.
     2. 요약, 특정 데이터베이스에서의 검색, 다른 정보 스트림을 합치는 것 등
   - 💡 **Predict Module**
   - 💡 **Task Adapter Module**

---

#### [STEPS]

1. `RETRIEVAL`
   - **Source**
     - `Data structure`
      1. semi-structured data : PDF ... (text + table)

         처리하기 가장 어려운 데이터 타입..
         - Text-2-SQL queries: TableGPT
         - transform tables into text format
      2. structured data : knowledge graph (KG)
         - KnowledGPT : generates KB search queries and stores knowledge in a bersonalized base
         - G-Retriever : GNNs + LLMs + RAG // Prize-Collecting Steiner Tree (PCST) optimization prob. for targeted graph retrieval.
      3. unstructured data : text
      4. LLMs-Generated Content
         - SKR: classifies questions as `known` or `unknown`, applying retrieval enhancement selectively.
         - GenRead: replaces the retriever with an LLM genenrator ; better alignment with the pre-training objectives of causal lang. modeling.
         - Selfmem: iteratively creates an unbounded memory pool with a retrieval-enhanced generator
      - `Retrieval Granularity` : 청크를 너무 잘게 쪼개면 리트리버가 제대로 기능하기 더 어렵고, 청크 크기가 너무 크면 임베딩이 적절히 의미를 내포하지 못합니다.
        1. 토큰, 구, 문장, `명제`, 청크, 문서 ...
           > **Proposition(명제)**: atomic expressions in the text, each encapsuating a unique factual segment and presented in a concise, self-contained natural language format.
        2. KG : Entity, Triplet, sub-Graph
     
2. `GENERATION`
   - 
3. `AUGMENTATION`
   -
   
---

## 📄 A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models
> Wenqi Fan et al.
>
> 17 Jun 2024
>
> https://arxiv.org/pdf/2405.06211

![image](https://github.com/yuneun92/personal_study/assets/101092482/3c7b49d5-42f6-4549-8aba-cb6328ac8a05)

