from huggingface_hub import notebook_login
notebook_login()
token = ""

# 원본 llama3 instruct
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=token
)

def generate_response(system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate( #하이퍼파라미터 조
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6, 
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response, skip_special_tokens=True)


# llama, gpt 답변 생성하기 (배치)
data = "" # 파일 불러오기
import openai
openai.api_key = ""
len_data = len(data)

system_message = "Referencing the text, "
import re
import pandas as pd
chunk_responses = {}
expert_llama = dict()
expert_gpt = dict()

# 데이터프레임 초기화
expert_data = {'File': [], 'Chunk_ID': [], 'Question': [], 'Response_gpt': [], 'Response_llama': []}

for i in range(102):
    row = rest_filled.iloc[i]
    query = row['질문'].split('\n')[0]
    file = row['문서'].split(r'\s(')[0]
    file_path = 'gpt_' + file.split(r'\s(')[0] + '.txt'
    chunk_id = str(row['청크']).split(',')[0]
    
    if file not in llama_ans:
        expert_llama[file] = {}

    llama_response = generate_response(system_message + query, data[file_path][chunk_id])
    
    if file not in gpt_ans:
        expert_gpt[file] = {}

    gpt_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message + query},
            {"role": "user", "content": data[file_path][chunk_id]},
        ]
    )
    
    # 응답 추가
    expert_data['File'].append(file)
    expert_data['Chunk_ID'].append(chunk_id + str(i))
    expert_data['Question'].append(system_message + query)
    expert_data['Response_llama'].append(llama_response)
    expert_data['Response_gpt'].append(gpt_response.choices[0].message.content)

    expert_llama[file][chunk_id + str(i)] = llama_response
    expert_gpt[file][chunk_id + str(i)] = gpt_response.choices[0].message.content
    
    print(f'{i+1}번째 처리 완료')

# 데이터프레임 생성
expert_response_df = pd.DataFrame(expert_data)

# CSV 파일로 저장
expert_response_df.to_csv("./data/expert_responses.csv", index=False)

print("chunk_responses가 chunk_responses.csv 파일로 저장되었습니다.")

for i in range(11):
    row = expert_DB_filled.iloc[i]
    query = row['질문'].split('\n')[0]
    file = 
    file_path = 'gpt_' + file + '.txt'
    chunk_id = str(row['청크']).split('\n')[0]
    
    if file not in llama_ans:
        expert_llama[file] = {}

    llama_response = generate_response(system_message + query, data[file_path][chunk_id])
    
    if file not in gpt_ans:
        expert_gpt[file] = {}

    gpt_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message + query},
            {"role": "user", "content": data[file_path][chunk_id]},
        ]
    )
    
    # 응답 추가
    expert_data['File'].append(file)
    expert_data['Chunk_ID'].append(chunk_id + str(i))
    expert_data['Question'].append(system_message + query)
    expert_data['Response_llama'].append(llama_response)
    expert_data['Response_gpt'].append(gpt_response.choices[0].message.content)

    expert_llama[file][chunk_id + str(i)] = llama_response
    expert_gpt[file][chunk_id + str(i)] = gpt_response.choices[0].message.content
    
    print(f'{i+1}번째 처리 완료')

# 데이터프레임 생성
expert_response_df = pd.DataFrame(expert_data)

# CSV 파일로 저장
expert_response_df.to_excel("./data/expert_responses.xlsx", index=False)


