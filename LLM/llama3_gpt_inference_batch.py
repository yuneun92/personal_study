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
gpt_ans = dict()
llama_ans = dict()

system_message = "Referencing the text, "
import re
for i in range(len_data):
    row = rest_filled.iloc[i]
    query = row['질문'].split('\n')[0]
    file = row['문서'].split(r'\s(')[0]
    file_path = 'gpt_' + file.split(r'\s(')[0] + '.txt'
    chunk_id = str(row['청크']).split(',')[0]
    
    if file not in llama_ans:
        llama_ans[file] = {}기
    llama_response = generate_response(system_message + query, chunks[file_path][chunk_id])
    llama_ans[file][chunk_id] = llama_response

    if file not in gpt_ans:
        gpt_ans[file] = {}

    gpt_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message + query},
            {"role": "user", "content": chunks[file_path][chunk_id]},
        ]
    )
    gpt_ans[file][chunk_id] = gpt_response.choices[0].message.content
    
    print(f'{i+1}번째 처리 완료')
