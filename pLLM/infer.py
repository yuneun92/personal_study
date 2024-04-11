# dependencies

# langchain
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# ctransformers: 트랜스포머 모델을 c 기반으로 실행할 수 있는 라이브러리
from ctransformers import AutoModelForCausalLM

import json, time

def ctrans_infer(user_prompt: list[str]) -> list[list[str]]:
    with open('./configs/c_model_config.json', 'r') as file:
        config = json.load(file)
        
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    result = []
    for model in config:
        model_config = config[model]
        llm = LlamaCpp(
            model_path=model_config["model_path"],
            n_gpu_layers=-1,
            callback_manager=callback_manager, #for streaming
            verbose=True,  # Verbose is required to pass to the callback manager
        )

        prompt = PromptTemplate.from_template(config['prompt_template'])
        systemp_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    
        llm_chain = prompt | llm
        
        for input_prompt in user_prompt:
            start = time.time()
            output = llm_chain.invoke({"systemp_message": systemp_message, "prompt": input_prompt})
            end = time.time()

            result.append([output, (end-start)])
    return result