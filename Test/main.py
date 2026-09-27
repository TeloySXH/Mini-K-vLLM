import os

from transformers import AutoTokenizer

from KvChat.engine.llm_engine import LLMEngine as LLM
from KvChat.sampling_params import SamplingParams

config = {
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 16,
    "max_model_len": 128,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 1,
    "enforce_eager": True,
    "model_name_or_path": "~/huggingface/Qwen3-4B",  # 本地模型路径，使用 ~ 表示用户主目录
}


def main():
    model_path = os.path.expanduser(config.get("model_name_or_path", "~/huggingface/Qwen3-4B"))
    
    # 验证本地模型路径存在
    if not os.path.isdir(model_path):
        raise ValueError(f"本地模型路径不存在: {model_path}")
    
    # 移除model_name_or_path，使用model参数初始化LLM
    config_copy = {k: v for k, v in config.items() if k != "model_name_or_path"}
    llm = LLM(model=model_path, **config_copy)
    
    # 从本地路径加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # max_tokens is the max number of generated tokens
    sampling_params = SamplingParams(
        temperature=0.6, max_tokens=256
    )
    prompts = [
        "介绍你自己",  # * 15,
        "列举20以内的质数",  # * 15,
        "为什么推荐冬天喝羊肉汤",  # * 15,
    ]  # * 30
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs is a list of dicts with 'text' and 'token_ids' keys
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
