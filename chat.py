import os
import logging
from transformers import AutoTokenizer
from KvChat import LLM, SamplingParams

# Suppress torch inductor auto-tune warning
logging.getLogger('torch._inductor').setLevel(logging.ERROR)


def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
    )
    temperature = 1.0
    max_tokens = 32000
    messages = []
    compress_enabled = True
    print("   Mini-K vLLM Qwen3-4B 多轮次对话   ")
    print("/exit            退出对话")
    print("/reset           清空历史记录")
    print("/compress on     启用KV压缩（默认）")
    print("/compress off    禁用KV压缩")
    print("")
    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            print("【已退出】")
            break
        if user_input == "/reset":
            messages = []
            print("【会话已清空】")
            continue
        if user_input == "/compress on":
            compress_enabled = True
            llm.model_runner.config.kv_compress_enabled = True
            print("【KV压缩已启用】")
            continue
        if user_input == "/compress off":
            compress_enabled = False
            llm.model_runner.config.kv_compress_enabled = False
            print("【KV压缩已禁用】")
            continue
        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        llm.model_runner.config.kv_compress_enabled = compress_enabled
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm.add_request(prompt, sampling_params)
        print("Assistant: ", end="", flush=True)
        generated_token_ids = []
        while not llm.is_finished():
            outputs, num_tokens, step_token_ids, _, _ = llm.step()
            if step_token_ids is not None:
                for tid in step_token_ids:
                    generated_token_ids.append(tid)
                    text = tokenizer.decode([tid], skip_special_tokens=False)
                    print(text, end="", flush=True)
        print()
        assistant_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
