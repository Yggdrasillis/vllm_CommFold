from vllm import LLM


llm = LLM(model="../model/DeepSeek-V3",
        trust_remote_code=True, tensor_parallel_size=2, 
        enable_expert_parallel=True, enforce_eager=True,
        hf_overrides={"moe_pipe_degree": 2})


with open('4k_token.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()
outputs = llm.generate(input_text)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")