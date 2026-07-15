from airllm import AutoModel

# Using a small model for a quick first test
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-7B")

input_text = ['What is the capital of France?']

input_tokens = model.tokenizer(input_text,
                                return_tensors="pt",
                                return_attention_mask=False,
                                truncation=True,
                                max_length=128,
                                padding=False)

generation_output = model.generate(
    input_tokens['input_ids'],
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print("MODEL RESPONSE:", output)