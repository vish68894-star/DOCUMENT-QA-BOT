from airllm import AutoModel

# Using a small but capable model
MODEL = "garage-bAInd/Platypus2-7B"

print("Loading model via AirLLM (first run will download ~13GB)...")
print("This may take 10-20 minutes on first run.")
print("="*60)

model = AutoModel.from_pretrained(MODEL)

input_text = ["What is the purpose of KYC policy in banks?"]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=256,
    padding=False
)

print("Generating response...")
generation_output = model.generate(
    input_tokens['input_ids'],
    max_new_tokens=200,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print("\nMODEL RESPONSE:")
print(output)
print("="*60)
print("AirLLM test complete.")
