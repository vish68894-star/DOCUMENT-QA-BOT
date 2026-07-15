from flask import Flask, request, jsonify
from airllm import AutoModel

app = Flask(__name__)

MODEL = "garage-bAInd/Platypus2-7B"

print("="*60)
print("Loading AirLLM model...")
print("="*60)

model = AutoModel.from_pretrained(MODEL, device="cpu")
print("Model loaded successfully.")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_tokens", 150)

    # Tokenize with strict limit
    inputs = model.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )

    generation_output = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_dict_in_generate=True
    )

    full_output = model.tokenizer.decode(
        generation_output.sequences[0],
        skip_special_tokens=True
    )

    # Strip the input prompt from the output
    input_text = model.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    answer = full_output[len(input_text):].strip()
    if not answer:
        answer = full_output.strip()

    return jsonify({"response": answer})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=False)