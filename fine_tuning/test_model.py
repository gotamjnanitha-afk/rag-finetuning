
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "university_model"

print("Loading fine-tuned model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")

while True:
    question = input("\nAsk a university question (type exit to stop): ")

    if question.lower() == "exit":
        print("Testing completed!")
        break

    prompt = "Answer this university question clearly: " + question

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        num_beams=4,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nFine-Tuned Model Answer:")
    print(answer)
