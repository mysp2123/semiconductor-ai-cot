import json
from transformers import pipeline
import time

# Tải mô hình GPT-2 hoặc GPT-Neo từ Hugging Face
generator = pipeline("text-generation", model="gpt2", pad_token_id=50256)  # 50256 là EOS token ID cho GPT-2

# Đọc câu hỏi từ file JSON
def load_questions(file_path):
    """Đọc dữ liệu câu hỏi từ file JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Trả lời câu hỏi bằng GPT-2 hoặc GPT-Neo
def query_model(question, method="standard"):
    """Gửi câu hỏi tới mô hình GPT-2 hoặc GPT-Neo với phương pháp mong muốn"""
    if method == "standard":
        prompt = f"Answer the following question concisely: {question}"
    elif method == "cot":
        prompt = f"Think step by step and then answer: {question}"
    else:
        prompt = f"{question}"  # Dự phòng cho các phương pháp khác

    try:
        # Sử dụng mô hình GPT-2 hoặc GPT-Neo để trả lời câu hỏi
        response = generator(prompt, max_length=200, num_return_sequences=1)
        return response[0]["generated_text"].strip()
    except Exception as e:
        print(f"⚠️ Lỗi khi gọi mô hình: {e}")
        return "[ERROR]"

# Đánh giá mô hình GPT-2 hoặc GPT-Neo trên bộ dữ liệu câu hỏi
def evaluate_questions(input_file, output_file):
    """Chạy đánh giá mô hình GPT-2 hoặc GPT-Neo trên bộ dữ liệu câu hỏi"""
    data = load_questions(input_file)
    results = []

    for item in data:
        question = item["question"]
        gt_answer = item["answer"]

        print(f"🧠 Đang xử lý: {question}")
        standard_answer = query_model(question, method="standard")
        cot_answer = query_model(question, method="cot")

        results.append({
            "question": question,
            "ground_truth": gt_answer,
            "standard_answer": standard_answer,
            "cot_answer": cot_answer
        })
        time.sleep(1)  # Tránh gửi quá nhiều request cùng lúc

    # Lưu kết quả vào file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu kết quả vào {output_file}")

# Chạy đánh giá mô hình
if __name__ == "__main__":
    input_file = "db/questions/cot_questions_clean.json"
    output_file = "results/evaluated_results.json"

    print("🚀 Bắt đầu đánh giá mô hình GPT-2 hoặc GPT-Neo...")
    evaluate_questions(input_file, output_file)
    print("✅ Hoàn thành đánh giá!")
