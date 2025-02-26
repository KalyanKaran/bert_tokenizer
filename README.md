# 🚀 IMDb Sentiment Classification with BERT (CPU-Optimized)

This project fine-tunes the **BERT (bert-base-uncased)** model on the **IMDb movie review dataset** to perform **binary sentiment classification** (positive/negative). It is designed to run efficiently on **CPU** inside a **Dockerized environment**.

---

## 📂 Project Structure

├── train.py           # Training script
├── Dockerfile              # Docker setup
├── requirements.txt        # Dependencies
├── README.md               # Documentation (this file)
└── results/                # Saved fine-tuned model will be genarated here (generated after training)

---

## 📌 Features

✅ Fine-tunes **bert-base-uncased** on IMDb movie reviews  
✅ Uses **Hugging Face Transformers** for training  
✅ **Optimized for CPU** (small batch size, `no_cuda=True`)  
✅ **Dockerized** for easy setup and portability  
✅ Logs training progress every **100 steps**  
✅ Evaluates the model and performs a **sample prediction** after training  

---

## 🚀 Setup Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/KalyanKaran/bert_tokenizer.git
cd bert_tokenizer

2️⃣ Build the Docker Image

docker build -t bert_tokenizer .

3️⃣ Run the Training Process

docker run --rm bert_tokenizer

🎯 Expected Output
	•	Training progress (loss, accuracy) printed every 100 steps
	•	Final evaluation results on the IMDb test dataset
	•	A sample sentiment prediction after training

Example Output:

🚀 Training started on CPU...
✅ Evaluation Results: {'eval_loss': 0.30, 'eval_accuracy': 0.89}
📝 Sample Text: This movie was absolutely fantastic!
🔮 Predicted Sentiment: Positive
✅ Model fine-tuning completed and saved!

📊 Model Evaluation

After training, the model is saved inside the fine_tuned_model/ directory.

To manually test it, run:

docker run --rm -it my-ai-ml-project /bin/sh
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model'); tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model'); text = 'This movie was amazing!'; inputs = tokenizer(text, return_tensors='pt'); print(model(**inputs).logits)"

⚙️ Modifications & Customization
	•	To increase accuracy, modify train.py to:
	•	Increase epochs (num_train_epochs=3 → 5)
	•	Adjust learning rate (2e-5 → 3e-5)
	•	Use larger batch size if you switch to GPU
	•	To use a different dataset, replace:

dataset = load_dataset("imdb")

with:

dataset = load_dataset("csv", data_files={"train": "data/train.csv", "validation": "data/val.csv"})

and provide your own dataset in data/.

📌 Requirements

✔ Docker
✔ Python 3.9+
✔ Hugging Face transformers, datasets, torch (installed via requirements.txt)

📜 License

This project is MIT Licensed – you are free to use, modify, and distribute it.

💡 Credits
	•	Built using Hugging Face Transformers 🚀
	•	IMDb dataset from Hugging Face Datasets
	•	Model: BERT (bert-base-uncased)

❓ Need Help?

💬 Reach out via Issues or Discussions!

🚀 Happy Fine-Tuning!

This **README.md** is now in a **single file** while keeping it detailed and well-structured. 🚀🔥 Let me know if you need any changes!
