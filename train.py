import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # Load IMDb dataset from Hugging Face
    dataset = load_dataset("imdb")

    # Use pre-trained BERT model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # ✅ Reduced for CPU efficiency
        per_device_eval_batch_size=4,  # ✅ Reduced for CPU efficiency
        num_train_epochs=2,  # ✅ Reduce epochs for faster testing
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,  # ✅ Log less frequently for CPU
        save_strategy="epoch",  # ✅ Save only at the end of each epoch
        report_to="none",  # ✅ Disable extra logging tools like WandB
        no_cuda=True,  # ✅ Ensure no GPU is used
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Start training
    print("🚀 Training started on CPU...")
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print("✅ Evaluation Results:", eval_results)

    # Test the model with a sample text
    sample_text = "This movie was absolutely fantastic!"
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

    print(f"📝 Sample Text: {sample_text}")
    print(f"🔮 Predicted Sentiment: {'Positive' if predicted_label == 1 else 'Negative'}")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("✅ Model fine-tuning completed and saved!")

if __name__ == "__main__":
    main()