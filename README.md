# RAG vs Fine-Tuning: A Practical Comparison for Domain Knowledge

## Project Overview

This project compares two approaches used in Artificial Intelligence:

1. **Retrieval-Augmented Generation (RAG)**
2. **Fine-Tuning**

The project uses sample university knowledge to test how both approaches respond to questions about admissions, courses, examinations, library resources, and attendance.

The main objective is to understand their differences in factual accuracy, response time, updateability, and practical usage.

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- PyTorch
- NumPy

## Project Structure

```text
Finetuning-Rag/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── documents/
│   └── university_knowledge.txt
│
├── dataset/
│   └── training_data.json
│
├── rag/
│   └── rag.py
│
├── fine_tuning/
│   ├── train.py
│   ├── test_model.py
│   └── university_model/
│
└── evaluation/
    ├── compare.py
    └── results.md
```

## RAG Implementation

RAG retrieves relevant information from the university knowledge document.

### Workflow

1. Load the university knowledge document.
2. Divide the document into smaller text sections.
3. Convert the sections into numerical embeddings.
4. Store the embeddings in a FAISS index.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant information.

### Run RAG

```powershell
python -m streamlit run app.py
```

## Fine-Tuning Implementation

Fine-Tuning trains a pre-trained language model using a small university question-and-answer dataset.

### Model Used

`google/flan-t5-small`

### Workflow

1. Prepare question-and-answer training examples.
2. Load the pre-trained model and tokenizer.
3. Tokenize the training data.
4. Train the model using the dataset.
5. Save the trained model.
6. Test the model with university questions.

### Train the Model

```powershell
python .\fine_tuning\train.py
```

### Test the Model

```powershell
python .\fine_tuning\test_model.py
```

## Experimental Comparison

Both approaches were tested using five university-related questions.

| Evaluation | RAG | Fine-Tuning |
|---|---|---|
| Attendance information | Correct | Incorrect |
| Courses offered | Correct | Incomplete |
| Library resources | Correct | Incomplete |
| Examination information | Correct | Incomplete |
| Admission information | Correct | Incomplete |

### Observations

- RAG retrieved relevant information for all five test questions.
- Fine-Tuning generated answers, but several answers were inaccurate or incomplete.
- RAG retrieval was faster in the observed test runs.
- Fine-Tuned generation took longer per question in the observed test runs.

The timings measure different operations: RAG retrieval versus fine-tuned answer generation. They should not be treated as a complete end-to-end performance comparison.

## Conclusion

The experiment showed that RAG was more suitable for answering questions that required specific information from the university knowledge document.

Fine-Tuning can help a model learn a task or response style. However, the small training dataset used in this project was not sufficient to produce consistently accurate university-specific answers.

RAG is useful when information needs to be retrieved and updated regularly, while Fine-Tuning is useful when adapting a model's behaviour or response format.

## Limitations

- The project uses a small sample university dataset.
- The university information is synthetic and intended only for demonstration.
- Fine-Tuning uses only 12 question-and-answer examples.
- The model was trained on CPU.
- Only five questions were used in the comparison.
- The results are educational observations, not production-level benchmarks.

## Future Improvements

- Add more domain-specific training examples.
- Test with a larger dataset.
- Improve the fine-tuned model's factual accuracy.
- Use an LLM with RAG to generate answers from retrieved context.
- Evaluate more questions and compare additional performance metrics.

## Author

**Jnanitha**

## Project Purpose

Academic project demonstrating a practical comparison between Retrieval-Augmented Generation and Fine-Tuning.
