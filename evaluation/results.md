
# RAG vs Fine-Tuning – Experimental Results

## 1. Experiment Overview

The experiment compares Retrieval-Augmented Generation (RAG) and Fine-Tuning using sample university knowledge.

Both approaches were tested with the same five questions.

## 2. Comparison Table

| Question | RAG Result | Fine-Tuning Result |
|---|---|---|
| Minimum attendance | Correct: 75% | Incorrect: 30 minutes |
| Courses offered | Correctly lists courses | Incomplete answer |
| Library resources | Correctly lists resources | Incomplete answer |
| Examinations | Correctly identifies both types | Incomplete answer |
| Admission process | Correctly explains application process | Unsupported detail |

## 3. Performance Observations

- RAG retrieved relevant information for all five questions.
- Fine-Tuning generated answers but several were inaccurate or incomplete.
- RAG retrieval time ranged from approximately 0.0065 to 0.0271 seconds in the second run.
- Fine-Tuned generation time ranged from approximately 0.1907 to 0.3105 seconds.
- Fine-Tuning used a small training dataset of 12 question-answer examples.

## 4. Conclusion

In this experiment, RAG performed better for answering questions that required specific university information.

RAG retrieves information directly from the knowledge document, making it easier to update when new information is added.

Fine-Tuning can help a model learn a particular response style or task, but a small training dataset may not provide enough factual knowledge.

Therefore, RAG was more suitable for this university knowledge question-answering demonstration.

## 5. Limitations

- The experiment used a small synthetic university dataset.
- Fine-Tuning was performed using a small model on CPU.
- Only five questions were used for comparison.
- RAG retrieval and fine-tuned generation times measure different operations.
- The experiment does not represent production-level performance.
