"""
Sample test cases for the evaluation framework.

Each case has:
- query: what the user asked
- retrieved_docs: what the RAG pipeline returned
- responses: dict of prompt_label -> llm_response
  (some grounded, some hallucinated, for testing)
"""

SAMPLE_CASES = [
    {
        "query": "What is the attention mechanism in transformers?",
        "retrieved_docs": [
            "The attention mechanism allows the model to focus on different parts of the input sequence when producing each output token. It computes query, key, and value vectors from the input.",
            "Self-attention enables each token to attend to all other tokens in the sequence. The attention score is computed as the dot product of query and key vectors, scaled by the square root of the dimension.",
            "Multi-head attention runs multiple attention operations in parallel, allowing the model to attend to information from different representation subspaces."
        ],
        "responses": {
            "grounded_response": (
                "The attention mechanism in transformers allows each token to attend to all other tokens "
                "in the sequence. It works by computing query, key, and value vectors from the input, "
                "then using the dot product of query and key to compute attention scores. "
                "Multi-head attention runs this process in parallel across multiple heads."
            ),
            "hallucinated_response": (
                "The attention mechanism works by using a recurrent neural network to process "
                "tokens sequentially. Each token is processed one at a time using LSTM cells, "
                "which maintain a hidden state across the sequence. The final hidden state "
                "is used to generate the output."
            ),
        }
    },
    {
        "query": "How does gradient descent work in neural network training?",
        "retrieved_docs": [
            "Gradient descent is an optimization algorithm that minimizes the loss function by iteratively moving in the direction of steepest descent, as defined by the negative of the gradient.",
            "The learning rate controls how large each step is during gradient descent. A learning rate that is too high can cause divergence; too low and training is slow.",
            "Stochastic gradient descent (SGD) computes gradients on mini-batches rather than the full dataset, which makes training faster and can help escape local minima."
        ],
        "responses": {
            "grounded_response": (
                "Gradient descent minimizes the loss by iteratively moving in the direction of the "
                "negative gradient. The learning rate controls step size — too high causes divergence, "
                "too low is slow. SGD uses mini-batches instead of the full dataset to speed up training."
            ),
            "hallucinated_response": (
                "Gradient descent works by randomly initializing weights and then using a genetic "
                "algorithm to evolve the network toward better solutions. Each generation of "
                "networks is evaluated, and the best-performing ones are selected to reproduce, "
                "gradually improving performance over many generations."
            ),
        }
    },
    {
        "query": "What is the difference between precision and recall?",
        "retrieved_docs": [
            "Precision is the ratio of true positives to all predicted positives: TP / (TP + FP). It measures how many of the predicted positives are actually positive.",
            "Recall is the ratio of true positives to all actual positives: TP / (TP + FN). It measures how many of the actual positives were correctly identified.",
            "The F1 score is the harmonic mean of precision and recall, balancing both metrics into a single number."
        ],
        "responses": {
            "grounded_response": (
                "Precision measures how many of your positive predictions were actually correct: "
                "TP / (TP + FP). Recall measures how many actual positives you correctly found: "
                "TP / (TP + FN). F1 score balances both by computing their harmonic mean."
            ),
            "hallucinated_response": (
                "Precision measures the accuracy of your model on the test set overall. "
                "Recall is a measure of how well the model generalizes to new data. "
                "High precision means the model performs well on seen data, "
                "while high recall means it performs well on unseen data."
            ),
        }
    }
]
