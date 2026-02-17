import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Lightweight Model for Analogy Only
# -------------------------------


model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# -------------------------------
# Hybrid Explanation Generator
# -------------------------------

def generate_explanation(topic, gap_type):

    topic_lower = topic.lower()

    # ===== CASE 1: Learning Rate =====
    if "learning rate" in topic_lower:

        simple = """
Learning rate controls how big a step we take while updating model weights.
If the step is too large, the algorithm may jump over the minimum.
If the step is too small, learning becomes very slow.
"""

        deeper = """
In gradient descent, weights are updated using:

w = w - learning_rate × gradient

The learning rate scales the gradient.
A large learning rate can cause oscillation or divergence.
A small learning rate leads to slow but stable convergence.
The correct learning rate ensures smooth convergence toward the minimum.
"""

        analogy_prompt = """
Give a short real-world analogy explaining learning rate in optimization.
Keep it under 3 sentences.
"""

        inputs = tokenizer(analogy_prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        analogy = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return f"""
### Simple Explanation:
{simple}

### Deeper Explanation:
{deeper}

### Real-World Analogy:
{analogy}
"""

    # ===== CASE 2: Overfitting =====
    elif "overfitting" in topic_lower:

        simple = """
Overfitting happens when a model memorizes training data instead of learning patterns.
It performs well on training data but poorly on new data.
"""

        deeper = """
Overfitting occurs when a model has too many parameters or trains too long.
It captures noise instead of the underlying distribution.
Techniques like regularization, dropout, and cross-validation help reduce overfitting.
"""

        analogy = "It is like memorizing exam answers instead of understanding the concepts."

        return f"""
### Simple Explanation:
{simple}

### Deeper Explanation:
{deeper}

### Real-World Analogy:
{analogy}
"""

    # ===== DEFAULT FALLBACK =====
    else:
        return """
### Explanation:
This topic is not yet in the structured knowledge base.

For demo purposes, please try:
- Learning rate
- Overfitting
"""


# -------------------------------
# Visual Explanation (Improved)
# -------------------------------

def generate_visual_explanation():

    x = np.linspace(-5, 5, 100)
    y = x**2

    def gradient_descent(start, lr, steps):
        w = start
        path = [w]
        for _ in range(steps):
            grad = 2 * w
            w = w - lr * grad
            path.append(w)
        return path

    small_lr_path = gradient_descent(start=4, lr=0.1, steps=10)
    large_lr_path = gradient_descent(start=4, lr=0.9, steps=10)

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.scatter(small_lr_path, [w**2 for w in small_lr_path])
    ax.scatter(large_lr_path, [w**2 for w in large_lr_path])

    ax.set_title("Learning Rate Affects Convergence")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Loss")

    ax.legend(["Loss Curve", "Small LR (Stable)", "Large LR (Overshoot)"])

    st.pyplot(fig)
# I understand gradient descent but not why learning rate affects convergence
# I understand model training but not overfitting
