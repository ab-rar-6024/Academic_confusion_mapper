Academic Confusion Mapper
Detect Where a Student Is Confused — and Explain It Clearly

An AI-powered hybrid educational assistant that identifies specific conceptual confusion in student input and generates structured, technically accurate explanations with visual learning support.

Project Overview

Academic Confusion Mapper is a hybrid Natural Language Understanding (NLU) system designed to:

Detect the concept a student understands

Identify the exact confusion node

Classify the knowledge gap type

Generate structured explanations

Provide visual learning demonstrations

Unlike pure LLM systems, this project uses a hybrid architecture combining structured domain knowledge with controlled AI generation to ensure accuracy and prevent hallucination.

🧠 Example Input
I understand gradient descent but not why learning rate affects convergence

🔍 System Output
✅ Concept Understood

gradient descent

❓ Confusion Node

why learning rate affects convergence

📚 Knowledge Gap Type

Application confusion

📘 Explanation

Simple Explanation

Deeper Technical Explanation

Real-world Analogy

📊 Visual Demonstration

Loss curve visualization

Small learning rate (stable convergence)

Large learning rate (overshooting behavior)

🏗 Architecture
Hybrid Design (College Demo Optimized)

The system avoids pure LLM hallucination by using:

🔹 Rule-based structured explanations (core accuracy)

🔹 LLM-based analogy generation (intelligent enhancement)

🔹 Mathematical visualization using Matplotlib

🔹 Zero-shot classification for knowledge gap detection

⚙️ Tech Stack

Python

Streamlit

HuggingFace Transformers (FLAN-T5)

PyTorch

Matplotlib

NumPy

📂 Project Structure
academic_confusion_mapper/
│
├── app.py
├── confusion_engine.py
├── explanation_engine.py
├── requirements.txt
└── README.md

▶️ How to Run
1️⃣ Clone the Repository
git clone https://github.com/yourusername/academic-confusion-mapper.git
cd academic-confusion-mapper

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py


Open in browser:

http://localhost:8501

🎯 Supported Topics (Demo Version)

Learning Rate & Convergence

Overfitting

Gradient Descent Basics

More topics can be easily added using structured knowledge templates.

🧪 Demo Use Cases

Try inputs like:

I understand model training but not overfitting

I understand gradient descent but not why learning rate affects convergence

🧠 Key Features

Confusion Node Detection

Knowledge Gap Classification

Structured Multi-Level Explanation

Real-world Analogies

Mathematical Visualization

Hybrid AI Architecture

Streamlit Interactive Interface

🎓 Academic Significance

This project demonstrates:

Applied Natural Language Understanding

Hybrid AI System Design

Explainable AI (XAI) principles

Educational AI architecture

Controlled generative modeling

🚀 Future Improvements

Add more ML topic modules

Add user confusion history tracking

Add confidence calibration

Add knowledge graph integration

Add interactive learning rate slider

Deploy as web app

👨‍💻 Author

Mohamed Abrar
AI & Data Science Student
