import streamlit as st
from confusion_engine import extract_concepts, classify_gap
from explanation_engine import generate_explanation, generate_visual_explanation

st.set_page_config(page_title="Academic Confusion Mapper")

st.title("🧠 Academic Confusion Mapper")
st.write("Detect where exactly a student is confused and explain it clearly.")

user_input = st.text_area("Enter your academic doubt:")

if st.button("Analyze Confusion"):

    if user_input.strip() == "":
        st.warning("Please enter a doubt.")
    else:
        understood, confused = extract_concepts(user_input)

        if confused is None:
            st.error("Could not detect confusion structure. Try: 'I understand X but not Y'")
        else:
            gap_type, confidence = classify_gap(user_input)

            st.subheader("🔍 Detected Analysis")
            st.write(f"✅ Concept Understood: {understood}")
            st.write(f"❓ Confusion Node: {confused}")
            st.write(f"📚 Knowledge Gap Type: {gap_type}")
            st.write(f"Confidence Score: {confidence}")

            st.subheader("📘 Explanation")
            explanation = generate_explanation(confused, gap_type)
            st.write(explanation)

            st.subheader("📊 Visual Explanation")
            generate_visual_explanation()
