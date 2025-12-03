import streamlit as st

st.title("Streamlit Deployment Test 🚀")

st.write("If you're seeing this on your deployed app, everything is working!")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}! Your Streamlit app is live 🎉")

st.line_chart({"data": [1, 5, 3, 6, 2, 8]})
