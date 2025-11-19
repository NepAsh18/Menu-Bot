import requests
import streamlit as st

st.title("Menu RAG Search System")

query = st.text_input("Search anything...")

if st.button("Search"):
    payload = {"query": query}
    res = requests.post("http://localhost:5000/search", json=payload)

    st.write("### Results:")
    for item in res.json():
        st.write(f"**{item['name']}**")
        st.write(item["description"])
        st.write("---")
