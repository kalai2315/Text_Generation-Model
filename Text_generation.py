import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
    return tokenizer, model

def generate_text(prompt, max_length=100):
    tokenizer, model = load_model()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("GPT-2 Text Generator")

prompt = st.text_input("Enter a prompt:", "Once upon a time")

max_length = st.slider("Max length of generated text", min_value=50, max_value=300, value=100)

if st.button("Generate Text"):
    with st.spinner("Generating text..."):
        generated_text = generate_text(prompt, max_length)
        st.subheader("Generated Text")
        st.write(generated_text)
