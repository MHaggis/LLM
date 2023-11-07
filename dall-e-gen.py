# This script uses OpenAI's GPT-4-Turbo and DALL-E models to generate creative prompts and images.
# It first asks GPT-4-Turbo to generate a creative prompt for DALL-E image generation.
# Then, it uses the generated prompt to ask DALL-E to generate an image.
# The generated image is then displayed on the Streamlit app.
# Parallel image generation for DALL-E 3

import streamlit as st
import openai
from PIL import Image
import requests
from io import BytesIO
import os
import threading
from queue import Queue


def ask_gpt(prompt, api_key, gpt_model):
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=gpt_model,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": "Generate creative prompts for DALL-E image generation."
                },
                {
                    "role": "user",
                    "content": "I need a prompt for a DALL-E image."
                },
                {
                    "role": "assistant",
                    "content": "Sure, could you give me a theme or a subject?"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )   
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        st.error(f"GPT-4-Turbo API error: {e}")
        return None

def generate_dalle_image(prompt, api_key, dalle_model, n, quality, response_format, size, style):
    images = []
    try:
        openai.api_key = api_key
        response = openai.Image.create(
            model=dalle_model,
            prompt=prompt,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style
        )
        for data in response['data']:
            image_url = data['url']
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            images.append(image)
    except openai.error.OpenAIError as e:
        st.error(f"DALL-E API error: {e}")
    return images

def generate_images_in_parallel(dalle_prompt, api_key, dalle_model, n, quality, response_format, size, style):
    image_queue = Queue()

    def generate_and_add_to_queue():
        try:
            images = generate_dalle_image(dalle_prompt, api_key, dalle_model, 1, quality, response_format, size, style)
            for image in images:
                image_queue.put(image)
        except Exception as e:
            print(f"Error in thread: {e}") 

    threads = []
    for _ in range(n):
        thread = threading.Thread(target=generate_and_add_to_queue)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    images_to_display = []
    while not image_queue.empty():
        images_to_display.append(image_queue.get())

    for image in images_to_display:
        st.image(image, caption="Generated Art")

st.title("AI Art Assistant")
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    api_key = st.text_input("Enter your OpenAI API key", type="password")

dalle_prompt = st.text_input("DALL-E Prompt", key="dallePrompt", max_chars=4000)

dalle_model = st.selectbox(
    'Select a DALL-E model',
    (
        'dall-e-2',
        'dall-e-3'
    ),
    index=1 
)

n = st.number_input("Number of images to generate", min_value=1, max_value=10, value=1, step=1)

quality = st.selectbox(
    'Select image quality',
    (
        'standard',
        'hd'
    ),
    index=0 
)

response_format = st.selectbox(
    'Select response format',
    (
        'url',
        'b64_json'
    ),
    index=0 
)

size = st.selectbox(
    'Select image size',
    (
        '1024x1024',
        '1792x1024',
        '1024x1792'
    ),
    index=0 
)

style = st.selectbox(
    'Select image style',
    (
        'vivid',
        'natural'
    ),
    index=0 
)

gpt_model = st.selectbox(
    'Select a GPT-4 model',
    (
        'gpt-4-1106-preview',
        'gpt-4-vision-preview',
        'gpt-4',
        'gpt-4-32k',
        'gpt-4-32k-0613',
        'gpt-3.5-turbo-1106'
    )
)

st.header("Ask GPT-4 for Prompt Suggestions")
user_input = st.text_input("Ask a question to get prompt suggestions", key="gptPrompt")

if user_input and api_key:
    if st.button("Get Prompt Suggestions", key="gptButton"):
        with st.spinner("Waiting for GPT-4-Turbo response..."):
            gpt_response = ask_gpt(user_input, api_key)
            if gpt_response:
                st.text(gpt_response)

if dalle_prompt and api_key and st.button("Create Art", key="dalleButton"):
    with st.spinner("Waiting for DALL-E to generate the images..."):
        generate_images_in_parallel(dalle_prompt, api_key, dalle_model, n, quality, response_format, size, style)
