# from langchain_openai import OpenAI
import openai
import requests
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS
# Initialize the OpenAI model
openai.api_key = ""


# Function to complete sentences from words
def complete_words_to_sentences(input_list):
    completed_sentences = []
    for word in input_list:
        # Form the prompt for the model
        prompt = f"Use the word please only use this '{word}' to form a complete only one korean one sentence no english."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart linguist assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50  # 생성할 토큰 수를 설정하세요
        )
        completed_sentence = response['choices'][0]['message']['content'].strip()
        completed_sentences.append(completed_sentence)

    # Join all completed sentences into a single string
    result = ' '.join(completed_sentences)
    return result


# Example usage
input_list = ["안녕하세요", "만나다", "너", "나","맞다","아니다","감사합니다","미안합니다","반갑다","괜찮다","좋다","행복","사랑","지금","끝"]
output_string = complete_words_to_sentences(input_list)

# print(output_string)
def text_to_speech(text):
    tts = gTTS(text=text, lang='ko')
    tts.save('ex1.mp3')
    audio = AudioSegment.from_mp3('ex1.mp3')
    play(audio)