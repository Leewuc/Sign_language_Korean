import os

from pydub import AudioSegment

from api_chat import complete_words_to_sentences, text_to_speech
from modules import *
from MP_holistic_styled_landmarks import mp_holistic,draw_styled_landmarks
from mediapipe_detection import mediapipe_detection
from keypoints_extraction import extract_keypoints
import keras
from folder_setup import *
from visualization import prob_viz,colors
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from playsound import playsound
import openai
from pathlib import Path
from pydub.playback import play

tts_ENDPOINT = 'https://api.openai.com/v1/audio/speech'
openai.api_key = ""
'''
from langchain.chains import LLMchain
from langchain_community.llms import GPT4ALL
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from getpass import getpass
'''
sequence = []
sentence = []
text_buffer = []
buffer_length = 10
threshold = 0.8
video_path = 'sign_data_fin.mp4'
model = keras.models.load_model('lstm_model_fin3.h5')
def put_korean(img, text, position, font_path='fonts/MaruBuri-Bold.ttf', font_size=20, font_color=(0, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=font_color)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return img
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            image = prob_viz(res, actions, image, colors)
            if len(sentence) > 2:
                api_str = complete_words_to_sentences(sentence[1])
                # image = put_korean(image, api_str, (400, 850), 'fonts/MaruBuri-Bold.ttf', 32, (255, 255, 255))
                print(api_str)
                text_to_speech(api_str)
                #text_to_speech1(api_str,openai.api_key,tts_ENDPOINT)
                sentence = sentence[-2:]



        #api_str = complete_words_to_sentences(sentence)
        image = put_korean(image,' '.join(sentence),(500,650),'fonts/MaruBuri-Bold.ttf',32,(255,255,255))

        cv2.imshow('Sign',image)
        #if len(sentence) > 4:
        #    print(api_str)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()