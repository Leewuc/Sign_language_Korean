import cv2
import numpy as np
import keras

from utils import (
    mp_holistic,
    draw_styled_landmarks,
    mediapipe_detection,
    extract_keypoints,
    prob_viz,
    colors,
)
from folder_setup import actions
from api_chat import complete_words_to_sentences, text_to_speech
from PIL import ImageFont, ImageDraw, Image


def put_korean(img, text, position, font_path='fonts/MaruBuri-Bold.ttf', font_size=20, font_color=(0, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=font_color)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return img


def run(model_path='lstm_model_fin3.h5', threshold=0.8):
    model = keras.models.load_model(model_path)
    sequence, sentence = [], []
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                image = prob_viz(res, actions, image, colors)
                if len(sentence) > 2:
                    api_str = complete_words_to_sentences(sentence[1])
                    print(api_str)
                    text_to_speech(api_str)
                    sentence = sentence[-2:]

            image = put_korean(image, ' '.join(sentence), (500, 650), 'fonts/MaruBuri-Bold.ttf', 32, (255, 255, 255))
            cv2.imshow('Sign', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()

