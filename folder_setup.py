from modules import *


DATA_PATH = os.path.join('Sign_data_finfin')

#actions = np.array(['hello', 'thanks', 'iloveyou','stop'])
actions = np.array(["안녕하세요", "만나다", "너", "나","맞다","아니다","감사합니다","미안합니다","반갑다","괜찮다","좋다","행복","사랑","지금","끝"])

number_sequences = 30 #30
sequence_length = 30 #30

for action in actions: 
    for sequence in range(number_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass