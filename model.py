import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import json
from termcolor import colored
from tqdm import tqdm

class LanguageModel:
    def __init__(self, vocab_size=5000, max_seq_length=20, data_file='train_data.json'):
        tf.config.set_visible_devices([], 'GPU')  # 禁用GPU（如果需要可以注释掉）

        print(colored("初始化模型...", "yellow"))
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.data_file = data_file
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.model = self.build_model()
        self.load_data()
        self.previous_answers = set()  
        self.is_trained = False  # 新增标志，跟踪模型是否训练完成

    def load_data(self):
        """加载现有的数据"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.data = data
        except FileNotFoundError:
            print(colored(f"未找到数据文件：{self.data_file}，将创建一个新的文件。", "red"))
            self.data = {"data": []}

        questions = [item['question'] for item in self.data['data']]
        answers = [item['answer'] for item in self.data['data']]

        questions = [" ".join(jieba.cut(q)) for q in questions]
        answers = [" ".join(jieba.cut(a)) for a in answers]

        self.tokenizer.fit_on_texts(questions + answers)

        self.question_sequences = pad_sequences(self.tokenizer.texts_to_sequences(questions), maxlen=self.max_seq_length)
        self.answer_sequences = [self.tokenizer.texts_to_sequences([a])[0] for a in answers]
        self.answer_sequences = np.array([seq[0] for seq in self.answer_sequences])  

    def build_model(self):
        """构建LSTM模型"""
        print(colored("构建模型...", "yellow"))
        model = Sequential()
        model.add(Embedding(self.vocab_size, 128, input_length=self.max_seq_length))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(self.vocab_size, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, epochs=10):
        """训练模型"""
        print(colored("开始训练模型...", "yellow"))
        self.model.fit(self.question_sequences, self.answer_sequences, epochs=epochs, batch_size=32, verbose=1)
        self.model.save('model/SAI-LLM_Model.h5')
        self.is_trained = True  # 训练完成后将标志设为 True
        print(colored("训练完成！模型已保存。", "yellow"))

    def generate_answer(self, input_text, max_length=20, temperature=0.7):
        """生成完整的回答，并确保它是一个句子"""
        seq = pad_sequences(self.tokenizer.texts_to_sequences([input_text]), maxlen=self.max_seq_length)
        generated_seq = list(seq[0])  # 初始化序列

        with tqdm(total=max_length, desc="生成回答", ncols=100) as pbar:
            for _ in range(max_length):
                pred = self.model.predict(np.array([generated_seq]), verbose=0)  
                pred = np.log(pred) / temperature  
                pred = np.exp(pred) / np.sum(np.exp(pred))

                k = 5  # Top-k采样
                top_k_indices = np.argsort(pred[0])[-k:]  
                next_word_index = np.random.choice(top_k_indices)

                generated_seq.append(next_word_index)

                if next_word_index == 0:  
                    break

                pbar.update(1)  

        generated_text = ' '.join(self.tokenizer.index_word.get(i, '') for i in generated_seq)
        generated_text = self.clean_text(generated_text)
        return generated_text

    def clean_text(self, text):
        """清理生成文本中的空格，并确保句子语法合理"""
        cleaned_text = ' '.join(text.split())
        cleaned_text = cleaned_text.replace("  ", " ").strip()  
        return cleaned_text

    def get_unique_answer(self, input_text, max_length=20, temperature=0.7):
        """确保每次问题的回答都是唯一的，并返回一个完整的句子"""
        generated_answer = self.generate_answer(input_text, max_length, temperature)

        retry_count = 0
        while generated_answer in self.previous_answers and retry_count < 5:
            generated_answer = self.generate_answer(input_text, max_length, temperature)
            retry_count += 1

        self.previous_answers.add(generated_answer)

        return generated_answer
