#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random

import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import spacy
import en_core_web_sm
import pysbd
from lemminflect import getAllInflections
import inflect

MAX_CH = 10000


# class for generating exercises
class ExerciseGenerator:
    def __init__(self):
        self.__file_content = ''
        self.__nlp = spacy.load('en_core_web_sm')
        self.__data_frame = None

    #upload text from the form
    def load_text(self, text): 
         self.__file_content = text
            
    #split text by sentence
    def split_by_sentence(self):
        if self.__nlp(self.__file_content):
            seg = pysbd.Segmenter(language="en", clean=False)
            document = seg.segment(self.__file_content)
            document = [i.strip() for i in document]
            sentences = []
            for index, sentence in enumerate(document):
                sentences.append({'id': index, 'sentence': str(sentence)})
            self.__data_frame = pd.DataFrame(sentences)
            self.__data_frame.set_index('id', inplace=True)
            
        return self.__data_frame    

    #split sentence by word
    def split_sentence_by_word(self):
        if self.__data_frame is not None:
            self.__data_frame['words'] = ''
            self.__data_frame['lemma'] = ''
            self.__data_frame['pos'] = ''
            self.__data_frame['tag'] = ''
            self.__data_frame['dep'] = ''
            for index, row in self.__data_frame.iterrows():
                current_doc = self.__nlp(row['sentence'])
                text = []
                lemma = []
                pos = []
                tag = []
                dep = []
                for i, token in enumerate(current_doc):
                    text.append(token.text) 
                    lemma.append(token.lemma_) 
                    pos.append(token.pos_) 
                    tag.append(token.tag_)  
                    dep.append(token.dep_) 
                self.__data_frame.at[index, 'words'] = text
                self.__data_frame.at[index, 'num_of_words'] = len(text)
                self.__data_frame.at[index, 'lemma'] = lemma
                self.__data_frame.at[index, 'pos'] = pos
                self.__data_frame.at[index, 'tag'] = tag
                self.__data_frame.at[index, 'dep'] = dep
            
            self.__data_frame = self.__data_frame.loc[self.__data_frame['num_of_words'] > 4]
        
        return self.__data_frame
    
    #choose correct verb
    def choose_correct_verb(self, count=5, ):
        if self.__data_frame is not None:
            required_parts_of_speech = ['VERB']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['pos'] for x  in required_parts_of_speech), axis=1)].sample(count, random_state=42)
            
            exercises_data_frame.reset_index(drop=True, inplace=True)
            exercises_data_frame = exercises_data_frame.apply(self.__choose_correct_verb, axis=1)
            exercises_data_frame.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)
        
            return exercises_data_frame
    
    @staticmethod
    def __choose_correct_verb(row):
        verb_index = row['pos'].index('VERB')
        verb = row['words'][verb_index]
        lemma = row['lemma'][verb_index]
        sentence_without_word = ' '.join(row['words'][:verb_index] + ['...'] + row['words'][verb_index + 1:])
        
        options = [verb]
        
        inflections = getAllInflections(lemma)
        
        for key in inflections:
            if key.startswith('V'):
                options.append(inflections[key][0])
                
        options = list(set(options))
        
        row['sentence_without_word'] = sentence_without_word
        row['answer'] = verb
        row['options'] = options
        row['result'] = ['']
        row['total'] = 0
        
        return row
    
    #choose correct adj exercise
    def choose_correct_adj(self, count=5):
        if self.__data_frame is not None:
            required_parts_of_speech = ['ADJ']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['pos'] for x  in required_parts_of_speech), axis=1)].sample(count, random_state=42)
            
            exercises_data_frame.reset_index(drop=True, inplace=True)
            exercises_data_frame = exercises_data_frame.apply(self.__choose_correct_adj, axis=1)
            exercises_data_frame.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)
        
            return exercises_data_frame
    
    @staticmethod
    def __choose_correct_adj(row):
        adj_index = row['pos'].index('ADJ')
        adj = row['words'][adj_index]
        lemma = row['lemma'][adj_index]
        sentence_without_word = ' '.join(row['words'][:adj_index] + ['...'] + row['words'][adj_index + 1:])
        
        options = [adj]
        
        nlp = spacy.load('en_core_web_sm')
        
        inflections = nlp(lemma)
        
        for token in inflections:
            if token.pos_=='ADJ':
                options.append(token._.inflect('JJR'))
                options.append(token._.inflect('JJS'))
                
        options = list(set(options))
        
        row['sentence_without_word'] = sentence_without_word
        row['answer'] = adj
        row['options'] = options
        row['result'] = ['']
        row['total'] = 0
        
        return row
    
    #choose correct det exercise
    def choose_correct_det(self, count=5):
        if self.__data_frame is not None:
            required_parts_of_speech = ['det']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['dep'] for x  in required_parts_of_speech), axis=1)].sample(count, random_state = 42)
            
            exercises_data_frame.reset_index(drop=True, inplace=True)
            exercises_data_frame = exercises_data_frame.apply(self.__choose_correct_det, axis=1)
            exercises_data_frame.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)
        
            return exercises_data_frame
    
    @staticmethod
    def __choose_correct_det(row):
        det_index = row['dep'].index('det')
        det = row['words'][det_index]
        sentence_without_word = ' '.join(row['words'][:det_index] + ['...'] + row['words'][det_index + 1:])
        options = ['a', 'an', 'the']
        options.append(det)
        
        if det.istitle():
            options = [x.title() for x in options]
        
        options = list(set(options))
       
        row['sentence_without_word'] = sentence_without_word
        row['answer'] = det
        row['options'] = options
        row['result'] = ['']
        row['total'] = 0
        
        return row
    
    #word formation exercise
    def make_up_word(self, count=5):
        if self.__data_frame is not None:
            required_parts_of_speech = ['NOUN']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['pos'] for x in required_parts_of_speech), axis=1)].sample(count, random_state=42)

            exercises_data_frame.reset_index(drop=True, inplace=True)
            exercises_data_frame = exercises_data_frame.apply(self.__make_up_word, axis=1)
            exercises_data_frame.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)

            return exercises_data_frame
    
    @staticmethod
    def __make_up_word(row):
        noun_index = row['pos'].index('NOUN')
        word = row['words'][noun_index]
        sentence_without_word = ' '.join(row['words'][:noun_index] + ['...'] + row['words'][noun_index + 1:])
        options = ''.join(random.sample(word,len(word)))

        row['sentence_without_word'] = sentence_without_word
        row['answer'] = word
        row['options'] = options
        row['result'] = ['']
        row['total'] = 0
        
        return row
    
    #plural exercise
    def make_up_plural(self, count=5):
        if self.__data_frame is not None:
            required_parts_of_speech = ['NOUN']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['pos'] for x in required_parts_of_speech), axis=1)].sample(count, random_state=42)
            
            exercises_data_frame.reset_index(drop=True, inplace=True)
            exercises_data_frame = exercises_data_frame.apply(self.__make_up_plural, axis=1)
            exercises_data_frame.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)
        
            return exercises_data_frame
    
    @staticmethod
    def __make_up_plural(row):
        noun_index = row['pos'].index('NOUN')
        word = row['lemma'][noun_index]
        
        if word.istitle():
            word = word.lower()
            
        engine = inflect.engine()
        answer = engine.plural(word)
                
        row['answer'] = answer
        row['options'] = word
        row['result'] = ['']
        row['total'] = 0
        
        return row
    
# streamlit app
def disable():
    if len(text) > 0:
        st.session_state.disabled = True

if "disabled" not in st.session_state:
    st.session_state.disabled = False
    
#main header
st.header('Генератор упражнений по английскому')

#streamlit form 
with st.form("my_form"):
    st.subheader('Вставьте текст на английском языке')
    text = st.text_area(f"Максимальное кол-во символов в тексте не должно превышать {MAX_CH}",
                placeholder="Вставьте текст...",
                max_chars=MAX_CH,
                height=100)
    
    example = st.radio("Или выберите из предложенных ниже 👇",
                       ("Не использовать предложенные тексты",
                        "'Little Red Cap' Jacob_and_Wilhelm_Grimm", 
                        "'Little Red Riding' Hood Charles Perrault"))
    
    if example == "'Little Red Cap' Jacob_and_Wilhelm_Grimm":
        with open(r"https://github.com/EVD-23/pet-projects/blob/main/english_exercises_app/red_hat/Little_Red_Cap_%20Jacob_and_Wilhelm_Grimm.txt") as f:
            text = f.read()
    elif example == "'Little Red Riding' Hood Charles Perrault":
        with open(r"https://github.com/EVD-23/pet-projects/blob/main/english_exercises_app/red_hat/Little_Red_Riding_Hood_Charles_Perrault.txt") as f:
            text = f.read()
    else:
        pass
    
    st.subheader('Выберите тип упражнений')
    option_one = st.checkbox('Выбор правильной формы глагола', value=True)
    option_two = st.checkbox('Выбор правильной формы прилагательного')
    option_three = st.checkbox('Выбор правильного артикля')
    option_four = st.checkbox('Составление слов из букв')
    option_five = st.checkbox('Образуйте множественное число существительного')
    
    st.subheader('Выберите количество упражнений:')
    num_of_exercise = st.slider('Установите диапазон от 1-10', 1, 10, 1)
    
    submit_text = st.form_submit_button('Сгенерировать', on_click=disable, disabled=st.session_state.disabled)
    st.caption('*Нажмите кнопку что-бы сгенерировать упражнения')
    
    if len(text) == 0:
        st.write("*Вставьте английский текст в поле выше или выберите один из предложенных вариантов")

'---' 

#split sentence by word using class ExerciseGenerator
try:
    if text:
        num = num_of_exercise
        eg = ExerciseGenerator()
        eg.load_text(text)
        eg.split_by_sentence()
        eg.split_sentence_by_word()
except:
    pass

#output exercise type one
try:
    if option_one:
        exercise_one = eg.choose_correct_verb(num)
        tasks_one = exercise_one.to_dict('records')
        st.subheader('Выберите правильную форму глаголов:')
        count_one = 0

        for task in tasks_one:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel', 
                                             ['–––'] + option, 
                                             label_visibility="hidden", 
                                             key=count_one)
                count_one += 1

                if task['result'] == '–––':
                    pass
                elif task['result'] == task['answer']:
                    st.success('', icon="✅")
                else:
                    st.error('', icon="😟")

        '---' 
except:
    pass

#output exercise type two
try:
    if option_two:
        exercise_two = eg.choose_correct_adj(num)
        tasks_two = exercise_two.to_dict('records')
        st.subheader('Выберите правильную форму прилагательного:')
        count_two = 11
        for task in tasks_two:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel_two', 
                                             ['–––'] + option, 
                                             label_visibility="hidden", 
                                             key=count_two)
                count_two +=1

                if task['result'] == '–––':
                    pass
                elif task['result'] == task['answer']:
                    st.success('', icon="✅")
                    task['total'] = 1
                else:
                    st.error('', icon="😟")

        '---' 
except:
    pass

#output exercise type three
try:
    if option_three:
        exercise_three = eg.choose_correct_det(num)
        tasks_three = exercise_three.to_dict('records')
        st.subheader('Выберите правильный артикль')
        count_three = 21

        for task in tasks_three:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel_three', 
                                             ['–––'] + option, 
                                             label_visibility="hidden", 
                                             key=count_three)
                count_three +=1

                if task['result'] == '–––':
                    pass
                elif task['result'] == task['answer']:
                    st.success('', icon="✅")
                    task['total'] = 1
                else:
                    st.error('', icon="😟")                    

        '---'
except:
    pass

#output exercise type four
try:
    if option_four: 
        exercise_four = eg.make_up_word(num)
        tasks_four = exercise_four.to_dict('records')
        st.subheader('Составьте слово')
        count_four = 31
        
        for task in tasks_four:
            col1, col2 = st.columns(2)
            with col1:
                option = task['options']
                st.write(f'Буквы  - {option}')
                st.write(str(task['sentence_without_word']))

            with col2:
                answer = task['answer']
                text_input = st.text_input("Введите слово 👇", key=count_four)   
                count_four +=1
                
                if text_input == answer:
                    st.success('', icon="✅")
                    task['total'] = 1
                elif len(text_input) < 1:
                    pass
                else:
                    st.error(f'Это было слово {answer}', icon="😟")
                    
        '---'

except:
    pass

#output exercise type five
try:
    if option_five: 
        exercise_five = eg.make_up_plural(num)
        tasks_five = exercise_five.to_dict('records')
        st.subheader('Образуйте множественное число')
        count_five = 41
        
        for task in tasks_five:
            col1, col2 = st.columns(2)
            with col1:
                option = task['options']
                st.write(f'Слово  - {option}')
                
            with col2:
                answer = task['answer']
                text_input = st.text_input("Введите слово или '-' в поле 👇", key=count_five)   
                count_five +=1
                
                if text_input == answer:
                    st.success('', icon="✅")
                    task['total'] = 1
                elif len(text_input) < 1:
                    pass
                else:
                    st.error(f'Это было слово {answer}', icon="😟")
                    
        '---'
except:
    pass

#reset button
if st.button('Загрузить новый текст'):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
