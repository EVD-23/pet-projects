#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st

import pandas as pd
import numpy as np

import en_core_web_sm

import random

import spacy
from lemminflect import getAllInflections

import pysbd

import gensim.downloader as api


# In[2]:


MAX_CH = 10000


# In[3]:


# класс для генерации упражнений

class ExerciseGenerator:
    def __init__(self):
        self.__file_content = ''
        self.__nlp = spacy.load('en_core_web_sm')
        self.__data_frame = None
    
#     def load_text(self, file_path): # предварительно загруженные файл 
#         with open(file_path, 'r') as file:
#             self.__file_content = file.read()    
     
    def load_text(self, text): # загрузка файда из формы 
         self.__file_content = text
            
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
                    text.append(token.text) # делим на слова
                    lemma.append(token.lemma_) # выделяем первоначальную форму слова
                    pos.append(token.pos_) # выделяем части речи
                    tag.append(token.tag_) # выделяем мелкие части речи 
                    dep.append(token.dep_) # выделяем роль в редложении
                self.__data_frame.at[index, 'words'] = text
                self.__data_frame.at[index, 'num_of_words'] = len(text)
                self.__data_frame.at[index, 'lemma'] = lemma
                self.__data_frame.at[index, 'pos'] = pos
                self.__data_frame.at[index, 'tag'] = tag
                self.__data_frame.at[index, 'dep'] = dep
            
            self.__data_frame = self.__data_frame.loc[self.__data_frame['num_of_words'] > 4]
        
        return self.__data_frame
    
    def choose_correct_verb(self, count=5):
        if self.__data_frame is not None:
            required_parts_of_speech = ['VERB']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['pos'] for x  in required_parts_of_speech), axis=1)].sample(count, random_state = 42)
            
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
        row['description'] = 'Выберите слово'
        row['exercise_type'] = 'select_word'
        row['result'] = ['']
        row['total'] = 0
        
        return row
    
    def choose_correct_adj(self, count=5):
        if self.__data_frame is not None:
            required_parts_of_speech = ['ADJ']
            exercises_data_frame = self.__data_frame.loc[self.__data_frame.apply(
                lambda p: all(x in p['pos'] for x  in required_parts_of_speech), axis=1)].sample(count, random_state = 42)
            
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
        row['description'] = 'Выберите слово'
        row['exercise_type'] = 'select_word'
        row['result'] = inflections
        row['total'] = 0
        
        return row
    


# In[4]:


# streamlit приложение

st.header('Генератор упражнений по английскому')

with st.form("my_form"):
    text = st.text_area(f"Максимальное кол-во символов в тексте не должно превышать {MAX_CH}",
                placeholder="Вставьте текст...",
                max_chars=MAX_CH,
                height=100)
    submit_text = st.form_submit_button('Сгенерировать')
    st.caption('*Нажмите кнопку что-бы сгенерировать упражнения')
    
try:
    if text:
        with st.form("my_form_checkbox"):
            st.subheader('Выберите тип упражнений')
            option_one = st.checkbox('Выбор правильной формы глагола')
            option_two = st.checkbox('Выбор правильной формы прилагательного')
            option_three = st.checkbox('Создание неправильных предложений')
            option_four = st.checkbox('Задача с пропусками')
            st.subheader('Выберите количество упражнений:')
            num_of_exercise = st.slider('Установите диапазон от 1-10', 1, 10, 1)
            submit_type = st.form_submit_button('Далее')
            if submit_type:
                num = num_of_exercise
except:
    pass

try:
    eg = ExerciseGenerator()
    eg.load_text(text)
    eg.split_by_sentence()
    eg.split_sentence_by_word()
except:
    pass
    
try:
    if option_one:
        
        '---'
        
        exercise_one = eg.choose_correct_verb(num)
        tasks_one = exercise_one.to_dict('records')
        st.subheader('Выберите правильную форму глаголов:')
        for task in tasks_one:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel', 
                                                 ['–––'] + option, 
                                                 label_visibility="hidden")
                if task['result'] == '–––':
                    pass
                elif task['result'] == task['answer']:
                    st.success('', icon="✅")
                    task['total'] = 1
                else:
                    st.error('', icon="😟")
        
                    
    if option_two:
        
        '---'
        
        exercise_two = eg.choose_correct_adj(num)
        tasks_two = exercise_two.to_dict('records')
        st.subheader('Выберите правильную форму прилагательного:')
        for task in tasks_two:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel_two', 
                                                 ['–––'] + option, 
                                                 label_visibility="hidden")
                if task['result'] == '–––':
                    pass
                elif task['result'] == task['answer']:
                    st.success('', icon="✅")
                    task['total'] = 1
                else:
                    st.error('', icon="😟")
        

    if option_three:
        
        '---'
        
        exercise_three = eg.choose_correct_verb(num)
        tasks_three = exercise_three.to_dict('records')
        st.subheader('Выберите правильные предложения')

        for task in tasks_three:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel_three', 
                                                 ['–––'] + option, 
                                                 label_visibility="hidden")
                if task['result'] == '–––':
                    pass
                elif task['result'] == task['answer']:
                    st.success('', icon="✅")
                    task['total'] = 1
                else:
                    st.error('', icon="😟")
        
    
    if option_four: 
        
        '---'
        
        exercise_four = eg.choose_correct_verb(num)
        tasks_four = exercise_four.to_dict('records')
        st.subheader('Заполните пропуски')
        for task in tasks_four:
            col1, col2 = st.columns(2)
            with col1:
                st.write('')
                st.write(str(task['sentence_without_word']))

            with col2:
                option = task['options']
                task['result'] = st.selectbox('nolabel_four', 
                                                 ['–––'] + option, 
                                                 label_visibility="hidden")
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


# In[ ]:




