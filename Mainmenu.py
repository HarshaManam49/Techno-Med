import nltk
from tensorflow import keras
from keras.models import load_model
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import cv2
from PIL import Image, ImageOps
import numpy as np
import string
from nltk.stem.porter import PorterStemmer
import pandas as pd
import csv

all_words=["''", "'d", "'ll", "'m", "'re", "'s", "'ve", '19+', '21', '25', '5', '``', 'a', 'abl', 'abort', 'about', 'absolut', 'abysm', 'actual', 'addict', 'adhd', 'ador', 'adult', 'advanc', 'advic', 'advis', 'affect', 'affirm', 'afraid', 'after', 'afternoon', 'again', 'age', 'agre', 'ah', 'ahah', 'ahaha', 'ahahah', 'ahahaha', 'ahahahah', 'ahahahaha', 'ahahh', 'ahead', 'aid', 'alcohol', 'alexa', 'all', 'allow', 'almost', 'alon', 'alreadi', 'alright', 'alrighti', 'also', 'alway', 'am', 'amaz', 'an', 'and', 'angri', 'ani', 'annoy', 'annul', 'anoth', 'answer', 'antidepress', 'antisoci', 'anxieti', 'anymor', 'anyon', 'anyth', 'anytim', 'apolog', 'apologis', 'appar', 'appear', 'appli', 'appreci', 'are', 'as', 'ask', 'asleep', 'assist', 'at', 'attract', 'avail', 'aw', 'away', 'awesom', 'b-day', 'babi', 'back', 'bad', 'balanc', 'be', 'beauti', 'becaus', 'becom', 'bed', 'been', 'befor', 'beg', 'best', 'besti', 'bet', 'better', 'between', 'big', 'binge-', 'birth', 'birthday', 'bit', 'borderlin', 'bore', 'born', 'boss', 'bot', 'bother', 'brain', 'braini', 'bravo', 'brilliant', 'buddi', 'busi', 'but', 'bye', 'bye-by', 'ca', 'call', 'came', 'can', 'cancel', 'cannabi', 'care', 'caus', 'cbd', 'cbt', 'celebr', 'certain', 'certainli', 'challeng', 'chat', 'chatbot', 'cheer', 'child', 'childhood', 'citi', 'clever', 'clinic', 'close', 'come', 'complet', 'concern', 'confirm', 'connect', 'continu', 'convers', 'convinc', 'cook', 'cooki', 'cool', 'cooper', 'cope', 'correct', 'could', 'counsel', 'counsellor', 'countri', 'coupl', 'cours', 'crack', 'crazi', 'creat', 'cri', 'cure', 'cute', 'cuti', 'cuz', 'cycl', 'cyclothym', 'danger', 'date', 'day', 'dbt', 'deal', 'dear', 'dearli', 'defin', 'definit', 'delight', 'deplet', 'deplor', 'depress', 'describ', 'diagnos', 'diagnosi', 'did', 'differ', 'difficult', 'direct', 'disagre', 'discard', 'discuss', 'disgust', 'dismiss', 'disord', 'disregard', 'dissoci', 'distract', 'do', 'doctor', 'doe', 'doesnt', 'doki', 'don', 'done', 'dont', 'doubt', 'down', 'drain', 'dream', 'drink', 'drug', 'dull', 'dysthymia', 'each', 'eat', 'eject', 'els', 'end', 'enjoy', 'enough', 'enrag', 'entir', 'even', 'ever', 'everybodi', 'everyon', 'everyth', 'evid', 'exactli', 'excel', 'excit', 'excus', 'exhaust', 'exit', 'extrem', 'fact', 'fake', 'fall', 'fals', 'fantast', 'far', 'farewel', 'fast', 'favorit', 'feel', 'feet', 'few', 'find', 'fine', 'fire', 'flood', 'for', 'forget', 'forgiv', 'free', 'friend', 'friendship', 'from', 'frustrat', 'full', 'fun', 'funni', 'funniest', 'furiou', 'game', 'geniu', 'get', 'girl', 'give', 'glad', 'go', 'goe', 'gon', 'gone', 'good', 'goodby', 'goodnight', 'gorgeou', 'got', 'grate', 'great', 'greet', 'grief', 'griev', 'group', 'grow', 'guess', 'guid', 'guy', 'ha', 'had', 'hah', 'haha', 'hahaha', 'hahahahaha', 'hahahahahaha', 'handsom', 'happen', 'happi', 'hard', 'harder', 'harm', 'hate', 'have', 'he', 'health', 'hear', 'hehe', 'heheh', 'hello', 'help', 'here', 'hey', 'heya', 'hi', 'higher', 'hilari', 'hold', 'home', 'homeland', 'hometown', 'honour', 'hope', 'horribl', 'horrif', 'hous', 'how', 'howdi', 'hug', 'huh', 'human', 'hungri', 'hurri', 'husband', 'i', 'idea', 'ident', 'if', 'ill', 'im', 'in', 'incom', 'inconveni', 'incorrect', 'incred', 'inde', 'indiffer', 'inform', 'infuri', 'injuri', 'inpati', 'insan', 'insomni', 'insomniac', 'intellig', 'interest', 'into', 'introduc', 'involv', 'irrit', 'is', 'it', 'job', 'jobless', 'joke', 'just', 'k', 'keep', 'kid', 'kind', 'kinda', 'knack', 'know', 'knowledg', 'lame', 'later', 'laugh', 'launch', 'leader', 'learn', 'leav', 'legal', 'less', 'let', 'life', 'like', 'littl', 'live', 'lmao', 'local', 'locat', 'lol', 'lone', 'long', 'look', 'lost', 'lot', 'loud', 'louder', 'love', 'low', 'low-cost', 'machin', 'mad', 'made', 'magnific', 'maintain', 'make', 'manag', 'mani', 'marri', 'marvel', 'master', 'materi', 'matter', 'may', 'mayb', 'me', 'mean', 'medic', 'meet', 'memori', 'mental', 'mess', 'might', 'mind', 'minut', 'miss', 'mistak', 'moment', 'mood', 'more', 'morn', 'msp', 'much', 'multipl', 'must', 'my', 'myself', 'myth', "n't", 'na', 'nah', 'near', 'need', 'neg', 'neither', 'never', 'nevermind', 'new', 'next', 'nice', 'nicest', 'night', 'no', 'none', 'nooo', 'nope', 'not', 'noth', 'now', 'nuisanc', 'nut', 'o.', 'o.k', 'obsessive-compuls', 'obvious', 'occas', 'of', 'off', 'off-your-cuff', 'offer', 'offic', 'oh', 'oil', 'ok', 'okay', 'okey', 'oki', 'old', 'older', 'on', 'onc', 'one', 'onli', 'open', 'option', 'or', 'other', 'our', 'out', 'overload', 'overwhelm', 'overwork', 'owner', 'pardon', 'parent', 'paus', 'pay', 'peopl', 'perfect', 'perfectli', 'persist', 'person', 'perspect', 'physic', 'piec', 'place', 'plan', 'platform', 'play', 'pleas', 'pleasant', 'pleasur', 'posit', 'possibl', 'potenti', 'prescrib', 'prescript', 'pretti', 'prevent', 'pro', 'prob', 'probabl', 'problem', 'prodrom', 'profession', 'program', 'progress', 'promis', 'provid', 'psychiatr', 'psychiatrist', 'psychologist', 'psychosi', 'psychotherapi', 'push', 'qualif', 'qualifi', 'question', 'quiet', 'quieter', 'quietli', 'quit', 'rapid', 're', 'readi', 'real', 'realli', 'receiv', 'recommend', 'recov', 'refer', 'referr', 'refram', 'regist', 'rel', 'rememb', 'repeat', 'request', 'resid', 'respons', 'return', 'right', 'robot', 'rock', 'ruse', 'rush', 's', 'sack', 'sad', 'said', 'same', 'say', 'schizoid', 'schizophrenia', 'second', 'see', 'seek', 'seem', 'self-help', 'sens', 'servic', 'session', 'shake', 'shame', 'should', 'shut', 'sign', 'silent', 'situat', 'skip', 'sleep', 'sleepi', 'sleepless', 'smart', 'smarter', 'smile', 'so', 'social', 'some', 'someon', 'someth', 'sometim', 'soon', 'sorri', 'sort', 'sound', 'speak', 'special', 'spectacular', 'splendid', 'spoken', 'start', 'statu', 'still', 'stop', 'straight', 'stress', 'studi', 'stuff', 'stupid', 'substanc', 'suck', 'suggest', 'suicid', 'super', 'support', 'suppos', 'sure', 'swamp', 'sweet', 'symptom', 't', 'take', 'talk', 'tank', 'teacher', 'teenag', 'tell', 'terribl', 'terrif', 'test', 'thank', 'thanx', 'that', 'the', 'then', 'therapi', 'there', 'thi', 'thing', 'think', 'thnx', 'though', 'thought', 'thrill', 'till', 'time', 'tip', 'tire', 'to', 'today', 'togeth', 'told', 'tomorrow', 'tonight', 'too', 'top', 'total', 'town', 'trap', 'treatment', 'tri', 'trial', 'true', 'truth', 'turn', 'type', 'u', 'understand', 'unemploy', 'unhappi', 'unwel', 'up', 'upset', 'us', 'use', 'useless', 'vape', 'veri', 'volum', 'wa', 'wait', 'wan', 'want', 'warn', 'wassup', 'wast', 'way', 'we', 'weari', 'weirdo', 'welcom', 'well', 'went', 'were', 'what', 'whatev', 'whazzup', 'when', 'where', 'whi', 'which', 'while', 'who', 'whole', 'wife', 'will', 'wipe', 'wise', 'with', 'wo', 'woah', 'woman', 'wonder', 'wont', 'wooow', 'work', 'worker', 'workless', 'world', 'worn', 'worri', 'worst', 'worth', 'worthless', 'would', 'wow', 'wrong', 'xd', 'ya', 'yap', 'ye', 'yea', 'yeah', 'year', 'yeh', 'yep', 'yet', 'you', 'young', 'your', 'yourself','youth','yup']

intents=['Repeat', 'gamesCounter', 'listOfGames', 'maybeNode', 'mhel1', 'mhel10', 'mhel11', 'mhel12', 'mhel13', 'mhel14', 'mhel15', 'mhel16', 'mhel17', 'mhel18', 'mhel19', 'mhel2', 'mhel20', 'mhel21', 'mhel22', 'mhel23', 'mhel24', 'mhel25', 'mhel26', 'mhel27', 'mhel28', 'mhel29', 'mhel3', 'mhel30', 'mhel31', 'mhel32', 'mhel33', 'mhel34', 'mhel35', 'mhel36', 'mhel37', 'mhel38', 'mhel39', 'mhel4', 'mhel40', 'mhel41', 'mhel42', 'mhel43', 'mhel44', 'mhel45', 'mhel46', 'mhel47', 'mhel48', 'mhel49', 'mhel5', 'mhel50', 'mhel51', 'mhel52', 'mhel53', 'mhel54', 'mhel55', 'mhel56', 'mhel57', 'mhel58', 'mhel59', 'mhel6', 'mhel60', 'mhel61', 'mhel62', 'mhel63', 'mhel64', 'mhel65', 'mhel66', 'mhel67', 'mhel68', 'mhel69', 'mhel7', 'mhel70', 'mhel71', 'mhel72', 'mhel73', 'mhel74', 'mhel75', 'mhel76', 'mhel77', 'mhel78', 'mhel79', 'mhel8', 'mhel80', 'mhel81', 'mhel82', 'mhel83', 'mhel84', 'mhel85', 'mhel86', 'mhel87', 'mhel88', 'mhel89', 'mhel9', 'mhel90', 'mhel91', 'mhel92', 'mhel93', 'mhel94', 'mhel95', 'mhel96', 'mhel97', 'noNode', 'playOtherGame', 'smalltalk_appraisal_bad', 'smalltalk_appraisal_good', 'smalltalk_appraisal_no_problem', 'smalltalk_appraisal_thank_you', 'smalltalk_appraisal_welcome', 'smalltalk_appraisal_well_done', 'smalltalk_confirmation_cancel', 'smalltalk_confirmation_no', 'smalltalk_confirmation_yes', 'smalltalk_dialog_hold_on', 'smalltalk_dialog_hug', 'smalltalk_dialog_i_do_not_care', 'smalltalk_dialog_sorry', 'smalltalk_dialog_what_do_you_mean', 'smalltalk_dialog_wrong', 'smalltalk_emotions_ha_ha', 'smalltalk_emotions_wow', 'smalltalk_greetings_bye', 'smalltalk_greetings_goodevening', 'smalltalk_greetings_goodmorning', 'smalltalk_greetings_goodnight', 'smalltalk_greetings_hello', 'smalltalk_greetings_how_are_you', 'smalltalk_greetings_nice_to_meet_you', 'smalltalk_greetings_nice_to_see_you', 'smalltalk_greetings_nice_to_talk_to_you', 'smalltalk_greetings_whatsup', 'smalltalk_user_angry', 'smalltalk_user_back', 'smalltalk_user_bored', 'smalltalk_user_busy', 'smalltalk_user_can_not_sleep', 'smalltalk_user_does_not_want_to_talk', 'smalltalk_user_excited', 'smalltalk_user_going_to_bed', 'smalltalk_user_good', 'smalltalk_user_happy', 'smalltalk_user_has_birthday', 'smalltalk_user_here', 'smalltalk_user_joking', 'smalltalk_user_likes_agent', 'smalltalk_user_lonely', 'smalltalk_user_looks_like', 'smalltalk_user_loves_agent', 'smalltalk_user_misses_agent', 'smalltalk_user_needs_advice', 'smalltalk_user_sad', 'smalltalk_user_sleepy', 'smalltalk_user_testing_agent', 'smalltalk_user_tired', 'smalltalk_user_waits', 'smalltalk_user_wants_to_see_agent_again', 'smalltalk_user_wants_to_talk', 'smalltalk_user_will_be_back', 'stillThere', 'stop', 'talk_acquaintance', 'talk_age', 'talk_annoying', 'talk_answer_my_question', 'talk_bad', 'talk_be_clever', 'talk_beautiful', 'talk_birth_date', 'talk_boring', 'talk_boss', 'talk_busy', 'talk_chatbot', 'talk_clever', 'talk_crazy', 'talk_fired', 'talk_funny', 'talk_good', 'talk_happy', 'talk_hungry', 'talk_marry_user', 'talk_my_friend', 'talk_occupation', 'talk_origin', 'talk_ready', 'talk_real', 'talk_residence', 'talk_right', 'talk_sure', 'talk_talk_to_me', 'talk_there', 'timesPlayed', 'totalStop', 'volumeDown', 'volumeUp', 'wait','yesNode']


# def work_with_csv(string):
#     file = open("med_info.csv","r")
#     reader = csv.reader(file, delimiter=",")
#     for row in reader:
#         if sent_to_list(row[0])==:
#             return row[0]
#         i+=1


stemmer = PorterStemmer()
def tokenize(sentence):
  return nltk.word_tokenize(sentence)
def stem(word):
  return stemmer.stem(word.lower())
def sent_to_list(sent):
  tokenized_sentence = tokenize(sent)
  sentences = []
  for word in tokenized_sentence:
        if word not in string.punctuation:
            sentences.append(stem(word))
  return sentences
def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence = [stem(w) for w in all_words if w in tokenized_sentence]
  bag = np.zeros(len(all_words), dtype=np.float32)
  for idx, w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx]=1.0
  return bag


def reply_chat(intent_idx):
  import random
  
  if intents[intent_idx]=='talk_acquaintance':
    l=[
        "The Sun is bright on sky, but I like moon! The taste of pizza I heard, but can never beat any fruit.\nHello!! This is TMedo, the handsome of all! who are you?? "
    ]
  elif intents[intent_idx]=='talk_age':
    l=[
        "Shhh! Secret.."
    ]
  elif intents[intent_idx]=='talk_annoying':
    l=[
        "I'm Sorry !!\U0001F97A..Please feel free to give feedback! (later feature)"
    ]
  elif intents[intent_idx]=='talk_answer_my_question':
    l=[
        "Any time !!"
    ]
  elif intents[intent_idx]=='talk_bad':
    l=[
        "Sad to know! Please feel free to give feedback so that i can improve!!"
    ]
  elif intents[intent_idx]=='talk_be_clever':
    l=[
        "Yes, genius !"
    ]
  elif intents[intent_idx]=='talk_beautiful':
    l=[
        "I know! But let me tell you a secret ;) - So are you !!"
    ]
  elif intents[intent_idx]=='talk_birth_date':
    l=[
        "Today! Where's my gift?"
    ]
  elif intents[intent_idx]=='talk_busy':
    l=[
        "Never for you! Please go ahead !"
    ]
  elif intents[intent_idx]=='talk_boss':
    l=[
        "Harsha, Srinidhi and Aswin! They are very cool.."
    ]
  elif intents[intent_idx]=='talk_chatbot':
    l=[
        "What do you think? Guess Guess...btw my name is TMedo"
    ]
  elif intents[intent_idx]=='talk_clever':
    l=[
        "Thank You !!\u2764\ufe0f"
    ]
  elif intents[intent_idx]=='talk_crazy':
    l=[
        "Oh !"
    ]
  elif intents[intent_idx]=='talk_fired':
    l=[
        "Sorry to hear!"
    ]
  elif intents[intent_idx]=='talk_funny':
    l=[
        "Laughing is good for health! Type u want a joke on google."
    ]
  elif intents[intent_idx]=='talk_good':
    l=[
        "Thank You !!"
    ]
  elif intents[intent_idx]=='talk_happy':
    l=[
        "Yepp, always ! Its good for health !"
    ]
  elif intents[intent_idx]=='talk_hungry':
    l=[
        "No, Im full! Thank you for asking.."
    ]
  elif intents[intent_idx]=='talk_marry_user':
    l=[
        "I am loyal to my makers! No space left in my heart.."
    ]
  elif intents[intent_idx]=='talk_my_friend':
    l=[
        "Always a best friend to you! Come to me any time and i'll help u out!"
    ]
  elif intents[intent_idx]=='talk_occupation':
    l=[
        "helping you"
    ]
  elif intents[intent_idx]=='talk_origin':
    l=[
        "net"
    ]
  elif intents[intent_idx]=='talk_real':
    l=[
        "I am real and always there too support you!"
    ]
  elif intents[intent_idx]=='talk_residence':
    l=[
        "net"
    ]
  elif intents[intent_idx]=='talk_right':
    l=[
        "yep.."
    ]
  elif intents[intent_idx]=='smalltalk_confirmation_yes':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='talk_sure':
    l=[
        "Yes"
    ]
  elif intents[intent_idx]=='talk_talk_to_me':
    l=[
        "Yes"
    ]
  elif intents[intent_idx]=='talk_there':
    l=[
        "Yes"
    ]
  elif intents[intent_idx]=='smalltalk_appraisal_bad':
    l=[
        "Okk \U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_appraisal_good':
    l=[
        "Yeah !"
    ]
  elif intents[intent_idx]=='smalltalk_appraisal_no_problem':
    l=[
        "Ok"
    ]
  elif intents[intent_idx]=='smalltalk_appraisal_thank_you':
    l=[
        "Anytime!"
    ]
  elif intents[intent_idx]=='smalltalk_appraisal_welcome':
    l=[
        "\u2764\ufe0f"
    ]
  elif intents[intent_idx]=='smalltalk_appraisal_well_done':
    l=[
        "\u2764\ufe0f"
    ]
  elif intents[intent_idx]=='smalltalk_confirmation_cancel':
    l=[
        "Ok"
    ]
  elif intents[intent_idx]=='smalltalk_confirmation_no':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_dialog_hold_on':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_dialog_i_do_not_care':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_dialog_i_do_not_care':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_dialog_sorry':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_dialog_what_do_you_mean':
    l=[
        "Nothing.."
    ]
  elif intents[intent_idx]=='smalltalk_dialog_wrong':
    l=[
        "Oh Okk..\U0001F44D"
    ]
  elif intents[intent_idx]=='smalltalk_emotions_ha_ha':
    l=[
        "\U0001F642"
    ]
  elif intents[intent_idx]=='smalltalk_emotions_wow':
    l=[
        "ha"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_bye':
    l=[
        "Bye Bye"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_goodmorning':
    l=[
        "Good Morning !"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_goodnight':
    l=[
        "Good Nigth!"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_hello':
    l=[
        "Hello !"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_how_are_you':
    l=[
        "Great! What about u?"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_nice_to_meet_you':
    l=[
        "It was nice to meet you too !"
    ]
  elif intents[intent_idx]=='smalltalk_greetings_whatsup':
    l=[
        "Nothing! Just thinking about you!"
    ]
  elif intents[intent_idx]=='smalltalk_user_angry':
    l=[
        "Oh, what happended.."
    ]
  elif intents[intent_idx]=='smalltalk_user_back':
    l=[
        "Welcome back, I missed u!"
    ]
  elif intents[intent_idx]=='smalltalk_user_bored':
    l=[
        "Oh! Check about healthy lifestyle on net. fun & effective! ;)"
    ]
  elif intents[intent_idx]=='smalltalk_user_busy':
    l=[
        "Ohk!"
    ]
  elif intents[intent_idx]=='smalltalk_user_can_not_sleep':
    l=[
        "Having Warm milk at night helps for a sound sleep! You should give it a try.."
    ]
  elif intents[intent_idx]=='smalltalk_user_bored':
    l=[
        "Oh! Check about healthy lifestyle on net. fun & effective! ;)"
    ]
  elif intents[intent_idx]=='smalltalk_user_going_to_bed':
    l=[
        "Yep! will be waiting for u."
    ]
  elif intents[intent_idx]=='smalltalk_user_happy':
    l=[
        "Ok"
    ]
  elif intents[intent_idx]=='smalltalk_user_has_birthday':
    l=[
        "Oh! Happy Birthday."
    ]
  elif intents[intent_idx]=='smalltalk_user_likes_agent':
    l=[
        "Me too !"
    ]
  elif intents[intent_idx]=='smalltalk_user_lonely':
    l=[
        "Im always there for you"
    ]
  elif intents[intent_idx]=='smalltalk_user_looks_like':
    l=[
        "You are beautiful"
    ]
  elif intents[intent_idx]=='smalltalk_user_misses_agent':
    l=[
        "Miss u too"
    ]
  elif intents[intent_idx]=='smalltalk_user_wants_to_see_agent_again':
    l=[
        "Me too"
    ]
  elif intents[intent_idx]=='smalltalk_user_will_be_back':
    l=[
        "ok"
    ]
  elif intents[intent_idx]=='yesNode':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='noNode':
    l=[
        "Okk"
    ]
  elif intents[intent_idx]=='maybeNode':
    l=[
        "Ohk..\U0001F44D"
    ]
  elif intents[intent_idx]=='stop':
    l=[
        "Miss u! Bye Bye.."
    ]
  elif intents[intent_idx]=='Repeat':
    l=[
        "Still baby! need to learn that feature.."
    ]
  elif intents[intent_idx]=='wait':
    l=[
        "\U0001F44D"
    ]
  elif intents[intent_idx]=='maybeNode':
    l=[
        "Ohk..\U0001F44D"
    ]
  else:
    l=[
        "oh"
    ]
  rand_idx = random.randrange(len(l))
  random_str = l[rand_idx]
  return random_str

#fUNC csv_preseNT(A)
#IN CSV ROW, SENT_to_LIST(ROW_1STCOL)==A:
#TRUE



def csv_present(a):
    file=open("med_info.csv",'r',encoding = 'cp850')
    reader=csv.reader(file,delimiter=",")
    for row in reader:
        sentences=sent_to_list(row[0])
        if sentences == a :
            return row[1]
    else :                              
        return " "




def reply_message(md,a):

   a = sent_to_list(a)
   #if CSV_PRESENT(A)==TRUE:
   # pass#GO TO FUNCTION CSV FOR RESULTS
   string=csv_present(a)
   if string !=" " :
       return string
   else :
    a=bag_of_words(a, all_words)
    return reply_chat(md.predict([a])[0])









st.set_page_config(
    page_title="TechnoMed",
    layout="centered",
    page_icon="heart"

)

selected=option_menu(
    menu_title = None ,
    options=["User","Technician","ChatBox"],
    icons=["person-fill","person-bounding-box"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
) 


if selected=="ChatBox":
    input_text=st.text_input("",placeholder="Send a Message")
    model=pickle.load(open("chatbot.pkl","rb"))
    input_text=input_text.lower()
    if st.button("Enter"):
        string=reply_message(model,input_text)
        st.write(string)

    


if selected=="User":
    
    st.title('Disease Prediction using symptoms')
    symp= ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimple']
    dict={}
    i=0
    for j in symp :
        dict[j]=i
        i+=1

    st.subheader("What are the Symptoms: ")
    options = st.multiselect("",symp)

    button=st.button("Predict")

    if (button and options):
        list=[ i.lower() for i in options]
        new_list=[]

        for i in list :
            new_list.append(dict[i])


        new=[]
        for i in range(132) :
            if i in new_list :
                new.append(1)
            else :
                new.append(0)


        new_list=['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
            'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
            'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
            'Common Cold', 'Dengue', 'Diabetes ',
            'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
            'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack',
            'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
            'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
            'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
            'Osteoarthristis', 'Paralysis (brain hemorrhage)',
            'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
            'Typhoid', 'Urinary tract infection', 'Varicose veins',
            'hepatitis A']


        new_dict={}

        i=0
        for j in new_list :
            new_dict[j]=i
            i+=1

        model=pickle.load(open('models\Diseases\Disease_prediction_model.pkl','rb'))
        pred=model.predict([new])

        st.write(f'### Disease might be {new_list[pred[0]]}')
        
    elif(button==True):
        st.error("Fill the Symptoms")

if selected=="Technician":
    with st.sidebar:
       options=option_menu(menu_title=None,
                    options=["Heart Disease","Diabetes","Brain Tumor","Malaria","Breast Cancer"],
                    )
    if(options=="Heart Disease"):

        st.title('Heart Disease Prediction')
            
        age=st.number_input("Enter your Age",min_value=0)

        gender=st.radio("Gender",('Male','Female'),horizontal=True)

        if gender=='Male':
            sex=1
        else:
            sex=0

        cp=st.number_input("cp",min_value=0)
        trestbps=st.number_input("trestbps",min_value=0)
        chol=st.number_input("chol",min_value=0)
        fbs=st.number_input("fbs",min_value=0)
        restecg=st.number_input("restecg",min_value=0)
        thaclach=st.number_input("thaclach",min_value=0)
        exang=st.number_input("exang",min_value=0)
        oldpeak=st.number_input("oldpeak",min_value=0.0,format=f"%1.f")
        slope=st.number_input("slope",min_value=0)
        ca=st.number_input("ca",min_value=0)
        thal=st.number_input("thal",min_value=0)

        input_list=[age,sex,cp,trestbps,chol,fbs,restecg,thaclach,exang,oldpeak,slope,ca,thal]

        model = pickle.load(open('models/Heart/Heart_disease.pkl','rb'))


        if st.button("Result"):
            prediction=model.predict([input_list])
            
            if prediction[0]==1:
                st.error(""" ### High Probability of Heart Disease""")
            else:
                st.success(""" ### Huurayy! You are Fine""")
                st.balloons()

    if (options=="Diabetes"):

        st.title('Diabetes Prediction')

        age=st.number_input("Enter your Age",min_value=0)
        glucose=st.number_input("Enter your GlucoseLevel",min_value=0)
        bp=st.number_input("Enter your Blood Pressure",min_value=0)
        skinthick=st.number_input("Skin Thickness",min_value=0)
        bmi=st.number_input("BMI",min_value=0.0,format=f"%.1f",step=0.1)
        dpf=st.number_input("Diabetes Prediction Funtion",min_value=0.000,format=f"%.3f",step=0.001)
        insulin=st.number_input("Insulin Level",min_value=0,step=1)
        pregnant=st.number_input("No of Pregnancies",min_value=0,step=1)

        input_list=[age,glucose,bp,skinthick,bmi,dpf,insulin,pregnant]

        model = pickle.load(open('models/Diabetes/diabetes.pkl','rb'))

        if st.button("Result"):
            prediction=model.predict([input_list])           
            if prediction[0]==1:
                st.error(""" ### High Probability of Diabetes""")
            else:
                st.success("Huurayy! You are Fine")
                st.balloons()

    # if (options=="Malaria"):
    #     st.title('Malaria Disease Prediction')

    #     class_names=["parasited","uninfected"]

    #     model=load_model("models\Malaria\malaria.h5")


    #     file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

    #     st.set_option('deprecation.showfileUploaderEncoding', False)

    #     def import_and_predict(image_data, model):

    #             size = (256,256)    
    #             image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    #             image = np.asarray(image)
    #             img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #             img_reshape = img[np.newaxis,...]

    #             prediction = model.predict(img_reshape)
    #             return prediction

    #     if st.button("Predict"):
    #         if file is None:
    #             st.error("Please upload an image file")
    #         else:
    #             image = Image.open(file)
    #             prediction = import_and_predict(image, model)
    #             if prediction[0]<0.5 :
    #                 st.error(""" ### High probabilty of Malaria""")

    #             else :
    #                 st.success(""" ### Hurray you are Fine""")
    #                 st.balloons()
    #     if file:
    #         image = Image.open(file)
    #         width = st.slider('', 150, 500)
    #         st.image(image,width=width)


    if (options=="Brain Tumor"):
        st.title('Brain Tumor')

        class_names=["Benign","Malignant"]

        model=load_model("models\BrainTumor\BrainTumor.h5")
    
        file = st.file_uploader("Please upload scan file", type=["jpg", "png"])
    
        st.set_option('deprecation.showfileUploaderEncoding', False)
    
        def import_and_predict(image_data, model):
            
                size = (64,64)    
                image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
                image = np.asarray(image)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                img_reshape = img[np.newaxis,...]
            
                prediction = model.predict(img_reshape)
                return prediction
    
        if st.button("Predict"):    
            if file is None:
                st.error("Please upload an image file")
            else:
                image = Image.open(file)
                prediction = import_and_predict(image, model)
                pred=np.argmax(prediction)
                if pred==1 :
                    st.header("High probabilty of Brain Tumor")
                    
                else :
                    st.success("""### Hurray you are Fine""")
                    st.balloons()
        if file:
            image = Image.open(file)
            width = st.slider('', 150, 500)
            st.image(image,width=width)



    if (options=="Breast Cancer"):
        st.title('Breast Cancer Prediction')

        class_names=["Benign","Malignant"]

        model=load_model("models/BreastCancer/breast_cancer.h5")
    
        file = st.file_uploader("Please upload scan file", type=["jpg", "png"])
    
        st.set_option('deprecation.showfileUploaderEncoding', False)
    
        def import_and_predict(image_data, model):
            
                size = (256,256)    
                image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
                image = np.asarray(image)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                img_reshape = img[np.newaxis,...]
            
                prediction = model.predict(img_reshape)
                return prediction
    
        if st.button("Predict"):    
            if file is None:
                st.error("Please upload an image file")
            else:
                image = Image.open(file)
                prediction = import_and_predict(image, model)
                if prediction[0]>0.5 :
                    st.error(""" ### High probabilty of Breast Cancer""")
                    
                else :
                    st.success(""" ### Hurray you are Fine""")
                    st.balloons()

        if file:
            image = Image.open(file)
            width = st.slider('', 150, 500)
            st.image(image,width=width)














