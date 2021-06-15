import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")


# used a dictionary to represent an intents JSON file
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up"],
              "responses": ["Fine", "Hello", "Hello, How may I help you?", "Greetings!", "How do you do?"],
             },
             {"tag": "acne symptoms",
              "patterns": ["what are the symptoms of acne?", "symptoms of acne", "acne's symptoms"],
              "responses": ["Small red, tender bumps & pus-filled lumps under the skin are the main symptoms of acne.",
                            "Whiteheads or Blackheads with Large, solid, painful lumps under the skin.",
                            "Painful, pus-filled lumps under the skin with Whiteheads or Blackheads"]
             },
             {"tag": "acne appear",
              "patterns": ["where is the appearance of acne on body?","tell me appearance of acne?", "where is the acne effect on body?"],
              "responses": ["Acne usually appears on the face, forehead, chest, upper back and shoulders.",
                            "usually appears on the face, forehead, chest, upper back and shoulders",
                            "face, forehead, chest, upper back and shoulders"]
             },
             {"tag": "acne causes",
              "patterns": ["what is the cause of acne?", "tell me about cause of acne disease?", "acne cause?"],
              "responses": ["Excess oil (sebum) production, Hair follicles clogged by oil and dead skin cells, Bacteria Inflammation are main causes",
                            "Excess oil (sebum) production & Hair follicles clogged by oil",
                            "dead skin cells & Bacteria Inflammation"]
             },
             {"tag": "insect bite symptoms",
              "patterns": ["what are the symptoms of insect bite?", "symptoms of insect's bite", "insect's bite symptoms"],
              "responses": ["redness, itching, stinging or minor swelling.",
                            "redness, stinging",
                            "Painful, minor swelling"]
             },
             {"tag": "insect bite causes",
              "patterns": ["what is the cause of insect's bite?", "tell me about cause of insect'bite?", "insect bite causes?"],
              "responses": ["causing little more than redness, itching, stinging or minor swelling",
                            ]
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "g2g", "see ya", "adios", "cya"],
              "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
             },
             
]}


# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()
# Each list to create
words = []
classes = []
doc_X = []
doc_y = []
# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    # add the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))

print(words)


# list for training data
training = []
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


# defining some parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])

model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens:
    for idx, word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels):
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json):
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents:
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result


def text_send(message):
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    return result
