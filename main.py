import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
import traitlets
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import roc_auc_score
warnings.simplefilter("ignore")

#Helper Functions
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
 tokenizer.enable_truncation(max_length=maxlen)
 tokenizer.enable_padding(max_length=maxlen)
 all_ids = []

 for i in tqdm(range(0, len(texts), chunk_size)):
 text_chunk = texts[i:i+chunk_size].tolist()
 encs = tokenizer.encode_batch(text_chunk)
 all_ids.extend([enc.ids for enc in encs])

 return np.array(all_ids)
 
 def build_model(transformer, loss='binary_crossentropy', max_len=512):
 input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids
")
 sequence_output = transformer(input_word_ids)[0]
 cls_token = sequence_output[:, 0, :]
 x = tf.keras.layers.Dropout(0.35)(cls_token)
 out = Dense(1, activation='sigmoid')(x)

 model = Model(inputs=input_word_ids, outputs=out)
 model.compile(Adam(lr=3e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])

 return model
 
 import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt') # if necessary...
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def stem_tokens(tokens):
 return [stemmer.stem(item) for item in tokens]
'''remove punctuation, lowercase, stem'''
def normalize(text):
 return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuati
on_map)))
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
def cosine_sim(text1, text2):
 tfidf = vectorizer.fit_transform([text1, text2])
 return ((tfidf * tfidf.T).A)[0,1]

#TPU Configs
AUTO = tf.data.experimental.AUTOTUNE
# Create strategy from tpu
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
Create fast tokenizer
# First load the real tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
# Save the loaded tokenizer locally
save_path = '/kaggle/working/distilbert_base_uncased/'
if not os.path.exists(save_path):
 os.makedirs(save_path)
tokenizer.save_pretrained(save_path)
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowe
rcase=False)
fast_tokenizer

#Load text data into memory
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classificat
ion/jigsaw-toxic-comment-train.csv")
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classificat
ion/jigsaw-unintended-bias-train.csv")
valid = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')
test1 = pd.read_csv('/kaggle/input/test-en-df/test_en.csv')
test2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigs
aw_miltilingual_test_translated.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification
/sample_submission.csv')

#Fast encode
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=512
)
x_valid = fast_encode(valid.comment_text_en.astype(str), fast_tokenizer, maxlen=5
12)
x_test1 = fast_encode(test1.content_en.astype(str), fast_tokenizer, maxlen=512)
x_test2 = fast_encode(test2.translated.astype(str), fast_tokenizer, maxlen=512)
y_train = train1.toxic.values
y_valid = valid.toxic.values

#Build datasets objects
train_dataset = (
 tf.data.Dataset
 .from_tensor_slices((x_train, y_train))
 .repeat()
 .shuffle(2048)
 .batch(64)
 .prefetch(AUTO)
)
valid_dataset = (
 tf.data.Dataset
 .from_tensor_slices((x_valid, y_valid))
 .batch(64)
 .cache()
 .prefetch(AUTO)
)
test_dataset = [(
 tf.data.Dataset
 .from_tensor_slices(x_test1)
 .batch(64)
),
 (
 tf.data.Dataset
 .from_tensor_slices(x_test2)
 .batch(64)
)]

#Focal Loss
from tensorflow.keras import backend as K
def focal_loss(gamma=2., alpha=.2):
 def focal_loss_fixed(y_true, y_pred):
 pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
 pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
 return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - a
lpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
 return focal_loss_fixed
Load model into the TPU
%%time
with strategy.scope():
 transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-cased
')
 model = build_model(transformer_layer, loss=focal_loss(gamma=1.5), max_len=51
2)
model.summary()

#RocAuc Callback
from tensorflow.keras.callbacks import Callback
class RocAucCallback(Callback):
 def __init__(self, test_data, score_thr):
 self.test_data = test_data
 self.score_thr = score_thr
 self.test_pred = []

 def on_epoch_end(self, epoch, logs=None):
 if logs['val_auc'] > self.score_thr:
 print('\nRun TTA...')
 for td in self.test_data:
 self.test_pred.append(self.model.predict(td))

#LrScheduler
def build_lrfn(lr_start=0.000001, lr_max=0.000004,
 lr_min=0.0000001, lr_rampup_epochs=7,
 lr_sustain_epochs=0, lr_exp_decay=.87):
 lr_max = lr_max * strategy.num_replicas_in_sync
 def lrfn(epoch):
 if epoch < lr_rampup_epochs:
 lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
 elif epoch < lr_rampup_epochs + lr_sustain_epochs:
 lr = lr_max
 else:
 lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr
_sustain_epochs) + lr_min
 return lr
return lrfn

#Train Model
roc_auc = RocAucCallback(test_dataset, 0.935)
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
train_history = model.fit(
 train_dataset,
 steps_per_epoch=150,
 validation_data=valid_dataset,
 callbacks=[lr_schedule, roc_auc],
 epochs=35
)
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_valid, y_valid, test_size
=0.15, shuffle=True, random_state=123, stratify=y_valid)
train_dataset = (
 tf.data.Dataset
 .from_tensor_slices((x_train, y_train))
 .repeat()
 .shuffle(2048)
 .batch(64)
 .prefetch(AUTO)
)
valid_dataset = (
 tf.data.Dataset
 .from_tensor_slices((x_valid, y_valid))
 .batch(64)
 .cache()
 .prefetch(AUTO)
)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
train_history = model.fit(
 train_dataset,
 steps_per_epoch=75,
 validation_data=valid_dataset,
 callbacks=[lr_schedule, roc_auc],
 epochs=8
)
#Submission
sub['toxic'] = np.mean(roc_auc.test_pred, axis=0)
sub.to_csv('submission.csv', index=False)
