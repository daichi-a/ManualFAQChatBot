# -*- coding: utf-8 -*-
# chat_bot_server.py
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.template

# BERTの特徴量抽出に必要なライブラリ
import torch
from transformers import BertJapaneseTokenizer, BertModel

import numpy as np
import pandas as pd

# TF-IDF用
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer

# 必要なライブラリ
# pipの場合（pipだと色々地獄を見そう）
#!apt install aptitude swig
#!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y
#!pip install mecab-python3==0.996.5
#!pip install unidic-lite
#!pip install fugashi
#!pip install ipadic
#!pip install transformers==4.7.0-py3

# condaの場合は、以下
# Ubuntu18.04LTSのconda環境は3.6で作る必要がある
# 3.7だとglibcのバージョンで2.29を要求される。Ubuntu 18.04LTSは標準が2.27
# 20.04LTSなら，glibcが2.29なので3.7でよい

# conda create -n 環境名 python=3.9
# 18.04LTSなら conda create -n 環境名 python=3.6
# conda activate 環境名
# conda install -c conda-forge mecab-python3
## mecab==0.996, mecab-python3==1.0.3
# conda install -c conda-forge unidic-lite
## unidic-lite-1.0.8
# conda install -c pytorch cpuonly pytorch torchvision
## pytorch-1.10.1-py3.9_cpu_0, torchvision-0.11.2-py39_cpu
# conda install -c huggingface transformers
## huggingface/noarch::transformers-4.11.3-py_0
# conda install tornado
# conda install pandas
# conda install scikit-learn
# pip install fugashi ipadic janome
## fugashi-1.1.1 ipadic-1.0.0 janome-0.4.1

# numpyがダメになることがあるので、一度
# conda uninstall numpyしておいて、
# conda install numpy
# conda install -c huggingface transformers
# で再インストールする

class chat_bot_server(tornado.websocket.WebSocketHandler):
    # Override Event Functions
    def open(self):
        print('connection opened...')
        self.write_message('<p>kibacoのサポートチャットボットへようこそ</p><p>ここではチャットボットが、マニュアルやFAQの検索を助けてくれます。</p><p>調べたいことを入力して、「送信」ボタンを押してください。</p>')

    def on_message(self, message):
        print('received:', message)

        self.write_message('<p>ご質問は</p><p><strong>' + message + '</strong></p><p>ですね？</p><p><strong>検索中です……</strong></p>')
        # BERTの検索を行う
        result_text_bert = self.search_items_by_bert_embedding(message)

        # TF-DFの検索を行う      
        result_text_tfidf = self.search_items_by_tfidf(message)

        # 両方の検索のかぶっている所を持ってくる。
        # かぶっている所がなければ、TF-IDFの方を優先
        # とりあえず両方表示

        self.write_message('<p>ご質問は</p><p><strong>' + message + '</strong></p><p>ですね？</p>' + result_text_bert + result_text_tfidf)



    def on_close(self):
        print('connection closed...')

    def check_origin(self, origin):
    # クロスオリジン(Cross-Origin)ポリシーの問題を解決する
	# 通常JavaScirptは「自分がダウンロードされてきたサーバのドメイン」にしか接続できない
	# ここで何もチェックしないでTrueを返すことで，
	# このサーバプログラムが動いているサイト以外(例えばローカルで実行している時)から
	# ダウンロードされたJavaSciprtプログラムからも，接続を許す
        return True

    def search_items_by_bert_embedding(self, text):
        global df_embedding, df_paragraph
        print('seraching items... ')

        # 既存のタイトルと一番近い距離のタイトルを検索

        target_text = text
        target_embedding = calc_embedding_last_layer(target_text)

        #target_embedding_copied = np.full(target_embedding, 768)
        #distances = np.linalg.norm(target_embedding_copied - df_embedding)        

        embedding_dist_list = []
        for a_vec in df_embedding.values:
            embedding_dist_list.append(np.linalg.norm(target_embedding - a_vec))
        distances = np.array(embedding_dist_list)
        #print(distances)

        # これで計算したのは単純なユークリッド距離なので、0に近いほど近い
        # なので昇順
        min_dist_indexes = np.argsort(distances)
        min_n_indexes = min_dist_indexes[:10]
        #print(min_6_indexes)
        
        paragraph_title_list = df_paragraph.loc[:, 'Title']
        paragraph_title_min_n_list = []
        paragraph_url_list = df_paragraph.loc[:, 'URL']
        paragraph_url_min_n_list = []
        paragraph_key_sentence_list = df_paragraph.loc[:, 'Paragraph']
        paragraph_key_sentence_min_n_list = []

        result_text = "<h2>BERTの結果</h2>"
        for i, index in enumerate(min_n_indexes):
            paragraph_title_min_n_list.append(paragraph_title_list[index])
            paragraph_url_min_n_list.append(paragraph_url_list[index])
            paragraph_key_sentence_min_n_list.append(paragraph_key_sentence_list[index])

            current_result = '<p>候補' + str(i) + '</p><p>' + paragraph_title_list[index] + '</p><p>マッチした文:' + paragraph_key_sentence_list[index] + '</p><p><a target="_blank" href="' + paragraph_url_list[index] + '">' + paragraph_url_list[index] + '</a></p>'
            # print(current_result)

            result_text = result_text + current_result

        return result_text

    def search_items_by_tfidf(self, text):
        global tfidf, tfidf_vectorizer, tfidf_transformer, df_body

        t = Tokenizer()
        # 1文だけ入れる時は、半角スペースで分かち書きしたものを、1要素のリストに入れて渡す
        text_tf = tfidf_vectorizer.transform([wakachi_str(t, text)])
        text_tfidf = tfidf_transformer.transform(text_tf)
        # コサイン類似度の計算。IF-IDFは登場頻度と非登場頻度なので、コサイン類似度との相性が良いらしい。
        similarity = cosine_similarity(text_tfidf, tfidf)[0]

        # TOP10を持ってくる。コサイン類似度は1に近いほど類似しているので、降順になる。
        top_n_indexes = np.argsort(similarity)[::-1][:10]

        body_title_list = df_body.loc[:, 'Title']
        body_title_top_n_list = []
        body_url_list = df_body.loc[:, 'URL']
        body_url_top_n_list = []

        result_text = "<h2>TFIDFの結果</h2>"
        for i, index in enumerate(top_n_indexes):
            body_title_top_n_list.append(body_title_list[index])
            body_url_top_n_list.append(body_url_list[index])

            current_result = '<p>候補' + str(i) + '</p><p>' + body_title_list[index] + '</p><p><a target="_blank" href="' + body_url_list[index] + '">' + body_url_list[index] + '</a></p>'
            # print(current_result)

            result_text = result_text + current_result

        return result_text



def calc_embedding_last_layer(text):
    global bert_tokenizer, model_bert, df_embedding, df_paragraph
    # 特徴量抽出の関数の定義
    # やってることは最終層の出力を見るだけ
    # 最終層のレイヤーの出力。これをそのまま使うのは良くないと公式ドキュメントにあるそうな
    # 「隠れ層を平均するなりプーリングするなりして使ってくれ」
    bert_tokens = bert_tokenizer.tokenize(text)
    ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])
    tokens_tensor = torch.tensor(ids).reshape(1, -1)

    with torch.no_grad():
        output = model_bert(tokens_tensor)
        return output[1].numpy() # これが最終層のみの出力を指定している部分か


def initialize_bert_pre_traind_model():
    global bert_tokenizer, model_bert, df_embedding, df_paragraph
    # 東北大の。Tokenizerの形態素解析にMeCabを使用（MeCabを経由しているだけだが）
    bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # 訓練モードに入る
    # model_bert.train()

    # predictモードに入る
    model_bert.eval()

    # CSVの読み込み
    df_paragraph = pd.read_csv('./url_title_paragraph_data.csv')

    # 'Paragraph'列だけを抽出
    df_paragraph_seq = df_paragraph.loc[:, 'Paragraph']

    # 特徴量を格納するarrayを作り
    embedding_array = np.zeros((len(df_paragraph_seq.values), 768), dtype=float)

    print('BERT Calc Embedding')
    # 特徴量の抽出
    for index, data in np.ndenumerate(df_paragraph_seq.values):
        embedding_array[index] = calc_embedding_last_layer(data)

    print('BERT Finish Calc Embedding')
    df_embedding = pd.DataFrame(embedding_array)


    return bert_tokenizer, model_bert, df_embedding


#わかち書き関数
def wakachi_list(t, text):

    tokens = t.tokenize(text)
    docs=[]
    for token in tokens:
        docs.append(token.surface)
    return docs

def wakachi_str(t, text):

    tokens = t.tokenize(text, wakati=True)
    docs=[]
    for token in tokens:
        docs.append(token)
    return ' '.join(docs)

def initialize_tfidf():
    global tfidf, tfidf_vectorizer, tfidf_transformer, df_body

    df_body = pd.read_csv('./url_title_body_data.csv')

    t = Tokenizer()
    corpus0 = df_body.loc[:, 'Body'].values
    corpus_list = [wakachi_list(t, sentence) for sentence in corpus0]
    corpus_str = list(wakachi_str(t, sentence) for sentence in corpus0)

    tfidf_vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    tfidf_vectorizer.fit(corpus_str) # 複数を一気に入れる時は半角スペース区切りを取る
    tf = tfidf_vectorizer.transform(corpus_str) 
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(tf)

    #cs_array = cosine_similarity(tfidf, tfidf)

    return t, tfidf_vectorizer, tfidf_transformer, tfidf, 


if __name__ == '__main__':
    # 複数のWebSocketサーバのインスタンスから見るためにグローバル変数にしておく
    # 読み込みと計算は起動じの1回でよいので
    global bert_tokenizer, model_bert, df_embedding, df_paragraph, tfidf_vectorizer, tfidf_transformer, tfidf, df_body

    print('Start Initializing BERT')
    # ここで、CSVのマニュアルとFAQのデータを読み見込んで、初期化、各項目のタイトルと本文のBERT特徴量を算出しておく
    bert_tokenizer, model_bert, df_embedding = initialize_bert_pre_traind_model()

    print('End: Initializing BERT.')

    print('Start: Initialize TF-IDF')
    t, tfidf_vectorizer, tfidf_transformer, tfidf = initialize_tfidf()


    print('End: Initialize TF-IDF')

    # 特徴量の算出が終わったら、WebSocket Serverを起動
    # 応答できるようになる

    print('Start WebScoket Server.')
    application = tornado.web.Application([('/kibaco_chat_bot', chat_bot_server)])
    application.listen(30005)
    tornado.ioloop.IOLoop.instance().start()
    print('kibaco Chat Bot: Boot Process Finished')