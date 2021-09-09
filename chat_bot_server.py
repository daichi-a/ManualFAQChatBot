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

# conda create -n 環境名 python=3.6
# conda activate 環境名
# conda install -c conda-forge mecab-python3
# conda install -c conda-forge unidic-lite
# conda install -c pytorch cpuonly pytorch torchvision
# conda install -c huggingface transformers
# conda install tornado

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

        self.search_items_by_bert_embedding(message)

        self.write_message('<p>ご質問は</p><p><strong>' + message + '</strong></p><p>ですね？</p>')



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
        print('serach item')

        # 既存のタイトルと一番近い距離のタイトルを検索

        target_text = text
        target_embedding = calc_embedding_last_layer(target_text)

        all_embedding = np.vstack((embedding_array, target_embedding))
        # お尻にvstackする。

        # ベクトル間の距離の計算
        # https://sekailab.com/wp/2018/06/11/numpy-combinatorial-calculation-in-array/

        tmp_index = np.arange(all_embedding.shape[0])
        xx, yy = np.meshgrid(tmp_index, tmp_index)
        distances = np.linalg.norm(all_embedding[xx]-all_embedding[yy], axis=2)[-1][:-1]
        # 当たり前だがケツは検索文同士の距離なので取り除いて、
        print(len(distances))
        print(distances)

        # 項目が増えると指数関数的にオーダーが増えてその大部分はいらない距離なので
        # 2つの配列を作って距離を取るアルゴリズムに変更する必要がある
        # xxとyyの配列をコントロールすることでforで回さなくてもよさそう

        min_dist_index = np.argmin(distances)
        print(min_dist_index)

        # 1番マッチした項目を持ってくる
        print(target_text, 'に対する答えは……')
        print(df.loc[:, 'Title'][min_dist_index])
        print(df.loc[:, 'Body'][min_dist_index])
        print(df.loc[:, 'URL'][min_dist_index])

        # 2番目にマッチした項目を持ってくる
        min_dist_indexes = np.argsort(distances)[::-1]
        print(min_dist_indexes)
        print('2番目に適合したindexは', min_dist_indexes[-2])
        second_min_dist_index = min_dist_indexes[-2]

        print(df.loc[:, 'Title'][second_min_dist_index])
        print(df.loc[:, 'Body'][second_min_dist_index])
        print(df.loc[:, 'URL'][second_min_dist_index])




def calc_embedding_last_layer(text):
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
    # 東北大の。Tokenizerの形態素解析にMeCabを使用（MeCabを経由しているだけだが）
    bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # 訓練モードに入る
    # model_bert.train()

    # predictモードに入る
    model_bert.eval()

    # CSVの読み込み
    df = pd.read_csv('faqmanual_itemized.csv')

    # 'Title'列だけを抽出
    df_title_seq = df.loc[:, 'Title']

    # 特徴量を格納するarrayを作り
    embedding_array = np.zeros((len(df_title_seq.values), 768), dtype=float)

    # 特徴量の抽出
    for index, title in np.ndenumerate(df_title_seq.values):
        embedding_array[index] = calc_embedding_last_layer(title)


    df_embedding = pd.DataFrame(embedding_array)


    return bert_tokenizer, model_bert, df_embedding


if __name__ == '__main__':
    # ここで、CSVのマニュアルとFAQのデータを読み見込んで、初期化、各項目のタイトルと本文のBERT特徴量を算出しておく
    initialize_bert_pre_traind_model()
    # 特徴量の算出が終わったら、WebSocket Serverを起動
    # 応答できるようになる
    application = tornado.web.Application([('/kibaco_chat_bot', chat_bot_server)])
    application.listen(30005)
    tornado.ioloop.IOLoop.instance().start()