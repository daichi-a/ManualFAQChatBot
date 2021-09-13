# -*- coding: utf-8 -*-
import urllib.request 
from html.parser import HTMLParser

import re
import csv

class get_manual_pages_url(HTMLParser): #HTMLParserを継承したクラスを定義
    def __init__(self, my_domain):
        # コンストラクタ
        super().__init__() #親クラスのコンストラクタを実行
        self.__counter = 0
        self.__link_url_array = [] #リンク先を格納するlistの初期化
        self.__my_domain = my_domain

    # Override
    def handle_starttag(self, tag, attrs):
        # tagはタグ名文字列
        # attrsはアトリビュート(要素)が入れ子配列
        # ([['href', 'http://...'], ['title', 'こーひーをぶんなぐれ'], ...]
        # のような形で出てくる
        if tag == 'a':
            # 入れ子配列で出てくるアトリビュートをハッシュテーブルに変換
            attrs_of_a_tag = dict(attrs) 
            # ハッシュテーブルにhref
            if 'href' in attrs_of_a_tag:
                link_address = attrs_of_a_tag['href']
                if 'https://' in link_address:
                    # 文字列'https://www.comp.tmu.ac.jp/e-learning/kibaco/documents/'を含んでいて
                    if 'https://www.comp.tmu.ac.jp/e-learning/kibaco/documents/' in link_address:
                        # kibacoのドキュメントである場合
                        # 配列に突っ込んでカウンタを増やす
                        self.__link_url_array.append(link_address)
                        self.__counter += 1
                        print(link_address)

                        

    # Override
    def handle_data(self, data):
        new_data = data.replace(' ', '')
        if len(new_data) > 0:
            pass

    # Override
    def handle_endtag(self, tag):
        pass

    def get_counter(self):
        return self.__counter

    def get_url_list(self):
        return self.__link_url_array

class paragraph_parser(HTMLParser): #HTMLParserを継承したクラスを定義

    # 定数やインスタンス内で値が変わらないstaticメンバ変数はクラス宣言の後に宣言する
    # 頭に「__」が付くと変数も関数もprivate扱いになる
    def __init__(self, url):
        # コンストラクタ
        super().__init__() #親クラスのコンストラクタを実行
        self.__data_list = []
        self.__url = url
        self.__in_tag = False

    # Override
    def handle_starttag(self, tag, attrs):
        # print('Start Tag:', tag)
        if tag == 'h2' or tag == 'h3' or tag == 'p':
            self.__in_tag = True

    # Override
    def handle_data(self, data):
        if self.__in_tag == True:
            # h2とh3とpの中には具体的にやりたいことの内容が入っているので、これをリンクと一緒に取り出す
            # テキストデータがタグで囲まれている時に呼び出される
            # どうやらスペースが入ってても呼び出されるらしい
            # なので最初に，' '(半角スペース)を''(何も入っていない文字)で置き換える
            new_data = data.replace(' ', '')
            if new_data != 'kibacoログイン':
                if len(new_data) > 0:
                    print('Some Data:', new_data)
                    self.__data_list.append(new_data)

    # Override
    def handle_endtag(self, tag):
        # print('End Tag:', tag)
        self.__in_tag = False

    def get_paragraph_list(self):
        #print(self.__data_list)
        return self.__data_list

                    
if __name__ == "__main__":
    url = "https://www.comp.tmu.ac.jp/e-learning/kibaco/documents/index.html"
    gotten_http_response = urllib.request.urlopen(url) 
    # 指定されたURLのページを取得する命令を出し，
    # その結果をgotten_http_responseという名前の
    # HTTPResponseクラス型の変数に突っ込む
    
    # 各種情報の出力
    print(gotten_http_response.code)
    #print(gotten_http_response('content-type'))
    charset = gotten_http_response.info().get_content_charset()
    print(charset)

    # GETしてきたHTML自体を出力
    # HTTPResponseクラスはread()という関数で中身を読めるのだが，
    # これはbyte型(つまりバイナリデータ)なので，文字列に変換するため
    # utf-8エンコードを指定してデコードする
    # print(gotten_http_response.read().decode('utf-8'))

    get_urls = get_manual_pages_url(url) #上で宣言していたクラスのオブジェクトインスタンス化
    get_urls.feed(gotten_http_response.read().decode('utf-8')) 
    #パーサにHTMLを渡してjson構造をイテレータ動作で読んでいく
    
    print('hyper link to other site appears', get_urls.get_counter(), 'times')

    manual_url_list = get_urls.get_url_list()

    row_list = []

    title_url_list = []
    max_row_length = 0
    for a_url in manual_url_list:
        gotten_http_response = urllib.request.urlopen(a_url)
        get_paragraph = paragraph_parser(url)
        get_paragraph.feed(gotten_http_response.read().decode('utf-8'))
        data_list = get_paragraph.get_paragraph_list()
        #print(data_list)

        for index, a_paragraph in enumerate(data_list):
            paragraph_string = re.sub(r'[0-9]+', '', a_paragraph)
            paragraph_string = re.sub(r'-', '', paragraph_string)
            paragraph_string = re.sub(r'\n', '', paragraph_string)
            paragraph_string = re.sub(r'\xa9', '', paragraph_string)
            paragraph_string = re.sub(r'\uf06c', '', paragraph_string)
            paragraph_string = re.sub(r'\u2003', '', paragraph_string)
            paragraph_string = re.sub(r'\u3000', '', paragraph_string)

            paragraph_string = re.sub(r'Copyright2015TokyoMetropolitanUniversity', '', paragraph_string)
            paragrah_string = paragraph_string.replace('.', '')
            data_list[index] = paragraph_string

        page_title = data_list[0]
        print(page_title)

        for a_paragraph in data_list:
            row_list.append([a_url, page_title, a_paragraph])

        #print(row_list)

        get_paragraph.close()
        gotten_http_response.close()

    get_urls.close() #デストラクタを呼ぶ
    gotten_http_response.close()

    # CSVファイルへ書き込み

    head_row = ['URL', 'Title', 'Data']

    with open('./url_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(head_row)
        writer.writerows(row_list)
    f.close()

    print('最大列幅', max_row_length)
    print(row_list)