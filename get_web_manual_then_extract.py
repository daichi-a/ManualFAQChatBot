# -*- coding: utf-8 -*-
import urllib.request 
from html.parser import HTMLParser

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

class get_headings(HTMLParser): #HTMLParserを継承したクラスを定義

    # 定数やインスタンス内で値が変わらないstaticメンバ変数はクラス宣言の後に宣言する
    # 頭に「__」が付くと変数も関数もprivate扱いになる
    def __init__(self, url):
        # コンストラクタ
        super().__init__() #親クラスのコンストラクタを実行
        self.__heading_list = []
        self.__url = url
        self.__in_heading = False

    # Override
    def handle_starttag(self, tag, attrs):
        # print('Start Tag:', tag)
        if tag == 'h2' or tag == 'h3':
            self.__in_heading = True

    # Override
    def handle_data(self, data):
        if self.__in_heading == True:
            # h2とh3には具体的にやりたいことの内容が入っているので、これをリンクと一緒に取り出す
            # ついでに受講者向けか教員向けかを取り出す
            # テキストデータがタグで囲まれている時に呼び出される
            # どうやらスペースが入ってても呼び出されるらしい
            # なので最初に，' '(半角スペース)を''(何も入っていない文字)で置き換える
            new_data = data.replace(' ', '')
            if len(new_data) > 0:
                # print('Some Data:', new_data)
                self.__heading_list.append(new_data)

    # Override
    def handle_endtag(self, tag):
        # print('End Tag:', tag)
        self.__in_heading = False

    def get_heading_list(self):
        return self.__heading_list

                    
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

    title_url_list = []
    for a_url in manual_url_list:
        gotten_http_response = urllib.request.urlopen(a_url)
        get_titles = get_headings(url)
        get_titles.feed(gotten_http_response.read().decode('utf-8'))
        title_url_list = get_titles.get_heading_list()
        title_url_list.insert(0, a_url)

        print(title_url_list)
        get_titles.close()
        gotten_http_response.close()

    get_urls.close() #デストラクタを呼ぶ
    gotten_http_response.close()

    print(title_url_list)