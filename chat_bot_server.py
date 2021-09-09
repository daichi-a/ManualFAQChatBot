# -*- coding: utf-8 -*-
# chat_bot_server.py
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.template

class chat_bot_server(tornado.websocket.WebSocketHandler):
    # Override Event Functions
    def open(self):
        print('connection opened...')
        self.write_message('<p>kibacoのサポートチャットボットへようこそ</p><p>ここではチャットボットが、マニュアルやFAQの検索を助けてくれます。</p><p>調べたいことを入力して、「送信」ボタンを押してください。</p>')

    def on_message(self, message):
        print('received:', message)
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

if __name__ == '__main__':
    application = tornado.web.Application([('/kibaco_chat_bot', chat_bot_server)])
    application.listen(30005)
    tornado.ioloop.IOLoop.instance().start()