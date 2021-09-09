# -*- coding: utf-8 -*-
# WSTextMessageTornadoServer.py
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.template

class chat_bot_server(tornado.websocket.WebSocketHandler):
    # Override Event Functions
    def open(self):
        print('connection opened...')
        self.write_message("The server says: 'Hello'. Connection was accepted.")

    def on_message(self, message):
        print('received:', message)
        self.write_message("The server return your message: " + message)

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