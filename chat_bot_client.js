'use strict';

// chat_bot_client.js

window.addEventListener('load', function(){
    // 内部で使うプロパティは全てグローバルのものなので，アローによる定義の必要はない
    // グローバル変数で(windowオブジェクトのプロパティとして)
    // WebSocketクラスのインスタンスを保持する変数を作っておく
    window.webSocket = null;
    window.wsServerUrl = 'ws://nt-d.sd.tmu.ac.jp:30005/kibaco_chat_bot';
}, false);

window.addEventListener('unload', function(event){
    window.exitConnection();
}, false);

window.startConnection = function(event){
    // WebSocketクラスのインスタンス化
    window.webSocket = new WebSocket(wsServerUrl);
    // Override Methods
    // WebSocketクラスのインスタンスのプロパティとして
    // WebSocket通信に必要な関数を上書きしていく
    // つまり関数オブジェクトを「プロパティ」として追加していく
    window.webSocket.onopen = function(event){
       window.document.getElementById("view_field").innerHTML =
            '<p>Success to connect WebSocket Server!</p>'
    };

    window.webSocket.onmessage = function(event){
	//alert(event.data);
	window.document.getElementById("view_field").innerHTML =
	    '<div><p>チャットボット:</p>' + event.data + '</div>'
    };

    window.webSocket.onerror = function(error){
	document.getElementById("view_field").innerHTML = 
        '<div><p>チャットボット:</p> <p>エラーが発生しました</p>' + error + '</div>'
    };

    window.webSocket.onclose = function(event){
	if(event.wasClean){ //切断が完全に完了したかどうかを確認
	    document.getElementById("view_field").innerHTML =
	    '<p>切断完了</p><dl><dt>Close code</dt><dd>' + 
		event.Code + 
		'</dd><dt>Close Reason</dt><dd>' + 
		event.reason +
		'</dd></dl>';
	    webSocket = null;
	}
	else
	    document.getElementById("view_field").innerHTML =
	    '<p>切断未完了</p>';
    };

}

window.exitConnection = function(event){
    if(window.webSocket != null)
	window.webSocket.close(1000, '通常終了'); //onclose関数が呼ばれる
}

window.sendMessage = function(sendingMessage){
    if(window.webSocket != null)
	window.webSocket.send(sendingMessage);
}