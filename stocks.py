import socket
import time
import re

server = 'irc.chat.twitch.tv'
port = 6667
nickname = 'jongoose1'
token = open('auth.token', 'r').read().rstrip()
channel = '#jongoose1'

sock = socket.socket()
sock.connect((server, port))
sock.send(f"PASS {token}\n".encode('utf-8'))
sock.send(f"NICK {nickname}\n".encode('utf-8'))
sock.send(f"JOIN {channel}\n".encode('utf-8'))


regex = re.compile('[^a-zA-Z]')
dick = {}

while True:
	resp = sock.recv(2048).decode('utf-8')
	#remove values older than 10 mins
	tenago = time.time() - 600
	for tickler in dick:
		dick[tickler] = [ts for ts in dick[tickler] if tenago < ts]
	for word in resp.split():
		if "$" in word:
			tickler = regex.sub('', word).upper()
			if tickler in dick:
				dick[tickler].append(time.time())
			else:
				dick[tickler] = [time.time()]
	for tickler in dick:
		print("{}: {}".format(tickler, len(dick[tickler])))
