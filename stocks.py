import socket
import time
import re
import os
import yfinance as yf
from colorama import Fore
from datetime import datetime
import pytz
import logging
import sys

logging.getLogger('yfinance').disabled = True

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


regex = re.compile('[^a-zA-Z-]')
dick = {}
period = 10
last_printed = 0
changed = False
votes = {}

while True:
	resp = sock.recv(2048).decode('utf-8')
	if resp == "":
		sock = socket.socket()
		sock.connect((server, port))
		sock.send(f"PASS {token}\n".encode('utf-8'))
		sock.send(f"NICK {nickname}\n".encode('utf-8'))
		sock.send(f"JOIN {channel}\n".encode('utf-8'))
		continue
	now = time.time()
	for tickler in dick:
		dick[tickler] = [ts for ts in dick[tickler] if now - 600 < ts]
	for word in resp.split():
		if "$" in word:
			changed = True
			tickler = regex.sub('', word).upper()
			if tickler in dick:
				dick[tickler].append(time.time())
			else:
				dick[tickler] = [time.time()]
	

	if now - last_printed > period and changed == True:
		changed = False
		last_printed = now
		votes = {tickler: len(dick[tickler]) for tickler in dick}
		votes = dict(sorted(votes.items(), key=lambda x:x[1], reverse=True))
		print(Fore.YELLOW)
		print("--{}--".format(datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S")))
		printed = 0
		for tickler in votes:
			if printed >= 10:
				break
			try:
				quote = yf.Ticker(tickler).info
				price = (quote['bid'] + quote['ask']) / 2
				percent_change =100 * (price - quote['previousClose'])/ quote['previousClose']
				print("|", end="")
				if percent_change > 0:
					print(Fore.GREEN, end="")
				else:
					print(Fore.RED, end="")
				print("{:5} ${:<7.2f} {:<+3.2f}%".format(tickler, price, percent_change), end=Fore.YELLOW)
				print("|")
				printed += 1
			except:
				del dick[tickler]
		for x in range(10 - printed):
			print('|                     |')
		print(Fore.YELLOW, end="")
		print('-----------------------')
