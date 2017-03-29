import requests
from bs4 import BeautifulSoup
from telegram.ext import Updater
import configparser
import random
import sys
import datetime
import time
import os

cf = configparser.ConfigParser()
cf.read('config.cfg')

VALID_USER=cf.get('telegram','VALID_USER')
TOKEN=cf.get('telegram','TOKEN')


RETRY_LIMIT_CNT=5
RETRY_LIMIT={datetime.date.today(): RETRY_LIMIT_CNT}
ACTION_TIME = [7,16]

seen_set = set()

while True:
	try :
		n_hour = datetime.datetime.now().hour
		#if n_hour in ACTION_TIME: 
                result = os.popen('python ten_worker.py').read()
		SLEEP_SEC = 60 * 60 * 1
		time.sleep(SLEEP_SEC)
	except :
		print('unexpected error', sys.exc_info()[0])
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
		if is_break():
			break
		continue


