import requests
from bs4 import BeautifulSoup
from telegram.ext import Updater
import configparser
import random
import sys

cf = configparser.ConfigParser()
cf.read('config.cfg')

VALID_USER=cf.get('telegram','VALID_USER')
TOKEN=cf.get('telegram','TOKEN')

sindongaUrl = cf.get('land_url','sindongaUrl')
ggachiUrl = cf.get('land_url', 'ggachiUrl')
daechiUrl = cf.get('land_url', 'daechiUrl')
suseoUrl = cf.get('land_url', 'suseoUrl')

LIMIT_PRICE = 20000
SLEEP_SEC = random.randint(120,300)
BREAK_LIMIT = 5
BREAK_TIME = 0

urls = {'SIN-DONG-A': sindongaUrl, 'GGA-CHI': ggachiUrl, 'DAE-CHI':daechiUrl, 'SU-SEO':suseoUrl}




def get_sale_products(findUrl) :
	soup = BeautifulSoup(requests.get(findUrl).text, "html.parser")
	table = soup.find("table", { "class" : "sale_list _tb_site_img NE=a:cpm"})
	trs = table.find("tbody").find_all('tr')
	results = list()

	for tr in trs:
		try :
			price = tr.find('td', {'class':'num align_r'}).find('strong').text
			dong = tr.find_all('td', {'class':"num2"})[0].text
			floor = tr.find_all('td', {'class':"num2"})[1].text
			budongsan = tr.find('td', {'class':'contact'}).find_all('span')[0]['title']
			contact = tr.find('td', {'class':'contact'}).find_all('span')[1].text
			crol = {'price':price,'dong':dong,'floor':floor,'budongsan':budongsan,'contact':contact}
			results.append(crol)
		except AttributeError:
			continue
	return results

def get_line_up(products):
	result = ''
	for key in products:
		result += '[PRODUCT]:: ' + key + '\n'
		for product in products[key]:
			result += '\t'+str(product) + '\n'
	return result

def get_new():
	products = {}
	for key in urls.keys():
		products[key] = get_sale_products(urls[key])[0:3]
	return products



import time

check_flag = False
seen_set = set()

while True:
	try :
		products = get_new()	
		for key in products:
			product = products[key]
			for each in product:
				check_key = key + each['price'] + each['contact'] + each['floor']
				if (check_key not in seen_set) and (int(each['price'].replace(',','')) <= LIMIT_PRICE): 
					check_flag = True
					seen_set.add(check_key)
			if check_flag is True:
				print('break')
				break
		if check_flag is True:
			msg = get_line_up(products)
			print(msg)
			updater = Updater(TOKEN)
			updater.bot.sendMessage(chat_id=VALID_USER, text=msg)
			check_flag = False
		else :
			print('none')
		time.sleep(SLEEP_SEC)
	except :
		print('unexpect error.', sys.exc_info())
		if BREAK_LIMIT < BREAK_TIME :
			break
		else :
			BREAK_TIME += 1
			continue
