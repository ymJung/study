import requests
from bs4 import BeautifulSoup
import configparser
import random
import sys
import datetime
import time

cf = configparser.ConfigParser()
cf.read('config.cfg')


landUrls = cf.get('land_url','URLS')
landUrls = landUrls.split(',')

if sys.argv[0] is not None:
    LIMIT_PRICE = sys.argv[0]
else:
	LIMIT_PRICE = 20000

SLEEP_SEC = random.randint(120,300)

RETRY_LIMIT_CNT=5
RETRY_LIMIT={datetime.date.today(): RETRY_LIMIT_CNT}



def get_sale_products(findUrl) :
	soup = BeautifulSoup(requests.get(findUrl).text, "html.parser")
	table = soup.find("table", { "class" : "sale_list _tb_site_img NE=a:cpm"})
	trs = table.find("tbody").find_all('tr')
	name = soup.find(id='complexListLayer').find('a', {'class':'on'}).text.strip()
	results = list()

	for tr in trs:
		try :
			price = tr.find('td', {'class':'num align_r'}).find('strong').text
			dong = tr.find_all('td', {'class':"num2"})[0].text
			floor = tr.find_all('td', {'class':"num2"})[1].text
			budongsan = tr.find('td', {'class':'contact'}).find_all('span')[0]['title']
			contact = tr.find('td', {'class':'contact'}).find_all('span')[1].text
			crol = {'name':name,'price':price,'dong':dong,'floor':floor,'budongsan':budongsan,'contact':contact}
			results.append(crol)
		except AttributeError:
			continue
	return results


def get_line_up(products):
	result = ''
	for product in products:
		result += '[' + product['name'] + '] \t' + str(product) + '\n'
	return result

def get_new():
	products=[] 
	for landUrl in landUrls: 
		getProducts = get_sale_products(landUrl)[0:3]
		products.extend(getProducts)

	return products

def is_break():
	retryCnt = get_date_retry_limit(datetime.date.today())
	if retryCnt<0:
		return True
	return False
def get_date_retry_limit(date):
	dateStr = str(date)
	if dateStr in RETRY_LIMIT:
		print('reduce today limit ', dateStr, RETRY_LIMIT[dateStr])
		RETRY_LIMIT[dateStr] -= 1
	else:
		print('make today limit ', dateStr)
		RETRY_LIMIT.update({dateStr: RETRY_LIMIT_CNT})
	return RETRY_LIMIT[dateStr]

check_flag = False
seen_set = set()
while True:
	try :
		products = get_new()	
		for product in products:
			check_key = product['name'] + product['price'] + product['contact'] + product['floor']
			if (check_key not in seen_set) and (int(product['price'].replace(',','')) <= LIMIT_PRICE): 
				check_flag = True
				seen_set.add(check_key)
		if check_flag is True:
			msg = get_line_up(products)
			print(msg)
			check_flag = False
		else :
			print('none')
			time.sleep(SLEEP_SEC)
	except :
		print('unexpected error', sys.exc_info()[0])
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
		if is_break():
			break
		continue


