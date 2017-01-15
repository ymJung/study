import requests
from bs4 import BeautifulSoup
from telegram.ext import Updater
import configparser

cf = configparser.ConfigParser()
cf.read('config.cfg')

VALID_USER=cf.get('telegram','VALID_USER')
TOKEN=cf.get('telegram','TOKEN')

sindongaUrl = cf.get('land_url','sindongaUrl')
ggachiUrl = cf.get('land_url', 'ggachiUrl')
daechiUrl = cf.get('land_url', 'daechiUrl')
suseoUrl = cf.get('land_url', 'suseoUrl')

urls = {'SIN-DONG-A': sindongaUrl, 'GGA-CHI': ggachiUrl, 'DAE-CHI':daechiUrl, 'SU-SEO':suseoUrl}




def getSaleProducts(findUrl) :
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

def printall(products):
	result = ''
	for key in products:
		result += '[PRODUCT]:: ' + key + '\n'
		for product in products[key]:
			result += '\t'+str(product) + '\n'
	return result

products = {}
for key in urls.keys():
	products[key] = getSaleProducts(urls[key])[0:3]

print(printall(products))
