import requests
from bs4 import BeautifulSoup

sindongaUrl = "http://land.naver.com/article/articleList.nhn?rletTypeCd=A01&tradeTypeCd=B1&rletNo=671&cortarNo=1168011500&hscpTypeCd=A01%3AA03%3AA04&mapX=127.0984011&mapY=37.4878068&mapLevel=13&page=&articlePage=&ptpNo=&rltrId=&mnex=&bildNo=&articleOrderCode=3&cpId=&period=&prodTab=&atclNo=&atclRletTypeCd=&location=&bbs_tp_cd=&sort=&siteOrderCode=&schlCd=&tradYy=&exclsSpc=&splySpcR=&cmplYn="
ggachiUrl = "http://land.naver.com/article/articleList.nhn?rletTypeCd=A01&tradeTypeCd=B1&rletNo=641&cortarNo=1168011500&hscpTypeCd=A01%3AA03%3AA04&mapX=127.0878162&mapY=37.485059&mapLevel=13&page=&articlePage=&ptpNo=&rltrId=&mnex=&bildNo=&articleOrderCode=3&cpId=&period=&prodTab=&atclNo=&atclRletTypeCd=&location=&bbs_tp_cd=&sort=&siteOrderCode=&schlCd=&tradYy=&exclsSpc=&splySpcR=&cmplYn="
daechiUrl = "http://land.naver.com/article/articleList.nhn?rletTypeCd=A01&tradeTypeCd=B1&rletNo=483&cortarNo=1168010300&hscpTypeCd=A01%3AA03%3AA04&mapX=127.0754858&mapY=37.4946581&mapLevel=13&page=&articlePage=&ptpNo=&rltrId=&mnex=&bildNo=&articleOrderCode=3&cpId=&period=&prodTab=&atclNo=&atclRletTypeCd=&location=&bbs_tp_cd=&sort=&siteOrderCode=&schlCd=&tradYy=&exclsSpc=&splySpcR=&cmplYn="
suseoUrl = "http://land.naver.com/article/articleList.nhn?rletTypeCd=A01&tradeTypeCd=B1&rletNo=827&cortarNo=1168011400&hscpTypeCd=A01%3AA03%3AA04&mapX=127.0909994&mapY=37.4930967&mapLevel=13&page=&articlePage=&ptpNo=&rltrId=&mnex=&bildNo=&articleOrderCode=3&cpId=&period=&prodTab=&atclNo=&atclRletTypeCd=&location=&bbs_tp_cd=&sort=&siteOrderCode=&schlCd=&tradYy=&exclsSpc=&splySpcR=&cmplYn="

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
