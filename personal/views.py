from django.shortcuts import render
from django.http import HttpResponse

import json

# Create your views here.

import urllib
import BeautifulSoup

def getHtml() :
	data = urllib.urlopen('http://m.comic.naver.com/webtoon/list.nhn?titleId=119874&week=fri')
	soup = BeautifulSoup.BeautifulSoup(data)
	titles = soup.findAll('strong')
	return titles
def personal_view(request) :

    return HttpResponse(getHtml(), status=200, content_type='application/text; charset=utf-8') 
