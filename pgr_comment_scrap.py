from bs4 import BeautifulSoup
import time
import random

import urllib.request
import urllib.parse
from urllib import parse


search_str = '2019 우리은행 챔피언스 코리아 서머 플레이오프 2R - DAMWON vs. SKT'
parse_str = parse.quote(search_str)
search_url = 'https://pgr21.com/pb/pb.php?id=bulpan&ss=on&sc=on&keyword=' + parse_str
pan_url = 'https://pgr21.com/pb/pb.php?id=bulpan&no='

def get_soup(url) :        
    with urllib.request.urlopen(url) as response:
        time.sleep(random.randrange(0, 5)) # 너무 자주 긁어서 디도스로 오인받아 차단당할라..
        html = response.read()
        return BeautifulSoup(html, 'html.parser')

soup = get_soup(search_url)
a_tags = soup.find_all('a')

pan_urls = []
for at in a_tags:
    if search_str in  at.text :
        pan_no = at['href'].split('no=')[1].split('&')[0]
        pan_urls.append(pan_url + pan_no)

f = open(search_str, 'w', encoding='utf8')

for pan_url in pan_urls:
    psoup = get_soup(pan_url)
    users = psoup.find_all(class_='ctName')
    comments = psoup.find_all(class_='cmemo')
    c_times = psoup.find_all(class_='time')
    lens = len(users)
    for idx in range(lens):
        idx = (lens - idx - 1)
        text = '[' + c_times[idx].text + ']' + users[idx].text + '\t' + comments[idx].text + '\n'
        f.write(text)
        
f.close()
