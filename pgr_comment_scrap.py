from bs4 import BeautifulSoup
import time
import random

import urllib.request
import urllib.parse
from urllib import parse


search_str = '' #검색할 불판명 쓸것 (숫자제외)
parse_str = parse.quote(search_str)
search_url = 'https://pgr21.com/pb/pb.php?id=bulpan&page=1&ss=on&sc=on&keyword=' + parse_str

pan_url = 'https://pgr21.com/pb/pb.php?id=bulpan&no='
default_page = 'https://pgr21.com/'
def get_soup(url) :        
    with urllib.request.urlopen(url) as response:
        print('crawlling.. ' + url)
        time.sleep(random.randrange(0, 5)) # 너무 자주 긁어서 디도스로 오인받아 차단당할라..
        html = response.read()
        return BeautifulSoup(html, 'html.parser')

a_tags = []

soup = get_soup(search_url)
pages = soup.find(class_='pagination')
DEFAULT_PAGE_A_TAG_CNT = 3

if len(pages) > DEFAULT_PAGE_A_TAG_CNT:
    for page in pages:
        if 'page=' in page['href'] :
            page_per_pan_soup = get_soup(default_page + page['href'])
            p_p_tags = page_per_pan_soup.find_all('a')
            a_tags.extend(p_p_tags)
first_page_tags = soup.find_all('a')

a_tags.extend(first_page_tags)

pan_urls = []
for at in a_tags:
    if search_str in  at.text :
        href_url = at['href']
        pan_no = href_url[href_url.rfind('/') + 1: href_url.find('?')]        
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
