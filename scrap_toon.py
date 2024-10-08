from selenium import webdriver
import os
from selenium.webdriver.chrome.service import Service
import time
from urllib.request import Request, urlopen
from urllib.parse import quote


SCROLL_PAUSE_SEC = 1  # 스크롤 간격 설정 (초)
def scroll_end(driver):
    
    screen_height = driver.execute_script("return window.screen.height;")  # 화면 높이 가져오기
    i = 1
    while True:
        # 현재 스크롤 위치 가져오기
        scroll_height = driver.execute_script("return document.body.scrollHeight;")        
        # 스크롤을 페이지의 끝까지 내리기
        driver.execute_script(f"window.scrollTo(0, {i * screen_height});")
        # 잠시 기다리기
        time.sleep(SCROLL_PAUSE_SEC)
        # 스크롤이 끝에 도달하면 종료
        if i * screen_height > scroll_height:
            time.sleep(SCROLL_PAUSE_SEC)
            break    
        i += 1

def get_src_urls(url):
    driver.get(url)
    scroll_end(driver)    
    tot_src = driver.page_source 
    splits = tot_src.split('data-original=')
    res = []
    for idx in range(1, len(splits)):
        each = splits[idx]
        if 'jpg' in each:
            img_url = each[each.find('"') + 1:each.find('jpg')] + 'jpg'        
        elif 'png' in each:
            img_url = each[each.find('"') + 1:each.find('png')] + 'png'        
        if '/undefined' in img_url or '11toon5' in img_url :
            continue    
        res.append(img_url)
    return res

def get_src_url_from_home(driver):
    scroll_end(driver)    
    tot_src = driver.page_source 
    splits = tot_src.split('<a href="/webtoons')
    res = []
    for idx in range(1, len(splits) -1):
        each = splits[idx]
        a_tag = each[each.find('<a href="') + 1:each.find('html')] + 'html' 
   
        res.append(a_tag)
    return res
def mk_dir(d_path):
    if not os.path.exists(d_path):
        os.makedirs(d_path)        
        



chrome_driver_path = 'D:\Dev\chromedriver.exe' 
options = webdriver.ChromeOptions() 
options.add_argument('--headless')  # (headless 모드)


home = '.com'#
topic =  '' #
# 웹 페이지 URL 
main_url = "https://{}/webtoons"
home_url = "https://{}/webtoon/{}.html".format(home, topic)

mk_dir(topic)
driver = webdriver.Chrome(service=Service(), options=options)
driver.get(home_url)
tot_srcs = get_src_url_from_home(driver)
# 한번 초기화
driver = webdriver.Chrome(service=Service(), options=options)
f = open('srcs.txt', 'w')
tot_srcs.reverse()
os.chdir(os.path.join(os.getcwd(), topic))
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

last_index = 0
start = ''# 중간부터 안할거면 비워둠
if last_index > 0: now = last_index 
else : now = 0
print(tot_srcs)
if len(start) > 0 : tot_srcs = tot_srcs[tot_srcs.index(start)+1:] 
print(tot_srcs)

for page_url in tot_srcs:
    format_url = main_url.format(home) + page_url
    now += 1
    print(now , '/', len(tot_srcs), ':', format_url)
    src_urls = get_src_urls(format_url)    
    for img_src in src_urls: #download
        now+=1
        img_name = img_src[img_src.rfind('/')+1:]        
        file_nm = str(now) + '_'+ img_name
        print(img_src)
        with open(file_nm, 'wb') as img_f:
            img_url = quote(img_src, safe=':/')
            request = Request(img_url, headers=headers)
            response = urlopen(request)
            img_f.write(response.read())

        f.write(img_src + '\n')
driver.quit()
