from selenium import webdriver
import os
from selenium.webdriver.chrome.service import Service
import time

def scroll_end(driver):
    scroll_pause_time = 0.3  # 스크롤 간격 설정 (초)
    screen_height = driver.execute_script("return window.screen.height;")  # 화면 높이 가져오기
    i = 1
    while True:
        # 현재 스크롤 위치 가져오기
        scroll_height = driver.execute_script("return document.body.scrollHeight;")        
        # 스크롤을 페이지의 끝까지 내리기
        driver.execute_script(f"window.scrollTo(0, {i * screen_height});")
        # 잠시 기다리기
        time.sleep(scroll_pause_time)
        # 스크롤이 끝에 도달하면 종료
        if i * screen_height > scroll_height:
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
        img_url = each[each.find('"') + 1:each.find('jpg')] + 'jpg'        
        res.append(img_url)
    return res

def get_src_url_from_home(home_url):
    driver.get(home_url)
    scroll_end(driver)    
    tot_src = driver.page_source 
    splits = tot_src.split('<a href="/webtoons')
    res = []
    for idx in range(1, len(splits) -1):
        each = splits[idx]
        a_tag = each[each.find('<a href="') + 1:each.find('html')] + 'html'        
        res.append(a_tag)
    return res





chrome_driver_path = 'chromedriver.exe' 
options = webdriver.ChromeOptions() 
options.add_argument('--headless')  # (headless 모드)

driver = webdriver.Chrome(service=Service(), options=options)

home = ''# 
topic =  '' #
# 웹 페이지 URL 
main_url = "https://{}/webtoons"
home_url = "https://{}/webtoon/{}.html".format(home, topic)

if not os.path.exists(str(topic)):
    os.makedirs(str(topic))
tot_srcs = get_src_url_from_home(home_url)
# 한번 초기화
driver = webdriver.Chrome(service=Service(), options=options)
idx = 0
for page_url in tot_srcs:
    idx+=1
    format_url = main_url.format(home) + page_url
    print('call ', format_url)
    src_urls = get_src_urls(format_url)
    tot_srcs += src_urls
    for img_src in src_urls: #download
        img_name = img_src[img_src.rfind('/')+1:]
        img_path = os.path.join(str(topic), img_name)
        os.system('curl ' + img_src + ' > ' + str(idx) + '_'+ img_path) 
driver.quit()
    
f = open('srcs.txt', 'w')
for each in tot_srcs: f.write(each + '\n')

