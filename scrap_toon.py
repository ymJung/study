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

chrome_driver_path = 'chromedriver.exe' 
options = webdriver.ChromeOptions() 
options.add_argument('--headless')  # (headless 모드)

driver = webdriver.Chrome(service=Service(), options=options)

home = #"https://blacktoon260.com/" 
topic =  #
first_page = #
last_page = #

# 웹 페이지 URL 
url = "https://{}/webtoons/{}/{}.html"
if not os.path.exists(str(topic)):
    os.makedirs(str(topic))
tot_srcs = []
for idx in range(first_page, last_page + 1):
    format_url = url.format(home, topic, idx)
    print('call ', format_url)
    src_urls = get_src_urls(format_url)
    tot_srcs += src_urls
    for src in src_urls: #download
        img_name = src[src.rfind('/')+1:]
        img_path = os.path.join(str(topic), img_name)
        os.system('curl ' + src + ' > ' + str(idx) + '_'+ img_path) 
driver.quit()
    
f = open('srcs.txt', 'w')
for each in tot_srcs: f.write(each + '\n')

