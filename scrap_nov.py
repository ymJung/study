from selenium import webdriver
import os
from selenium.webdriver.chrome.service import Service
import time
from selenium.webdriver.common.by import By

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

def get_txt_from_url(page_url):
    driver.get(page_url)
    #time.sleep(5) # capcha가 나온다면 칠것.
    scroll_end(driver)    
    div_element = driver.find_element(By.ID, 'novel_content') 
    text = div_element.text
    text = text.replace('</p>', '').replace('</br>', '')
    text = text.replace('<p>', '\n').replace('<br>', '\n')
    return text

def get_src_url_from_home(home_url):
    driver.get(home_url)
    scroll_end(driver)    
    tot_src = driver.page_source 
    splits = tot_src.split('<a rel=')
    res = []
    for idx in range(1, len(splits)):
        each = splits[idx]
        a_tag = each[each.find('href="') + 6:each.find('class=')-2]
        res.append(a_tag)
    return res
def mk_dir(d_path):
    if not os.path.exists(d_path):
        os.makedirs(d_path)        
        



chrome_driver_path = 'D:\Dev\chromedriver.exe' 
options = webdriver.ChromeOptions() 
options.add_argument('--headless')  # (headless 모드)

driver = webdriver.Chrome(service=Service(), options=options)

topic = ''
home = 'https://.com/novel/{}'
home_url = home.format(topic)
# 웹 페이지 URL 

mk_dir(topic)

tot_srcs = get_src_url_from_home(home_url)
# 한번 초기화
driver = webdriver.Chrome(service=Service(), options=options)
tot_srcs.reverse()
os.chdir(os.path.join(os.getcwd(), topic))

last_index = 0
start = ''
if last_index > 0: now = last_index 
else : now = 0
print(tot_srcs)
if len(start) > 0 : tot_srcs = tot_srcs[tot_srcs.index(start)+1:] 
print(tot_srcs)

for page_url in tot_srcs:
    now += 1
    print(now , '/', len(tot_srcs), ':', page_url)
    txts = get_txt_from_url(page_url)
    with open(topic + '_' + str(now) + '.txt', 'w') as txt_f:
        txt_f.write(txts)
driver.quit()
    

