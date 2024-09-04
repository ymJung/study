from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
import time

# 웹 드라이버 설정
driver = webdriver.Chrome()

# 웹페이지 열기
url = "https://m.yes24.com/Momo/Templates/FTLogin.aspx?ReturnURL=http://m.ticket.yes24.com/Perf/Detail/PerfInfo.aspx?IdPerf=45476"
driver.get(url)


id_input = driver.find_element(By.ID, 'SMemberID')
id_input.send_keys('')

pwd_input = driver.find_element(By.ID, 'SMemberPassword')
pwd_input.send_keys('')
# time.sleep(5)
driver.find_element(By.ID, 'btn_login').click()
time.sleep(10)
actions = ActionChains(driver)
actions.move_by_offset(255, 303).click().perform() # 예매
time.sleep(2)
actions.move_by_offset(874, 299).click().perform() # 날짜
time.sleep(10)
actions.move_by_offset(57, 624).click().perform() # 시간
time.sleep(10)
actions.move_by_offset(103, 73).click().perform() # 구역

screenshot = driver.get_screenshot_as_png()
target_color = (0, 204, 204)
img = Image.open(BytesIO(screenshot))

pixels = img.load()
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if pixels[i, j] == target_color:
            print(f"Found target color at ({i}, {j})")


# 브라우저 닫기 0 204 204
# driver.quit()

