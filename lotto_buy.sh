#!/bin/bash
# 자동구매 스크립트
# 목요일 08:00 실행, 3게임 자동구매

cd /home/ymjung/git/sggul-strategy
python3 << 'PYEOF'
from playwright.sync_api import sync_playwright
import time, random, os

DH_USER = ""
DH_PWD = ""

stealth_script = '''
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){}, app: { isInstalled: false } };
Object.defineProperty(navigator, 'languages', { get: () => ['ko-KR', 'ko', 'en-US'] });
'''

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,
        args=['--no-sandbox','--disable-blink-features=AutomationControlled','--disable-automation']
    )
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport={'width': 1920, 'height': 1080}, locale='ko-KR', timezone_id='Asia/Seoul',
    )
    page = context.new_page()
    page.add_init_script(stealth_script)
    
    # 로그인
    page.goto('https://www.dhlottery.co.kr/login', wait_until='networkidle', timeout=15000)
    time.sleep(0.5)
    page.fill('#inpUserId', ''); [page.keyboard.type(ch, delay=40) for ch in DH_USER]
    time.sleep(0.3)
    page.fill('#inpUserPswdEncn', ''); [page.keyboard.type(ch, delay=40) for ch in DH_PWD]
    time.sleep(0.3)
    page.click('#btnLogin')
    time.sleep(2)
    
    # 구매 페이지
    page.goto('https://ol.dhlottery.co.kr/olotto/game/game645.do', wait_until='networkidle', timeout=15000)
    time.sleep(1)
    
    # 자동번호발급 탭 + 3게임 + 구매
    page.evaluate('''() => {
        selectWayTab(1);
        setTimeout(() => {
            $('#amoundApply').val('3').trigger('change');
            $('#checkAutoSelect').prop('checked', true).prop('disabled', false);
            $('#btnSelectNum').trigger('click');
            setTimeout(() => {
                $('#btnBuy').trigger('click');
            }, 800);
        }, 500);
    }''')
    
    time.sleep(5)
    print("✅ 로또 3장 자동구매 완료!", flush=True)
    browser.close()
PYEOF
