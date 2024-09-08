

SRT_MOBILE = "https://app.srail.or.kr:443"
API_ENDPOINTS = {
    "main": f"{SRT_MOBILE}/main/main.do",
    "login": f"{SRT_MOBILE}/apb/selectListApb01080_n.do",
    "logout": f"{SRT_MOBILE}/login/loginOut.do",
    "search_schedule": f"{SRT_MOBILE}/ara/selectListAra10007_n.do",
    "reserve": f"{SRT_MOBILE}/arc/selectListArc05013_n.do",
    "tickets": f"{SRT_MOBILE}/atc/selectListAtc14016_n.do",
    "ticket_info": f"{SRT_MOBILE}/ard/selectListArd02017_n.do?",
    "cancel": f"{SRT_MOBILE}/ard/selectListArd02045_n.do",
    "standby_option": f"{SRT_MOBILE}/ata/selectListAta01135_n.do",
    "payment": f"{SRT_MOBILE}/ata/selectListAta09036_n.do",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 5.1.1; LGM-V300K Build/N2G47H) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Version/4.0 Chrome/39.0.0.0 Mobile Safari/537.36SRT-APP-Android V.1.0.6"
    ),
    "Accept": "application/json",
}

import requests
import json
import datetime
import random
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def login(session, srt_id, srt_pw):
    url = API_ENDPOINTS["login"]
    data = {
            "auto": "Y",
            "check": "Y",
            "page": "menu",
            "deviceKey": "-",
            "customerYn": "",
            "login_referer": API_ENDPOINTS["main"],
            "srchDvCd": "1", # membership type
            "srchDvNm": srt_id,
            "hmpgPwdCphd": srt_pw,
        }
    r = session.post(url=url, data=data, verify=False)
    membership_number = json.loads(r.text).get("userMap").get("MB_CRD_NO")
    return membership_number


def search_train(session, 
                 dep_code, # 출발역
                 arr_code, # 도착역
                 date, # 출발 날짜 (yyyyMMdd)
                 time, # 출발 시각 (hhmmss)
                 ):    
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    if time is None:
        time = "000000"
    url = API_ENDPOINTS["search_schedule"]
    data = {
            "chtnDvCd": "1",
            "arriveTime": "N",
            "seatAttCd": "015",
            "psgNum": 1, # passenger number
            "trnGpCd": 109,
            "stlbTrnClsfCd": "17",
            "dptDt": date,
            "dptTm": time,
            "arvRsStnCd": arr_code,
            "dptRsStnCd": dep_code,
        }
    r = session.post(url=url, data=data)
    res = r.json()
    return res

def filter(res):
    output = res['outDataSets']['dsOutput1']
    
    results = []
    for each in output:
        state = each['gnrmRsvPsbCdNm']
        if "예약" in state:
            time = int(each['dptTm'][:2])
            if time < 18:
                results.append({'time':each['dptTm'], 'state': state})
    return results


def post_message(token, channel, text):    
    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': token,
        'channel': channel,
        'text': text
    }).json()


o_auth = ''
srt_id = '' #membership id
srt_pw = '!' # pwd
channel = ''
start = ''
dest = ''

def main():
    session = requests.session()
    session.headers.update(DEFAULT_HEADERS)

    login(session, srt_id, srt_pw)
    while (True) :
        res = search_train(session, start, dest, "20240908", "130000")
        results = filter(res)
        if len(results) > 0:
            post_message(o_auth, channel, '확인필요'+ str(results))
            break
        else:
            sleepsec = random.randint(30,60)
            print("예약이 없어요 ㅠㅠ" , sleepsec , "만큼 쉽니다")
            time.sleep(sleepsec)
        
main()





