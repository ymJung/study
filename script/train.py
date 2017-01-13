import os
import sys
import time
from bs4 import BeautifulSoup
import random
from slacker import Slacker

#slackApiKey.txt < msg를 받을 slack message api
#trainCurl.txt < 일반승차권 조회 curl 

slack = Slacker(open('slackApiKey.txt','r').read().strip())
curlTxt = open('trainCurl.txt', 'r').read().strip()


def check_ticket(curlTxt):
    result = os.popen(curlTxt).read()
    soup = BeautifulSoup(result, 'html.parser')
    html = soup.prettify()
    statusTexts = list()
    for img in soup.find('table').find_all('img'):
        try :
            statusTexts.append(img.attrs['alt'])
        except KeyError:
            continue
    print('length : ', len(statusTexts))
    for statusText in statusTexts:
        if "예약하기" == statusText:
            print(statusText)
            return True
    return False
def send_slack_msg(msg):
    slack.chat.post_message('@zero', msg)


def main():
    while True:
        if(check_ticket(curlTxt)) :
            now = time.localtime()
            msg = '떳다 예약 !' , now.tm_hour , now.tm_min , now.tm_sec
            print(msg)
            send_slack_msg(msg)
        else :
            sleepsec = random.randint(10,25)
            print("예약이 없어요 ㅠㅠ" , sleepsec , "만큼 쉽니다")
            time.sleep(sleepsec)
            
            
try :   
    send_slack_msg("시작합니데이!")
    print('시작합니다')
    main()
except :
    send_slack_msg('오류났서예')
    print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
