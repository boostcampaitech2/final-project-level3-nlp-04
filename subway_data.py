import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import tqdm
import re

# 지하철 역 정보 가져오는 함수
def subway_crawling():
    subway_url = 'http://www.seoulmetro.co.kr/kr/cyberStation.do'
    subway_driver = webdriver.Chrome('./chromedriver')
    subway_driver.get(subway_url)
    response = requests.get(subway_url)

    if response.status_code == 200:
        html = subway_driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        station_info_list = soup.find_all(class_='mapInfo')
        station_info_dict = dict()
        station_name_tag = station_info_list[0].find_all('span')

        for index in range(len(station_name_tag)):
            station_name = station_info_list[0].find_all('span')[index].get_text()
            station_info = station_info_list[0].find_all('li')[index].find('div').get_text()
            station_info_dict[station_name] = station_info

        df = pd.DataFrame.from_dict(station_info_dict, orient='index')
        df.rename(columns={0 : 'station_info'}, inplace=True)
        df.to_csv('subway_data.csv', encoding='utf-8')
    else :
        print(response.status_code)

if __name__ == '__main__':
    subway_crawling()