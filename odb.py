# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 09:23:07 2023

@author: ruben

Selenium - Chromedriver
-- get latest version from
https://googlechromelabs.github.io/chrome-for-testing/
https://storage.googleapis.com/chrome-for-testing-public/123.0.6312.105/win64/chromedriver-win64.zip
"""

#%%
from selenium import webdriver
#from selenium.webdriver.common.by import By
#from selenium.common.exceptions import StaleElementReferenceException, UnexpectedAlertPresentException
#from selenium.webdriver.firefox.options import Options
import time, os, glob, random
import regex as re
import requests
from tqdm import trange, tqdm_notebook
from bs4 import BeautifulSoup, NavigableString
from pathlib import Path, PurePath
from urllib.parse import urlparse
from selenium.webdriver.support.ui import Select
from datetime import datetime, timedelta, date
import sys
from selenium.webdriver.chrome.service import Service
import winsound

options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features")
options.add_argument("--disable-blink-features=AutomationControlled")

options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

service = Service(executable_path="C:/Python/Scripts/chromedriver.exe")
driver = webdriver.Chrome(options=options, service=service)

#%%

urls = {
        'zh': "https://traditional-odb.org/",
        'en': "https://odb.org/",
        'ja': "https://japanese-odb.org/",
        'es': "https://nuestropandiario.org/"
       }
# Test Chinese website
lang = 'zh'
#driver = webdriver.Chrome()
#driver.get(urls[lang])

#%%
# Define your root folder to store your downloded HTML files
root_folder = Path("D:/NLP/ODB/html")

# Pick your staart and end dates
start_date = datetime.today()
end_date   = datetime(2019, 12, 31)

start_date  = datetime(2010, 10, 18)
start_date  = datetime(2009, 12, 31)
end_date    = datetime(2004, 12, 31)

start_date = datetime.today()
end_date   = start_date - timedelta(days=9600)

# Set "current_date" to "state_date" and go backward in time, one day at a time
current_date = start_date
while True:

    if current_date <= end_date:  # if end_date is reached, break out of the loop and stop
        break

    for lang in ['en', 'zh']:  # ['zh', 'en'] these are the languages I'm interested in
        
        print(f"LANGUAGE: {lang}")
        print('='*70)

        url = urls[lang]  # get URL for that language
        datestr = current_date.strftime('%Y/%m/%d')  # YYYY/MM/DD format
        print(url)
        # Create subfolders YYYY/MM/DD under the root folder
        fon_folder = root_folder / datestr
        fon_folder.mkdir(parents=True, exist_ok=True)

        # set up the file name to write to with downloaded HTML content
        fon = fon_folder / f"{lang}.html"
        url_date = f"{url}{datestr}" # URL for a particular day/date
        if not fon.exists():
            while True:
                driver.get(url_date)
                print(f"Retrieving: {url_date}")
                time.sleep(20)
                html = driver.page_source
                if "sticky-parent" in html and "canonical" in html:
                    break
                else:
                    winsound.Beep(440, 800); winsound.Beep(880, 800); winsound.Beep(880, 800); winsound.Beep(880, 800)
                    print(f"Trying again: {url_date}")

            fon.open('w', encoding='utf-8', newline='\n').write(html)
            print(f"Downloaded: {fon}")
        else:
            print(f"Skipping: {url_date}")

    # subtract one day
    current_date -= timedelta(days=1)
    #time.sleep(4+random.random()*6)

sys.exit(0)
#%%
def getParagraphs(soup):

    para = []
    for p in soup.find('div', {'class': "sticky-parent"}).find('div', {'id': "today-devo"}).find_all('p'):
        if p.text.strip() == '': continue
        para.append(p.text.strip())

    return para

def getTodaysScriptures(soup):
    
    btn_text = None
    bts = soup.find_all('span', {'class': 'btn-text'})
    if bts:
        ts = bts[1].text.strip()
        
    if ts == 'Sign Up For Emails':
        # <div class=""><a class="rtBibleRef" href="https://biblia.com/bible/niv/Acts%209.1%E2%80%9319" data-reference="Acts 9.1–19" data-version="niv" data-purpose="bible-reference" target="_blank">Acts 9:1–19</a></div>
        # <div class="devo-scriptureinsight"><span class="scripture">Today's Scripture  :</span><div><div><div class=""><a class="rtBibleRef" href="https://biblia.com/bible/niv/1%20Thess%203" data-reference="1 Thess 3" data-version="niv" data-purpose="bible-reference" target="_blank">1 Thessalonians 3</a></div></div></div></div>
        # <div class="devo-scriptureinsight"><span class="scripture">Today's Scripture  :</span><div><div><div class="">3 John</div></div></div></div>
        try:
            ts = soup.find('div', {'class': "devo-scriptureinsight"}).find('a', {'class': "rtBibleRef"})['data-reference'].strip()
        except:
            ts = soup.find('div', {'class': "devo-scriptureinsight"}).find('div').text.strip()

    return ts

def getScripturesHighlights(soup):
    '''<p class="PTSerifFont">死被得勝吞滅。—哥林多前書15章54節</p>'''
    highlight = None
    #highlights = soup.find_all('p', {'class': "PTSerifFont"})
    #if highlights:
    #    highlight = highlights[0].text.strip()
    
    highlights = soup.find_all('div', {'class': "verseArea"})
    if highlights:
        highlight = highlights[0].text.strip()
    
    return highlight

def getTitle(soup):
    '''<h1 class="devo-title">人生意義</h1>'''
    
    title = None
    titles = soup.find_all('h1', {'class': "devo-title"})
    if titles:
        title = titles[0].text.strip()
        
    return title
#%%
def getPubDate(soup):
    '''
    <link rel="canonical" href="https://odb.org/2025/04/21/deep-clean" data-react-helmet="true">
    '''
    canonical_links = soup.find_all('link', {'rel': "canonical"})
    if canonical_links:
        pubdate = re.findall(r"\d{4}/\d{2}/\d{2}", canonical_links[0]['href'])
        if pubdate:
            return pubdate[0]
    return None

#%%
def getAuthor(soup):
    '''
    <p class="author devo-author">作者: <span>&nbsp;<a href="https://traditional-odb.org/author/patriciaraybon/">雷翠霞</a></span></p>
    '''
    authors = soup.find_all('p', {'class': "author devo-author"})
    if authors:
        author     = authors[0].find('a').text.strip()
        author_url = authors[0].find('a')['href'].strip()
        return author, author_url
    return None, None
#%%
en_files = sorted(Path(root_folder).rglob("en.html"), reverse=True)
zh_files = sorted(Path(root_folder).rglob("zh.html"), reverse=True)

# There are more en files than zh files
en_cnt = len(en_files)
zh_cnt = len(zh_files)

#%%
cnt = 0
for f in zh_files:
    
    cnt += 1
    if cnt > 15: break  # only the first 15 files are processed
    html = f.read_text(encoding='utf-8')  # read contents of HTML file
    soup = BeautifulSoup(html, 'lxml')    # parse with BeautifulSoup
    
    ## Get scriptures
    ## Sample: <span class="btn-text">哥林多前書15章50-58節</span>
    bts = soup.find_all('span', {'class': 'btn-text'})
    if bts:
        btn_text = bts[1].text
        print(f"[{cnt}] {f}")
        print(f"Scripture: {btn_text}")
        print('-'*50)

    ## Get paragraphs
    ## Sample:
    ## <div class="sticky-parent">
    ## <div id="today-devo">
    for p in soup.find('div', {'class': "sticky-parent"}).find('div', {'id': "today-devo"}).find_all('p'):
        if p.text.strip() == '': continue
        print(p.text)
        print('-'*50)
    print('='*50)
        
#%%
cnt = 0
F, G = len(en_files), len(zh_files)
H = min(F, G)

fone = Path(root_folder) / 'odb.en.txt'
fonz = Path(root_folder) / 'odb.zh.txt'

with open(fone, 'w', encoding='utf-8', newline='\n') as foe, open(fonz, 'w', encoding='utf-8', newline='\n') as foz:

    for f, g in zip(en_files[:H], zh_files[:H]):

        cnt += 1
        #if cnt < 220: continue
        #if cnt > 5: break  # only the first 15 files are processed
        # English
        htmlf = f.read_text(encoding='utf-8')  # read contents of HTML file
        soupf = BeautifulSoup(htmlf, 'lxml')    # parse with BeautifulSoup
        title_en   = getTitle(soupf)
        pubdate_en = getPubDate(soupf)
        if not pubdate_en:
            pubdate_en = '/'.join(f.parent.parts[-3:])
        author_en, author_url_en = getAuthor(soupf)
        ts_en      = getTodaysScriptures(soupf)
        sh_en      = getScripturesHighlights(soupf)
        paras_en   = getParagraphs(soupf)
        foe.write(title_en + '\n')
        foe.write(pubdate_en + '\n')
        foe.write(f"{author_en}|{author_url_en}\n")
        foe.write(ts_en + '\n')
        foe.write(sh_en + '\n')
        foe.write('\n'.join(paras_en) + '\n')
        foe.write('#####\n')
        # Chinese
        htmlg = g.read_text(encoding='utf-8')  # read contents of HTML file
        soupg = BeautifulSoup(htmlg, 'lxml')    # parse with BeautifulSoup
        title_zh   = getTitle(soupg)
        pubdate_zh = getPubDate(soupg)
        if not pubdate_zh:
            pubdate_zh = '/'.join(f.parent.parts[-3:])
        author_zh, author_url_zh = getAuthor(soupg)
        ts_zh      = getTodaysScriptures(soupg)
        sh_zh      = getScripturesHighlights(soupg)
        paras_zh   = getParagraphs(soupg)
        foz.write(title_zh + '\n')
        foz.write(pubdate_zh + '\n')
        foz.write(f"{author_zh}|{author_url_zh}\n")
        foz.write(ts_zh + '\n')
        foz.write(sh_zh + '\n')
        foz.write('\n'.join(paras_zh) + '\n')
        foz.write('#####\n')
        if cnt % 500 == 0:
            print(f"processed {cnt} articles")
            foe.flush()
            foz.flush()
    
#%%
# find files that contain no "canonical"
nocanons = sorted(Path(root_folder).rglob("*.html"), reverse=True)

cnt = 0
bad_cnt = 0
bad_files = []
for f in nocanons:
    cnt += 1
    #if cnt < 200: continue
    txt = f.read_text(encoding='utf-8')
    if txt.count('canonical') < 1:
        #print(f)
        bad_cnt += 1
        bad_files.append(f.parent.parts[-3:])

    if cnt % 50 == 0:
        print(f"processed {cnt} files.")








