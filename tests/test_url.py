from abstract_webtools import *
from abstract_utilities import *
import re
import json
import requests
import urllib.parse
from typing import Tuple, List, Dict, Any
from pathlib import Path

approved_headers = [{
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Maxthon/4.4.6.1000 Chrome/30.0.1599.101 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Sec-CH-UA": "\"Chromium\";v=\"116\", \"Brave\";v=\"116\", \"Not A;Brand\";v=\"99\"",
            "Sec-CH-UA-Platform": "macOS",
            "Sec-CH-UA-Mobile": "?0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:39.0) Gecko/20100101 Firefox/39.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8,es;q=0.6",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Sec-CH-UA": "\"Chromium\";v=\"116\", \"Brave\";v=\"116\", \"Not A;Brand\";v=\"99\"",
            "Sec-CH-UA-Platform": "macOS",
            "Sec-CH-UA-Mobile": "?0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2503.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Sec-CH-UA": "\"Chromium\";v=\"116\", \"Brave\";v=\"116\", \"Not A;Brand\";v=\"99\"",
            "Sec-CH-UA-Platform": "Windows",
            "Sec-CH-UA-Mobile": "?0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:35.0) Gecko/20100101 Firefox/35.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8,es;q=0.6",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Sec-CH-UA": "\"Chromium\";v=\"118\", \"Google Chrome\";v=\"118\", \"Not A;Brand\";v=\"99\"",
            "Sec-CH-UA-Platform": "macOS",
            "Sec-CH-UA-Mobile": "?0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8,es;q=0.6",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "close",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Sec-CH-UA": "\"Chromium\";v=\"116\", \"Brave\";v=\"116\", \"Not A;Brand\";v=\"99\"",
            "Sec-CH-UA-Platform": "macOS",
            "Sec-CH-UA-Mobile": "?0"
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:36.0) Gecko/20100101 Firefox/36.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "close",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Sec-CH-UA": "\"Chromium\";v=\"120\", \"Microsoft Edge\";v=\"120\", \"Not A;Brand\";v=\"99\"",
            "Sec-CH-UA-Platform": "Windows",
            "Sec-CH-UA-Mobile": "?0"
        }
]
WEBSITE = "https://thedailydialectics.com/pdfs/FIOA/41786769082578/"
USER_AGENT = approved_headers[3]
S = requests.Session()
S.headers.update({"User-Agent": USER_AGENT})
def get_session(url):
    session = requests.Session()
    session.headers.update(USER_AGENT)
    return session
def fetch_html(url,session = None):
    session = session or get_session(url)
    response = session.get(url)
    return response.text
def get_source_code(url,session = None):
    session = session or get_session(url)
    return fetch_html(url=url,session=session)
def get_soup(url,session = None):
    source_code = get_source_code(url,session = session)
    return BeautifulSoup(source_code, "html.parser")
def get_soup_text(url,session = None):
    source_soup = get_soup(url,session = session)
    return source_soup.text
def get_headers(url,session = None):
    source_soup = get_soup(url,session = session)
def get_all_js(url=None,html=None):
    if not html:
        html = get_html(url)
    all_js = []
    for line in html.split('.js'):
        all_js.append(f"""{line.split('"')[-1]}.js""")
    return all_js

