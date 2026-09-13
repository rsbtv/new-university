import re
import requests
from bs4 import BeautifulSoup
import csv

URL = "https://msk.spravker.ru/avtoservisy-avtotehcentry/"

def get_html(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.encoding = "utf-8"
    resp.raise_for_status()
    return resp.text

def clean_html_tags(text):
    # удаляем теги hmtl
    return re.sub(r'<[^>]+>', '', text)

def regex_parse_block(block_html):
    # название
    name_match = re.search(
        r'<a[^>]*class="\[?[\w\-]*org-widget-header__title-link[\w\-]*\]?"[^>]*>(.*?)</a>',
        block_html, re.DOTALL | re.IGNORECASE
    )
    name = name_match.group(1).strip() if name_match else ""

    # адрес
    addr_match = re.search(
        r'<span[^>]*class="\[?[\w\- ]*org-widget-header__meta--location[\w\- ]*\]?"[^>]*>(.*?)</span>',
        block_html, re.DOTALL | re.IGNORECASE
    )
    address = addr_match.group(1).strip() if addr_match else ""

    # телефон
    phone_match = re.search(
        r'<dt[^>]*>\s*<span[^>]*>\s*Телефон\s*</span>\s*</dt>\s*<dd[^>]*>(.*?)</dd>',
        block_html, re.DOTALL | re.IGNORECASE
    )
    phone = phone_match.group(1).strip() if phone_match else ""

    #часы работы
    hours_match = re.search(
        r'<dt[^>]*>\s*<span[^>]*>\s*Часы\s*работы\s*</span>\s*</dt>\s*<dd[^>]*>(.*?)</dd>',
        block_html, re.DOTALL | re.IGNORECASE
    )
    hours = hours_match.group(1).strip() if hours_match else ""

    name    = clean_html_tags(name)
    address = clean_html_tags(address)
    phone   = clean_html_tags(phone)
    hours   = clean_html_tags(hours)

    return {"Название": name, "Адрес": address, "Телефон": phone, "Часы работы": hours}

def parse_services(html):
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all("div", class_="org-widget")
    services = []
    for block in blocks:
        block_html = str(block)
        service_data = regex_parse_block(block_html)
        services.append(service_data)
    return services

def save_to_csv(services, filename="autocenters_regex.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Название", "Адрес", "Телефон", "Часы работы"], delimiter=";")
        writer.writeheader()
        writer.writerows(services)

def main():
    html = get_html(URL)
    services = parse_services(html)
    for s in services:
        print(s)
    save_to_csv(services)

main()