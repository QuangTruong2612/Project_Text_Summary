from bs4 import BeautifulSoup
import requests
import urllib.parse

from config import SITE_CONFIGS


class CrawlNews:
    def __init__(self):
        self.news_config = SITE_CONFIGS
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/'
        }

    def crawl(self, url: str) -> dict:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8' 
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            domain = urllib.parse.urlparse(url).netloc
            if domain not in SITE_CONFIGS:
                print(f"Domain {domain} not supported.")
                return None
        
            config = SITE_CONFIGS[domain]

            # ========== GET TITLE ==========
            title = soup.select_one(config['title']).get_text(strip=True) if soup.select_one(config['title']) else ''

            # ========== GET PUBLISHED DATE ==========
            published_date = soup.select_one(config['published-date']).get_text(strip=True) if soup.select_one(config['published-date']) else ''

            # ========== GET DESCRIPTION ==========
            description_tag = soup.select_one(config['description'])
            description = description_tag.get_text(strip=True) if description_tag else ''

            # ========== GET LOCATION ==========
            if 'location' in config:
                location_tag = soup.select_one(config['location'])
                location = location_tag.get_text(strip=True) if location_tag else ''
                description = description.replace(location, '').strip() if location else description
            else:
                location = ''

            # ========== GET CONTENT (Đã sửa để lấy nhiều thẻ p.Normal) ==========
            content_tags = soup.select(config['content']) # Sử dụng select() thay vì select_one()
            content_text = ""
            if content_tags:
                texts = []
                for tag in content_tags:
                    # Xóa các tag rác bên trong mỗi đoạn
                    for trash in tag.select('table, figure, div.z-news-mini, .more-news'):
                        trash.decompose()
                    
                    # Giữ lại text trong thẻ <a>
                    for a in tag.find_all('a'):
                        a.unwrap()
                    
                    texts.append(tag.get_text(strip=True))
                
                # Nối các đoạn văn lại với nhau
                content_text = "\n".join(texts)
                
            return {
                'title': title,
                'location': location,
                'published_date': published_date,
                'description': description,
                'content': content_text
            }
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
