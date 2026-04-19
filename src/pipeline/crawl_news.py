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

    def _get_text(self, soup: BeautifulSoup, selector: str) -> str:
        """Trả về text của selector đầu tiên tìm được, hoặc chuỗi rỗng."""
        tag = soup.select_one(selector)
        return tag.get_text(strip=True) if tag else ''

    def crawl(self, url: str) -> dict | None:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8'
            if response.status_code != 200:
                return None

            domain = urllib.parse.urlparse(url).netloc
            if domain not in SITE_CONFIGS:
                print(f"Domain '{domain}' not supported.")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            config = SITE_CONFIGS[domain]

            # ========== TITLE ==========
            title = self._get_text(soup, config['title'])

            # ========== CATEGORY ==========
            category = self._get_text(soup, config['category'])

            # ========== DESCRIPTION ==========
            description = self._get_text(soup, config['description'])

            # ========== CONTENT ==========
            content_tags = soup.select(config['content'])
            content_text = ''
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
                content_text = '\n'.join(texts)

            return {
                'title': title,
                'category': category,
                'description': description,
                'content': content_text,
            }

        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
