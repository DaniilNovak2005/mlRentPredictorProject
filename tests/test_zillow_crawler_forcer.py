import unittest
from unittest.mock import patch
from scraper import zillow_crawler_forcer

class TestZillowCrawlerForcer(unittest.TestCase):
    @patch('scraper.zillow_crawler_forcer.main')
    def test_main(self, mock_main):
        zillow_crawler_forcer.main()
        mock_main.assert_called_once()

if __name__ == '__main__':
    unittest.main()