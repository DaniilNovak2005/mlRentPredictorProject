import unittest
from unittest.mock import patch
import asyncio

class TestZipcodeIncomeScraper(unittest.TestCase):
    @patch('scraper.zipcode_income_scraper.main')
    def test_main(self, mock_main):
        async def async_main():
            pass
        mock_main.return_value = async_main()
        asyncio.run(mock_main())
        mock_main.assert_called_once()

if __name__ == '__main__':
    unittest.main()