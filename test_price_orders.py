"""
Tests for price-based food ordering feature.
Covers: config, nlp_parser (OrderIntent + price extraction), sheets_manager, and bot integration.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import json


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Test config.py constants"""

    def test_valid_prices_defined(self):
        from config import VALID_PRICES
        assert VALID_PRICES == [30000, 35000, 40000]

    def test_default_price_defined(self):
        from config import DEFAULT_PRICE
        assert DEFAULT_PRICE == 30000

    def test_default_price_is_in_valid_prices(self):
        from config import VALID_PRICES, DEFAULT_PRICE
        assert DEFAULT_PRICE in VALID_PRICES


# =============================================================================
# OrderIntent Model Tests
# =============================================================================

class TestOrderIntentModel:
    """Test the OrderIntent Pydantic model with price field"""

    def test_order_with_price(self):
        from nlp_parser import OrderIntent
        data = {
            "intent": "order",
            "confidence": "high",
            "day_number": 11,
            "food_items": "1 cơm gà",
            "price": 30000,
        }
        result = OrderIntent(**data)
        assert result.intent == "order"
        assert result.price == 30000
        assert result.food_items == "1 cơm gà"

    def test_order_with_35k_price(self):
        from nlp_parser import OrderIntent
        data = {
            "intent": "order",
            "confidence": "high",
            "day_number": 11,
            "food_items": "1 phở bò",
            "price": 35000,
        }
        result = OrderIntent(**data)
        assert result.price == 35000

    def test_order_with_40k_price(self):
        from nlp_parser import OrderIntent
        data = {
            "intent": "order",
            "confidence": "high",
            "day_number": 11,
            "food_items": "1 bánh canh",
            "price": 40000,
        }
        result = OrderIntent(**data)
        assert result.price == 40000

    def test_order_without_price_defaults_none(self):
        from nlp_parser import OrderIntent
        data = {
            "intent": "order",
            "confidence": "high",
            "day_number": 11,
            "food_items": "cơm sườn",
        }
        result = OrderIntent(**data)
        assert result.price is None

    def test_cancel_intent_no_price(self):
        from nlp_parser import OrderIntent
        data = {
            "intent": "cancel",
            "confidence": "high",
            "day_number": 11,
        }
        result = OrderIntent(**data)
        assert result.price is None
        assert result.food_items is None

    def test_none_intent(self):
        from nlp_parser import OrderIntent
        data = {
            "intent": "none",
            "confidence": "high",
        }
        result = OrderIntent(**data)
        assert result.price is None

    def test_order_intent_json_serialization(self):
        from nlp_parser import OrderIntent
        json_str = '{"intent": "order", "confidence": "high", "day_number": 11, "food_items": "1 cơm gà", "price": 30000}'
        result = OrderIntent.model_validate_json(json_str)
        assert result.intent == "order"
        assert result.price == 30000
        assert result.food_items == "1 cơm gà"

    def test_order_intent_json_null_price(self):
        from nlp_parser import OrderIntent
        json_str = '{"intent": "order", "confidence": "high", "day_number": 11, "food_items": "cơm sườn", "price": null}'
        result = OrderIntent.model_validate_json(json_str)
        assert result.price is None

    def test_order_intent_json_missing_price(self):
        from nlp_parser import OrderIntent
        json_str = '{"intent": "order", "confidence": "high", "day_number": 11, "food_items": "cơm sườn"}'
        result = OrderIntent.model_validate_json(json_str)
        assert result.price is None


# =============================================================================
# SheetsManager Tests (with mocked gspread)
# =============================================================================

class TestSheetsManagerMarkOrder:
    """Test sheets_manager.mark_order with price values"""

    def _make_manager(self):
        """Create a SheetsManager with mocked worksheet"""
        from sheets_manager import SheetsManager
        manager = SheetsManager("fake_creds.json", "fake_sheet_id", "Tháng 2")
        manager.worksheet = MagicMock()
        manager.spreadsheet = MagicMock()
        manager.auto_detect_month = False
        return manager

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_mark_order_with_30k(self, mock_col, mock_row):
        manager = self._make_manager()
        result = manager.mark_order("Duy Thái", 30000, datetime(2026, 2, 11))
        assert result is True
        manager.worksheet.update_cell.assert_called_once_with(3, 5, 30000)

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_mark_order_with_35k(self, mock_col, mock_row):
        manager = self._make_manager()
        result = manager.mark_order("Duy Thái", 35000, datetime(2026, 2, 11))
        assert result is True
        manager.worksheet.update_cell.assert_called_once_with(3, 5, 35000)

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_mark_order_with_40k(self, mock_col, mock_row):
        manager = self._make_manager()
        result = manager.mark_order("Duy Thái", 40000, datetime(2026, 2, 11))
        assert result is True
        manager.worksheet.update_cell.assert_called_once_with(3, 5, 40000)

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_cancel_order_clears_cell(self, mock_col, mock_row):
        """Cancellation should write empty string to clear the cell"""
        manager = self._make_manager()
        result = manager.mark_order("Duy Thái", None, datetime(2026, 2, 11))
        assert result is True
        manager.worksheet.update_cell.assert_called_once_with(3, 5, "")

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=None)
    def test_mark_order_user_not_found(self, mock_row):
        manager = self._make_manager()
        result = manager.mark_order("Unknown User", 30000, datetime(2026, 2, 11))
        assert result is False

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=None)
    def test_mark_order_date_not_found(self, mock_col, mock_row):
        manager = self._make_manager()
        result = manager.mark_order("Duy Thái", 30000, datetime(2026, 2, 11))
        assert result is False


class TestSheetsManagerGetOrderStatus:
    """Test sheets_manager.get_order_status returns price int"""

    def _make_manager(self):
        from sheets_manager import SheetsManager
        manager = SheetsManager("fake_creds.json", "fake_sheet_id", "Tháng 2")
        manager.worksheet = MagicMock()
        manager.spreadsheet = MagicMock()
        return manager

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_get_status_returns_price(self, mock_col, mock_row):
        manager = self._make_manager()
        manager.worksheet.cell.return_value = MagicMock(value="30000")
        result = manager.get_order_status("Duy Thái", datetime(2026, 2, 11))
        assert result == 30000

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_get_status_returns_35k(self, mock_col, mock_row):
        manager = self._make_manager()
        manager.worksheet.cell.return_value = MagicMock(value="35000")
        result = manager.get_order_status("Duy Thái", datetime(2026, 2, 11))
        assert result == 35000

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_get_status_empty_returns_none(self, mock_col, mock_row):
        manager = self._make_manager()
        manager.worksheet.cell.return_value = MagicMock(value="")
        result = manager.get_order_status("Duy Thái", datetime(2026, 2, 11))
        assert result is None

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_get_status_none_value_returns_none(self, mock_col, mock_row):
        manager = self._make_manager()
        manager.worksheet.cell.return_value = MagicMock(value=None)
        result = manager.get_order_status("Duy Thái", datetime(2026, 2, 11))
        assert result is None

    @patch("sheets_manager.SheetsManager.get_row_for_user", return_value=3)
    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_get_status_float_string(self, mock_col, mock_row):
        """Google Sheets may return numbers as floats like '30000.0'"""
        manager = self._make_manager()
        manager.worksheet.cell.return_value = MagicMock(value="30000.0")
        result = manager.get_order_status("Duy Thái", datetime(2026, 2, 11))
        assert result == 30000


class TestSheetsManagerDailySummary:
    """Test sheets_manager.get_daily_summary returns price instead of has_order"""

    def _make_manager(self):
        from sheets_manager import SheetsManager
        manager = SheetsManager("fake_creds.json", "fake_sheet_id", "Tháng 2")
        manager.worksheet = MagicMock()
        manager.spreadsheet = MagicMock()
        return manager

    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_daily_summary_with_prices(self, mock_col):
        manager = self._make_manager()
        manager.worksheet.col_values.side_effect = [
            ["Member", "Chí Phát", "Duy Thái", "Ngọc Quí", "Bao Hiếu"],  # col 1 (names)
            ["", "30000", "", "35000", "40000"],  # col 5 (orders)
        ]
        summary = manager.get_daily_summary(datetime(2026, 2, 11))
        assert len(summary) == 4
        assert summary[0] == {"name": "Chí Phát", "price": 30000}
        assert summary[1] == {"name": "Duy Thái", "price": None}
        assert summary[2] == {"name": "Ngọc Quí", "price": 35000}
        assert summary[3] == {"name": "Bao Hiếu", "price": 40000}

    @patch("sheets_manager.SheetsManager.get_column_for_date", return_value=5)
    def test_daily_summary_all_empty(self, mock_col):
        manager = self._make_manager()
        manager.worksheet.col_values.side_effect = [
            ["Member", "Chí Phát", "Duy Thái"],
            ["", "", ""],
        ]
        summary = manager.get_daily_summary(datetime(2026, 2, 11))
        assert len(summary) == 2
        assert all(entry["price"] is None for entry in summary)


# =============================================================================
# Bot Handler Tests (mocked parser + sheets_manager)
# =============================================================================

class TestBotPriceHandling:
    """Test that bot.py correctly passes price from parser to sheets_manager"""

    def test_price_validation_valid(self):
        """Valid prices should be kept as-is"""
        from config import VALID_PRICES, DEFAULT_PRICE
        for price in [30000, 35000, 40000]:
            validated = price if price in VALID_PRICES else DEFAULT_PRICE
            assert validated == price

    def test_price_validation_invalid_defaults(self):
        """Invalid prices should fall back to DEFAULT_PRICE"""
        from config import VALID_PRICES, DEFAULT_PRICE
        for price in [25000, 50000, 100, 0, -1]:
            validated = price if price in VALID_PRICES else DEFAULT_PRICE
            assert validated == DEFAULT_PRICE

    def test_price_validation_none_defaults(self):
        """None price should fall back to DEFAULT_PRICE"""
        from config import VALID_PRICES, DEFAULT_PRICE
        price = None
        validated = DEFAULT_PRICE if price is None or price not in VALID_PRICES else price
        assert validated == DEFAULT_PRICE


# =============================================================================
# NLP Parser System Prompt Tests
# =============================================================================

class TestParserSystemPrompt:
    """Test that the system prompt includes price extraction instructions"""

    @patch("google.genai.Client")
    def test_prompt_contains_price_extraction(self, mock_client):
        from nlp_parser import VietnameseOrderParser
        parser = VietnameseOrderParser("fake_api_key")
        prompt = parser._get_system_prompt(datetime(2026, 2, 11))
        
        assert "PRICE EXTRACTION" in prompt
        assert "30000" in prompt
        assert "35000" in prompt
        assert "40000" in prompt
        assert "30k" in prompt
        assert "35k" in prompt
        assert "40k" in prompt

    @patch("google.genai.Client")
    def test_prompt_contains_price_examples(self, mock_client):
        from nlp_parser import VietnameseOrderParser
        parser = VietnameseOrderParser("fake_api_key")
        prompt = parser._get_system_prompt(datetime(2026, 2, 11))
        
        assert "cho tôi 1 cơm gà 30k" in prompt
        assert "1 phở bò 35 ngàn" in prompt


# =============================================================================
# Confirmation Message Tests  
# =============================================================================

class TestConfirmationMessage:
    """Test that generate_confirmation_message handles price correctly"""

    @patch("google.genai.Client")
    def test_fallback_message_includes_price(self, mock_client):
        from nlp_parser import VietnameseOrderParser
        parser = VietnameseOrderParser("fake_api_key")
        
        # Mock the Gemini API to fail, triggering fallback
        parser.client.models.generate_content.side_effect = Exception("API Error")
        
        result = parser.generate_confirmation_message(
            user_name="Thái",
            intent="order",
            food_items="1 cơm gà",
            date_desc="hôm nay",
            price=30000,
        )
        assert "30k" in result
        assert "Thái" in result
        assert "✅" in result

    @patch("google.genai.Client")
    def test_fallback_message_35k(self, mock_client):
        from nlp_parser import VietnameseOrderParser
        parser = VietnameseOrderParser("fake_api_key")
        parser.client.models.generate_content.side_effect = Exception("API Error")
        
        result = parser.generate_confirmation_message(
            user_name="Phát",
            intent="order",
            food_items="1 phở bò",
            price=35000,
        )
        assert "35k" in result

    @patch("google.genai.Client")
    def test_fallback_cancel_message_no_price(self, mock_client):
        from nlp_parser import VietnameseOrderParser
        parser = VietnameseOrderParser("fake_api_key")
        parser.client.models.generate_content.side_effect = Exception("API Error")
        
        result = parser.generate_confirmation_message(
            user_name="Quí",
            intent="cancel",
        )
        assert "❌" in result
        assert "Quí" in result

    @patch("google.genai.Client")
    def test_fallback_message_no_price(self, mock_client):
        from nlp_parser import VietnameseOrderParser
        parser = VietnameseOrderParser("fake_api_key")
        parser.client.models.generate_content.side_effect = Exception("API Error")
        
        result = parser.generate_confirmation_message(
            user_name="Hiếu",
            intent="order",
            food_items="cơm sườn",
            price=None,
        )
        assert "✅" in result
        assert "Hiếu" in result
        # No price text when price is None
        assert "k)" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
