"""
Google Sheets Manager
Handles reading from and writing to Google Sheets for order tracking
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from typing import Optional, List, Dict
import logging
import unicodedata

logger = logging.getLogger(__name__)


def remove_vietnamese_accents(text: str) -> str:
    """
    Remove Vietnamese diacritics from text for matching
    Example: "Nguyễn Duy Thái" -> "nguyen duy thai"

    Args:
        text: Text with Vietnamese diacritics

    Returns:
        Text without diacritics, lowercase
    """
    # Normalize unicode to decomposed form (separate base + accent)
    nfd = unicodedata.normalize("NFD", text)
    # Filter out accent marks (combining characters)
    without_accents = "".join(
        char for char in nfd if unicodedata.category(char) != "Mn"
    )

    # Handle special Vietnamese characters that don't decompose
    replacements = {
        "đ": "d",
        "Đ": "d",
        "ð": "d",
        "Ð": "d",
    }
    for viet_char, latin_char in replacements.items():
        without_accents = without_accents.replace(viet_char, latin_char)

    return without_accents.lower().strip()


class SheetsManager:
    """Manages Google Sheets operations for food order tracking"""

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    def __init__(
        self,
        credentials_file: str,
        sheet_id: str,
        sheet_name: str = None,
        auto_detect_month: bool = True,
    ):
        """
        Initialize the Sheets Manager

        Args:
            credentials_file: Path to Google service account JSON file
            sheet_id: Google Sheet ID
            sheet_name: Name of the sheet tab (default: auto-detect based on current month)
            auto_detect_month: If True, automatically detect sheet based on current month (default: True)
        """
        self.credentials_file = credentials_file
        self.sheet_id = sheet_id
        self.sheet_name = sheet_name
        self.auto_detect_month = auto_detect_month
        self.client = None
        self.worksheet = None
        self.spreadsheet = None

    def connect(self):
        """Establish connection to Google Sheets"""
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_file, scopes=self.SCOPES
            )
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(self.sheet_id)

            # Auto-detect sheet name based on current month if enabled
            if self.auto_detect_month and not self.sheet_name:
                self.sheet_name = self._get_sheet_name_for_date(datetime.now())
                logger.info(f"Auto-detected sheet name: {self.sheet_name}")

            self.worksheet = self.spreadsheet.worksheet(self.sheet_name)
            logger.info(
                f"Successfully connected to Google Sheets (sheet: {self.sheet_name})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            return False

    def _get_sheet_name_for_date(self, date: datetime) -> str:
        """Get the sheet name for a given date based on month

        Args:
            date: The date to get the sheet name for

        Returns:
            Sheet name in format "Tháng X" (e.g., "Tháng 1" for January)
        """
        month = date.month
        return f"Tháng {month}"

    def _ensure_correct_worksheet(self, date: datetime) -> bool:
        """Ensure we're connected to the correct worksheet for the given date

        Args:
            date: The date to check

        Returns:
            True if successful, False otherwise
        """
        if not self.auto_detect_month:
            return True

        required_sheet_name = self._get_sheet_name_for_date(date)

        # If we're already on the correct sheet, no need to switch
        if self.worksheet and self.worksheet.title == required_sheet_name:
            return True

        # Switch to the correct sheet
        try:
            self.worksheet = self.spreadsheet.worksheet(required_sheet_name)
            logger.info(f"Switched to worksheet: {required_sheet_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to worksheet '{required_sheet_name}': {e}")
            return False

    def get_column_for_date(self, date: datetime) -> Optional[int]:
        """
        Find the column index for a specific date
        Supports both full date formats and day-only formats (e.g., "22" for day 22)

        Args:
            date: The date to search for

        Returns:
            Column index (1-based) or None if not found
        """
        try:
            # Get the first row (header row with dates)
            header_row = self.worksheet.row_values(1)

            # Get the day of month (e.g., 22 for Jan 22)
            day_number = str(date.day)
            day_padded = f"{date.day:02d}"
            month_number = str(date.month)
            month_padded = f"{date.month:02d}"
            year = str(date.year)

            # Build date formats (Windows-compatible, no %-d)
            date_formats = [
                str(date.day),  # "23"
                day_padded,  # "23"
                f"{date.day}/{date.month}/{date.year}",  # "23/1/2026"
                f"{day_padded}/{month_padded}/{year}",  # "23/01/2026"
                f"{date.month}/{date.day}/{date.year}",  # "1/23/2026"
                f"{month_padded}/{day_padded}/{year}",  # "01/23/2026"
                f"{date.day}/{date.month}",  # "23/1"
                f"{day_padded}/{month_padded}",  # "23/01"
                date.strftime("%d/%m/%Y"),  # "23/01/2026"
                date.strftime("%m/%d/%Y"),  # "01/23/2026"
                date.strftime("%d/%m"),  # "23/01"
            ]

            logger.info(
                f"Searching for date column for day {date.day} (formats: {day_number}, {day_padded})"
            )

            # Search for the date in header row
            for idx, cell_value in enumerate(header_row):
                cell_value = str(cell_value).strip()

                # Skip empty cells
                if not cell_value:
                    continue

                # Direct match with any format
                if cell_value in date_formats:
                    logger.info(
                        f"✓ Found date column at index {idx + 1} with value '{cell_value}'"
                    )
                    return idx + 1  # 1-based index

                # Try to match just the day number
                if cell_value == day_number or cell_value == day_padded:
                    logger.info(
                        f"✓ Found day column at index {idx + 1} with value '{cell_value}'"
                    )
                    return idx + 1

            logger.warning(f"✗ Date column not found for day {date.day}")
            logger.info(f"Available header values: {[v for v in header_row[:20] if v]}")
            return None
        except Exception as e:
            logger.error(f"Error finding date column: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def get_row_for_user(self, user_name: str) -> Optional[int]:
        """
        Find the row index for a specific user
        Supports Vietnamese names with diacritics by removing accents for comparison

        Args:
            user_name: The user's name (Telegram first name)

        Returns:
            Row index (1-based) or None if not found
        """
        try:
            # Get the first column (names column)
            names_column = self.worksheet.col_values(1)

            # Normalize user name (remove accents, lowercase)
            user_name_normalized = remove_vietnamese_accents(user_name)
            logger.info(
                f"Looking for user: '{user_name}' (normalized: '{user_name_normalized}')"
            )

            # Search for user name with multiple matching strategies
            for idx, cell_value in enumerate(names_column):
                if not cell_value or not cell_value.strip():
                    continue

                cell_normalized = remove_vietnamese_accents(cell_value)

                # Strategy 1: Exact match (without accents)
                if cell_normalized == user_name_normalized:
                    logger.info(
                        f"✓ Exact match: '{user_name}' → '{cell_value}' at row {idx + 1}"
                    )
                    return idx + 1

                # Strategy 2: Full name contains the search name
                # e.g., "Duy Thai" matches "Nguyễn Duy Thái"
                if user_name_normalized in cell_normalized:
                    logger.info(
                        f"✓ Partial match: '{user_name}' found in '{cell_value}' at row {idx + 1}"
                    )
                    return idx + 1

                # Strategy 3: Check if all words in user_name are in cell_value
                user_words = user_name_normalized.split()
                cell_words = cell_normalized.split()
                if all(word in cell_words for word in user_words):
                    logger.info(
                        f"✓ Word match: '{user_name}' matches '{cell_value}' at row {idx + 1}"
                    )
                    return idx + 1

            logger.warning(f"✗ User not found in sheet: {user_name}")
            logger.info(
                f"Available names: {[name for name in names_column[:10] if name.strip()]}"
            )
            return None
        except Exception as e:
            logger.error(f"Error finding user row: {e}")
            return None

    def mark_order(
        self, user_name: str, has_order: bool, date: Optional[datetime] = None
    ) -> bool:
        """
        Mark a user's order status for a specific date

        Args:
            user_name: The user's name
            has_order: True to mark as ordered, False to mark as not ordered
            date: The date (defaults to today)

        Returns:
            True if successful, False otherwise
        """
        if date is None:
            date = datetime.now()

        try:
            # Ensure we're on the correct worksheet for this date
            if not self._ensure_correct_worksheet(date):
                logger.error(
                    f"Cannot access worksheet for date {date.strftime('%d/%m/%Y')}"
                )
                return False

            row = self.get_row_for_user(user_name)
            if row is None:
                logger.error(
                    f"Cannot mark order: User '{user_name}' not found in sheet"
                )
                return False

            col = self.get_column_for_date(date)
            if col is None:
                logger.error(
                    f"Cannot mark order: Date column not found for {date.strftime('%d/%m/%Y')}"
                )
                return False

            # Update the cell with TRUE or FALSE
            value = "TRUE" if has_order else "FALSE"
            self.worksheet.update_cell(row, col, value)
            logger.info(
                f"Updated {user_name}'s order to {value} for {date.strftime('%d/%m/%Y')} in sheet '{self.worksheet.title}'"
            )
            return True

        except Exception as e:
            logger.error(f"Error marking order: {e}")
            return False

    def get_order_status(
        self, user_name: str, date: Optional[datetime] = None
    ) -> Optional[bool]:
        """
        Get a user's order status for a specific date

        Args:
            user_name: The user's name
            date: The date (defaults to today)

        Returns:
            True if ordered, False if not ordered, None if not found or error
        """
        if date is None:
            date = datetime.now()

        try:
            row = self.get_row_for_user(user_name)
            if row is None:
                return None

            col = self.get_column_for_date(date)
            if col is None:
                return None

            # Get the cell value
            value = self.worksheet.cell(row, col).value

            if value and value.upper() == "TRUE":
                return True
            elif value and value.upper() == "FALSE":
                return False
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    def get_daily_summary(
        self, date: Optional[datetime] = None
    ) -> List[Dict[str, any]]:
        """
        Get summary of all orders for a specific date

        Args:
            date: The date (defaults to today)

        Returns:
            List of dicts with 'name' and 'has_order' keys
        """
        if date is None:
            date = datetime.now()

        summary = []
        try:
            col = self.get_column_for_date(date)
            if col is None:
                return summary

            # Get all names and order statuses
            names = self.worksheet.col_values(1)[1:]  # Skip header
            order_statuses = self.worksheet.col_values(col)[1:]  # Skip header

            for i, name in enumerate(names):
                if name.strip():  # Skip empty rows
                    has_order = False
                    if i < len(order_statuses) and order_statuses[i].upper() == "TRUE":
                        has_order = True
                    summary.append({"name": name, "has_order": has_order})

            return summary

        except Exception as e:
            logger.error(f"Error getting daily summary: {e}")
            return summary
