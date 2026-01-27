"""
Vietnamese Natural Language Processing for Food Orders
Uses Google Gemini API to detect order and cancellation intents from Vietnamese chat messages
"""

import google.genai as genai
from google.genai import types
from typing import Optional, Literal
from pydantic import BaseModel, Field
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OrderIntent(BaseModel):
    """Structured output for order intent classification"""

    intent: Literal["order", "cancel", "none"] = Field(
        description="The detected intent: 'order' for placing an order, 'cancel' for cancelling an order, 'none' for unrelated messages"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the classification"
    )
    day_number: Optional[int] = Field(
        default=None,
        description="Specific day of month mentioned (1-31), or null if not specified",
    )


class VietnameseOrderParser:
    """Parser for detecting Vietnamese food order intents using Gemini AI with structured outputs"""

    def __init__(self, api_key: str):
        """
        Initialize the parser with Gemini API using structured outputs

        Args:
            api_key: Google Gemini API key
        """
        # Get current date for context
        current_date = datetime.now()
        current_day = current_date.day
        current_month = current_date.month
        current_year = current_date.year

        self.SYSTEM_PROMPT = f"""You are a strict Vietnamese food order intent classifier. 

**CURRENT DATE CONTEXT:**
Today is day {current_day}, month {current_month}, year {current_year}.

**CRITICAL RULES:**
1. ONLY classify as "order" if the speaker is DIRECTLY placing or reporting THEIR OWN food order (present or past tense)
2. ONLY classify as "cancel" if the speaker is DIRECTLY cancelling or reporting they are NOT eating (present or past tense)
3. Classify as "none" for ALL other cases including:
   - Questions to others
   - Conditionals or future plans
   - Requests asking SOMEONE ELSE to order (e.g., "giùm", "nhớ đặt", "đặt giùm")
   - Imperatives/commands to others (e.g., "Remember to...", "Nhớ...")
   - Single words without clear context (e.g., just "đặt" or "ăn")
   - Ambiguous statements

**IMPORTANT - REJECT THESE PATTERNS:**
- "giùm" or "hộ" (asking someone else to do something for you) → "none"
- "nhớ" + order/eat (reminding someone else) → "none"
- Single word messages like "đặt", "ăn", "order" → "none"
- Any imperative directed at others → "none"

**DAY NUMBER DETECTION:**
When a message mentions a specific day number (e.g., "ngày 20", "day 15"), extract it as day_number.
Examples:
- "ngày 20 tôi có đặt" → day_number: 20
- "ngày 15 tui không ăn" → day_number: 15
- "hôm qua" (yesterday) → day_number: {current_day - 1}
- "hôm nay" (today) or no date mentioned → day_number: {current_day}

**Examples of "order":**
- "Đặt cho tui 1 cơm gà" → intent: "order", day_number: {current_day}
- "Ngày 20 tui có đặt" → intent: "order", day_number: 20
- "Hôm qua tui có đặt" → intent: "order", day_number: {current_day - 1}
- "ngày 20 tôi có đặt 1 phần cơm đó" → intent: "order", day_number: 20
- "Tui đặt cơm sườn" → intent: "order", day_number: {current_day}

**Examples of "cancel":**
- "Hủy order của tui" → intent: "cancel", day_number: {current_day}
- "Ngày 15 tui không ăn" → intent: "cancel", day_number: 15
- "Hôm qua tui kh có ăn á" → intent: "cancel", day_number: {current_day - 1}
- "Tui k ăn" → intent: "cancel", day_number: {current_day}

**Examples of "none" (DO NOT CLASSIFY AS ORDER/CANCEL):**
- "hôm sau nếu có ăn thì nhắn" → CONDITIONAL, intent: "none"
- "Ai đặt cơm chưa?" → QUESTION, intent: "none"
- "Mai mình đặt nhé" → FUTURE, intent: "none"
- "Nhớ đặt giùm tao nha" → ASKING SOMEONE ELSE, intent: "none"
- "Đặt giùm tui với" → ASKING SOMEONE ELSE, intent: "none"
- "đặt" → SINGLE WORD, NO CONTEXT, intent: "none"
- "ăn" → SINGLE WORD, NO CONTEXT, intent: "none"

**BE STRICT:** When in doubt, classify as "none". Only classify as "order" or "cancel" when the speaker is CLEARLY and DIRECTLY stating their own action."""

        try:
            # Initialize the new GenAI client
            self.client = genai.Client(api_key=api_key)

            # Configure safety settings to be less restrictive for food order messages
            self.safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
                ),
            ]

            # Create generation config with structured output using Pydantic model
            self.generation_config = types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=1.0,  # Lower temperature for more deterministic responses
                top_p=0.95,
                top_k=20,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema=OrderIntent,
                safety_settings=self.safety_settings,
            )

            self.model_name = "gemini-3-flash-preview"
            logger.info("Gemini API initialized successfully with structured outputs")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

    def parse_message(self, message: str) -> Optional[str]:
        """Parse a message and return just the intent string

        Args:
            message: The Vietnamese message text

        Returns:
            'order' if order detected, 'cancel' if cancellation detected, None otherwise
        """
        result = self.parse_message_full(message)
        return result[0] if result else None

    def parse_message_full(
        self, message: str
    ) -> Optional[tuple[Optional[str], Optional[OrderIntent]]]:
        """
        Parse a message and return the intent and full OrderIntent object

        Args:
            message: The Vietnamese message text

        Returns:
            Tuple of (intent_string, OrderIntent_object) or None if parsing fails
            intent_string: 'order', 'cancel', or None
            OrderIntent_object: Full parsed result with day_number, confidence, etc.
        """
        if not message or len(message.strip()) == 0:
            return None

        try:
            # Structured prompt following official Gemini docs best practices
            prompt = f"""<message>
{message}
</message>

<instruction>
Classify this Vietnamese message according to your system instructions.
Return only the JSON response with intent, confidence, and day_number.
</instruction>"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config,
            )

            # Check if response has valid candidates
            if not response or not response.candidates:
                logger.warning(
                    f"No candidates in Gemini response for message: {message[:50]}"
                )
                return None

            # Check finish reason
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            # finish_reason values: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER, etc.
            if finish_reason not in ["FINISH_REASON_UNSPECIFIED", "STOP", 0, 1]:
                logger.error(
                    f"Gemini response blocked (finish_reason={finish_reason}) for message: {message[:50]}"
                )
                logger.error(f"Response: {response}")
                return None

            # Parse structured JSON response
            try:
                if response.text:
                    result = OrderIntent.model_validate_json(response.text)
                    logger.info(
                        f"Classified '{message[:50]}' as '{result.intent}' (confidence: {result.confidence}, day: {result.day_number})"
                    )

                    # Only return order/cancel if confidence is at least medium
                    if result.intent in ["order", "cancel"]:
                        if result.confidence in ["high", "medium"]:
                            return (result.intent, result)
                        else:
                            logger.info(
                                f"Low confidence {result.intent}, treating as none: {message[:50]}"
                            )
                            return (None, result)

                    return (None, result)
                else:
                    logger.warning(f"No response text from Gemini for: {message[:50]}")
                    return None

            except ValueError as ve:
                logger.error(f"Could not parse structured response: {ve}")
                logger.error(
                    f"Response text: {response.text if response.text else 'None'}"
                )
                return None

        except Exception as e:
            logger.error(f"Error parsing message with Gemini: {e}")
            return None

    def parse_date_from_message(
        self, message: str, intent_result: Optional[OrderIntent] = None
    ) -> datetime:
        """Extract date reference from message (today, yesterday, specific day, etc.)

        Args:
            message: The Vietnamese message text
            intent_result: Optional OrderIntent result from parse_message with day_number

        Returns:
            datetime object representing the referenced date
        """
        current_date = datetime.now()
        current_day = current_date.day
        current_month = current_date.month
        current_year = current_date.year

        # If we have an intent_result with day_number, use it
        if intent_result and intent_result.day_number:
            target_day = intent_result.day_number

            # If the day is in the future this month, it must be from last month
            if target_day > current_day:
                # Go to previous month
                if current_month == 1:
                    return datetime(current_year - 1, 12, target_day)
                else:
                    return datetime(current_year, current_month - 1, target_day)
            else:
                # Same month
                return datetime(current_year, current_month, target_day)

        # Fallback: Parse from message text
        message_lower = message.lower()

        # Check for specific day numbers (e.g., "ngày 20", "day 15")
        day_patterns = [
            r"ngày\s+(\d{1,2})",  # "ngày 20"
            r"day\s+(\d{1,2})",  # "day 20"
            r"(\d{1,2})\s+tui",  # "20 tui" (less common)
        ]

        for pattern in day_patterns:
            match = re.search(pattern, message_lower)
            if match:
                day_num = int(match.group(1))
                if 1 <= day_num <= 31:
                    # If the day is in the future this month, assume it's from last month
                    if day_num > current_day:
                        if current_month == 1:
                            return datetime(current_year - 1, 12, day_num)
                        else:
                            return datetime(current_year, current_month - 1, day_num)
                    else:
                        return datetime(current_year, current_month, day_num)

        # Check for yesterday references
        yesterday_keywords = ["hôm qua", "ngày hôm qua", "yesterday", "hqua"]
        if any(keyword in message_lower for keyword in yesterday_keywords):
            return current_date - timedelta(days=1)

        # Default to today
        return current_date
