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


class ConfirmationMessage(BaseModel):
    """Structured output for confirmation messages"""
    
    message: str = Field(
        description="Casual Vietnamese confirmation message with emoji. Sometimes include jokes or health comments about the food."
    )


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
    food_items: Optional[str] = Field(
        default=None,
        description="Food items mentioned in the order (e.g., '1 bánh canh', '2 cơm gà', 'cơm sườn'). Only extract if intent is 'order'."
    )


class VietnameseOrderParser:
    """Parser for detecting Vietnamese food order intents using Gemini AI with structured outputs"""

    def __init__(self, api_key: str):
        """
        Initialize the parser with Gemini API using structured outputs

        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key


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

            self.model_name = "gemini-3-flash-preview"
            logger.info("Gemini API initialized successfully with structured outputs")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

    def _get_system_prompt(self, current_date: datetime) -> str:
        """Generate system prompt with current date context
        
        Args:
            current_date: The date to use for context (usually message timestamp)
            
        Returns:
            System prompt string with date context
        """
        current_day = current_date.day
        current_month = current_date.month
        current_year = current_date.year
        
        return f"""You are a Vietnamese food order intent classifier with deep understanding of casual Vietnamese communication.

**CURRENT DATE CONTEXT:**
Today is day {current_day}, month {current_month}, year {current_year}.

**CRITICAL RULES:**
1. ACCEPT MINIMAL VIETNAMESE ORDERS - Vietnamese speakers often use shorthand without subjects or verbs:
   - "1 bánh canh" → intent: "order", food_items: "1 bánh canh" ✅
   - "2 cơm gà" → intent: "order", food_items: "2 cơm gà" ✅
   - "phở bò" → intent: "order", food_items: "phở bò" ✅
   - "cơm sườn" → intent: "order", food_items: "cơm sườn" ✅
   
   These are DIRECT orders even without "đặt" or "tui". If it mentions a Vietnamese food dish (with or without quantity), classify as "order".

2. FULL SENTENCE ORDERS also work:
   - "Đặt cho tui 1 cơm gà" → intent: "order", food_items: "1 cơm gà" ✅
   - "Tui đặt cơm sườn" → intent: "order", food_items: "cơm sườn" ✅
   - "Tui có đặt bánh mì" → intent: "order", food_items: "bánh mì" ✅

3. ONLY classify as "cancel" if speaker is DIRECTLY cancelling or NOT eating:
   - "Tui k ăn" → intent: "cancel" ✅
   - "Hủy order" → intent: "cancel" ✅
   - "Không ăn" → intent: "cancel" ✅

4. Classify as "none" for:
   - Questions to others: "Ai đặt cơm chưa?" → "none"
   - Conditionals: "nếu có ăn thì nhắn" → "none"
   - Future plans: "Mai mình đặt" → "none"
   - Asking someone else: "Đặt giùm tui", "nhớ đặt giùm" → "none"
   - Menu inquiries: "menu hôm nay", "có gì ăn" → "none"
   - Single words without food context: "đặt", "ăn" alone → "none"

**DAY NUMBER DETECTION:**
- "ngày 20 tôi có đặt" → day_number: 20
- "ngày 15 tui không ăn" → day_number: 15
- "hôm qua" (yesterday) → day_number: {current_day - 1}
- "hôm nay" (today) or no date mentioned → day_number: {current_day}

**FOOD EXTRACTION:**
When intent is "order", extract the food items mentioned:
- "1 bánh canh" → food_items: "1 bánh canh"
- "Tui đặt 2 cơm gà và 1 phở" → food_items: "2 cơm gà và 1 phở"
- "cơm sườn" → food_items: "cơm sườn"

**BE LENIENT WITH ORDERS, STRICT WITH NONE:** 
If it looks like food with or without quantity, it's likely an order. Only classify as "none" when clearly not placing an order."""


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
        self, message: str, message_date: Optional[datetime] = None
    ) -> Optional[tuple[Optional[str], Optional[OrderIntent]]]:
        """
        Parse a message and return the intent and full OrderIntent object

        Args:
            message: The Vietnamese message text
            message_date: Optional message timestamp (timezone-aware). Defaults to now() if not provided.

        Returns:
            Tuple of (intent_string, OrderIntent_object) or None if parsing fails
            intent_string: 'order', 'cancel', or None
            OrderIntent_object: Full parsed result with day_number, confidence, food_items, etc.
        """
        if not message or len(message.strip()) == 0:
            return None

        # Use message timestamp if provided, otherwise use current time
        if message_date is None:
            message_date = datetime.now()

        try:
            # Generate system prompt with message date context
            system_prompt = self._get_system_prompt(message_date)
            
            # Create generation config with structured output using Pydantic model
            generation_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema=OrderIntent,
                safety_settings=self.safety_settings,
            )
            
            # Structured prompt following official Gemini docs best practices
            prompt = f"""<message>
{message}
</message>

<instruction>
Classify this Vietnamese message according to your system instructions.
Return only the JSON response with intent, confidence, day_number, and food_items (if order).
</instruction>"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
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

    def generate_confirmation_message(
        self,
        user_name: str,
        intent: str,
        food_items: Optional[str] = None,
        date_desc: str = "hôm nay",
    ) -> str:
        """Generate casual confirmation message using Gemini
        
        Args:
            user_name: User's name
            intent: 'order' or 'cancel'
            food_items: Food items ordered (if any)
            date_desc: Date description in Vietnamese (e.g., "hôm nay", "hôm qua")
            
        Returns:
            Casual Vietnamese confirmation message
        """
        try:
            # Create confirmation prompt
            if intent == "order":
                food_context = f" (món: {food_items})" if food_items else ""
                prompt_content = f"""Generate a casual Vietnamese confirmation message for a food order.

Context:
- User: {user_name}
- Intent: Placing order
- Food: {food_items if food_items else "không rõ"}
- Date: {date_desc}

Requirements:
- Start with ✅ emoji
- Use casual Vietnamese tone (tui, nha, etc.)
- Include the food item if provided
- Sometimes add a light joke or comment about the food (healthy, yummy, etc.)
- Sometimes just be straightforward
- Keep it short (1-2 sentences max)

Examples:
- "✅ Đã ghi nhận order {food_items} cho {user_name} {date_desc}! Healthy choice nha 💪"
- "✅ Ok noted! {user_name} đặt {food_items} {date_desc}. Ngon lành cành đào luôn 😋"
- "✅ Roger that! {user_name} - {food_items} cho {date_desc} nhé!"
- "✅ Ghi nhận rồi nhen! {user_name} ăn {food_items} {date_desc}. Nhớ ăn rau nữa nha 🥗"

Generate one similar message NOW:"""
            else:  # cancel
                prompt_content = f"""Generate a casual Vietnamese cancellation confirmation message.

Context:
- User: {user_name}
- Intent: Cancelling order
- Date: {date_desc}

Requirements:
- Start with ❌ emoji
- Use casual Vietnamese tone
- Sometimes add sympathy or joke
- Keep it short (1 sentence)

Examples:
- "❌ Đã hủy order của {user_name} cho {date_desc}. Tiết kiệm tiền đi ăn sang hơn 💰"
- "❌ Ok cancel! {user_name} không ăn {date_desc}. Giảm cân à? 😄"
- "❌ Noted! Đã hủy order {user_name} cho {date_desc}"

Generate one similar message NOW:"""

            # Create generation config for confirmation messages
            confirmation_config = types.GenerateContentConfig(
                temperature=1.3,  # Higher temperature for variety
                top_p=0.95,
                top_k=40,
                max_output_tokens=256,
                response_mime_type="application/json",
                response_schema=ConfirmationMessage,
                safety_settings=self.safety_settings,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=confirmation_config,
            )

            if response and response.text:
                result = ConfirmationMessage.model_validate_json(response.text)
                logger.info(f"Generated confirmation: {result.message[:50]}...")
                return result.message
            else:
                # Fallback to simple message
                logger.warning("Failed to generate confirmation, using fallback")
                if intent == "order":
                    food_text = f" - {food_items}" if food_items else ""
                    return f"✅ Đã ghi nhận order của {user_name} cho {date_desc}{food_text}!"
                else:
                    return f"❌ Đã hủy order của {user_name} cho {date_desc}!"

        except Exception as e:
            logger.error(f"Error generating confirmation message: {e}")
            # Fallback to simple message
            if intent == "order":
                food_text = f" - {food_items}" if food_items else ""
                return f"✅ Đã ghi nhận order của {user_name} cho {date_desc}{food_text}!"
            else:
                return f"❌ Đã hủy order của {user_name} cho {date_desc}!"
