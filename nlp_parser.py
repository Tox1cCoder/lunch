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
    price: Optional[int] = Field(
        default=None,
        description="Price of the order in VND. Only valid values are 30000, 35000, or 40000. Extract from patterns like '30k', '35K', '30 ngàn', '30000', etc. null if no price mentioned."
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

            # Use Gemini 3 Flash Preview with minimal thinking
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

**PRICE EXTRACTION:**
When intent is "order", extract the price if mentioned. ONLY valid prices are 30000, 35000, 40000.
- "30k", "30K", "30 ngàn", "30 nghìn", "30000" → price: 30000
- "35k", "35K", "35 ngàn", "35 nghìn", "35000" → price: 35000
- "40k", "40K", "40 ngàn", "40 nghìn", "40000" → price: 40000
- If no price mentioned → price: null
- If a different price is mentioned (not 30000/35000/40000), set price to null

Examples:
- "cho tôi 1 cơm gà 30k" → intent: "order", food_items: "1 cơm gà", price: 30000
- "1 phở bò 35 ngàn" → intent: "order", food_items: "1 phở bò", price: 35000
- "cơm sườn" → intent: "order", food_items: "cơm sườn", price: null

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
            # Use ThinkingConfig to minimize thinking for faster responses
            generation_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0,
                top_p=0.9,
                top_k=20,
                max_output_tokens=350,  # Account for minimal thinking overhead
                response_mime_type="application/json",
                response_schema=OrderIntent,
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            )
            
            prompt = f"""<message>
{message}
</message>

<instruction>
Classify this Vietnamese message according to your system instructions.
Return only the JSON response with intent, confidence, day_number, food_items (if order), and price (if order and price mentioned).
</instruction>"""

            # Call Gemini API
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
                        f"Classified '{message[:50]}' as '{result.intent}' (confidence: {result.confidence}, day: {result.day_number}, price: {result.price})"
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
        price: Optional[int] = None,
    ) -> str:
        """Generate casual confirmation message using Gemini
        
        Args:
            user_name: User's name
            intent: 'order' or 'cancel'
            food_items: Food items ordered (if any)
            date_desc: Date description in Vietnamese (e.g., "hôm nay", "hôm qua")
            price: Order price in VND (e.g., 30000, 35000, 40000)
            
        Returns:
            Casual Vietnamese confirmation message
        """
        try:
            # Create confirmation prompt
            if intent == "order":
                food_info = food_items if food_items else "món không rõ"
                price_info = f"{price // 1000}k" if price else "giá không rõ"
                system_instruction = """You are a casual Vietnamese food order bot assistant. 
Generate short, friendly confirmation messages in Vietnamese with emojis.
Return ONLY the message text, no JSON, no explanation."""

                prompt_content = f"""Generate a casual Vietnamese confirmation message for this order:
- User: {user_name}
- Food: {food_info}
- Price: {price_info}
- Date: {date_desc}

Requirements:
- Start with ✅ emoji
- Use casual Vietnamese (nha, nhé, luôn, etc.)
- Mention the food, price, and user
- 1-2 sentences max
- Sometimes add health comment or joke, sometimes be straightforward

Examples:
✅ Đã note {food_info} ({price_info}) cho {user_name} {date_desc}! Ngon lành 😋
✅ Roger! {user_name} - {food_info} {price_info} {date_desc} nhé
✅ Ghi nhận rồi nha! {user_name} ăn {food_info} {price_info} {date_desc}. Healthy đó 💪

Generate ONE message NOW (return only the message):"""

            else:  # cancel
                system_instruction = """You are a casual Vietnamese food order bot assistant. 
Generate short, friendly cancellation messages in Vietnamese with emojis.
Return ONLY the message text, no JSON, no explanation."""

                prompt_content = f"""Generate a casual Vietnamese cancellation message:
- User: {user_name}
- Date: {date_desc}

Requirements:
- Start with ❌ emoji
- Use casual Vietnamese
- 1 sentence
- Sometimes add sympathetic or joking comment

Examples:
❌ Đã cancel order {user_name} cho {date_desc}
❌ Ok noted! {user_name} không ăn {date_desc}. Tiết kiệm tiền nha 💰
❌ Hủy rồi nhen! {user_name} - {date_desc}

Generate ONE message NOW (return only the message):"""

            # Use plain text generation (not JSON)
            # Use ThinkingConfig to minimize thinking for faster responses
            plain_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
                top_p=0.9,
                max_output_tokens=200,  # Account for minimal thinking overhead
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            )

            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=plain_config,
            )

            if response and response.text:
                message = response.text.strip()
                
                # Clean up the response - remove common preamble patterns
                preamble_patterns = [
                    "Here is the JSON requested:",
                    "Here is the message:",
                    "Here's the message:",
                    "Message:",
                    "Response:",
                ]
                for pattern in preamble_patterns:
                    if message.startswith(pattern):
                        message = message[len(pattern):].strip()
                
                # Remove quotes if the LLM wrapped the message
                if message.startswith('"') and message.endswith('"'):
                    message = message[1:-1]
                if message.startswith("'") and message.endswith("'"):
                    message = message[1:-1]
                
                # Verify it starts with expected emoji
                if message and (message.startswith('✅') or message.startswith('❌')):
                    logger.info(f"Generated confirmation: {message[:50]}...")
                    return message
                else:
                    logger.warning(f"Generated message doesn't start with emoji: {message[:100]}")
                    # Fall through to fallback
            
            # Fallback to template
            logger.warning("Using fallback template for confirmation")
            if intent == "order":
                food_text = f" - {food_items}" if food_items else ""
                price_text = f" ({price // 1000}k)" if price else ""
                return f"✅ Đã ghi nhận order của {user_name} cho {date_desc}{food_text}{price_text}!"
            else:
                return f"❌ Đã hủy order của {user_name} cho {date_desc}!"

        except Exception as e:
            logger.error(f"Error generating confirmation message: {e}")
            # Fallback to simple message
            if intent == "order":
                food_text = f" - {food_items}" if food_items else ""
                price_text = f" ({price // 1000}k)" if price else ""
                return f"✅ Đã ghi nhận order của {user_name} cho {date_desc}{food_text}{price_text}!"
            else:
                return f"❌ Đã hủy order của {user_name} cho {date_desc}!"
