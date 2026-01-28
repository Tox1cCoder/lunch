"""
Telegram Food Order Bot
Tracks food orders in Vietnamese group chats and logs to Google Sheets
"""

import logging
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from nlp_parser import VietnameseOrderParser
from sheets_manager import SheetsManager

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global instances
parser = None
sheets_manager = None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming messages from the group chat

    Detects Vietnamese food orders and cancellations using Gemini AI, then updates Google Sheets
    """
    # Ignore if not a text message
    if not update.message or not update.message.text:
        return

    message_text = update.message.text
    user = update.effective_user
    user_name = user.first_name

    # Get message timestamp from Telegram (already timezone-aware in UTC)
    message_timestamp_utc = update.message.date
    # Convert to Vietnam timezone
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    message_date_vietnam = message_timestamp_utc.astimezone(vietnam_tz)

    logger.info(
        f"Received message from {user_name} at {message_date_vietnam.strftime('%Y-%m-%d %H:%M:%S %Z')}: {message_text}"
    )

    # Parse the message for order intent using Gemini AI (get full result with day_number and food_items)
    parse_result = parser.parse_message_full(message_text, message_date=message_date_vietnam)

    if not parse_result:
        return

    intent, intent_data = parse_result

    if not intent:
        return

    # Parse date reference from message (uses day_number from intent_data)
    target_date = parser.parse_date_from_message(message_text, intent_data)

    # Determine the date description for the reply
    date_desc = "hôm nay"
    if target_date.date() < message_date_vietnam.date():
        days_ago = (message_date_vietnam.date() - target_date.date()).days
        if days_ago == 1:
            date_desc = "hôm qua"
        else:
            date_desc = f"ngày {target_date.day}/{target_date.month}"

    if intent == "order":
        # User is placing an order
        success = sheets_manager.mark_order(user_name, True, target_date)
        if success:
            # Generate dynamic confirmation message
            confirmation = parser.generate_confirmation_message(
                user_name=user_name,
                intent="order",
                food_items=intent_data.food_items,
                date_desc=date_desc,
            )
            await update.message.reply_text(
                confirmation,
                reply_to_message_id=update.message.message_id,
            )
            logger.info(
                f"Marked order for {user_name} on {target_date.strftime('%d/%m/%Y')} - Food: {intent_data.food_items}"
            )
        else:
            await update.message.reply_text(
                f"⚠️ Không tìm thấy tên '{user_name}' trong bảng hoặc không tìm thấy cột ngày {date_desc}. "
                f"Vui lòng kiểm tra tên Telegram của bạn có trùng với tên trong sheet không.",
                reply_to_message_id=update.message.message_id,
            )
            logger.warning(
                f"Failed to mark order for {user_name} on {target_date.strftime('%d/%m/%Y')}"
            )

    elif intent == "cancel":
        # User is cancelling their order
        success = sheets_manager.mark_order(user_name, False, target_date)
        if success:
            # Generate dynamic cancellation message
            confirmation = parser.generate_confirmation_message(
                user_name=user_name,
                intent="cancel",
                date_desc=date_desc,
            )
            await update.message.reply_text(
                confirmation,
                reply_to_message_id=update.message.message_id,
            )
            logger.info(
                f"Cancelled order for {user_name} on {target_date.strftime('%d/%m/%Y')}"
            )
        else:
            await update.message.reply_text(
                f"⚠️ Không tìm thấy tên '{user_name}' trong bảng hoặc không tìm thấy cột ngày {date_desc}.",
                reply_to_message_id=update.message.message_id,
            )
            logger.warning(
                f"Failed to cancel order for {user_name} on {target_date.strftime('%d/%m/%Y')}"
            )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates"""
    logger.error(f"Update {update} caused error {context.error}")


def main():
    """Start the bot"""
    # Get configuration from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    # If SHEET_NAME is not set or set to "auto", auto-detect based on current month
    sheet_name_env = os.getenv("SHEET_NAME")
    sheet_name = (
        None
        if not sheet_name_env or sheet_name_env.lower() == "auto"
        else sheet_name_env
    )
    credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return

    if not sheet_id:
        logger.error("GOOGLE_SHEET_ID not found in environment variables")
        return

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return

    # Initialize Gemini-powered NLP parser
    global parser
    try:
        parser = VietnameseOrderParser(gemini_api_key)
        logger.info("Gemini AI parser initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini parser: {e}")
        return

    # Initialize Google Sheets connection
    global sheets_manager
    sheets_manager = SheetsManager(credentials_file, sheet_id, sheet_name)

    if not sheets_manager.connect():
        logger.error("Failed to connect to Google Sheets. Exiting.")
        return

    logger.info("Successfully connected to Google Sheets")

    # Create the Application
    application = Application.builder().token(bot_token).build()

    # Add message handler for all text messages in groups
    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND
            & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
            handle_message,
        )
    )

    # Add error handler
    application.add_error_handler(error_handler)

    # Start the bot
    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
