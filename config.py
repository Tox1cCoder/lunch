# Configuration file for the bot (if needed for advanced settings)

# Bot Configuration
BOT_NAME = "Lunch Order Bot"
BOT_VERSION = "1.0.0"

# Timezone (Vietnam)
TIMEZONE = "Asia/Ho_Chi_Minh"

# Response messages in Vietnamese
MESSAGES = {
    "order_confirmed": "✅ Đã ghi nhận order của {name} cho hôm nay!",
    "order_cancelled": "❌ Đã hủy order của {name} cho hôm nay!",
    "user_not_found": "⚠️ Không tìm thấy tên '{name}' trong bảng. Vui lòng kiểm tra tên Telegram của bạn có trùng với tên trong sheet không.",
    "error": "❗ Đã có lỗi xảy ra. Vui lòng thử lại sau.",
}

# Date formats to try when searching for date columns
DATE_FORMATS = [
    "%-d/%-m/%Y",  # 22/1/2026
    "%d/%m/%Y",  # 22/01/2026
    "%-m/%-d/%Y",  # 1/22/2026
    "%m/%d/%Y",  # 01/22/2026
]
