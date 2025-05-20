import logging
import os
import sys
import nltk
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# تكوين التسجيل مع حفظ السجلات في ملف
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot_log.txt", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# تحميل بيانات NLTK المطلوبة عند بدء التشغيل
try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK punkt data already downloaded")
except LookupError:
    logger.info("Downloading NLTK punkt data...")
    nltk.download('punkt')
    logger.info("NLTK punkt data downloaded successfully")

# استخدام التوكن من متغيرات البيئة أو القيمة الافتراضية
TOKEN = os.environ.get("BOT_TOKEN", "8093292228:AAEmQaJ_YTwq99s75O1bHIA0O-LVXWFoBF4")

# تحديد اللغة الافتراضية للتلخيص
DEFAULT_LANGUAGE = "arabic"
# عدد الجمل الافتراضي في التلخيص
DEFAULT_SENTENCES_COUNT = 3

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال رسالة عند تنفيذ الأمر /start."""
    await update.message.reply_text(
        'مرحباً! أنا بوت تلخيص النصوص. فقط أرسل لي نصًا وسأقوم بتلخيصه لك.\n\n'
        'يمكنك استخدام الأمر /help للحصول على مزيد من المعلومات.'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال رسالة عند تنفيذ الأمر /help."""
    help_text = (
        'كيفية استخدام بوت التلخيص:\n\n'
        '1. أرسل أي نص تريد تلخيصه\n'
        '2. سأقوم بتلخيصه إلى 3 جمل افتراضيًا\n\n'
        'أوامر إضافية:\n'
        '/summarize [عدد الجمل] [النص] - لتلخيص نص مع تحديد عدد الجمل\n'
        'مثال: /summarize 5 هذا هو النص الذي أريد تلخيصه...\n\n'
        '/method [الطريقة] - لتغيير طريقة التلخيص (lexrank, lsa, luhn)\n'
        'مثال: /method lsa'
    )
    await update.message.reply_text(help_text)

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """تلخيص النص باستخدام الأمر /summarize مع تحديد عدد الجمل."""
    if not context.args:
        await update.message.reply_text("الرجاء إدخال النص بعد الأمر. مثال: /summarize 3 النص الذي تريد تلخيصه")
        return
    
    try:
        # محاولة استخراج عدد الجمل من الأمر
        sentences_count = int(context.args[0])
        # استخراج النص بدءًا من العنصر الثاني
        text = ' '.join(context.args[1:])
    except ValueError:
        # إذا لم يكن العنصر الأول رقمًا، نستخدم العدد الافتراضي
        sentences_count = DEFAULT_SENTENCES_COUNT
        text = ' '.join(context.args)
    
    if not text:
        await update.message.reply_text("الرجاء إدخال نص للتلخيص")
        return
    
    # تلخيص النص
    summary = await summarize_text(text, sentences_count, context)
    await update.message.reply_text(summary)

async def set_method(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """تغيير طريقة التلخيص."""
    if not context.args:
        await update.message.reply_text("الرجاء تحديد طريقة التلخيص: lexrank, lsa, أو luhn")
        return
    
    method = context.args[0].lower()
    if method not in ["lexrank", "lsa", "luhn"]:
        await update.message.reply_text("طريقة غير صالحة. الطرق المتاحة: lexrank, lsa, luhn")
        return
    
    # حفظ طريقة التلخيص في بيانات المستخدم
    context.user_data["summarization_method"] = method
    await update.message.reply_text(f"تم تغيير طريقة التلخيص إلى {method}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """التعامل مع الرسائل النصية وتلخيصها."""
    text = update.message.text
    
    if len(text) < 100:
        await update.message.reply_text("النص قصير جدًا للتلخيص. الرجاء إدخال نص أطول (أكثر من 100 حرف).")
        return
    
    # إرسال رسالة انتظار
    wait_message = await update.message.reply_text("جاري تلخيص النص، يرجى الانتظار...")
    
    try:
        # تلخيص النص باستخدام العدد الافتراضي من الجمل
        summary = await summarize_text(text, DEFAULT_SENTENCES_COUNT, context)
        await wait_message.edit_text(summary)
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await wait_message.edit_text("حدث خطأ أثناء تلخيص النص. الرجاء المحاولة مرة أخرى.")

async def summarize_text(text, sentences_count, context):
    """تلخيص النص باستخدام المكتبة المحددة."""
    logger.info(f"Summarizing text of length {len(text)} with {sentences_count} sentences")
    
    # تحديد اللغة للتوكنايزر
    language = DEFAULT_LANGUAGE
    
    # إنشاء محلل للنص
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    
    # تحديد طريقة التلخيص (استخدام القيمة المخزنة في بيانات المستخدم أو القيمة الافتراضية)
    method = context.user_data.get("summarization_method", "lexrank")
    
    # إنشاء الملخص المناسب بناءً على الطريقة
    stemmer = Stemmer(language)
    if method == "lsa":
        summarizer = LsaSummarizer(stemmer)
    elif method == "luhn":
        summarizer = LuhnSummarizer(stemmer)
    else:  # lexrank (الافتراضي)
        summarizer = LexRankSummarizer(stemmer)
    
    # إضافة كلمات التوقف للغة المحددة
    summarizer.stop_words = get_stop_words(language)
    
    # إنشاء التلخيص
    summary = summarizer(parser.document, sentences_count)
    
    # تحويل التلخيص إلى نص
    summary_text = "\n\n".join([str(sentence) for sentence in summary])
    
    if not summary_text:
        return "لم أتمكن من تلخيص هذا النص. قد يكون النص قصيرًا جدًا أو غير مناسب للتلخيص."
    
    return summary_text

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """تسجيل الأخطاء وإرسال رسالة للمطور."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # تسجيل تفاصيل الخطأ
    import traceback
    traceback.print_exception(None, context.error, context.error.__traceback__)

def main() -> None:
    """بدء تشغيل البوت."""
    # تسجيل بدء تشغيل البوت
    logger.info("Starting bot...")
    
    # إنشاء التطبيق مع إعدادات محسنة
    application = Application.builder().token(TOKEN).connect_timeout(
        60.0
    ).pool_timeout(60.0).read_timeout(60.0).build()

    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("summarize", summarize_command))
    application.add_handler(CommandHandler("method", set_method))
    
    # إضافة معالج النصوص
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # إضافة معالج الأخطاء
    application.add_error_handler(error_handler)

    # استخدام polling للتشغيل المستمر مع إعدادات محسنة
    logger.info("Bot is running...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,  # تجاهل التحديثات المعلقة عند بدء التشغيل
        poll_interval=1.0,  # فترة الاستطلاع بالثواني
        timeout=30  # مهلة الاتصال بالثواني
    )

if __name__ == '__main__':
    main()
