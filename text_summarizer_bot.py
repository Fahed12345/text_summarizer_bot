import logging
import os
import sys
import nltk
import openai
import asyncio
from dotenv import load_dotenv
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

# تكوين OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

# تحديد اللغة الافتراضية للتلخيص
DEFAULT_LANGUAGE = "arabic"
# عدد الجمل الافتراضي في التلخيص
DEFAULT_SENTENCES_COUNT = 3
# طريقة التلخيص الافتراضية
DEFAULT_METHOD = "gpt"

# قاموس لتخزين إعدادات المستخدمين
user_settings = {}

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
        '/method [الطريقة] - لتغيير طريقة التلخيص (gpt, lexrank, lsa, luhn)\n'
        'مثال: /method gpt\n\n'
        'طريقة gpt توفر تلخيصًا أكثر ترابطًا وفهمًا للسياق.'
    )
    await update.message.reply_text(help_text)

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """تلخيص النص باستخدام الأمر /summarize مع تحديد عدد الجمل."""
    if not context.args:
        await update.message.reply_text("الرجاء إدخال النص بعد الأمر. مثال: /summarize 3 النص الذي تريد تلخيصه")
        return
    
    user_id = update.effective_user.id
    
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
    
    # إرسال رسالة انتظار
    wait_message = await update.message.reply_text("جاري تلخيص النص، يرجى الانتظار...")
    
    try:
        # تخزين عدد الجمل في إعدادات المستخدم
        if user_id not in user_settings:
            user_settings[user_id] = {"method": DEFAULT_METHOD, "sentences": DEFAULT_SENTENCES_COUNT}
        user_settings[user_id]["sentences"] = sentences_count
        
        # تلخيص النص
        summary = await summarize_text(text, sentences_count, user_id)
        # إضافة معلومات عن عدد الجمل في الرد
        await wait_message.edit_text(f"التلخيص (عدد الجمل: {sentences_count}):\n\n{summary}")
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await wait_message.edit_text("حدث خطأ أثناء تلخيص النص. الرجاء المحاولة مرة أخرى.")

async def set_method(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """تغيير طريقة التلخيص."""
    user_id = update.effective_user.id
    
    if user_id not in user_settings:
        user_settings[user_id] = {"method": DEFAULT_METHOD, "sentences": DEFAULT_SENTENCES_COUNT}
    
    if not context.args:
        # عرض الطريقة الحالية إذا لم يتم تحديد طريقة
        current_method = user_settings[user_id].get("method", DEFAULT_METHOD)
        await update.message.reply_text(
            f"الطريقة الحالية: {current_method}\n"
            "الرجاء تحديد طريقة التلخيص: gpt, lexrank, lsa, أو luhn"
        )
        return
    
    method = context.args[0].lower()
    if method not in ["gpt", "lexrank", "lsa", "luhn"]:
        await update.message.reply_text("طريقة غير صالحة. الطرق المتاحة: gpt, lexrank, lsa, luhn")
        return
    
    # حفظ طريقة التلخيص في إعدادات المستخدم
    user_settings[user_id]["method"] = method
    
    # تسجيل تغيير الطريقة
    logger.info(f"User {user_id} changed summarization method to {method}")
    await update.message.reply_text(f"تم تغيير طريقة التلخيص إلى {method}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """التعامل مع الرسائل النصية وتلخيصها."""
    text = update.message.text
    user_id = update.effective_user.id
    
    if len(text) < 100:
        await update.message.reply_text("النص قصير جدًا للتلخيص. الرجاء إدخال نص أطول (أكثر من 100 حرف).")
        return
    
    # إرسال رسالة انتظار
    wait_message = await update.message.reply_text("جاري تلخيص النص، يرجى الانتظار...")
    
    try:
        # التأكد من وجود إعدادات للمستخدم
        if user_id not in user_settings:
            user_settings[user_id] = {"method": DEFAULT_METHOD, "sentences": DEFAULT_SENTENCES_COUNT}
        
        # الحصول على طريقة التلخيص وعدد الجمل من إعدادات المستخدم
        method = user_settings[user_id].get("method", DEFAULT_METHOD)
        sentences_count = user_settings[user_id].get("sentences", DEFAULT_SENTENCES_COUNT)
        
        # تسجيل الإعدادات المستخدمة
        logger.info(f"User {user_id} using method: {method}, sentences: {sentences_count}")
        
        # تلخيص النص
        summary = await summarize_text(text, sentences_count, user_id)
        # إضافة معلومات عن طريقة التلخيص وعدد الجمل في الرد
        await wait_message.edit_text(f"التلخيص (الطريقة: {method}، عدد الجمل: {sentences_count}):\n\n{summary}")
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await wait_message.edit_text("حدث خطأ أثناء تلخيص النص. الرجاء المحاولة مرة أخرى.")

async def summarize_text(text, sentences_count, user_id):
    """تلخيص النص باستخدام المكتبة المحددة."""
    logger.info(f"Summarizing text of length {len(text)} with {sentences_count} sentences")
    
    # تحديد طريقة التلخيص من إعدادات المستخدم
    if user_id not in user_settings:
        user_settings[user_id] = {"method": DEFAULT_METHOD, "sentences": DEFAULT_SENTENCES_COUNT}
    
    method = user_settings[user_id].get("method", DEFAULT_METHOD)
    logger.info(f"Using summarization method: {method}")
    
    # استخدام OpenAI GPT للتلخيص
    if method == "gpt":
        logger.info("Using GPT summarizer")
        return await summarize_with_gpt(text, sentences_count)
    
    # تحديد اللغة للتوكنايزر
    language = DEFAULT_LANGUAGE
    
    # إنشاء محلل للنص
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    
    # إنشاء الملخص المناسب بناءً على الطريقة
    stemmer = Stemmer(language)
    if method == "lsa":
        summarizer = LsaSummarizer(stemmer)
        logger.info("Using LSA summarizer")
    elif method == "luhn":
        summarizer = LuhnSummarizer(stemmer)
        logger.info("Using Luhn summarizer")
    else:  # lexrank (الافتراضي إذا لم يكن gpt)
        summarizer = LexRankSummarizer(stemmer)
        logger.info("Using LexRank summarizer")
    
    # إضافة كلمات التوقف للغة المحددة
    summarizer.stop_words = get_stop_words(language)
    
    # إنشاء التلخيص
    summary = summarizer(parser.document, sentences_count)
    
    # تحويل التلخيص إلى نص
    summary_text = "\n\n".join([str(sentence) for sentence in summary])
    
    if not summary_text:
        return "لم أتمكن من تلخيص هذا النص. قد يكون النص قصيرًا جدًا أو غير مناسب للتلخيص."
    
    return summary_text

async def summarize_with_gpt(text, sentences_count):
    """تلخيص النص باستخدام OpenAI GPT."""
    if not OPENAI_API_KEY:
        return "لم يتم تكوين مفتاح API لـ OpenAI. الرجاء استخدام طريقة تلخيص أخرى."
    
    try:
        # إنشاء طلب إلى OpenAI API
        prompt = f"""لخص النص التالي في {sentences_count} جمل مترابطة باللغة العربية الفصحى:

{text}

الملخص:"""
        
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت مساعد متخصص في تلخيص النصوص العربية بشكل دقيق ومترابط."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.5
        )
        
        # استخراج الملخص من الرد
        summary = response.choices[0].message.content.strip()
        
        if not summary:
            return "لم أتمكن من تلخيص هذا النص. حاول مرة أخرى أو استخدم طريقة تلخيص أخرى."
        
        return summary
    
    except Exception as e:
        logger.error(f"Error with OpenAI API: {e}")
        return f"حدث خطأ أثناء استخدام OpenAI API: {str(e)}. حاول استخدام طريقة تلخيص أخرى."

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """إرسال رسالة عند تنفيذ الأمر /help."""
    help_text = (
        'كيفية استخدام بوت التلخيص:\n\n'
        '1. أرسل أي نص تريد تلخيصه\n'
        '2. سأقوم بتلخيصه إلى 3 جمل افتراضيًا\n\n'
        'أوامر إضافية:\n'
        '/summarize [عدد الجمل] [النص] - لتلخيص نص مع تحديد عدد الجمل\n'
        'مثال: /summarize 5 هذا هو النص الذي أريد تلخيصه...\n\n'
        '/method [الطريقة] - لتغيير طريقة التلخيص (gpt, lexrank, lsa, luhn)\n'
        'مثال: /method gpt\n\n'
        'طريقة gpt توفر تلخيصًا أكثر ترابطًا وفهمًا للسياق.'
    )
    await update.message.reply_text(help_text)

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
