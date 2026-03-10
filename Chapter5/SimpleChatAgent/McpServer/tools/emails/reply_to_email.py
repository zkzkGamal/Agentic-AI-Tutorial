"""
reply to email tool
"""

from server import mcp
import smtplib , logging , environ , pathlib
import imaplib
import email



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = pathlib.Path(__file__).parent.parent.parent
e = environ.Env()
e.read_env(str(base_path / ".env"))

# global email configuration
EMAIL_HOST = e('MAIL_HOST', default='smtp.example.com')
EMAIL_PORT = e('MAIL_PORT', default=587)
MAIL_USERNAME = e('MAIL_USERNAME',default=None)
MAIL_HOST_PASSWORD = e('MAIL_PASSWORD', default=None)
EMAIL_USE_SSL = e.bool('MAIL_ENCRYPTION', default=True)
MAIL_FROM_NAME = e('MAIL_FROM_NAME', default='Zkaria Gamal')

@mcp.tool()
def reply_to_email(email_id: str, body: str) -> bool:
    """Reply to a specific email by its ID."""
    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            if EMAIL_USE_SSL:
                server.starttls()
            server.login(MAIL_USERNAME, MAIL_HOST_PASSWORD)
            server.sendmail(MAIL_USERNAME, email_id, body)
        logger.info(f"Email sent successfully to {email_id}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False