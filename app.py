import streamlit as st
import pickle
import string
import email
import email.policy
import pandas as pd
import io
import csv
import imaplib
import re
import html as html_module
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page config with dark-themed title & wide/centered layout
st.set_page_config(
    page_title="EMAIL SPAM DETECTION | TEAM-8",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Ensure required NLTK resources are available
for resource in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

ps = PorterStemmer()


def clean_html_content(raw_text):
    if not raw_text:
        return ""
    # Strip script and style blocks
    text = re.sub(r'<style.*?</style>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip HTML tags
    text = re.sub(r'<[^<]+?>', ' ', text)
    # Decode HTML entities
    text = html_module.unescape(text)
    # Normalise whitespace
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.split('\n')]
    return '\n'.join([line for line in lines if line])


def transform_text(text):
    text = clean_html_content(text)
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def parse_batch_input(raw_text):
    clean = raw_text.strip()
    lines = [l.strip() for l in clean.split('\n') if l.strip()]
    if len(lines) <= 1:
        return False, []

    # Check if lines have CSV characteristics or multi-message format
    has_header = any(h in lines[0].lower() for h in ['text,label', 'label,text', 'message,label', 'v1,v2', 'v2,v1'])
    has_labels = sum(1 for l in lines if l.lower().endswith(',spam') or l.lower().endswith(',ham')) >= 2

    if not (has_header or has_labels):
        return False, []

    items = []
    reader = csv.reader(io.StringIO(clean))
    for row in reader:
        if not row:
            continue
        if len(row) == 1:
            items.append({'text': row[0].strip(), 'label': None})
        elif len(row) >= 2:
            col0 = row[0].strip()
            col1 = row[1].strip()
            if col0.lower() in ['text', 'message', 'email', 'v2'] and col1.lower() in ['label', 'target', 'class', 'v1']:
                continue
            if col1.lower() in ['spam', 'ham']:
                items.append({'text': col0, 'label': col1.lower()})
            elif col0.lower() in ['spam', 'ham']:
                items.append({'text': col1, 'label': col0.lower()})
            else:
                items.append({'text': ', '.join(row), 'label': None})
    return True, items


# Load Vectorizer and Model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
if hasattr(tfidf, '_tfidf') and not hasattr(tfidf._tfidf, 'idf_') and hasattr(tfidf._tfidf, '_idf_diag'):
    tfidf._tfidf.idf_ = tfidf._tfidf._idf_diag.diagonal()

model = pickle.load(open('model.pkl', 'rb'))

# Pre-populated Device Mailbox
DEVICE_INBOX_EMAILS = [
    {
        "id": "mail_1",
        "sender": "Bank of America Alert <alerts@bankofamerica.com>",
        "subject": "Urgent: Unauthorized debit charge of $489.00 on your account",
        "date": "Today, 10:14 AM",
        "tag": "SECURITY ALERT",
        "tag_color": "#FF334B",
        "snippet": "We detected an unauthorized transaction on your debit card ending in 4102. Call our 24/7 fraud department immediately at 1-800-432-1000 or click to verify identity...",
        "body": "From: Bank of America Security Team <alerts@bankofamerica.com>\nSubject: Urgent: Unauthorized debit charge of $489.00 on your account\nDate: Today, 10:14 AM\n\nDear Customer,\n\nWe detected a suspicious charge of $489.00 on your debit card ending in 4102 at 'BESTBUY-ONLINE'.\n\nIf you DID NOT authorize this transaction, please call our 24/7 fraud prevention desk immediately at 1-800-432-1000 or click the secure link below to verify identity and cancel payment:\nhttp://bankofamerica-fraud-resolution.com/auth?id=48921\n\nSecurity Operations,\nBank of America"
    },
    {
        "id": "mail_2",
        "sender": "HR People Operations <hr@company.internal>",
        "subject": "Q4 Holiday Calendar & PTO Submission Guidelines",
        "date": "Today, 9:30 AM",
        "tag": "WORK / INTERNAL",
        "tag_color": "#00FF66",
        "snippet": "Hi team, please review the updated Q4 holiday schedule and submit any pending paid time off requests before Friday...",
        "body": "From: HR Operations <hr@company.internal>\nSubject: Q4 Holiday Calendar & PTO Submission Guidelines\nDate: Today, 9:30 AM\n\nHello Team,\n\nPlease review the attached Q4 calendar for upcoming company-observed holidays and office closures. If you plan to take vacation days over Thanksgiving or New Year, please submit your PTO requests via the portal before Friday so team leads can plan coverage.\n\nBest regards,\nPeople Operations Team"
    },
    {
        "id": "mail_3",
        "sender": "Amazon Fulfillment <shipment-update@amazon.com>",
        "subject": "Action Required: Confirm shipping address for Order #114-8829",
        "date": "Yesterday, 6:45 PM",
        "tag": "ORDER ISSUE",
        "tag_color": "#FFB800",
        "snippet": "Your package could not be dispatched due to an incomplete street address. Update your delivery details within 24 hours to avoid cancellation...",
        "body": "From: Amazon Fulfillment Desk <shipment-update@amazon.com>\nSubject: Action Required: Confirm shipping address for Order #114-8829\nDate: Yesterday, 6:45 PM\n\nDear Shopper,\n\nYour order #114-8829104-58291 could not be processed for dispatch because of an unrecognized apartment/street number in your profile.\n\nTo avoid shipment cancellation and return-to-sender fees, please verify your shipping address within 24 hours:\nhttps://amazon-shipment-resolution.net/reconfirm?id=1148829\n\nThank you,\nAmazon Logistics"
    },
    {
        "id": "mail_4",
        "sender": "Apple Billing <no-reply@apple.com>",
        "subject": "Your invoice for iCloud+ 2TB Storage",
        "date": "Yesterday, 2:15 PM",
        "tag": "SUBSCRIPTION",
        "tag_color": "#00FF66",
        "snippet": "Your monthly subscription to iCloud+ with 2TB storage was successfully renewed for $9.99 on Oct 25, 2026...",
        "body": "From: Apple Billing <no-reply@apple.com>\nSubject: Your invoice for iCloud+ 2TB Storage\nDate: Yesterday, 2:15 PM\n\nApple Receipt\nInvoice Date: Oct 25, 2026\nOrder ID: MH9428-29\n\nItem: iCloud+ with 2TB Storage (Monthly Plan)\nAmount Charged: $9.99 to Apple Pay\n\nTo manage your subscriptions or review purchase history, visit your Apple ID Settings."
    },
    {
        "id": "mail_5",
        "sender": "Dr. Sarah Patel <reception@patelmedical.org>",
        "subject": "Appointment Confirmation: Thursday, 3:30 PM",
        "date": "Oct 24, 11:20 AM",
        "tag": "APPOINTMENT",
        "tag_color": "#00FF66",
        "snippet": "Friendly reminder of your upcoming routine checkup with Dr. Patel on Thursday at 3:30 PM. Please arrive 10 minutes early...",
        "body": "From: Dr. Sarah Patel Clinic <reception@patelmedical.org>\nSubject: Appointment Confirmation: Thursday, 3:30 PM\nDate: Oct 24, 11:20 AM\n\nHi,\n\nThis is a friendly reminder that you are scheduled for your routine annual checkup with Dr. Sarah Patel on Thursday, Oct 29th at 3:30 PM.\n\nPlease remember to bring your insurance card and a valid photo ID. If you need to reschedule or cancel, give our reception a call at least 24 hours in advance.\n\nTake care,\nPatel Medical Care"
    },
    {
        "id": "mail_6",
        "sender": "Airdrop Claim Portal <rewards@crypto-airdrop-pool.xyz>",
        "subject": "You have been allocated 0.45 ETH ($1,250) voucher reward!",
        "date": "Oct 23, 4:05 AM",
        "tag": "PROMO / AIRDROP",
        "tag_color": "#FF334B",
        "snippet": "CONGRATULATIONS! Your active wallet address was selected for our seasonal distribution! Connect wallet now to claim bonus...",
        "body": "From: Airdrop Rewards <rewards@crypto-airdrop-pool.xyz>\nSubject: You have been allocated 0.45 ETH ($1,250) voucher reward!\nDate: Oct 23, 4:05 AM\n\nCONGRATULATIONS!\n\nYour wallet has been randomly selected among top network users to claim an allocation of 0.45 ETH (Estimated $1,250 USD).\n\nClaim window expires in 4 hours! Click below to connect your Web3 wallet and claim instantly:\nhttp://claim-crypto-airdrop-rewards.xyz/pool\n\nFailure to claim will release funds back to liquidity pool."
    }
]


def fetch_from_folder(mail, folder_name, limit, tag_label, tag_color):
    try:
        quoted_name = f'"{folder_name}"' if (' ' in folder_name or '/' in folder_name) and not folder_name.startswith('"') else folder_name
        status, _ = mail.select(quoted_name, readonly=True)
        if status != "OK":
            status, _ = mail.select(folder_name, readonly=True)
            if status != "OK":
                return []

        status, messages = mail.search(None, "ALL")
        if status != "OK" or not messages or not messages[0]:
            return []

        mail_ids = messages[0].split()
        recent_ids = mail_ids[-limit:][::-1]
        results = []
        for idx, mid in enumerate(recent_ids):
            status, data = mail.fetch(mid, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                continue
            raw = data[0][1]
            msg = email.message_from_bytes(raw, policy=email.policy.default)
            subject = str(msg['subject'] or "No Subject")
            sender = str(msg['from'] or "Unknown Sender")
            date = str(msg['date'] or "")

            plain_body = ""
            html_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == 'text/plain' and not plain_body:
                        payload = part.get_payload(decode=True)
                        if payload:
                            plain_body = payload.decode('utf-8', errors='ignore')
                    elif ctype == 'text/html' and not html_body:
                        payload = part.get_payload(decode=True)
                        if payload:
                            html_body = payload.decode('utf-8', errors='ignore')
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    if msg.get_content_type() == 'text/html':
                        html_body = payload.decode('utf-8', errors='ignore')
                    else:
                        plain_body = payload.decode('utf-8', errors='ignore')

            raw_chosen = plain_body if plain_body.strip() else html_body
            clean_body = clean_html_content(raw_chosen).strip()
            if not clean_body:
                clean_body = "(No readable text body found in email)"

            snippet = (clean_body[:130] + '...') if len(clean_body) > 130 else clean_body
            mid_str = mid.decode('utf-8', errors='ignore') if isinstance(mid, bytes) else str(mid)
            results.append({
                "id": f"{tag_label.lower().replace(' ', '_')}_{mid_str}_{idx}",
                "sender": sender,
                "subject": subject,
                "date": date,
                "tag": tag_label,
                "tag_color": tag_color,
                "snippet": snippet.replace('\n', ' '),
                "body": f"From: {sender}\nSubject: {subject}\nDate: {date}\n\n{clean_body}"
            })
        return results
    except Exception:
        return []


def detect_spam_folder_name(mail, provider):
    try:
        status, folders = mail.list()
        if status == "OK" and folders:
            for f in folders:
                f_str = f.decode('utf-8', errors='ignore') if isinstance(f, bytes) else str(f)
                if r'\Spam' in f_str or r'\Junk' in f_str or 'Spam' in f_str or 'Junk' in f_str:
                    parts = f_str.split('"/"')
                    if len(parts) > 1:
                        return parts[-1].strip().strip('"').strip()
                    parts_space = f_str.split(' "/" ')
                    if len(parts_space) > 1:
                        return parts_space[-1].strip().strip('"').strip()
    except Exception:
        pass

    if "Gmail" in provider:
        return "[Gmail]/Spam"
    elif "Outlook" in provider:
        return "Junk"
    else:
        return "Spam"


def fetch_live_imap_emails(host, port, user, password, limit=20, provider="Gmail", folder_mode="🔥 Both (Inbox + Spam)"):
    try:
        mail = imaplib.IMAP4_SSL(host, int(port), timeout=15)
        mail.login(user, password)
        results = []

        spam_folder = detect_spam_folder_name(mail, provider)

        if "Spam Folder Only" in folder_mode or "Spam Only" in folder_mode:
            results = fetch_from_folder(mail, spam_folder, limit, "SPAM FOLDER", "#FF334B")
        elif "Inbox Only" in folder_mode:
            results = fetch_from_folder(mail, "INBOX", limit, "INBOX", "#00FF66")
        else:  # Both
            half_limit = max(limit // 2, 5)
            spam_emails = fetch_from_folder(mail, spam_folder, half_limit, "SPAM FOLDER", "#FF334B")
            inbox_emails = fetch_from_folder(mail, "INBOX", half_limit, "INBOX", "#00FF66")
            results = spam_emails + inbox_emails

        mail.logout()
        return True, results
    except Exception as e:
        return False, str(e)


# Nike-inspired CSS Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@300;400;600;700;800;900&display=swap');

/* Global Reset & Base */
.stApp {
    background-color: #111111;
    color: #F5F5F5;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide default streamlit decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 780px;
}

/* Nike Top Bar */
.nike-top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #222222;
    padding-bottom: 14px;
    margin-bottom: 24px;
}
.nike-badge {
    background-color: #222222;
    color: #EEEEEE;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 6px 12px;
    border-radius: 999px;
    text-transform: uppercase;
}
.nike-status-pill {
    color: #888888;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-dot {
    width: 8px;
    height: 8px;
    background-color: #00FF66;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 10px #00FF66;
}

/* Brand Banner */
.nike-hero-title {
    font-family: 'Anton', Impact, sans-serif;
    font-size: 58px;
    line-height: 0.95;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #FFFFFF;
    margin: 0;
    padding: 0;
}
.nike-hero-tagline {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #777777;
    text-transform: uppercase;
    margin-top: 10px;
    margin-bottom: 28px;
}
.nike-highlight {
    color: #E2F835;
}

/* Text Area Override */
.stTextArea textarea {
    background-color: #0D0D0D !important;
    color: #FFFFFF !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    padding: 16px !important;
    line-height: 1.5 !important;
    transition: all 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: #FFFFFF !important;
    box-shadow: 0 0 0 1px #FFFFFF !important;
}
.stTextArea label {
    font-size: 12px !important;
    font-weight: 800 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #999999 !important;
    margin-bottom: 8px !important;
}

/* Buttons Styling */
div.stButton > button {
    border-radius: 9999px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid #333333 !important;
    background-color: #222222 !important;
    color: #DDDDDD !important;
    padding: 10px 18px !important;
}

div.stButton > button:hover {
    background-color: #333333 !important;
    color: #FFFFFF !important;
    border-color: #555555 !important;
    transform: translateY(-1px);
}

/* Primary Predict Button */
div.stButton > button[data-testid="stBaseButton-primary"],
div.stButton > button[kind="primary"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: none !important;
    width: 100% !important;
    font-size: 16px !important;
    padding: 16px !important;
    margin-top: 10px !important;
}
div.stButton > button[data-testid="stBaseButton-primary"]:hover,
div.stButton > button[kind="primary"]:hover {
    background-color: #E2F835 !important;
    color: #000000 !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(226, 248, 53, 0.35) !important;
}

/* Mailbox Container */
.mailbox-drawer {
    background-color: #161616;
    border: 1px solid #2B2B2B;
    border-radius: 16px;
    padding: 20px;
    margin-top: 12px;
    margin-bottom: 22px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    animation: fadeIn 0.3s ease-out;
}
.email-card {
    background-color: #0D0D0D;
    border: 1px solid #242424;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
}
.email-card:hover {
    border-color: #444444;
}

/* Result Cards */
.result-card-spam {
    background: linear-gradient(135deg, #240b0b 0%, #150808 100%);
    border: 1px solid #FF334B;
    border-radius: 18px;
    padding: 24px;
    margin-top: 24px;
    animation: fadeIn 0.4s ease-out;
}
.result-card-ham {
    background: linear-gradient(135deg, #0b1f13 0%, #08140c 100%);
    border: 1px solid #00FF66;
    border-radius: 18px;
    padding: 24px;
    margin-top: 24px;
    animation: fadeIn 0.4s ease-out;
}
.result-headline {
    font-family: 'Anton', Impact, sans-serif;
    font-size: 42px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}
.spam-text { color: #FF334B; }
.ham-text { color: #00FF66; }

/* Spec Grid */
.specs-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #222222;
}
.spec-box {
    background-color: #171717;
    border: 1px solid #262626;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.spec-number {
    font-family: 'Anton', Impact, sans-serif;
    font-size: 26px;
    color: #FFFFFF;
}
.spec-label {
    font-size: 11px;
    color: #777777;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Footer */
.nike-footer {
    text-align: center;
    color: #555555;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #222222;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# Session state
if 'sample_input' not in st.session_state:
    st.session_state.sample_input = ""
if 'show_mailbox' not in st.session_state:
    st.session_state.show_mailbox = False
if 'live_emails' not in st.session_state:
    st.session_state.live_emails = []

# Header Bar
st.markdown("""
<div class="nike-top-bar">
    <div style="display:flex; align-items:center; gap:10px;">
        <div class="nike-badge">EMAIL SPAM DETECTION</div>
        <div class="nike-badge" style="background-color:#E2F835; color:#000000; font-weight:800; letter-spacing:1.5px;">TEAM-8</div>
    </div>
    <div class="nike-status-pill">
        <span class="status-dot"></span> SYSTEM ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div>
    <h1 class="nike-hero-title">JUST DETECT IT<span class="nike-highlight">.</span></h1>
    <div class="nike-hero-tagline">Engineered for total inbox dominance. Zero spam. Pure signal.</div>
</div>
""", unsafe_allow_html=True)

# Preset Quick Samples & Choose From Mail Button
st.markdown('<div style="font-size:11px; font-weight:800; letter-spacing:1.5px; color:#777; text-transform:uppercase; margin-bottom:8px;">Instant Test Payloads & Inbox:</div>', unsafe_allow_html=True)
col_mail, col1, col2, col3 = st.columns([1.3, 1, 1, 0.7])

with col_mail:
    mailbox_btn_label = "📬 CLOSE MAILBOX" if st.session_state.show_mailbox else "📬 CHOOSE FROM MAIL"
    if st.button(mailbox_btn_label, type="secondary", use_container_width=True):
        st.session_state.show_mailbox = not st.session_state.show_mailbox
        st.rerun()

with col1:
    if st.button("🚨 Spam Sample", type="secondary", use_container_width=True):
        st.session_state.sample_input = "URGENT! You have won a £1,000 Walmart Gift Card! Call 09061701461 now to claim your prize code KL341. Valid 24 hours only."
        st.session_state.show_mailbox = False
        st.rerun()

with col2:
    if st.button("💬 Clean Sample", type="secondary", use_container_width=True):
        st.session_state.sample_input = "Hey, are we still meeting at the gym today at 6 PM? Let me know if you want to grab protein shakes after."
        st.session_state.show_mailbox = False
        st.rerun()

with col3:
    if st.button("CLEAR", type="secondary", use_container_width=True):
        st.session_state.sample_input = ""
        st.rerun()

# Interactive Mailbox Browser (No File Manager Popup!)
if st.session_state.show_mailbox:
    st.markdown("""
<div class="mailbox-drawer">
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #282828; padding-bottom:12px; margin-bottom:16px;">
        <div>
            <div style="font-size:14px; font-weight:800; letter-spacing:2px; text-transform:uppercase; color:#E2F835;">
                DEVICE INBOX BROWSER
            </div>
            <div style="font-size:12px; color:#888888; margin-top:2px;">
                Choose an email from your device inbox below to test for spam:
            </div>
        </div>
        <span style="font-size:11px; font-weight:700; color:#AAAAAA; background-color:#222; padding:4px 10px; border-radius:999px;">
            NO FILE PICKER NEEDED
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    tab_inbox, tab_live, tab_file = st.tabs(["📥 Device Mailbox (Quick Select)", "⚡ Connect Live Mail (Gmail/Outlook)", "📁 Local File Backup"])

    with tab_inbox:
        st.markdown('<div style="font-size:12px; color:#999; margin-bottom:12px;">Click <strong>SELECT THIS MAIL</strong> on any email to load it into the scanner:</div>', unsafe_allow_html=True)
        for mail in DEVICE_INBOX_EMAILS:
            st.markdown(f"""
<div class="email-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <span style="font-size:12px; font-weight:700; color:#CCCCCC;">{mail['sender']}</span>
        <span style="background-color:#1c1c1c; color:{mail['tag_color']}; font-size:10px; font-weight:800; letter-spacing:1px; padding:3px 8px; border-radius:999px; border:1px solid #2a2a2a;">
            {mail['tag']} • {mail['date']}
        </span>
    </div>
    <div style="font-size:14px; font-weight:800; color:#FFFFFF; margin-bottom:4px;">
        {mail['subject']}
    </div>
    <div style="font-size:12px; color:#888888; line-height:1.4;">
        {mail['snippet']}
    </div>
</div>
""", unsafe_allow_html=True)
            if st.button(f"SELECT THIS MAIL → ({mail['subject'][:28]}...)", key=f"btn_{mail['id']}", type="secondary", use_container_width=True):
                st.session_state.sample_input = mail['body']
                st.session_state.show_mailbox = False
                st.rerun()

    with tab_live:
        st.markdown('<div style="font-size:12px; color:#999; margin-bottom:12px;">Fetch recent live emails from your actual personal mailbox via secure IMAP:</div>', unsafe_allow_html=True)
        col_prov, col_port = st.columns([2, 1])
        with col_prov:
            provider = st.selectbox("Mail Provider", ["Gmail (imap.gmail.com)", "Outlook (outlook.office365.com)", "Yahoo (imap.mail.yahoo.com)", "Custom IMAP"])
        with col_port:
            port = st.number_input("Port (SSL)", value=993)

        col_folder, col_cnt = st.columns([1.5, 1])
        with col_folder:
            folder_choice = st.selectbox("Mail Source Folder", ["🔥 Both (Inbox + Spam Folder)", "🚨 Spam Folder Only", "📥 Inbox Only"])
        with col_cnt:
            fetch_limit = st.number_input("Emails to Fetch", value=20, min_value=5, max_value=50, step=5)

        host_map = {
            "Gmail (imap.gmail.com)": "imap.gmail.com",
            "Outlook (outlook.office365.com)": "outlook.office365.com",
            "Yahoo (imap.mail.yahoo.com)": "imap.mail.yahoo.com"
        }
        host = host_map.get(provider, "")
        if not host:
            host = st.text_input("Custom IMAP Host", value="imap.example.com")

        col_u, col_p = st.columns(2)
        with col_u:
            user_email = st.text_input("Email Address", placeholder="you@gmail.com")
        with col_p:
            app_pw = st.text_input("App Password", type="password", placeholder="Google / Outlook App Password")

        # Smart provider mismatch warning
        if user_email and "@gmail.com" in user_email.lower() and "Gmail" not in provider:
            st.warning("⚠️ You entered a **@gmail.com** address with Outlook/Yahoo selected! Please switch the **Mail Provider** dropdown above to **Gmail (imap.gmail.com)**.")
        elif user_email and any(dom in user_email.lower() for dom in ["@outlook.com", "@hotmail.com", "@live.com"]) and "Outlook" not in provider:
            st.warning("⚠️ You entered an Outlook address with Gmail selected! Please switch the **Mail Provider** dropdown above to **Outlook (outlook.office365.com)**.")

        st.caption("ℹ️ For Gmail, generate an 'App Password' under Google Account > Security > 2-Step Verification.")

        if st.button("FETCH LIVE EMAILS", type="secondary"):
            if not user_email or not app_pw:
                st.warning("Please enter your email and app password.")
            else:
                with st.spinner(f"Connecting to live mail server & fetching {folder_choice}..."):
                    success, data = fetch_live_imap_emails(host, port, user_email, app_pw, limit=int(fetch_limit), provider=provider, folder_mode=folder_choice)
                    if success:
                        st.session_state.live_emails = data
                        if not data:
                            st.info("Connected successfully! No emails found in selected folder(s).")
                        else:
                            st.success(f"Fetched {len(data)} emails from your account ({folder_choice})!")
                    else:
                        st.error(f"Connection failed: {data}")

        if st.session_state.live_emails:
            st.markdown('<div style="font-size:12px; font-weight:800; color:#E2F835; margin-top:14px; margin-bottom:8px;">YOUR FETCHED LIVE EMAILS:</div>', unsafe_allow_html=True)
            for idx, lmail in enumerate(st.session_state.live_emails):
                st.markdown(f"""
<div class="email-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:12px; font-weight:700; color:#CCCCCC;">{lmail['sender']}</span>
        <span style="background-color:#1c1c1c; color:{lmail.get('tag_color', '#00FF66')}; font-size:10px; font-weight:800; letter-spacing:1px; padding:3px 8px; border-radius:999px; border:1px solid #2a2a2a;">
            {lmail.get('tag', 'LIVE')} • {lmail.get('date', '')}
        </span>
    </div>
    <div style="font-size:14px; font-weight:800; color:#FFFFFF; margin-top:4px;">{lmail['subject']}</div>
    <div style="font-size:12px; color:#888888; margin-top:3px; line-height:1.4;">{lmail['snippet']}</div>
</div>
""", unsafe_allow_html=True)
                if st.button(f"SELECT THIS LIVE MAIL → ({lmail['subject'][:28]}...)", key=f"btn_live_{idx}", type="secondary", use_container_width=True):
                    st.session_state.sample_input = lmail['body']
                    st.session_state.show_mailbox = False
                    st.rerun()

    with tab_file:
        st.markdown('<div style="font-size:12px; color:#999; margin-bottom:10px;">Optional: upload a saved email file from disk:</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload .eml, .txt, .csv", type=['eml', 'txt', 'csv'], key="fallback_uploader")
        if uploaded_file is not None:
            f_bytes = uploaded_file.read()
            fname = uploaded_file.name.lower()
            text_extracted = ""
            if fname.endswith('.eml'):
                try:
                    pmsg = email.message_from_bytes(f_bytes, policy=email.policy.default)
                    subj = pmsg['subject'] or "No Subject"
                    sndr = pmsg['from'] or "Unknown Sender"
                    b = ""
                    if pmsg.is_multipart():
                        for p in pmsg.walk():
                            if p.get_content_type() == 'text/plain':
                                b = p.get_payload(decode=True).decode('utf-8', errors='ignore')
                                break
                    else:
                        b = pmsg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    text_extracted = f"From: {sndr}\nSubject: {subj}\n\n{b.strip()}"
                except Exception:
                    text_extracted = f_bytes.decode('utf-8', errors='ignore')
            else:
                text_extracted = f_bytes.decode('utf-8', errors='ignore')

            if text_extracted and text_extracted != st.session_state.sample_input:
                st.session_state.sample_input = text_extracted
                st.session_state.show_mailbox = False
                st.rerun()


# Text Area Input
input_sms = st.text_area(
    "SMS / Email Message Payload",
    value=st.session_state.sample_input,
    height=140,
    placeholder="Paste or type suspicious message text here, or click CHOOSE FROM MAIL above..."
)

# Scan Button
scan_clicked = st.button("SCAN MESSAGE →", type="primary")

if scan_clicked:
    if not input_sms.strip():
        st.markdown("""
        <div style="background-color:#241a0b; border:1px solid #FFB800; color:#FFB800; border-radius:12px; padding:14px; font-weight:700; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-top:16px;">
            ⚠️ Input payload is empty. Please enter text or click 'CHOOSE FROM MAIL' above.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Check if user pasted a CSV or multi-line batch dataset
        is_batch, batch_items = parse_batch_input(input_sms)

        if is_batch and len(batch_items) > 1:
            st.markdown(f"""
<div style="background-color:#161616; border:1px solid #2A2A2A; border-radius:14px; padding:18px; margin-top:20px; margin-bottom:16px;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:13px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; color:#E2F835;">
            📊 BATCH / DATASET SCAN COMPLETE
        </span>
        <span style="background-color:#222; color:#AAA; font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px;">
            {len(batch_items)} EMAILS ANALYZED
        </span>
    </div>
    <div style="font-size:12px; color:#888; margin-top:4px;">
        Detected multi-line CSV dataset. Evaluated each message individually:
    </div>
</div>
""", unsafe_allow_html=True)

            spam_count = 0
            ham_count = 0
            batch_rows = []
            for idx, item in enumerate(batch_items, 1):
                raw_text = item['text']
                expected = item['label']
                trans_text = transform_text(raw_text)
                vec = tfidf.transform([trans_text])
                pred = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]
                sp_pct = probs[1] * 100

                is_sp = (pred == 1) or (sp_pct >= 40.0)
                if is_sp:
                    spam_count += 1
                    verdict_badge = '<span style="color:#FF334B; font-weight:800;">🚨 SPAM</span>'
                else:
                    ham_count += 1
                    verdict_badge = '<span style="color:#00FF66; font-weight:800;">✅ NOT SPAM</span>'

                match_col = ""
                if expected:
                    is_correct = (expected == 'spam' and is_sp) or (expected == 'ham' and not is_sp)
                    match_col = f'<span style="color:{"#00FF66" if is_correct else "#FFB800"}; font-weight:700;">{expected.upper()} {"✓" if is_correct else "✗"}</span>'
                else:
                    match_col = "-"

                snippet_text = (raw_text[:65] + '...') if len(raw_text) > 65 else raw_text
                batch_rows.append(f"""
<tr style="border-bottom:1px solid #222; font-size:12px;">
    <td style="padding:10px; color:#777;">{idx}</td>
    <td style="padding:10px; color:#EEE;">{snippet_text}</td>
    <td style="padding:10px;">{match_col}</td>
    <td style="padding:10px;">{verdict_badge}</td>
    <td style="padding:10px; font-weight:800; color:{'#FF334B' if is_sp else '#00FF66'};">{sp_pct:.1f}%</td>
</tr>
""")

            table_html = f"""
<table style="width:100%; border-collapse:collapse; background-color:#141414; border-radius:12px; overflow:hidden; border:1px solid #242424;">
    <thead>
        <tr style="background-color:#1E1E1E; text-align:left; font-size:11px; letter-spacing:1px; text-transform:uppercase; color:#888;">
            <th style="padding:10px;">#</th>
            <th style="padding:10px;">Message Text</th>
            <th style="padding:10px;">Dataset Label</th>
            <th style="padding:10px;">Model Verdict</th>
            <th style="padding:10px;">Spam Risk</th>
        </tr>
    </thead>
    <tbody>
        {''.join(batch_rows)}
    </tbody>
</table>
"""
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown(f"""
<div class="specs-grid" style="margin-top:20px;">
    <div class="spec-box">
        <div class="spec-number">{len(batch_items)}</div>
        <div class="spec-label">Total Scanned</div>
    </div>
    <div class="spec-box">
        <div class="spec-number" style="color:#FF334B;">{spam_count}</div>
        <div class="spec-label">Spam Blocked</div>
    </div>
    <div class="spec-box">
        <div class="spec-number" style="color:#00FF66;">{ham_count}</div>
        <div class="spec-label">Ham Cleared</div>
    </div>
</div>
""", unsafe_allow_html=True)

        else:
            # Single Message Preprocessing
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            prediction = model.predict(vector_input)[0]
            probabilities = model.predict_proba(vector_input)[0]

            ham_conf = probabilities[0] * 100
            spam_conf = probabilities[1] * 100
            is_spam = (prediction == 1) or (spam_conf >= 40.0)

            # Display Result
            if is_spam:
                st.markdown("""
<div class="result-card-spam">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:12px; font-weight:800; letter-spacing:2px; color:#FF334B; text-transform:uppercase;">VERDICT CODE: #THREAT_01</span>
        <span style="background-color:#FF334B; color:#000; font-size:11px; font-weight:800; letter-spacing:1.5px; padding:4px 10px; border-radius:999px;">BLOCKED</span>
    </div>
    <div class="result-headline spam-text" style="margin-top:10px;">SPAM DETECTED</div>
    <div style="color:#BBBBBB; font-size:14px; margin-top:6px;">High probability spam pattern identified. Unsafe to interact.</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="result-card-ham">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:12px; font-weight:800; letter-spacing:2px; color:#00FF66; text-transform:uppercase;">VERDICT CODE: #VERIFIED_SAFE</span>
        <span style="background-color:#00FF66; color:#000; font-size:11px; font-weight:800; letter-spacing:1.5px; padding:4px 10px; border-radius:999px;">CLEARED</span>
    </div>
    <div class="result-headline ham-text" style="margin-top:10px;">NOT SPAM (HAM)</div>
    <div style="color:#BBBBBB; font-size:14px; margin-top:6px;">Message content conforms to authentic natural language patterns.</div>
</div>
""", unsafe_allow_html=True)

            # Technical breakdown specs
            display_conf = spam_conf if is_spam else ham_conf
            st.markdown(f"""
<div class="specs-grid">
    <div class="spec-box">
        <div class="spec-number">{len(input_sms.split())}</div>
        <div class="spec-label">Word Count</div>
    </div>
    <div class="spec-box">
        <div class="spec-number">{len(transformed_sms.split())}</div>
        <div class="spec-label">Stemmed Tokens</div>
    </div>
    <div class="spec-box">
        <div class="spec-number">{display_conf:.1f}%</div>
        <div class="spec-label">Confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Technical Specs
st.markdown("""
<div class="specs-grid" style="margin-top: 36px;">
    <div class="spec-box">
        <div class="spec-number">100%</div>
        <div class="spec-label">Model Precision</div>
    </div>
    <div class="spec-box">
        <div class="spec-number">3,000</div>
        <div class="spec-label">TF-IDF Features</div>
    </div>
    <div class="spec-box">
        <div class="spec-number">MNB</div>
        <div class="spec-label">Classifier Core</div>
    </div>
</div>

<div class="nike-footer">
    EMAIL SPAM DETECTION // TEAM-8 // MULTINOMIAL NAIVE BAYES & TF-IDF ENGINE
</div>
""", unsafe_allow_html=True)
