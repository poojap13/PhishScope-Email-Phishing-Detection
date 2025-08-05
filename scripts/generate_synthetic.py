import pandas as pd
from faker import Faker
import random
import os

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

faker = Faker()
n_samples = 500
synthetic_data = []

phishing_subjects = [
    "Urgent: Verify Your Account Now",
    "Security Alert: Unusual Login Detected",
    "Payment Failed – Update Details",
    "Your Account Has Been Locked",
    "Confirm Your Email to Continue Access"
]

phishing_bodies = [
    "Click here to verify your credentials immediately.",
    "Your bank account has been suspended. Log in here.",
    "You have an unpaid invoice. Open the attachment.",
    "We detected a suspicious login attempt. Please reset your password.",
    "Confirm your payment details to avoid service interruption."
]

for _ in range(n_samples):
    is_phish = random.choice([0, 1])
    subject = faker.sentence(nb_words=6) if not is_phish else random.choice(phishing_subjects)
    body = faker.paragraph() if not is_phish else random.choice(phishing_bodies)
    num_links = random.randint(0, 5) if not is_phish else random.randint(1, 8)
    links = ", ".join([faker.url() for _ in range(num_links)]) if num_links > 0 else ""
    synthetic_data.append({
        "sender": faker.email(),
        "subject": subject,
        "body": body,
        "links": links,
        "timestamp": faker.date_time_this_year(),
        "label": is_phish
    })

df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.to_csv("data/synthetic_emails.csv", index=False)
print("Synthetic dataset saved at data/synthetic_emails.csv with shape:", df_synthetic.shape)
