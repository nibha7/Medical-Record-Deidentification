
from faker import Faker
import random
import json
from pathlib import Path

# --------------------------------------------------
# 1. INITIAL SETUP
# --------------------------------------------------

fake = Faker("en_IN")
random.seed(42)
Faker.seed(42)

OUT = Path("data")
OUT.mkdir(exist_ok=True)

LABELS = [
    "O",
    "B-PATIENT", "I-PATIENT",
    "B-AGE", "I-AGE",
    "B-DATE", "I-DATE",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-ID", "I-ID",
    "B-LOCATION", "I-LOCATION",
    "B-HOSPITAL", "I-HOSPITAL",
    "B-DOCTOR", "I-DOCTOR"
]


# --------------------------------------------------
# 2. GENERATE ONE MEDICAL RECORD
# --------------------------------------------------

def make_record(i):
    patient = fake.name()
    doctor = "Dr. " + fake.name()

    hospital = random.choice([
        "Apollo Hospital",
        "Manipal Hospital",
        "Fortis Hospital",
        "Narayana Health",
        "Columbia Asia Hospital"
    ])

    city = random.choice([
        "Bengaluru",
        "Mangaluru",
        "Mysuru",
        "Chennai",
        "Hyderabad"
    ])

    age = str(random.randint(18, 82))

    # Generate a realistic date instead of a fixed date.
    date = fake.date_between(
        start_date="-5y",
        end_date="today"
    ).strftime("%d/%m/%Y")

    phone = "9" + "".join(
        str(random.randint(0, 9)) for _ in range(9)
    )

    email = (
        patient.lower().replace(" ", ".")
        + f"{i}@example.com"
    )

    mrn = f"MRN{random.randint(100000, 999999)}"

    text = ""
    entities = []

    # Add text and store exact character positions of each entity.
    def put(s, label=None):
        nonlocal text

        start = len(text)
        text += s
        end = len(text)

        if label:
            entities.append({
                "start": start,
                "end": end,
                "label": label
            })

    # --------------------------------------------------
    # 3. SELECT A SENTENCE STRUCTURE
    # --------------------------------------------------

    template = random.randint(1, 4)

    if template == 1:
        put("Patient ")
        put(patient, "PATIENT")
        put(" is ")
        put(age, "AGE")
        put(" years old and visited ")
        put(hospital, "HOSPITAL")
        put(" in ")
        put(city, "LOCATION")
        put(". The attending physician was ")
        put(doctor, "DOCTOR")
        put(". Appointment date: ")
        put(date, "DATE")
        put(". Medical record number: ")
        put(mrn, "ID")
        put(". Contact phone: ")
        put(phone, "PHONE")
        put(". Email: ")
        put(email, "EMAIL")

    elif template == 2:
        put("On ")
        put(date, "DATE")
        put(", ")
        put(patient, "PATIENT")
        put(", aged ")
        put(age, "AGE")
        put(", attended ")
        put(hospital, "HOSPITAL")
        put(" in ")
        put(city, "LOCATION")
        put(". The consultation was handled by ")
        put(doctor, "DOCTOR")
        put(". Contact details: ")
        put(phone, "PHONE")
        put(" and ")
        put(email, "EMAIL")
        put(". Patient reference: ")
        put(mrn, "ID")

    elif template == 3:
        put("Medical file ")
        put(mrn, "ID")
        put(" belongs to ")
        put(patient, "PATIENT")
        put(", age ")
        put(age, "AGE")
        put(". The patient lives in ")
        put(city, "LOCATION")
        put(" and received care at ")
        put(hospital, "HOSPITAL")
        put(". Responsible doctor: ")
        put(doctor, "DOCTOR")
        put(". Recorded date: ")
        put(date, "DATE")
        put(". Phone number ")
        put(phone, "PHONE")
        put(" | Email address ")
        put(email, "EMAIL")

    else:
        put("Please contact ")
        put(patient, "PATIENT")
        put(" regarding the visit to ")
        put(hospital, "HOSPITAL")
        put(", ")
        put(city, "LOCATION")
        put(". The patient is ")
        put(age, "AGE")
        put(" years old. For medical questions, ask ")
        put(doctor, "DOCTOR")
        put(". Date of visit: ")
        put(date, "DATE")
        put(". Phone: ")
        put(phone, "PHONE")
        put(". Email: ")
        put(email, "EMAIL")
        put(". Reference number: ")
        put(mrn, "ID")

    # Additional medical information
    put(
        ". Diagnosis: hypertension. "
        "Follow-up recommended in 2 weeks."
    )

    return {
        "id": i,
        "text": text,
        "entities": entities
    }


# --------------------------------------------------
# 4. SHUFFLE AND SPLIT THE DATASET
# --------------------------------------------------

def split_and_save(records):
    random.shuffle(records)

    total = len(records)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    datasets = [
        ("train.jsonl", train_records),
        ("val.jsonl", val_records),
        ("test.jsonl", test_records)
    ]

    for filename, data in datasets:
        filepath = OUT / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for record in data:
                f.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        print(f"Saved: {filepath} ({len(data)} records)")

    print("\nDataset generation completed.")
    print(f"Total records: {total}")
    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")


# --------------------------------------------------
# 5. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    records = [
        make_record(i)
        for i in range(1, 1201)
    ]

    split_and_save(records)
