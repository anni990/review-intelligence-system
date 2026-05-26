from transformers import pipeline
import torch

# ==========================
# DEVICE CONFIG
# ==========================

device = 0 if torch.cuda.is_available() else -1

print("GPU Available:", torch.cuda.is_available())

# ==========================
# LOAD MODEL
# ==========================

MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

classifier = pipeline(
    task="zero-shot-classification",
    model=MODEL_NAME,
    device=device,
    framework="pt"
)

# ==========================
# 50 CUSTOMER REVIEW BUCKETS
# ==========================

CATEGORIES = [

    "Late Service",
    "Poor Cleaning Quality",
    "Bad Customer Support",
    "Missed Appointment",
    "Rude Behavior",
    "Damage Complaint",
    "Incomplete Service",
    "Payment Issue",
    "Communication Issue",
    "Unprofessional Staff",

    "Overpriced Service",
    "Hidden Charges",
    "Vehicle Odor Issue",
    "Strong Chemical Smell",
    "Dirty Cleaning Equipment",
    "Weak Pressure Wash",
    "Poor Vacuum Cleaning",
    "Weak Foam Wash",
    "Wet Interior",
    "Incomplete Drying",

    "Wrong Service Package",
    "Repeated Rescheduling",
    "Long Waiting Time",
    "Poor Service Coordination",
    "No Follow Up",
    "No Verification Call",
    "No Service Reminder",
    "Unresponsive Staff",
    "Machine Issue",
    "Lack Of Training",

    "No Professionalism",
    "Cheap Cleaning Material",
    "Poor Product Quality",
    "Low Quality Cloth",
    "No Proper Tools",
    "No Attention To Details",
    "Dust Left Behind",
    "Stain Marks",
    "Uneven Cleaning",
    "Poor Interior Cleaning",

    "Unclean Seats",
    "Mess Left Behind",
    "Incorrect Billing",
    "Water Leakage Issue",
    "Bad First Experience",
    "Poor Overall Experience",
    "Service Rushed",
    "Glass Cleaning Issue",
    "Tyre Cleaning Issue",
    "Dashboard Cleaning Issue"
]

# ==========================
# CLASSIFICATION FUNCTION
# ==========================

def classify_review(
    review,
    top_k=5
):

    result = classifier(
        review,
        CATEGORIES,
        multi_label=True,
        hypothesis_template="This customer complaint is about {}."
    )

    predictions = []

    for label, score in zip(
        result["labels"],
        result["scores"]
    ):

        if score > 0.30:
            predictions.append({
                "category": label,
                "confidence": round(score,4)
            })

    return predictions[:top_k]


# ==========================
# EXAMPLES
# ==========================

sample_reviews = [

    "Vehicle was not cleaned properly. Dashboard still had dust and tyres were dirty.",

    "Executive arrived 2 hours late and customer support never answered my calls.",

    "Paid for premium package but polishing and vacuum cleaning were missing.",

    "Payment deducted twice and no response from support.",

    "Cleaner behaved very rudely and scratched my car."
]

for i, review in enumerate(sample_reviews):

    print("\n")
    print("="*70)

    print("Review:", review)

    output = classify_review(review)

    print("\nPredictions:")

    for x in output:

        print(
            f"{x['category']} : {x['confidence']}"
        )