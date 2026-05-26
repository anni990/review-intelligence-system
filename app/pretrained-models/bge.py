from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# ======================
# DEVICE
# ======================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Running on:", device)

# ======================
# LOAD MODEL
# ======================

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=device
)

# ======================
# CATEGORY LABELS
# ======================

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

# ======================
# EMBED LABELS ONCE
# ======================

category_embeddings = model.encode(
    CATEGORIES,
    normalize_embeddings=True
)

# ======================
# PREDICT
# ======================

def classify_review(
    review,
    top_k=5
):

    review_embedding = model.encode(
        review,
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [review_embedding],
        category_embeddings
    )[0]

    top_indices = np.argsort(
        similarity
    )[::-1][:top_k]

    results=[]

    for idx in top_indices:

        results.append({
            "category":CATEGORIES[idx],
            "score":round(
                float(similarity[idx]),
                4
            )
        })

    return results


# ======================
# TEST
# ======================

review="""
Vehicle was not cleaned properly.
Dashboard still had dust.
Support ignored complaint.
"""

results=classify_review(review)

print("\nResults:\n")

for r in results:
    print(r)