from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# ===========================
# DEVICE
# ===========================

device="cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# LOAD MODELS
# ===========================

embedder=SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device=device
)

reranker=CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device=device
)

# ===========================
# LABELS
# ===========================


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

# ===========================
# EMBED ONCE
# ===========================

label_embeddings=embedder.encode(
    CATEGORIES,
    normalize_embeddings=True
)

# ===========================
# PREDICTION
# ===========================

def classify_review(
        review,
        top_k=3
):

    review_embedding=embedder.encode(
        review,
        normalize_embeddings=True
    )

    similarity=cosine_similarity(
        [review_embedding],
        label_embeddings
    )[0]

    top5=np.argsort(
        similarity
    )[::-1][:5]

    candidates=[
        CATEGORIES[i]
        for i in top5
    ]

    pairs=[
        [review,label]
        for label in candidates
    ]

    scores=reranker.predict(
        pairs
    )

    ranked=sorted(
        zip(
            candidates,
            scores
        ),
        key=lambda x:x[1],
        reverse=True
    )

    return [

        {
            "category":label,
            "score":round(
                float(score),
                4
            )
        }

        for label,score in ranked[:top_k]
    ]


review="""
Vehicle was not cleaned properly,
dashboard still had dust,
and support ignored me.
"""

result=classify_review(
    review
)

print(result)