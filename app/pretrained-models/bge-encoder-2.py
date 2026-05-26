from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from scipy.special import softmax
import json


# ======================================
# DEVICE
# ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Running on:", device)


# ======================================
# LOAD MODELS
# ======================================

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device=device
)


# ======================================
# CATEGORY + DESCRIPTION
# ======================================

LABELS = {

"Late Service":
"Complaint about delayed arrival or service being very late",

"Poor Cleaning Quality":
"Vehicle cleaning quality poor, dirt remained after wash",

"Bad Customer Support":
"Support team ignored issue or poor response",

"Missed Appointment":
"Booked appointment but nobody arrived",

"Rude Behavior":
"Staff behaved badly or spoke rudely",

"Damage Complaint":
"Vehicle scratched or damaged during service",

"Payment Issue":
"Payment failure, duplicate payment or refund issue",

"Communication Issue":
"No updates or poor communication",

"Unprofessional Staff":
"Staff lacked professionalism",

"Overpriced Service":
"Customer thinks service price is too high",

"Hidden Charges":
"Unexpected charges added later",

"Wrong Service Package":
"Different package delivered than booked",

"Incomplete Service":
"Some parts of service skipped",

"Repeated Rescheduling":
"Service repeatedly postponed",

"Long Waiting Time":
"Customer waited too long",

"No Follow Up":
"No response after complaint",

"No Verification Call":
"No confirmation before appointment",

"No Service Reminder":
"No reminder notification received",

"Unresponsive Staff":
"Staff ignored calls/messages",

"Machine Issue":
"Machine problem affected service",

"No Professionalism":
"Service process looked unorganized",

"Poor Product Quality":
"Cleaning products gave poor result",

"No Proper Tools":
"Staff lacked proper equipment",

"No Attention To Details":
"Small areas ignored",

"Dust Left Behind":
"Dust remained after cleaning",

"Stain Marks":
"Cleaning caused visible marks",

"Uneven Cleaning":
"Some areas clean, some dirty",

"Poor Interior Cleaning":
"Dashboard seats or interiors not cleaned",

"Service Rushed":
"Service completed carelessly and too fast",

"Tyre Cleaning Issue":
"Tyres remained dirty"

}


CATEGORY_NAMES = list(
    LABELS.keys()
)

CATEGORY_DESCRIPTIONS = list(
    LABELS.values()
)


# ======================================
# EMBED LABELS ONCE
# ======================================

label_embeddings = embedder.encode(
    CATEGORY_DESCRIPTIONS,
    normalize_embeddings=True
)


# ======================================
# CLASSIFIER
# ======================================

def classify_reviews(
        reviews,
        top_k=3,
        candidate_pool=5
):


    review_embeddings = embedder.encode(
        reviews,
        normalize_embeddings=True
    )


    all_results=[]


    for review,review_embedding in zip(
            reviews,
            review_embeddings
    ):


        similarity=cosine_similarity(
            [review_embedding],
            label_embeddings
        )[0]


        top_indices=np.argsort(
            similarity
        )[::-1][:candidate_pool]


        candidates=[
            CATEGORY_NAMES[i]
            for i in top_indices
        ]


        pairs=[

            [review,LABELS[x]]

            for x in candidates

        ]


        logits=reranker.predict(
            pairs
        )


        probs=softmax(
            logits
        )


        ranked=sorted(
            zip(
                candidates,
                probs
            ),
            key=lambda x:x[1],
            reverse=True
        )


        output=[]

        for rank,(label,score) in enumerate(
                ranked[:top_k],
                start=1
        ):

            output.append({

                "rank":rank,

                "label":label,

                "confidence":
                round(
                    float(score),
                    4
                )

            })


        all_results.append({

            "review":review,

            "predictions":output

        })


    return all_results


# ======================================
# MULTIPLE REVIEWS
# ======================================

reviews=[

"Vehicle was not cleaned properly and dashboard still had dust.",

"Executive came 2 hours late and support ignored my calls.",

"Paid for premium package but vacuum cleaning was missing.",

"Payment deducted twice and still waiting for refund.",

"Cleaner behaved rudely and scratched my car.",

"Tyres remained dirty and wash quality was poor"

]


results=classify_reviews(
    reviews
)

print(
    json.dumps(
        results,
        indent=4
    )
)