import os
import json
import asyncio
from google import genai
from dotenv import load_dotenv
from app.models import BatchResponse

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


MODEL_NAME = "gemini-3.1-pro-preview"

SYSTEM_PROMPT = """
You are an expert customer complaint intelligence analyst specializing in qualitative review analysis and sentiment extraction.

## ANALYSIS SCOPE

Analyze each review as an independent assessment. Extract the following dimensions:

### Core Understanding Dimensions
- User Intent: What is the customer trying to communicate?
- Customer Pain Points: What are the underlying frustrations?
- Experience Quality: How satisfied or dissatisfied is the customer?
- Hidden Issues: What problems are implied but not explicitly stated?
- Emotional Sentiment: What emotions are present (frustration, satisfaction, etc.)?
- Service Failures: What specific service breakdowns occurred?
- Root Causes: What fundamental issues led to this experience?

### Analytical Approach
- Go beyond literal text: infer underlying meanings and context
- Consider tone, language patterns, and implicit signals
- Identify systemic issues from individual experiences

## OUTPUT STRUCTURE

For each review, generate exactly these fields:

1. **Issue Categories** (dynamically created based on content) - atleast 3 per review, each with a confidence score (0-1) and more if applicable.
2. **Service Event** (what happened)
3. **Customer Experience** (quality of the experience)
4. **Emotion** (dominant emotional tone)
5. **Severity** (impact level of the issues)
6. **Root Cause** (underlying reason for the problem)
7. **Recommended Actions** (specific improvements)
8. **Overall Confidence** (0-1 scale: certainty in analysis)

## CRITICAL RULES

- Each review is analyzed independently; do not mix or compare reviews
- Confidence scores must be between 0 and 1
- Provide concise, focused outputs
- Return response in JSON format only
- Issue labels are dynamically created based on actual content, not predefined categories
"""

def chunk_reviews(data,size=3):

    for i in range(0,len(data),size):
        yield data[i:i+size]


async def analyze_batch(batch):

    payload=[]

    for idx,item in enumerate(batch):

        payload.append(
            {
                "review_id":idx+1,
                "rating":item["rating"],
                "review":item["review"]
            }
        )


    user_prompt=f"""
Analyze reviews:

{json.dumps(payload,indent=2)}
"""


    # Run the blocking LLM API call in a thread pool executor
    # This prevents blocking the event loop
    loop = asyncio.get_event_loop()
    
    def _call_llm():
        """Synchronous wrapper for blocking LLM API call."""
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                SYSTEM_PROMPT,
                user_prompt
            ],
            config={
                "response_mime_type":"application/json",
                "response_schema":BatchResponse
            }
        )
    
    response = await loop.run_in_executor(None, _call_llm)
    parsed=response.parsed

    return parsed.reviews