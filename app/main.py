import json
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from app.llm_service import analyze_batch
from app.llm_service import chunk_reviews
from app.models import FinalResponse


app=FastAPI(

 title="Review Intelligence API",

 description="LLM based qualitative review understanding service",

 version="1.0"
)


@app.post(
    "/upload-review-json",
    response_model=FinalResponse
)
async def upload_reviews(
    file:UploadFile=File(...)
):

    content=await file.read()

    reviews=json.loads(content)


    final=[]

    batches=0


    for batch in chunk_reviews(
            reviews,
            size=3
    ):

        result=await analyze_batch(batch)

        final.extend(result)

        batches+=1


    return {
        "total_reviews":len(reviews),
        "processed_batches":batches,
        "results":final
    }


@app.get("/")
def home():

    return {
      "message":"Review intelligence service running"
    }