import json
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.models import JobResponse, Job, JobStatus
from app.queue_service import JobQueue
from app.worker import JobWorker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Review Intelligence System",
    description="LLM based qualitative review understanding service with job queue",
    version="2.0"
)


# Global job queue and worker instances
job_queue: JobQueue = None
job_worker: JobWorker = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize job queue and start background worker on app startup.
    """
    global job_queue, job_worker
    
    job_queue = JobQueue()
    job_worker = JobWorker(job_queue)
    
    # Start background worker task
    asyncio.create_task(job_worker.start())
    
    # Start cleanup task (runs every hour)
    asyncio.create_task(cleanup_task())
    
    logger.info("Application startup: job queue and worker initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Gracefully shut down the background worker.
    """
    global job_worker
    
    if job_worker:
        await job_worker.stop()
    
    logger.info("Application shutdown: job worker stopped")


async def cleanup_task():
    """
    Background task that removes completed jobs older than 24 hours.
    Runs periodically every hour.
    """
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            deleted = await job_queue.cleanup_old_jobs(hours=24)
            if deleted > 0:
                logger.info(f"Cleanup: removed {deleted} old jobs")
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")


@app.post("/upload-review-json")
async def upload_reviews(file: UploadFile = File(...)):
    """
    Upload a JSON file with reviews and create an asynchronous analysis job.
    
    Returns immediately with job ID and status (queued).
    Poll the /job/{job_id} endpoint to check processing status and results.
    
    Expected JSON format:
    [
        {"rating": 5, "review": "Great service!"},
        {"rating": 2, "review": "Poor experience."}
    ]
    """
    try:
        content = await file.read()
        reviews = json.loads(content)
        
        # Validate input
        if not isinstance(reviews, list):
            raise ValueError("JSON must be an array of reviews")
        
        if not reviews:
            raise ValueError("Reviews array cannot be empty")
        
        # Validate each review has required fields
        for idx, review in enumerate(reviews):
            if not isinstance(review, dict):
                raise ValueError(f"Review {idx} must be a dictionary")
            if "rating" not in review or "review" not in review:
                raise ValueError(
                    f"Review {idx} missing 'rating' or 'review' field"
                )
        
        # Create and queue the job
        job_id = await job_queue.add_job(reviews, len(reviews))
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Job created. Poll /job/{job_id} for status and results."
        }
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get the status and details of a specific job.
    
    Returns:
    - status: one of "queued", "processing", "completed", "failed"
    - results: (if completed) List of ReviewAnalysis objects
    - error: (if failed) Error message
    """
    job = await job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Convert Job to JobResponse (excludes internal fields)
    return JobResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        total_reviews=job.total_reviews,
        processed_batches=job.processed_batches,
        results=job.results,
        error=job.error
    )


@app.get("/jobs")
async def list_all_jobs():
    """
    List all jobs with their current status.
    Ordered by creation time (newest first).
    """
    jobs = await job_queue.list_jobs()
    
    return {
        "total_jobs": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "total_reviews": job.total_reviews,
                "processed_batches": job.processed_batches
            }
            for job in jobs
        ]
    }


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job (cancel if queued/processing, or remove if completed).
    """
    deleted = await job_queue.delete_job(job_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return {"message": f"Job {job_id} deleted successfully"}


# @app.get("/queue-stats")
# async def queue_stats():
#     """
#     Get current queue statistics.
#     """
#     queue_size = await job_queue.queue_size()
#     total_jobs = await job_queue.total_jobs()
    
#     return {
#         "queued_jobs": queue_size,
#         "total_jobs": total_jobs
#     }


@app.get("/")
def home():
    """
    Health check endpoint.
    """
    return {
        "message": "Review intelligence service running",
        "version": "2.0",
        "endpoints": {
            "upload": "POST /upload-review-json",
            "job_status": "GET /job/{job_id}",
            "list_jobs": "GET /jobs",
            "delete_job": "DELETE /job/{job_id}",
            "queue_stats": "GET /queue-stats"
        }
    }
