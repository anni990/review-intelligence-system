import asyncio
import logging
from typing import TYPE_CHECKING

from app.models import JobStatus
from app.llm_service import analyze_batch, chunk_reviews

if TYPE_CHECKING:
    from app.queue_service import JobQueue


logger = logging.getLogger(__name__)


class JobWorker:
    """
    Background worker that processes jobs from the queue sequentially.
    Handles job state transitions and error handling.
    """

    def __init__(self, queue: "JobQueue"):
        """
        Initialize the worker with a job queue.
        
        Args:
            queue: JobQueue instance to process jobs from
        """
        self.queue = queue
        self._running = False

    async def start(self) -> None:
        """
        Start the background worker loop.
        Runs continuously and processes jobs sequentially.
        """
        self._running = True
        logger.info("Job worker started")
        
        try:
            while self._running:
                job_id = await self.queue.get_next_job()
                
                if job_id:
                    await self.process_job(job_id)
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            logger.info("Job worker stopped")

    async def process_job(self, job_id: str) -> None:
        """
        Process a single job: analyze reviews and update job status.
        
        Args:
            job_id: The job ID to process
        """
        try:
            # Get job from queue
            job = await self.queue.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return

            # Update status to processing
            await self.queue.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"Processing job {job_id} with {job.total_reviews} reviews")

            # Process reviews in batches
            all_results = []
            batch_count = 0

            for batch in chunk_reviews(job.reviews, size=3):
                try:
                    batch_results = await analyze_batch(batch)
                    all_results.extend(batch_results)
                    batch_count += 1
                except Exception as batch_error:
                    logger.error(f"Batch processing error for job {job_id}: {batch_error}")
                    await self.queue.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error=f"Batch {batch_count + 1} processing failed: {str(batch_error)}"
                    )
                    return

            # Update job with results
            await self.queue.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                results=all_results
            )
            
            # Update processed_batches in job
            job = await self.queue.get_job(job_id)
            if job:
                job.processed_batches = batch_count
            
            logger.info(f"Completed job {job_id}: {batch_count} batches, {len(all_results)} results")

        except Exception as e:
            logger.error(f"Job processing error for {job_id}: {e}")
            await self.queue.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=f"Unexpected error: {str(e)}"
            )

    async def stop(self) -> None:
        """Stop the background worker loop."""
        self._running = False
        logger.info("Stopping job worker...")
