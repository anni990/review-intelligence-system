import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app.models import Job, JobStatus


class JobQueue:
    """
    In-memory job queue manager for handling asynchronous job processing.
    Manages job lifecycle: creation, status updates, retrieval, and cleanup.
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def add_job(
        self,
        reviews: list,
        total_reviews: int
    ) -> str:
        """
        Create a new job and add to queue.
        
        Args:
            reviews: List of review dictionaries to analyze
            total_reviews: Total number of reviews in the batch
            
        Returns:
            Job ID (UUID string)
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow(),
            total_reviews=total_reviews,
            reviews=reviews
        )

        async with self._lock:
            self._jobs[job_id] = job

        await self._queue.put(job_id)
        return job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Job object or None if not found
        """
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        results: Optional[List] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update job status and optionally add results or error.
        
        Args:
            job_id: Unique job identifier
            status: New JobStatus
            results: List of ReviewAnalysis results (if completed)
            error: Error message (if failed)
            
        Returns:
            True if successful, False if job not found
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            job.status = status
            
            if status == JobStatus.COMPLETED:
                job.completed_at = datetime.utcnow()
                job.results = results

            if status == JobStatus.FAILED:
                job.completed_at = datetime.utcnow()
                job.error = error

            return True

    async def get_next_job(self) -> Optional[str]:
        """
        Get the next job ID from the queue (blocking).
        
        Returns:
            Job ID or None if queue is empty
        """
        try:
            job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            return job_id
        except asyncio.TimeoutError:
            return None

    async def list_jobs(self) -> List[Job]:
        """
        Get all jobs, ordered by creation time (newest first).
        
        Returns:
            List of Job objects
        """
        async with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )

    async def cleanup_old_jobs(self, hours: int = 24) -> int:
        """
        Remove completed jobs older than specified hours.
        
        Args:
            hours: Age threshold in hours (default 24)
            
        Returns:
            Number of jobs deleted
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        deleted_count = 0

        async with self._lock:
            to_delete = []
            
            for job_id, job in self._jobs.items():
                if (
                    job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
                    and job.completed_at
                    and job.completed_at < cutoff_time
                ):
                    to_delete.append(job_id)

            for job_id in to_delete:
                del self._jobs[job_id]
                deleted_count += 1

        return deleted_count

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job immediately (for cancellation).
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    async def queue_size(self) -> int:
        """Get current queue size (number of queued jobs)."""
        return self._queue.qsize()

    async def total_jobs(self) -> int:
        """Get total number of jobs (including completed/failed)."""
        async with self._lock:
            return len(self._jobs)
