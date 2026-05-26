# Review Intelligence System

## What

An LLM-powered REST API service that analyzes customer reviews using Google's Gemini AI. It extracts insights like issues, sentiment, root causes, and recommended actions from customer feedback.

## How

1. **Upload Reviews** - Submit a JSON file containing customer reviews
2. **Batch Processing** - Reviews are processed in batches (default: 3 reviews per batch)
3. **LLM Analysis** - Each batch is sent to Gemini for intelligent analysis
4. **Structured Output** - Returns detailed analysis including:
   - Issue categories
   - Service events and customer experience
   - Emotional sentiment
   - Severity and root causes
   - Recommended actions
   - Confidence scores

## How to Run

### Prerequisites
- Python 3.11+
- Google API Key (Gemini API)

### Setup

1. **Clone/navigate to project**
   ```bash
   cd d:\Projects\review-intelligence-system
   ```

2. **Activate virtual environment**
   ```bash
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or with uv:
   ```bash
   uv sync
   ```

4. **Set environment variables**
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the service**
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://localhost:8000`

### API Endpoints

- **POST** `/upload-review-json` - Upload and analyze reviews
  - Accept a JSON file with array of reviews
  - Each review should have: `rating` and `review` fields

- **GET** `/` - Health check endpoint

### Example Review JSON
```json
[
  {
    "rating": 2,
    "review": "The service was slow and staff was unhelpful. Waited 30 minutes for a simple request."
  },
  {
    "rating": 5,
    "review": "Excellent experience! Fast, friendly, and professional service throughout."
  }
]
```
