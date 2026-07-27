from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi import Body
from fastapi.staticfiles import StaticFiles
from LLM_QA_CLI import preprocess_text, query_llm  # Your existing functions

app = FastAPI(title="LLM Q&A Web App (Gemini-Powered)")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (place index.html in a 'templates' folder)
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(data: dict = Body(...)):
    question = data.get('question', '').strip()
    
    if not question:
        return JSONResponse(
            status_code=400,
            content={'error': 'No question provided'}
        )

    # Process the question (same as CLI)
    processed_question = preprocess_text(question)
    
    # Query Gemini LLM
    answer = query_llm(processed_question)
    
    return {
        'processed_question': processed_question,
        'answer': answer
    }
