from fastapi import FastAPI

app = FastAPI()

@app.post("/test")
async def test_post():
    return {"message": "POST works!"}

@app.get("/")
async def test_get():
    return {"message": "GET works!"}