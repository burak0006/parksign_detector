from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model.api import extract, _read
from pydantic import BaseModel
from pathlib import Path
import os
import uvicorn

app = FastAPI()

CURRENTDIR = Path(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(CURRENTDIR, "templates", "index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")


async def validate_image(file: UploadFile):
    content_type = file.content_type.split("/")[0].lower()
    if content_type != "image":
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed.")


class Result(BaseModel):
    filename: str
    result: str


class ImageUpload(BaseModel):
    image: UploadFile


@app.get("/")
def home():
    with open(TEMPLATE, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/upload", response_model=Result)
async def upload_image(file: UploadFile = File(...)):
    await validate_image(file)

    contents = file.file.read()
    image = _read(contents)
    result = extract(image)
    print(result)

    image_path = os.path.join("static", file.filename)
    with open(image_path, "wb") as f:
        f.write(contents)

    ocr_path = os.path.join("static", file.filename.split(".")[0] + ".txt")
    with open(ocr_path, "wb") as f:
        f.write(str(result).encode())

    return {"filename": file.filename, "result": result}


@app.get("/result/{filename}")
async def get_result(filename):
    ocr_path = os.path.join("static", filename.split(".")[0] + ".txt")
    with open(ocr_path, "r") as f:
        ocr_text = f.read()

    return HTMLResponse(content=f"<h3>OCR Results for {filename}:</h3><p>{ocr_text}</p>")


@app.get("/results")
async def get_results():
    html_content = "<html><body>"
    # Get a list of all files in the "static" directory
    files = os.listdir("static")
    # Filter files by extension (.png and .jpg)
    img_files = [f for f in files if f.endswith(".png") or f.endswith(".jpg")]
    # Loop through the image files
    for img in img_files:
        # Get the OCR results file path
        ocr_path = os.path.join("static", os.path.splitext(img)[0] + ".txt")
        # Read the OCR results from the file
        with open(ocr_path, "r") as f:
            ocr_text = f.read()
        # Add the image and OCR results to the HTML content
        html_content += f"<h3>OCR Results for {img}:</h3><img src='static/{img}' width='500' height='500'><p>{ocr_text}</p>"
    html_content += "</body></html>"
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
