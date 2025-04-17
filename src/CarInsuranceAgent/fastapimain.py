import sys
package = __import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import markdown  # Make sure this is installed: `pip install markdown`
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from tools.see_img import describe_image
from car_value_estimation_setup import CarValueEstimatorCrew
from insurance_info_response_setup import InsuranceInfoResponderCrew
from image_car_analysis_setup import ImageCarAnalysisCrew

app = FastAPI(title="🚗 Car Insurance Assistant API")

@app.post("/estimate-car-value")
async def estimate_car_value(car_details: str = Form(...)):
    try:
        result = CarValueEstimatorCrew().crew().kickoff(inputs={"car_details": car_details})
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask-insurance-question")
async def ask_insurance_question(user_question: str = Form(...)):
    try:
        result = InsuranceInfoResponderCrew().crew().kickoff(inputs={"user_question": user_question})
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/image-car-analysis")
async def image_car_analysis(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        description = describe_image(image_bytes)

        result = ImageCarAnalysisCrew().crew().kickoff(
            inputs={"image_description": description}
        )

        with open("report.md", "r") as f:
            md_content = f.read()

        return {
            "description": description,
            "agent_result": result,
            "report_markdown": md_content
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/report", response_class=HTMLResponse)
async def get_report_html():
    try:
        with open("report.md", "r") as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        return f"<html><body>{html}</body></html>"
    except Exception as e:
        return HTMLResponse(f"<p>Error: {e}</p>", status_code=500)

@app.get("/")
async def root():
    return {"message": "Welcome to the Car Insurance Assistant API!"}

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204)
