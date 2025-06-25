from fastapi import FastAPI, Request, Form
from .routes import router
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from config import forecast_options
from api.routes import forecast
from api.schemas import ForecastRequest

app = FastAPI(title='Freight Forecast API')
templates = Jinja2Templates(directory='templates')
app.include_router(router)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # return {"message": "Welcome to Freight Forecast API"}
    return templates.TemplateResponse("form.html", {
        "request": request,
        "forecast_options": list(forecast_options.keys()),
        "result": None
    })

@app.post("/submit", response_class=HTMLResponse)
async def handle_form(request: Request,
                      year: int = Form(...),
                      target_label: str = Form(...),
                      MEHOINUSGAA672N: float = Form(...),
                      GARETAILNQGSP: float = Form(...),
                      Population: float = Form(...)
                      ):

    data_dict = {
        "year": year,
        "target_label": target_label,
        "features": {
            "MEHOINUSGAA672N": MEHOINUSGAA672N,
            "GARETAILNQGSP": GARETAILNQGSP,
            "Population": Population
        }
    }

    request_obj = ForecastRequest(**data_dict)
    result = forecast(request_obj)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "forecast_options": list(forecast_options.keys()),
        "result": result
    })


@app.get("/version")
async def get_version():
    return {"version": "1.0.0"}
