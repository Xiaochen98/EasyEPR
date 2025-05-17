from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize

app = FastAPI()

# CORS 设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端页面（index.html）
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# 请求数据模型
class FitRequest(BaseModel):
    spacings: list[float]
    resistances: list[float]

# 模型拟合接口
@app.post("/fit")
def fit_soil_model(data: FitRequest):
    a_vals = np.array(data.spacings)
    resistances = np.array(data.resistances)
    rho_measured = resistances * a_vals * 2 * np.pi

    def solveRHO(s, p1, p2, h):
        if p1 == 0 or p2 == 0 or h == 0:
            return np.nan
        pi = np.pi
        epsilon = 1e-6
        K = (p2 - p1) / (p2 + p1)

        s1 = 0
        delta = 1
        i = 1
        while delta > epsilon and i <= 10000:
            prev = s1
            term = K**i / np.sqrt(s**2 + (2 * i * h)**2)
            s1 += term
            delta = abs(s1 - prev)
            i += 1

        s2 = 0
        delta = 1
        i = 1
        while delta > epsilon and i <= 10000:
            prev = s2
            term = K**i / np.sqrt(4 * s**2 + (2 * i * h)**2)
            s2 += term
            delta = abs(s2 - prev)
            i += 1

        Rw = p1 / (2 * pi * s) + (2 * p1 / pi) * s1 - (2 * p1 / pi) * s2
        return 2 * pi * s * Rw

    def model_vectorized(a_array, rho1, rho2, h):
        return np.array([solveRHO(a, rho1, rho2, h) for a in a_array])

    def loss(params):
        rho1, rho2, h = params
        pred = model_vectorized(a_vals, rho1, rho2, h)
        return np.sum(((pred - rho_measured) / rho_measured) ** 2)

    x0 = [100, 50, 1]
    bounds = [(1, 2000), (1, 2000), (0.01, 32)]
    result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    rho1_fit, rho2_fit, h_fit = result.x

    a_plot = np.linspace(0.1, 40, 200)
    rho_fit_plot = model_vectorized(a_plot, rho1_fit, rho2_fit, h_fit)

    return {
        "params": {"rho1": rho1_fit, "rho2": rho2_fit, "h": h_fit},
        "curve": [{"a": float(a), "rho": float(r)} for a, r in zip(a_plot, rho_fit_plot)],
        "loss": float(result.fun)
    }
