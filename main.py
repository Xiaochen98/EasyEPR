from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.optimize import minimize

app = FastAPI()

# 跨域（给其他测试者用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以限制为你的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 提供静态前端页面（根路径访问 index.html）
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


# POST 接口用于模型拟合
@app.post("/fit")
def fit_soil_model(data: dict):
    spacings = np.array(data["spacings"])
    resistances = np.array(data["resistances"])
    rho_measured = resistances * spacings * 2 * np.pi

    def solveRHO(s, p1, p2, h):
        if p1 == 0 or p2 == 0 or h == 0:
            return np.nan
        pi = np.pi
        epsilon = 1e-6
        K = (p2 - p1) / (p2 + p1)

        s1 = s2 = 0
        for i in range(1, 10000):
            t1 = K**i / np.sqrt(s**2 + (2 * i * h)**2)
            s1 += t1
            if t1 < epsilon:
                break
        for i in range(1, 10000):
            t2 = K**i / np.sqrt(4 * s**2 + (2 * i * h)**2)
            s2 += t2
            if t2 < epsilon:
                break

        Rw = p1 / (2 * pi * s) + (2 * p1 / pi) * s1 - (2 * p1 / pi) * s2
        return 2 * pi * s * Rw

    def model_vectorized(a_array, rho1, rho2, h):
        return np.array([solveRHO(a, rho1, rho2, h) for a in a_array])

    def loss(params):
        rho1, rho2, h = params
        pred = model_vectorized(spacings, rho1, rho2, h)
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
