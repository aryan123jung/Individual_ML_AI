from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.evaluation import router as evaluation_router
from app.routes.players import router as players_router
from app.routes.recommendations import router as recommendations_router
from app.routes.teams import router as teams_router


app = FastAPI(
    title="IPL Domestic Talent Recommendation API",
    version="1.0.0",
    description="Backend API for IPL domestic player recommendation, team gaps, and ML evaluation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations_router)
app.include_router(players_router)
app.include_router(teams_router)
app.include_router(evaluation_router)


@app.get("/")
def root():
    return {
        "message": "IPL Domestic Talent Recommendation API is running",
        "docs": "/docs",
        "routes": [
            "/recommendations",
            "/players/{player_name}",
            "/players/compare",
            "/teams",
            "/evaluation/metrics",
        ],
    }
