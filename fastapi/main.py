from fastapi import FastAPI

from router import auth

main = FastAPI(title="FastAPI", 
               description="A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.", 
               version="0.1.0")

API_PREFIX = "/api"
main.include_router(auth.router , prefix=API_PREFIX)