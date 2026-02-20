from pydantic import BaseModel
from fastapi import APIRouter, HTTPException


class User(BaseModel):
    username: str
    password: str
    
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
async def login(user: User):
    if user.username == "admin" and user.password == "password":
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
