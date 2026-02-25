from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uuid

security = HTTPBearer()

db = {}
active_tokens = {}

class User(BaseModel):
    username: str
    password: str


class RegisterUser(BaseModel):
    username: str
    password: str


class ChangePassword(BaseModel):
    old_password: str
    new_password: str

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register")
async def register(user: RegisterUser):
    if user.username in db:
        raise HTTPException(status_code=400, detail="User already exists")

    db[user.username] = user.password
    return {"message": "User registered successfully"}

@router.post("/login")
async def login(user: User):
    db_password = db.get(user.username)

    if not db_password or db_password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    active_tokens[token] = user.username

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer"
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    username = active_tokens.get(token)

    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return username

@router.get("/me")
async def get_me(username: str = Depends(get_current_user)):
    return {"username": username}

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    if token in active_tokens:
        del active_tokens[token]

    return {"message": "Logged out successfully"}

@router.post("/change-password")
async def change_password(
    data: ChangePassword,
    username: str = Depends(get_current_user)
):
    current_password = db.get(username)

    if current_password != data.old_password:
        raise HTTPException(status_code=400, detail="Old password incorrect")

    db[username] = data.new_password
    return {"message": "Password changed successfully"}