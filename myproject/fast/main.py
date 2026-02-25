from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy big data list
dummy_users = [
    {"id": i, "name": f"User {i}", "email": f"user{i}@example.com"}
    for i in range(1, 21)
]

@app.get("/htmx-get/", response_class=HTMLResponse)
def htmx_get(name: str = ""):
    
    if not name:
        return "<p>Type a name to search...</p>"

    name_lower = name.lower()

    # Add special custom data
    extended_users = dummy_users + [
        {"id": 101, "name": "Harsh", "email": "harsh@custom.com"},
        {"id": 102, "name": "Harshvardhan", "email": "hv@example.com"},
    ]

    # STRICT match
    matched_users = [
        user for user in extended_users
        if user["name"].lower() == name_lower
    ]

    if not matched_users:
        return f"<p>No exact match found for <b>{name}</b></p>"

    user_list_html = "".join(
        f"<li>{user['id']} - {user['name']} - {user['email']}</li>"
        for user in matched_users
    )

    return f"""
    <div>
        <h3>Exact match for: {name}</h3>
        <ul>{user_list_html}</ul>
    </div>
    """

#TODO : python-multipart version checking required for Form data parsing, ensure it's compatible with FastAPI version used

@app.post("/htmx-post/", response_class=HTMLResponse)
def htmx_post(name: str = Form(...), message: str = Form(...)):
    return f"""
    <div>
        <h3>Form Submitted</h3>
        <p><b>Name:</b> {name}</p>
        <p><b>Message:</b> {message}</p>
        <p><b>Status:</b> Success </p>
    </div>
    """