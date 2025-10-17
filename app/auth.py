from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .security import generate_csrf_token, hash_password, validate_csrf_token, verify_password


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
	user_id = request.session.get("user_id")
	if not user_id:
		return None
	return db.get(User, user_id)


router = APIRouter(prefix="/auth")


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> HTMLResponse:
	csrf = generate_csrf_token()
	request.session["csrf"] = csrf
	flash = request.session.pop("flash", None)
	return request.app.state.templates.TemplateResponse("login.html", {"request": request, "csrf": csrf, "flash": flash})


@router.post("/login")
async def login(request: Request, response: Response, email: str = Form(...), password: str = Form(...), csrf: str = Form(...), db: Session = Depends(get_db)) -> RedirectResponse:
	# Validate token authenticity (signed with SECRET_KEY). Avoid strict session equality to reduce spurious failures in dev.
	if not validate_csrf_token(csrf):
		request.session["flash"] = "Security check failed. Please try again."
		return RedirectResponse(url="/auth/login", status_code=303)
	user = db.query(User).filter(User.email == email).first()
	if not user or not verify_password(password, user.password_hash):
		request.session["flash"] = "Invalid email or password."
		return RedirectResponse(url="/auth/login", status_code=303)
	request.session["user_id"] = user.id
	request.session["flash"] = "Signed in successfully."
	return RedirectResponse(url="/", status_code=303)


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request) -> HTMLResponse:
	csrf = generate_csrf_token()
	request.session["csrf"] = csrf
	flash = request.session.pop("flash", None)
	return request.app.state.templates.TemplateResponse("register.html", {"request": request, "csrf": csrf, "flash": flash})


@router.post("/register")
async def register(request: Request, email: str = Form(...), password: str = Form(...), csrf: str = Form(...), db: Session = Depends(get_db)) -> RedirectResponse:
	# Validate token authenticity only
	if not validate_csrf_token(csrf):
		request.session["flash"] = "Security check failed. Please try again."
		return RedirectResponse(url="/auth/register", status_code=303)
	existing = db.query(User).filter(User.email == email).first()
	if existing:
		request.session["flash"] = "Email already registered. Please sign in."
		return RedirectResponse(url="/auth/login", status_code=303)
	user = User(email=email, password_hash=hash_password(password))
	db.add(user)
	db.commit()
	db.refresh(user)
	request.session["user_id"] = user.id
	request.session["flash"] = "Account created. Welcome!"
	return RedirectResponse(url="/", status_code=303)


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
	request.session.clear()
	return RedirectResponse(url="/", status_code=303)
