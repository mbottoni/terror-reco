from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from .db import get_db_session
from .models import User
from .security import hash_password, verify_password, generate_csrf_token, validate_csrf_token
from .settings import get_settings


def get_current_user(request: Request, db: Session = Depends(get_db_session)) -> Optional[User]:
	user_id = request.session.get("user_id")
	if not user_id:
		return None
	return db.get(User, user_id)


router = APIRouter(prefix="/auth")


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> HTMLResponse:
	csrf = generate_csrf_token()
	request.session["csrf"] = csrf
	return request.app.state.templates.TemplateResponse("login.html", {"request": request, "csrf": csrf})


@router.post("/login")
async def login(request: Request, response: Response, email: str = Form(...), password: str = Form(...), csrf: str = Form(...), db: Session = Depends(get_db_session)) -> RedirectResponse:
	if not validate_csrf_token(csrf) or csrf != request.session.get("csrf"):
		raise HTTPException(status_code=400, detail="Invalid CSRF")
	user = db.query(User).filter(User.email == email).first()
	if not user or not verify_password(password, user.password_hash):
		raise HTTPException(status_code=401, detail="Invalid credentials")
	request.session["user_id"] = user.id
	return RedirectResponse(url="/", status_code=303)


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request) -> HTMLResponse:
	csrf = generate_csrf_token()
	request.session["csrf"] = csrf
	return request.app.state.templates.TemplateResponse("register.html", {"request": request, "csrf": csrf})


@router.post("/register")
async def register(request: Request, email: str = Form(...), password: str = Form(...), csrf: str = Form(...), db: Session = Depends(get_db_session)) -> RedirectResponse:
	if not validate_csrf_token(csrf) or csrf != request.session.get("csrf"):
		raise HTTPException(status_code=400, detail="Invalid CSRF")
	existing = db.query(User).filter(User.email == email).first()
	if existing:
		raise HTTPException(status_code=409, detail="Email already registered")
	user = User(email=email, password_hash=hash_password(password))
	db.add(user)
	db.commit()
	db.refresh(user)
	request.session["user_id"] = user.id
	return RedirectResponse(url="/", status_code=303)


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
	request.session.clear()
	return RedirectResponse(url="/", status_code=303)
