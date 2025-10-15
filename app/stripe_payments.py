"""
Stripe payment integration for TerrorReco.
Handles "Buy me a coffee" payments.
"""

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .settings import get_settings

router = APIRouter(prefix="/stripe", tags=["stripe"])
settings = get_settings()

# Initialize Stripe
if settings.STRIPE_SECRET_KEY:
    stripe.api_key = settings.STRIPE_SECRET_KEY

# Get templates from app state
def get_templates():
    from fastapi import Request
    from fastapi.templating import Jinja2Templates
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    TEMPLATES_DIR = BASE_DIR / "templates"
    return Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/coffee", response_class=HTMLResponse)
async def buy_coffee_page(request: Request):
    """Display the buy coffee page."""
    if not settings.STRIPE_PUBLISHABLE_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")
    
    templates = get_templates()
    return templates.TemplateResponse(
        "coffee.html", 
        {
            "request": request,
            "stripe_publishable_key": settings.STRIPE_PUBLISHABLE_KEY,
            "price_id": settings.COFFEE_PRICE_ID
        }
    )


@router.post("/create-checkout-session")
async def create_checkout_session(request: Request):
    """Create a Stripe checkout session for coffee purchase."""
    if not settings.STRIPE_SECRET_KEY or not settings.COFFEE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe not configured")
    
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': settings.COFFEE_PRICE_ID,
                'quantity': 1,
            }],
            mode='payment',
            success_url=str(request.url_for('stripe_success')) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=str(request.url_for('stripe_cancel')),
            metadata={
                'app': 'terror-reco',
                'type': 'coffee'
            }
        )
        
        return {"checkout_url": checkout_session.url}
        
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/success", response_class=HTMLResponse)
async def stripe_success(request: Request, session_id: str = None):
    """Handle successful payment."""
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        templates = get_templates()
        
        return templates.TemplateResponse(
            "coffee_success.html",
            {
                "request": request,
                "session": session,
                "amount": session.amount_total / 100  # Convert from cents
            }
        )
    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/cancel", response_class=HTMLResponse)
async def stripe_cancel(request: Request):
    """Handle cancelled payment."""
    templates = get_templates()
    return templates.TemplateResponse(
        "coffee_cancel.html",
        {"request": request}
    )


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks."""
    if not settings.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        # Here you could save the payment to your database
        # For now, we'll just log it
        print(f"Payment completed: {session['id']}")
    
    return {"status": "success"}
