# 🧪 Local Stripe Testing Guide for TerrorReco

## Quick Start Testing

### 1. Install Dependencies
`make setup` installs everything the app needs, including `stripe`.

### 2. Set Up Environment Variables
Create a `.env` file in your project root:
```bash
# Stripe Test Keys (get from https://dashboard.stripe.com/test/apikeys)
STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
STRIPE_SECRET_KEY=sk_test_your_secret_key_here
COFFEE_PRICE_ID=price_your_price_id_here

# Optional for webhook testing
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here
```

### 3. Run the automated tests
```bash
pytest tests/test_stripe.py
```

These mock the Stripe SDK and need no keys. The old hand-run scripts that talk to Stripe are in `scripts/manual/` (`manual_stripe_local.py`, `manual_stripe_simple.py`, `debug_stripe.py`); they are not maintained.

## Manual Testing Steps

### Step 1: Start the Server
```bash
uvicorn app.main:app --reload
```
Server will start at: http://localhost:8000

### Step 2: Test the Homepage
1. Open http://localhost:8000
2. Scroll to the footer
3. Verify you see the **Buy me a coffee** link

### Step 3: Test Payment Page
1. Click the coffee link
2. Should redirect to: http://localhost:8000/stripe/coffee
3. Verify you see the payment page with the amounts $3, $5, $10 and "other"

### Step 4: Test Stripe Checkout
1. Pick an amount and click "☕ Support with $N"
2. Should redirect to Stripe checkout page
3. Use test card: **4242 4242 4242 4242**
4. Any future expiry date and CVC
5. Complete the payment

### Step 5: Verify Success
1. Should redirect to success page
2. Should see thank you message
3. Should show payment amount

## Test Card Numbers

| Card Number | Description |
|-------------|-------------|
| 4242 4242 4242 4242 | Visa (Success) |
| 4000 0000 0000 0002 | Visa (Declined) |
| 4000 0000 0000 9995 | Visa (Insufficient funds) |

## Troubleshooting

### "Stripe not configured" Error
- Check your environment variables are set
- Verify API keys are correct
- Make sure you're using test keys (pk_test_, sk_test_)

### "Price ID invalid" Error
- Check COFFEE_PRICE_ID is correct
- Verify the price exists in your Stripe dashboard
- Make sure you're using the right Stripe account (test vs live)

### Server Won't Start
- Check if port 8000 is available
- Try: `uvicorn app.main:app --reload --port 8001`
- Install dependencies: `pip install -e .`

### Payment Page Won't Load
- Check browser console for JavaScript errors
- Verify Stripe publishable key is set
- Make sure you're using HTTPS for Stripe (localhost is OK for testing)

## Webhook Testing (Advanced)

### Using ngrok
1. Install ngrok: https://ngrok.com/download
2. Start ngrok: `ngrok http 8000`
3. Copy the HTTPS URL
4. In Stripe Dashboard, create webhook:
   - URL: `https://your-ngrok-url.ngrok.io/stripe/webhook`
   - Events: `checkout.session.completed`, `charge.refunded`
5. Copy webhook secret to environment

### Using Stripe CLI
1. Install: `brew install stripe/stripe-cli/stripe`
2. Login: `stripe login`
3. Forward: `stripe listen --forward-to localhost:8000/stripe/webhook`
4. Use the webhook secret provided

## What to Look For

✅ **Footer**: Coffee link visible on every page
✅ **Payment Page**: Amount picker with $3, $5, $10 and a custom amount
✅ **Stripe Checkout**: Professional Stripe interface
✅ **Success Page**: Thank you message with amount
✅ **Cancel Page**: Graceful cancellation handling

## Next Steps After Testing

1. **Set the Stripe variables** in `/opt/terror-reco/.env.prod` on the server and restart the app
2. **Set up production webhook** pointing to your domain
3. **Test with real domain** and HTTPS
4. **Switch to live keys** when ready for real payments

## Support

- Stripe Docs: https://stripe.com/docs
- Stripe Test Cards: https://stripe.com/docs/testing
- TerrorReco Issues: Check your app logs

## Automated tests

`tests/test_stripe.py` covers checkout, the success page and the webhook (including signature checks and payment recording) with the Stripe SDK mocked. It runs in CI and needs no keys. The hand-run scripts that talk to Stripe live in `scripts/manual/`.
