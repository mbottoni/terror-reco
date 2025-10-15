# Stripe Integration Setup Guide for TerrorReco

## 🚀 Quick Setup Steps

### 1. Install Dependencies
```bash
pip install stripe>=7.0.0
```

### 2. Stripe Dashboard Setup
1. Go to [Stripe Dashboard](https://dashboard.stripe.com/)
2. Create a new account or log in
3. Get your API keys from the Developers section

### 3. Create a Product and Price
1. In Stripe Dashboard, go to Products
2. Create a new product called "Coffee" 
3. Set price to $3.00 USD
4. Copy the Price ID (starts with `price_`)

### 4. Environment Variables
Add these to your `.env` file or Render environment:

```bash
# Stripe Configuration
STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
STRIPE_SECRET_KEY=sk_test_your_secret_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here
COFFEE_PRICE_ID=price_your_price_id_here
```

### 5. Webhook Setup (Optional)
1. In Stripe Dashboard, go to Webhooks
2. Add endpoint: `https://your-domain.com/stripe/webhook`
3. Select events: `checkout.session.completed`
4. Copy the webhook secret

## 🎯 Features Added

✅ **Coffee Button**: Added to homepage with coffee mug emoji ☕
✅ **Payment Page**: Dedicated page for coffee purchase
✅ **Success Page**: Thank you message after payment
✅ **Cancel Page**: Graceful handling of cancelled payments
✅ **Webhook Support**: Ready for payment notifications

## 🔧 Testing

### Test Mode
- Use Stripe test keys (start with `pk_test_` and `sk_test_`)
- Use test card: `4242 4242 4242 4242`
- Any future expiry date and CVC

### Production Mode
- Use live keys (start with `pk_live_` and `sk_live_`)
- Real payments will be processed

## 📱 User Flow

1. User clicks "☕ Buy me a coffee" button on homepage
2. Redirected to `/stripe/coffee` page
3. Clicks payment button to create Stripe checkout session
4. Redirected to Stripe checkout page
5. Completes payment
6. Redirected to success page with thank you message

## 🚨 Security Notes

- Never commit real API keys to version control
- Use environment variables for all sensitive data
- Enable HTTPS in production
- Validate webhook signatures

## 🎨 Customization

The coffee button and pages can be customized by editing:
- `app/templates/index.html` - Homepage button
- `app/templates/coffee.html` - Payment page
- `app/templates/coffee_success.html` - Success page
- `app/templates/coffee_cancel.html` - Cancel page

## 📊 Monitoring

Check Stripe Dashboard for:
- Payment analytics
- Failed payments
- Webhook delivery status
- Customer information
