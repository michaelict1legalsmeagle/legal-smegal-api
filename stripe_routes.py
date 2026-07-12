
# ══════════════════════════════════════════════════════════════════════════════
# STRIPE BILLING — add these routes to app.py
# Add to env vars on Render: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
# ══════════════════════════════════════════════════════════════════════════════

# Add near the top of app.py with other imports:
# import stripe as _stripe

# Add near other os.getenv calls:
# STRIPE_SECRET_KEY     = (os.getenv("STRIPE_SECRET_KEY") or "").strip()
# STRIPE_WEBHOOK_SECRET = (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()

# ── PRICE → PLAN MAPPING (single source of truth) ────────────────────────────
# S-AUDIT-1 (Stripe entitlement fix): this map is the ONLY place plan names are
# derived from. Neither checkout creation nor the webhook trust a client- or
# metadata-supplied `plan` string anymore — plan is always looked up from the
# Stripe price ID that was actually confirmed purchased.
PRICE_TO_PLAN = {
    "price_1SGKAuACdQXaNPBV6Sxywnd4":    "report",
    "price_1TjH94ACdQXaNPBV4urMtc8o":    "starter",
    "price_1TjHA7ACdQXaNPBVTEemQBvu":    "professional",
    "price_1TjHAjACdQXaNPBVV6RUBg5q":    "portfolio",
}


# ── CREATE CHECKOUT SESSION ──────────────────────────────────────────────────
@app.route("/api/billing/checkout", methods=["POST"])
@require_auth
def create_checkout():
    """Create a Stripe Checkout session for subscription or one-off payment."""
    import stripe as _stripe
    _stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    if not _stripe.api_key:
        return jsonify({"error": "Billing not configured"}), 503

    data       = request.get_json(silent=True) or {}
    price_id   = data.get("price_id")
    mode       = data.get("mode", "subscription")   # "subscription" or "payment"
    deal_id    = data.get("deal_id")
    success_url = data.get("success_url", "https://legalsmegal-frontend.onrender.com/legalsmegal-dashboard.html?upgraded=1")
    cancel_url  = data.get("cancel_url",  "https://legalsmegal-frontend.onrender.com/legalsmegal-card.html")

    if not price_id:
        return jsonify({"error": "price_id required"}), 400

    # S-AUDIT-1: price_id must be one we recognise. We no longer accept a
    # client-supplied `plan` at all — it was possible to pair a cheap price_id
    # with an expensive plan string and have the webhook trust it verbatim.
    # The plan the user ends up on is derived exclusively from PRICE_TO_PLAN,
    # both here (for logging/metadata only) and again at webhook time (for
    # the actual entitlement write) using Stripe's own confirmed line items.
    if price_id not in PRICE_TO_PLAN:
        app.logger.warning(f"[STRIPE] checkout rejected — unrecognised price_id={price_id}")
        return jsonify({"error": "Unrecognised price_id"}), 400

    try:
        # Get or create Stripe customer
        profile = supabase.table("profiles").select("stripe_customer_id, email").eq("id", request.user_id).single().execute()
        customer_id = (profile.data or {}).get("stripe_customer_id")

        if not customer_id:
            # Create customer
            user = supabase.auth.admin.get_user(request.user_id)
            email = (user.user.email if user and user.user else None) or ""
            customer = _stripe.Customer.create(
                email=email,
                metadata={"user_id": request.user_id}
            )
            customer_id = customer.id
            supabase.table("profiles").update({"stripe_customer_id": customer_id}).eq("id", request.user_id).execute()

        # Build session params
        session_params = {
            "customer":         customer_id,
            "payment_method_types": ["card"],
            "line_items":       [{"price": price_id, "quantity": 1}],
            "mode":             mode,
            "success_url":      success_url,
            "cancel_url":       cancel_url,
            "metadata": {
                "user_id":  request.user_id,
                # Informational only — the webhook does NOT trust this value.
                # It re-derives the plan from the actual Stripe line items on
                # the completed session before writing any entitlement.
                "plan":     PRICE_TO_PLAN[price_id],
                "deal_id":  deal_id or "",
            },
        }

        session = _stripe.checkout.Session.create(**session_params)
        return jsonify({"ok": True, "session_id": session.id, "url": session.url}), 200

    except _stripe.error.StripeError as e:
        app.logger.error(f"Stripe checkout error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("create_checkout failed")
        return jsonify({"error": str(e)}), 500


# ── STRIPE WEBHOOK ────────────────────────────────────────────────────────────
@app.route("/api/billing/webhook", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events — upgrades user plan on successful payment."""
    import stripe as _stripe
    _stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    payload   = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = _stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except _stripe.error.SignatureVerificationError:
        app.logger.warning("Stripe webhook signature verification failed")
        return jsonify({"error": "Invalid signature"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    event_type = event["type"]
    app.logger.info(f"[STRIPE] event={event_type}")

    if event_type == "checkout.session.completed":
        session  = event["data"]["object"]
        user_id  = session.get("metadata", {}).get("user_id")
        deal_id  = session.get("metadata", {}).get("deal_id") or None
        sub_id   = session.get("subscription")
        customer = session.get("customer")

        if not user_id:
            app.logger.warning("[STRIPE] checkout.session.completed missing user_id")
            return jsonify({"received": True}), 200

        # S-AUDIT-1: derive plan from what Stripe confirms was actually
        # purchased on THIS session — never from metadata, which is
        # attacker-influenceable at checkout-creation time. list_line_items
        # hits Stripe's API directly, so this reflects the real charge.
        plan = None
        try:
            line_items = _stripe.checkout.Session.list_line_items(session["id"], limit=10)
            purchased_price_ids = {
                item["price"]["id"] for item in line_items.get("data", [])
                if item.get("price", {}).get("id")
            }
            resolved_plans = {PRICE_TO_PLAN[p] for p in purchased_price_ids if p in PRICE_TO_PLAN}
            if len(resolved_plans) == 1:
                plan = resolved_plans.pop()
            else:
                app.logger.error(
                    f"[STRIPE] could not resolve a single plan for session={session['id']} "
                    f"— purchased_price_ids={purchased_price_ids} resolved={resolved_plans}"
                )
        except Exception as e:
            app.logger.error(f"[STRIPE] list_line_items failed for session={session['id']}: {e}")

        if plan is None:
            # Fail SAFE, not fail open: an unrecognised or ambiguous purchase
            # must never silently grant entitlement. No plan is written; this
            # event needs manual review in the Stripe dashboard.
            app.logger.error(
                f"[STRIPE] REFUSING to grant entitlement — session={session['id']} "
                f"user={user_id} plan could not be verified from line items"
            )
            return jsonify({"received": True}), 200

        try:
            # Update profile plan
            update = {
                "plan": plan,
                "stripe_customer_id":     customer,
                "stripe_subscription_id": sub_id,
            }
            # For one-off report purchase: mark deal as summary_purchased
            if plan == "report" and deal_id:
                # Store in deal summary_json.meta
                deal = supabase.table("deals").select("summary_json").eq("id", deal_id).single().execute()
                sj = deal.data.get("summary_json") or {}
                sj.setdefault("meta", {})["summary_purchased"] = True
                supabase.table("deals").update({"summary_json": sj}).eq("id", deal_id).execute()
                app.logger.info(f"[STRIPE] summary_purchased set for deal={deal_id}")
            else:
                # Subscription — update user plan
                supabase.table("profiles").update(update).eq("id", user_id).execute()
                app.logger.info(f"[STRIPE] user={user_id} upgraded to plan={plan}")
        except Exception as e:
            app.logger.error(f"[STRIPE] webhook processing error: {e}")

    elif event_type in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub = event["data"]["object"]
        customer_id = sub.get("customer")
        status      = sub.get("status")  # active, canceled, past_due

        if customer_id:
            try:
                # Downgrade to free if subscription cancelled
                if status in ("canceled", "unpaid"):
                    supabase.table("profiles").update({"plan": "free"}).eq("stripe_customer_id", customer_id).execute()
                    app.logger.info(f"[STRIPE] customer={customer_id} downgraded to free (status={status})")
            except Exception as e:
                app.logger.error(f"[STRIPE] subscription update error: {e}")

    return jsonify({"received": True}), 200
