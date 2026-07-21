"""
Regression test for the stale stripe_customer_id auto-recovery fix
(S43-STALE-STRIPE-CUSTOMER) in create_checkout().

app.py itself can't be imported in this test environment (needs supabase,
anthropic, etc. not installed here), so this test extracts the retry logic
into an equivalent standalone function and exercises it with the REAL
stripe.error.InvalidRequestError class (stripe IS installed here), rather
than a hand-rolled stand-in exception — the exact attribute names
(code, param) matter and are worth checking against the real thing.
"""
import stripe


def checkout_with_customer_recovery(customer_id, create_session_fn, create_customer_fn, save_customer_id_fn):
    """Mirrors the retry logic added to create_checkout() in app.py."""
    try:
        return create_session_fn(customer_id)
    except stripe.error.InvalidRequestError as e:
        if getattr(e, "code", None) == "resource_missing" and (
            getattr(e, "param", None) == "customer" or "customer" in str(e).lower()
        ):
            new_customer_id = create_customer_fn()
            save_customer_id_fn(new_customer_id)
            return create_session_fn(new_customer_id)
        raise


class TestStaleStripeCustomerRecovery:
    def test_stale_customer_recovers_and_retries(self):
        calls = {"session": [], "customer_created": 0, "saved_id": None}

        def create_session(cust_id):
            calls["session"].append(cust_id)
            if cust_id == "cus_stale_old_one":
                raise stripe.error.InvalidRequestError(
                    "No such customer: 'cus_stale_old_one'",
                    param="customer",
                    code="resource_missing",
                )
            return {"id": "sess_ok", "url": "https://checkout.stripe.com/ok"}

        def create_customer():
            calls["customer_created"] += 1
            return "cus_fresh_new_one"

        def save_id(cid):
            calls["saved_id"] = cid

        result = checkout_with_customer_recovery(
            "cus_stale_old_one", create_session, create_customer, save_id
        )
        assert result["id"] == "sess_ok"
        assert calls["session"] == ["cus_stale_old_one", "cus_fresh_new_one"]
        assert calls["customer_created"] == 1
        assert calls["saved_id"] == "cus_fresh_new_one"

    def test_valid_customer_never_triggers_recovery(self):
        calls = {"session": 0, "customer_created": 0}

        def create_session(cust_id):
            calls["session"] += 1
            return {"id": "sess_ok"}

        def create_customer():
            calls["customer_created"] += 1
            return "should_never_be_called"

        result = checkout_with_customer_recovery(
            "cus_perfectly_valid", create_session, create_customer, lambda cid: None
        )
        assert result["id"] == "sess_ok"
        assert calls["session"] == 1
        assert calls["customer_created"] == 0

    def test_unrelated_resource_missing_is_not_treated_as_stale_customer(self):
        # resource_missing on a DIFFERENT param (e.g. a bad price_id) is a
        # real, different problem — must not be masked by blindly retrying
        # with a new customer, which would succeed and hide a real bug.
        def create_session(cust_id):
            raise stripe.error.InvalidRequestError(
                "No such price: 'price_totally_wrong'",
                param="line_items[0][price]",
                code="resource_missing",
            )

        try:
            checkout_with_customer_recovery(
                "cus_fine", create_session, lambda: "cus_x", lambda cid: None
            )
            assert False, "should have raised"
        except stripe.error.InvalidRequestError as e:
            assert e.code == "resource_missing"
            assert e.param != "customer"

    def test_non_resource_missing_stripe_error_is_not_retried(self):
        # A card decline, rate limit, etc. is not a stale-customer problem —
        # must propagate normally, not trigger a pointless customer-recreate.
        def create_session(cust_id):
            raise stripe.error.InvalidRequestError(
                "Your card was declined.", param=None, code="card_declined"
            )

        try:
            checkout_with_customer_recovery(
                "cus_fine", create_session, lambda: "cus_x", lambda cid: None
            )
            assert False, "should have raised"
        except stripe.error.InvalidRequestError as e:
            assert e.code == "card_declined"
