# services/solicitor_playbooks.py

PLAYBOOKS = {
    "SHORT_LEASE": {
        "category": "tenure",
        "severity": "critical",
        "what_it_means": (
            "The property is leasehold with very little time remaining on the lease. "
            "At this length, the lease is effectively running out."
        ),
        "why_it_matters": (
            "Most mortgage lenders will not lend on a lease of this length, which "
            "severely limits buyer demand and reduces the property's value."
        ),
        "cost_implications": (
            "Legal advice on a lease extension typically costs £500–£1,500. "
            "The premium to extend the lease itself can be substantial and often "
            "runs into tens of thousands of pounds."
        ),
        "unknowns": [
            "Whether the seller is eligible to start a statutory lease extension",
            "The freeholder’s likely premium and terms"
        ],
        "solicitor_actions": [
            "Review the lease and title",
            "Confirm eligibility for a lease extension",
            "Estimate the likely extension cost and timeline"
        ],
        "escalation_rule": "always"
    },

    "PLANNING_CONTRADICTION": {
        "category": "planning",
        "severity": "high",
        "what_it_means": (
            "The seller’s statements about planning or building regulations do not "
            "match the local authority records, and a completion certificate is missing."
        ),
        "why_it_matters": (
            "Without clear evidence of approval and sign-off, there is a risk of "
            "enforcement action, lender refusal, or resale difficulties."
        ),
        "cost_implications": (
            "Solicitor investigation typically costs £300–£800. "
            "Further costs may arise if retrospective approval or negotiations are required."
        ),
        "unknowns": [
            "Whether planning permission was required",
            "Whether building regulations were properly signed off"
        ],
        "solicitor_actions": [
            "Check local authority planning records",
            "Confirm building control sign-off",
            "Advise on risk mitigation options"
        ],
        "escalation_rule": "always"
    }
}
