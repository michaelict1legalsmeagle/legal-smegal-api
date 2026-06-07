# services/solicitor_qa_engine.py

from services.solicitor_playbooks import PLAYBOOKS
from services.llm_openrouter import rephrase_for_user

def answer_flag(flag_id, question=None):
    playbook = PLAYBOOKS.get(flag_id)

    if not playbook:
        return {
            "what_it_means": "This issue requires direct legal review.",
            "why_it_matters": "There is not enough information to explain this safely.",
            "cost_implications": "A solicitor would normally review this directly.",
            "what_is_unknown": ["Full legal documentation"],
            "what_a_solicitor_would_do": ["Review the legal pack"],
            "escalation_note": "This would usually be reviewed by a solicitor."
        }

    try:
        explanation = rephrase_for_user(playbook, question)
    except Exception:
        explanation = playbook["what_it_means"]

    return {
        "what_it_means": explanation,
        "why_it_matters": playbook["why_it_matters"],
        "cost_implications": playbook["cost_implications"],
        "what_is_unknown": playbook["unknowns"],
        "what_a_solicitor_would_do": playbook["solicitor_actions"],
        "escalation_note": (
            "This is usually the point where a solicitor would review the documents directly."
            if playbook["escalation_rule"] == "always"
            else ""
        )
    }


def _ensure_contract(resp: dict) -> dict:
    defaults = {
        "what_it_means": "",
        "why_it_matters": "",
        "cost_implications": "",
        "what_is_unknown": [],
        "what_a_solicitor_would_do": [],
        "escalation_note": "",
    }
    out = defaults.copy()
    out.update({k: v for k, v in resp.items() if v is not None})
    return out
