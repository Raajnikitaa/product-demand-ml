"""
Recommendations Module for DemandWise
======================================
Generates plain-language driver summaries and practical retail advisories
based on model predictions and recent sales indicators.

Advisory policy:
- Only recommend actions that are supported by available data.
- Do not claim inventory coverage is sufficient or aligned — the app has no inventory data.
- Do not claim a specific reorder quantity — the app has no demand forecast in unit terms.
- Surge probability is a ranked risk score, not a guaranteed frequency.
- WATCH status means the model's probability is near the decision threshold — not that
  a surge is likely; monitor before acting.
"""

from typing import Dict, Any, List


def generate_recommendation_and_explanation(
    pred_data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Translates model predictions and feature inputs into an understandable
    retail advisory and natural-language explanation.

    The explanation is based strictly on historical feature values.
    The advisory is cautious and does not fabricate inventory or supplier information.
    """
    is_surge = pred_data.get('is_surge', False)
    badge = pred_data.get('badge_status', 'NORMAL')
    prob = pred_data.get('probability', 0.0)
    thresh = pred_data.get('threshold_used', 0.50)

    feats = pred_data.get('feature_values', {})
    sales_today = feats.get('sales_today', 0.0)
    past_7d = feats.get('past_7d_avg', 0.0)
    past_30d = feats.get('past_30d_avg', 0.0)
    sales_vs_7d = feats.get('sales_vs_7d_avg', 0.0)
    onpromotion = feats.get('onpromotion', 0)
    category = pred_data.get('category', 'this category')

    # 1. Driver factors based on historical observations
    drivers: List[str] = []

    if sales_vs_7d > 0.15:
        pct_above = int(sales_vs_7d * 100)
        drivers.append(
            f"Recent daily volume ({sales_today:.0f} units) is running {pct_above}% above "
            f"the 7-day average ({past_7d:.1f} units/day)."
        )
    elif sales_vs_7d < -0.15:
        pct_below = int(abs(sales_vs_7d) * 100)
        drivers.append(
            f"Recent daily volume ({sales_today:.0f} units) is running {pct_below}% below "
            f"the 7-day average ({past_7d:.1f} units/day)."
        )
    else:
        drivers.append(
            f"Recent daily sales ({sales_today:.0f} units) are tracking closely with "
            f"the 7-day baseline ({past_7d:.1f} units/day)."
        )

    if past_30d > 0:
        ratio_30d = (past_7d - past_30d) / past_30d
        if ratio_30d > 0.10:
            drivers.append(
                f"The 7-day average is {int(ratio_30d * 100)}% above the 30-day baseline, "
                f"indicating upward momentum in recent demand."
            )
        elif ratio_30d < -0.10:
            drivers.append(
                f"The 7-day average is {int(abs(ratio_30d) * 100)}% below the 30-day baseline, "
                f"suggesting recent demand has softened."
            )

    if onpromotion == 1:
        drivers.append("An active promotional flag is recorded for this category on the most recent observation.")
    else:
        drivers.append("No active promotional flag was recorded for this category on the most recent observation.")

    explanation = " ".join(drivers)

    # 2. Suggested Action — cautious, evidence-based, no fabricated inventory claims
    if is_surge:
        advisory = (
            f"The model indicates a potential demand surge for {category}. "
            f"Review recent sales trends and current stock levels before adjusting replenishment. "
            f"Verify supplier delivery lead times if you are considering increasing order quantities."
        )
    elif badge == 'WATCH':
        advisory = (
            f"Demand indicators for {category} are near the model's detection threshold. "
            f"Continue monitoring daily sales closely over the next 2–3 days "
            f"before making any adjustments to orders or stock levels."
        )
    else:
        advisory = (
            f"No demand surge was detected for {category}. "
            f"Continue monitoring normal sales for this category. "
            f"Review the sales trend chart to confirm conditions remain stable."
        )

    return {
        'explanation': explanation,
        'advisory': advisory
    }
