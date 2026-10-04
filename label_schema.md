# Tactic Label Schema — Second Look

Multi-label classification. A message can carry **1–4 tactics**.

## T1 — Urgency & Scarcity
**Definition:** Creates artificial time pressure or fear of loss to prevent
the target from thinking or verifying.

**Signal phrases:** "within 24 hours", "immediately", "act now", "expires today",
"final warning", "last chance", "account will be suspended"

**Example:** *"Your account will be permanently closed within 24 hours unless
you verify now."*

**Negative example:** *"Sale ends Sunday — 20% off."* (marketing, not scam
unless combined with other tactics)

## T2 — Authority Impersonation
**Definition:** Claims to be from a trusted institution (bank, government,
police, tech company) to lower the target's guard.

**Signal phrases:** "SBI Security Team", "IRS", "Microsoft Support",
"Government of India", "Federal Trade Commission"

**Example:** *"This is the SBI fraud department. We detected unauthorized
activity on your account."*

**Key distinction from T1:** T1 is about *time pressure*. T2 is about
*who* is claiming to speak.

## T3 — Financial Lure
**Definition:** Promises unexpected money, refunds, prizes, or investment
returns to induce action.

**Signal phrases:** "refund", "you've won", "claim your prize",
"guaranteed returns", "double your investment", "unclaimed funds"

**Example:** *"You are eligible for a tax refund of Rs. 15,000. Click
here to claim."*

## T4 — Credential Harvesting
**Definition:** Directs the target to a fake login page or asks for
passwords, OTPs, PINs, or card numbers.

**Signal phrases:** "verify your password", "confirm your identity",
"enter your OTP", "update your KYC", "re-authenticate"

**Example:** *"Click here to verify your account credentials."*

**Note:** T4 often co-occurs with T1 and T2.

## T5 — Payment Redirection
**Definition:** Asks the target to send money via an untraceable or
unusual payment method, or changes payment details.

**Signal phrases:** "wire transfer", "gift card", "bitcoin", "USDT",
"updated bank details", "pay via UPI to this number"

**Example:** *"Please update our payment details — we've switched banks.
Wire to account 123456789."*

## T6 — Secrecy & Isolation
**Definition:** Instructs the target to keep the request confidential,
avoid telling family, or not contact official channels.

**Signal phrases:** "do not tell anyone", "keep this confidential",
"this is a secret investigation", "do not contact the bank"

**Example:** *"This is a covert investigation. Do not discuss with
anyone, including bank staff."*

## T7 — Tech Support / Remote Access
**Definition:** Claims the target's device is infected or compromised and
asks them to install software, grant remote access, or call a number.

**Signal phrases:** "your computer is infected", "call Microsoft support",
"install AnyDesk", "TeamViewer", "your device has a virus"

**Example:** *"Windows Defender detected 5 viruses. Call our toll-free
number immediately."*

## Labeling Guidelines

1. **Multi-label is the norm.** Most scam messages use 2–3 tactics.
2. **Only label tactics actually present**, not implied. If the message
   doesn't pressure you, don't label T1 even if scams usually do.
3. **Borderline cases:** label if a reasonable non-technical person
   would feel the tactic's effect. The target audience defines the threshold.
4. **Negative examples matter.** Label legitimate marketing, bank alerts,
   and delivery notifications as `no_tactic` to prevent false positives.
5. **Keep a "notes" column** for ambiguous cases — useful for the
   limitations section of the README.

## Annotation Format (CSV)

```
message_id,text,t1_urgency,t2_authority,t3_financial,t4_credential,t5_payment,t6_secrecy,t7_tech_support,notes
001,"Urgent: verify now...",1,1,0,1,0,0,0,""
002,"Your package is delayed...",0,0,0,0,0,0,0,"legit carrier"
```

## Inter-Annotator Note

You're solo, so you won't have a second annotator. Compensate by:
- **Labeling in two passes**, 24 hours apart.
- **Disagreement rate** between your two passes = your annotation
  noise estimate. Report it in the README.
- This is a genuine methodological strength, not a weakness —
  most hackathon projects don't measure their own label noise.
