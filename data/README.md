# Dataset

Two folders feed the ingestion pipeline:

## `synthetic/`

Six hand-written healthcare policy and patient-education documents created
for this prototype. They cover the topic areas suggested in the assignment:

| File | Topic |
| --- | --- |
| `appointment_scheduling_policy.md` | Booking, cancellations, specialty availability |
| `telehealth_policy.md` | Eligible visit types, refills, technical requirements |
| `medication_refill_policy.md` | Routine refills, controlled substances, after-hours |
| `hipaa_guidelines.md` | PHI definition, sharing rules, patient rights |
| `insurance_eligibility_faq.md` | Network status, referrals, pre-authorisation |
| `discharge_instructions.md` | Activity, wound care, when to seek urgent care |

Each file is structured into numbered sections so that the chunker preserves
natural section boundaries and source citations land on coherent passages.

**No real patient data is used. All content is synthetic and written for
this assignment.**

## `public/`

Optional drop-in folder for additional public healthcare documents — for
example a few short MedlinePlus articles or WHO fact sheets in `.txt`,
`.md`, or `.pdf` form. The ingestion pipeline picks up any supported file
inside `data/` recursively, so adding documents here and re-running
`POST /ingest` is enough to extend the knowledge base.

This folder is gitignored by default since downloaded content can be bulky
and licensing varies by source.
