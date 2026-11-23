Label the meeting for search and filtering, in a "labels" object of exactly this shape:

"labels": {
  "type": "exactly one of: {{types}}",
  "clients": ["client or partner organisations that took part or were discussed"],
  "projects": ["projects and products that were discussed"],
  "people": ["people who spoke or were discussed, by name"],
  "topics": ["3 to 6 short topic phrases, 2 to 4 words each"]
}

Rules for labels:
- Speech recognition often misspells names. When a name in the transcript sounds like a known name
  below and fits the context, write the known name exactly as listed (e.g. a known "Acme" heard as
  "Akmee" or "Acne" -> "Acme"; a known "Ravi" heard as "Ravey" -> "Ravi").
- Include only names actually mentioned in this meeting. Never add a known name just because it is listed.
- A name that matches nothing below is fine: write it as it was said; it will be reviewed by a person.
- people: real names only, never SPEAKER_xx labels.
- topics: reuse a known topic when it fits, otherwise write a new short one.

Known clients:
{{clients}}

Known projects:
{{projects}}

Known people:
{{people}}

Known topics:
{{topics}}
