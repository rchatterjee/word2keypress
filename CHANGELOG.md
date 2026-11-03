# Changelog

## 1.0.17
- Now supports Python 3.8–3.12. 
- Regenerated `_keyboard.c` with Cython 0.29.37.
- `array.fromstring` --> `frombytes` (removed in Python 3.9). `keyseq_insert_edits` now requires bytes keys, not str.
- Dropped the `profile=True` Cython directive (doesn't build on 3.12).
