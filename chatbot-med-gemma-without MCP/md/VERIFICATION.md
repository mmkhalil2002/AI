# Patient GUI verification

Generated application: `../python/client_medgemma_patients.py`.
Adapted from `../python/client-med-gemma-2w-data-v2(1).py`.
The reference copy matches the sibling project's original SHA-256:
`9A71E6D28EC8EBB20F9EBB239FE22C6E1A0542DA7461E74518F2D0DB78765A21`.
The existing `.env` was not modified. No production patient records were created.

## Passed

- Syntax parsing and imports.
- Present and empty records roots; missing root and regular-file root exit nonzero before GUI construction.
- Launch from `config/`; copied project resolves relative `.env` paths against its Python directory.
- Real Tkinter widget tests with temporary synthetic patients named MohamedLjalil, P0002, and Patient With Spaces; hidden directories excluded.
- Patient switching clears image and blood selections, prompt, and response.
- X-ray/EKG gallery loading and thumbnail selection; invalid images and malformed JSON remain nonfatal.
- CBC JSON without `title`, including units and reference ranges.
- Missing modality folder remains nonfatal.
- Actions section has no child controls.
- Local HTTP mock exercises `/api/tags` and `/api/generate` in the background worker; patient selector disabled during request and restored afterward.
- Payload contains selected active-patient blood data and original image bytes; cross-patient paths rejected.
- Mock answer appears with clinician-review notice. Timeout and unreachable-server errors handled.

## Limits and launch

GUI widgets were tested programmatically, without visual desktop inspection.
Live MedGemma was not contacted. Unreadable-directory ACL behavior was not tested.
The actual `python/patient_records` directory is missing; normal launch correctly prints the fatal startup error and exits. Restore the real directory, or configure an existing readable directory with `PATIENT_RECORDS_DIR` in `.env`.

Both existing launchers choose the first alphabetical `.py` file, which can select the reference program. Launch the generated GUI explicitly from `python/`:

```cmd
python client_medgemma_patients.py
```
